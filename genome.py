# genome.py
"""Defines the basic building blocks: NodeGene, ConnectionGene, and Genome."""

import random as rnd 
import copy
import math
from typing import Dict, List, Tuple
import config 
from config import mutate_float, mutate_string, clamp 

class NodeGene:
    """Represents a single node (neuron) in the genome."""
    def __init__(self, id: int, node_type: str):
        self.id: int = id
        self.type: str = node_type # INPUT, HIDDEN, OUTPUT
        self.bias: float = 0.0
        self.response: float = 1.0
        self.activation: str = config.activation_default 
        self.aggregation: str = config.aggregation_default

    def get_activation_function(self):
        """Returns the callable activation function."""
        func = config.ACTIVATION_FUNCTIONS.get(self.activation)
        if func is None:
            raise ValueError(f"Unknown activation function name: {self.activation}")
        return func

    def get_aggregation_function(self):
        """Returns the callable aggregation function."""
        func = config.AGGREGATION_FUNCTIONS.get(self.aggregation)
        if func is None:
            raise ValueError(f"Unknown aggregation function name: {self.aggregation}")
        return func

    def init_attributes(self, cfg):
        """Initialize attributes based on config."""
        self.bias = clamp(rnd.gauss(cfg.bias_init_mean, cfg.bias_init_stdev),
                          cfg.bias_min_value, cfg.bias_max_value)
        self.response = clamp(rnd.gauss(cfg.response_init_mean, cfg.response_init_stdev),
                              cfg.response_min_value, cfg.response_max_value)

    def mutate(self, cfg):
        """Mutate node attributes based on config."""
        self.bias = mutate_float(self.bias, cfg, "bias")
        self.response = mutate_float(self.response, cfg, "response")
        self.activation = mutate_string(self.activation, cfg, "activation")
        self.aggregation = mutate_string(self.aggregation, cfg, "aggregation")

    def distance(self, other, cfg):
        """Calculate distance between this node and another node gene."""
        d = abs(self.bias - other.bias) + abs(self.response - other.response)
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d # Raw difference count + attribute diffs

    def __repr__(self):
        return (f"NodeGene(id={self.id}, type={self.type}, bias={self.bias:.3f}, "
                f"resp={self.response:.3f}, act={self.activation}, agg={self.aggregation})")

class ConnectionGene:
    """Represents a connection between two nodes."""
    def __init__(self, in_node_id: int, out_node_id: int, innovation: int):
        self.in_node_id: int = in_node_id
        self.out_node_id: int = out_node_id
        self.key = (in_node_id, out_node_id)
        self.innovation: int = innovation
        self.weight: float = 0.0
        self.enabled: bool = config.enabled_default

    def init_attributes(self, cfg):
        """Initialize attributes based on config."""
        self.weight = clamp(rnd.gauss(cfg.weight_init_mean, cfg.weight_init_stdev),
                           cfg.weight_min_value, cfg.weight_max_value)

    def mutate(self, cfg):
        """Mutate connection attributes based on config."""
        self.weight = mutate_float(self.weight, cfg, "weight")
        if rnd.random() < cfg.enabled_mutate_rate:
            self.enabled = not self.enabled

    def distance(self, other, cfg):
        """Calculate distance based on weight and enabled status."""
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * cfg.compatibility_weight_coefficient # Apply weight coeff here

    def __repr__(self):
        return (f"ConnectionGene(key={self.key}, w={self.weight:.3f}, "
                f"en={self.enabled}, innov={self.innovation})")

class Genome:
    """Represents a complete neural network structure and weights."""
    def __init__(self, genome_id: int):
        self.genome_id: int = genome_id
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {} # Keyed by innovation number
        self.fitness: float = None 
        self.adjusted_fitness: float = None
        self.species_id: int = None
        self.config = None
        # Activation order cache - invalidated by structural mutations
        self.activation_order = None
        self._needs_sort = True

    def set_needs_sort(self):
        self._needs_sort = True
        self.activation_order = None

    def add_node(self, node: NodeGene):
        self.nodes[node.id] = node
        self.set_needs_sort()

    def add_connection(self, connection: ConnectionGene):
        self.connections[connection.innovation] = connection
        self.set_needs_sort()

    def get_new_node_key(self):
        return self.config.get_new_node_key() 

    def get_innovation_number(self, i, o):
        return self.config.get_innovation_number(i, o)
    
    def __getstate__(self):
            """Lightweight state for pickling (workers don’t need full config)."""
            state = self.__dict__.copy()

            cfg = state.get('config')
            if cfg is not None:
                cfg_shallow = copy.copy(cfg)     
                cfg_shallow.get_new_node_key = None
                cfg_shallow.get_innovation_number = None
                state['config'] = cfg_shallow       

            return state
    
    def remove_connection(self, innovation_number: int):
        if innovation_number in self.connections:
            del self.connections[innovation_number]
            self.set_needs_sort()

    def remove_node(self, node_id: int):
         if node_id in self.nodes:
             del self.nodes[node_id]
             conns_to_remove = [innov for innov, conn in self.connections.items()
                                if conn.in_node_id == node_id or conn.out_node_id == node_id]
             for innov in conns_to_remove:
                 if innov in self.connections: del self.connections[innov]
             self.set_needs_sort()

    @staticmethod
    def create_node(cfg, node_id, node_type):
        node = NodeGene(node_id, node_type)
        node.init_attributes(cfg)
        # Apply specific activation/aggregation if needed (e.g., for output layer)
        # if node_type == 'OUTPUT':
        #     node.activation = cfg.output_activation if hasattr(cfg,'output_activation') else cfg.activation_default
        #     node.aggregation = cfg.output_aggregation if hasattr(cfg,'output_aggregation') else cfg.aggregation_default
        return node

    @staticmethod
    def create_connection(cfg, in_id, out_id, innov):
        conn = ConnectionGene(in_id, out_id, innov)
        conn.init_attributes(cfg)
        return conn

    def configure_new(self, genome_cfg):
        """Configure a new genome based on the given configuration."""
        self.config = genome_cfg

        # Create node genes for the output pins
        for node_key in genome_cfg.output_keys:
            self.nodes[node_key] = self.create_node(genome_cfg, node_key, 'OUTPUT')

        # Add hidden nodes if requested
        for _ in range(genome_cfg.num_hidden):
            node_key = genome_cfg.get_new_node_key(self.nodes)
            node = self.create_node(genome_cfg, node_key, 'HIDDEN')
            self.nodes[node_key] = node

        # Initial connection logic
        if genome_cfg.initial_connection == 'minimal':
             if genome_cfg.num_hidden == 0:
                  for in_id in genome_cfg.input_keys:
                      for out_id in genome_cfg.output_keys:
                           innov = genome_cfg.get_innovation_number(in_id, out_id)
                           conn = self.create_connection(genome_cfg, in_id, out_id, innov)
                           self.add_connection(conn)
             else: # Connect inputs->hidden and hidden->outputs
                  hidden_keys = [k for k, n in self.nodes.items() if n.type == 'HIDDEN']
                  # Inputs -> Hidden
                  for in_id in genome_cfg.input_keys:
                       for hid_id in hidden_keys:
                            innov = genome_cfg.get_innovation_number(in_id, hid_id)
                            conn = self.create_connection(genome_cfg, in_id, hid_id, innov)
                            self.add_connection(conn)
                  # Hidden -> Outputs
                  for hid_id in hidden_keys:
                      for out_id in genome_cfg.output_keys:
                           innov = genome_cfg.get_innovation_number(hid_id, out_id)
                           conn = self.create_connection(genome_cfg, hid_id, out_id, innov)
                           self.add_connection(conn)
        else:
             print(f"Warning: Unsupported initial_connection type: {genome_cfg.initial_connection}")

        self.set_needs_sort()

    def configure_crossover(self, genome1: 'Genome', genome2: 'Genome', genome_cfg):
        """ Configure a new genome by crossover from two parent genomes. """
        self.config = genome_cfg
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit node genes
        for key, ng1 in parent1.nodes.items():
            ng2 = parent2.nodes.get(key)
            if ng2 is None:
                self.nodes[key] = copy.deepcopy(ng1)
            else:
                new_node = NodeGene(key, ng1.type)
                new_node.bias = rnd.choice([ng1.bias, ng2.bias])
                new_node.response = rnd.choice([ng1.response, ng2.response])
                new_node.activation = rnd.choice([ng1.activation, ng2.activation])
                new_node.aggregation = rnd.choice([ng1.aggregation, ng2.aggregation])
                self.nodes[key] = new_node

        # Inherit connection genes (using innovation numbers as keys)
        for innov, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(innov)
            if cg2 is None:
                self.connections[innov] = copy.deepcopy(cg1)
            else:
                # Create new connection gene, inheriting attributes randomly
                new_conn = ConnectionGene(cg1.in_node_id, cg1.out_node_id, innov)
                new_conn.weight = rnd.choice([cg1.weight, cg2.weight])
                # Handle enabled gene (neat-python style)
                inherit_enabled = rnd.choice([cg1.enabled, cg2.enabled])
                new_conn.enabled = inherit_enabled
                # Chance to re-enable if disabled in either parent
                if (not cg1.enabled or not cg2.enabled) and rnd.random() < 0.75:
                    new_conn.enabled = True
                self.connections[innov] = new_conn
        self.set_needs_sort()

    def mutate(self, genome_cfg):
        """ Mutates this genome. """
        self.config = genome_cfg

        # Structural Mutations
        # Use overall probabilities from config
        if rnd.random() < genome_cfg.PROB_ADD_NODE:
            self.mutate_add_node(genome_cfg)

        if rnd.random() < genome_cfg.PROB_ADD_CONNECTION:
            self.mutate_add_connection(genome_cfg)

        # Attribute Mutations
        # Mutate node attributes
        for ng in self.nodes.values():
             if rnd.random() < genome_cfg.PROB_MUTATE_NODE_ATTRS: # Overall chance to mutate *some* node attr
                 ng.mutate(genome_cfg) # Gene-level method applies individual attr rates

        # Mutate connection attributes
        for cg in self.connections.values():
             if rnd.random() < genome_cfg.PROB_MUTATE_CONN_ATTRS: # Overall chance to mutate *some* conn attr
                 cg.mutate(genome_cfg) # Gene-level method applies individual attr rates

    def mutate_add_node(self, genome_cfg):
        """Adds a new node by splitting an existing connection."""
        if not self.connections: return

        enabled_connections = {innov: conn for innov, conn in self.connections.items() if conn.enabled}
        if not enabled_connections: return

        innov_to_split = rnd.choice(list(enabled_connections.keys()))
        conn_to_split = self.connections[innov_to_split]

        new_node_id = genome_cfg.get_new_node_key(self.nodes) # Get new ID via config helper
        new_node = self.create_node(genome_cfg, new_node_id, 'HIDDEN')
        self.add_node(new_node) # Adds node and sets _needs_sort

        conn_to_split.enabled = False # Disable original connection

        i, o = conn_to_split.key
        innov1 = genome_cfg.get_innovation_number(i, new_node_id) # Get new innov via config helper
        innov2 = genome_cfg.get_innovation_number(new_node_id, o) # Get new innov via config helper

        conn1 = self.create_connection(genome_cfg, i, new_node_id, innov1)
        conn1.weight = 1.0
        self.add_connection(conn1)

        conn2 = self.create_connection(genome_cfg, new_node_id, o, innov2)
        conn2.weight = conn_to_split.weight
        self.add_connection(conn2)

    def mutate_add_connection(self, genome_cfg):
        """Attempt to add a new connection."""
        # Corrected logic to select from valid node IDs
        possible_outputs_ids = [k for k, n in self.nodes.items()] # Hidden + Output IDs
        possible_inputs_ids = possible_outputs_ids + genome_cfg.input_keys # Input + Hidden + Output IDs

        if not possible_outputs_ids or not possible_inputs_ids: return # Should not happen

        max_attempts = 10
        for _ in range(max_attempts):
            in_node_id = rnd.choice(possible_inputs_ids)
            out_node_id = rnd.choice(possible_outputs_ids) # Cannot be an input node

            # Check validity
            if in_node_id == out_node_id: continue # No self-loops (can be allowed if recurrent)
            node_in = self.nodes.get(in_node_id)
            node_out = self.nodes.get(out_node_id) # Will be None if in_node_id is an input key

            if node_in and node_in.type == 'OUTPUT': continue # Cannot connect from output node

            key = (in_node_id, out_node_id)

            # Check if connection already exists
            connection_exists = any(conn.key == key for conn in self.connections.values())
            if connection_exists: continue

            # Check for cycles if feed_forward
            if genome_cfg.feed_forward:
                # Need a robust cycle check - Using placeholder logic for now
                # Replace with call to a proper graph cycle check if needed
                if self._check_cycles_simple(key):
                     continue

            # Add the new connection
            innov = genome_cfg.get_innovation_number(in_node_id, out_node_id)
            new_conn = self.create_connection(genome_cfg, in_node_id, out_node_id, innov)
            self.add_connection(new_conn)
            return # Success

        # print(f"Genome {self.genome_id}: Failed add connection attempt.")


    def _check_cycles_simple(self, new_connection_key):
        """Placeholder for cycle check. A robust check should use graph traversal."""
        # This is very basic and likely insufficient for complex graphs.
        if not self.config.feed_forward: return False # Cycles allowed
        # Extremely simplified check - assumes activation order is somewhat valid
        i, o = new_connection_key
        if self.activation_order:
            try:
                idx_i = self.activation_order.index(i) if i in self.activation_order else -1
                idx_o = self.activation_order.index(o) if o in self.activation_order else -1
                # If both are in the order, input must come before output
                if idx_i != -1 and idx_o != -1 and idx_i >= idx_o:
                    return True
            except ValueError: pass 
        return False # Assume no cycle if check is inconclusive


    def distance(self, other: 'Genome', genome_cfg) -> float:
        """
        Returns the genetic distance between this genome and the other using neat-python's DefaultGenome method.
        """
        if not self.config: self.config = genome_cfg
        if not other.config: other.config = genome_cfg

        # Calculate node distance
        node_distance = 0.0
        disjoint_nodes = 0
        if self.nodes or other.nodes:
            node_keys_1 = set(self.nodes.keys())
            node_keys_2 = set(other.nodes.keys())

            # Homologous nodes (intersection)
            for k in node_keys_1.intersection(node_keys_2):
                n1 = self.nodes[k]
                n2 = other.nodes[k]
                node_distance += n1.distance(n2, genome_cfg)

            # Disjoint nodes (symmetric difference)
            disjoint_nodes = len(node_keys_1.symmetric_difference(node_keys_2))

            max_nodes = max(len(node_keys_1), len(node_keys_2))
            if max_nodes > 0:
                # Normalize based on neat-python formula
                node_distance = (node_distance +
                                 (genome_cfg.compatibility_disjoint_coefficient * disjoint_nodes)) / max_nodes
            else:
                node_distance = 0.0

        # Calculate connection distance
        connection_distance = 0.0
        disjoint_connections = 0
        if self.connections or other.connections:
            conn_keys_1 = set(self.connections.keys()) # Innovation numbers
            conn_keys_2 = set(other.connections.keys())

            # Homologous connections (intersection)
            for k in conn_keys_1.intersection(conn_keys_2):
                c1 = self.connections[k]
                c2 = other.connections[k]
                connection_distance += c1.distance(c2, genome_cfg) # distance method includes weight coeff

            # Disjoint connections (symmetric difference)
            disjoint_connections = len(conn_keys_1.symmetric_difference(conn_keys_2))

            max_conn = max(len(conn_keys_1), len(conn_keys_2))
            if max_conn > 0:
                # Normalize based on neat-python formula
                connection_distance = (connection_distance +
                                       (genome_cfg.compatibility_disjoint_coefficient * disjoint_connections)) / max_conn
            else:
                connection_distance = 0.0

        # Total distance
        distance = node_distance + connection_distance
        return distance

    def size(self) -> Tuple[int, int]:
        """ Returns genome 'complexity' (number of nodes, number of enabled connections) """
        num_enabled_connections = sum(1 for cg in self.connections.values() if cg.enabled)
        return len(self.nodes), num_enabled_connections

    def __repr__(self):
        fit_str = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return (f"Genome(id={self.genome_id}, nodes={len(self.nodes)}, "
                f"conns={len(self.connections)}, fitness={fit_str})")