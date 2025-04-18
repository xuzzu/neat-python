# mutation.py
"""Mutation operators for modifying genomes."""

import random
import copy
from genome import Genome, NodeGene, ConnectionGene
import config

def mutate(genome: Genome, population_info: dict):
    """Applies various mutations to a genome based on probabilities."""
    if random.random() < config.PROB_ADD_NODE:
        mutate_add_node(genome, population_info)

    if random.random() < config.PROB_ADD_CONNECTION:
        mutate_add_connection(genome, population_info)

    if random.random() < config.PROB_MUTATE_WEIGHT:
        mutate_weights(genome)

def mutate_weights(genome: Genome):
    """Mutates the weights of existing connections."""
    for conn in genome.connections.values():
        if random.random() < config.WEIGHT_MUTATION_RATE:
            # Perturb weight
            perturbation = random.gauss(0, config.WEIGHT_PERTURB_STD_DEV)
            conn.weight += perturbation
        else:
            # Assign new random weight
            conn.weight = random.uniform(*config.WEIGHT_RANDOM_RANGE)
        # Clamp weights if necessary

def mutate_add_connection(genome: Genome, population_info: dict):
    """Adds a new connection between two previously unconnected nodes."""
    possible_starts = [n.id for n in genome.nodes.values() if n.type != 'OUTPUT']
    possible_ends = [n.id for n in genome.nodes.values() if n.type != 'INPUT']

    if not possible_starts or not possible_ends:
        return # Cannot add connection

    max_attempts = 10 # Avoid infinite loops in dense graphs
    for _ in range(max_attempts):
        start_node_id = random.choice(possible_starts)
        end_node_id = random.choice(possible_ends)

        # Basic Checks
        if start_node_id == end_node_id: continue 
        if genome.nodes[start_node_id].type == 'OUTPUT': continue 
        if genome.nodes[end_node_id].type == 'INPUT': continue 

        # Check if connection already exists
        connection_exists = False
        for conn in genome.connections.values():
            if (conn.in_node_id == start_node_id and conn.out_node_id == end_node_id):
                connection_exists = True
                break
        if connection_exists: continue

        # Check for cycles (simple feed-forward check)
        if creates_cycle(genome, end_node_id, start_node_id):
            continue

        # Passed Checks - add connection
        new_weight = random.uniform(*config.WEIGHT_RANDOM_RANGE)
        innovation_number = population_info['get_innovation_number'](start_node_id, end_node_id)
        new_conn = ConnectionGene(start_node_id, end_node_id, new_weight, True, innovation_number)
        genome.add_connection(new_conn)
        # print(f"Added connection: {new_conn}")
        return # Success



def mutate_add_node(genome: Genome, population_info: dict):
    """Adds a new node by splitting an existing connection (NEAT-style)."""
    if not genome.connections:
        return 

    enabled_connections = [conn for conn in genome.connections.values() if conn.enabled]
    if not enabled_connections:
        # print(f"Genome {genome.genome_id}: No enabled connections to split.") 
        return 

    # Choose a connection to split
    conn_to_split = random.choice(enabled_connections)

    # Disable the old connection
    conn_to_split.enabled = False

    # Create the new node
    new_node_id = population_info['get_next_node_id']()
    new_node = NodeGene(new_node_id, 'HIDDEN', config.ACTIVATION_FUNCTION_HIDDEN)
    genome.add_node(new_node)

    # Create connection from original input to new node
    weight_in_new = 1.0 # Weight of 1 for the input connection to the new node
    innov_in_new = population_info['get_innovation_number'](conn_to_split.in_node_id, new_node_id)
    conn_in_new = ConnectionGene(conn_to_split.in_node_id, new_node_id, weight_in_new, True, innov_in_new)
    genome.add_connection(conn_in_new)

    # Create connection from new node to original output
    weight_new_out = conn_to_split.weight # Inherit original weight for the output connection
    innov_new_out = population_info['get_innovation_number'](new_node_id, conn_to_split.out_node_id)
    conn_new_out = ConnectionGene(new_node_id, conn_to_split.out_node_id, weight_new_out, True, innov_new_out)
    genome.add_connection(conn_new_out)

    # print(f"Added node {new_node_id} splitting {conn_to_split.innovation}")

def creates_cycle(genome: Genome, start_node_id: int, end_node_id: int) -> bool:
    """
    Performs a basic DFS check to see if a path exists from start_node_id to end_node_id.
    Used by mutate_add_connection to prevent creating immediate cycles in a feed-forward assumption.
    """
    if start_node_id == end_node_id:
        return True # Direct self-loop attempt

    path = {start_node_id}
    visited = {start_node_id}
    stack = [conn.out_node_id for conn in genome.connections.values() if conn.enabled and conn.in_node_id == start_node_id]

    while stack:
        current_node_id = stack.pop()
        if current_node_id == end_node_id:
            return True # Found a path back

        if current_node_id not in visited:
            visited.add(current_node_id)
            # Add neighbors to stack
            for conn in genome.connections.values():
                 if conn.enabled and conn.in_node_id == current_node_id:
                     if conn.out_node_id not in visited: # Avoid revisiting nodes already explored in this path
                         stack.append(conn.out_node_id)
    return False