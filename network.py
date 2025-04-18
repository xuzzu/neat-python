# network.py
"""Handles the activation of a genome to produce network outputs."""

import numpy as np
from typing import List, Dict, Set, Tuple
from genome import Genome 
import config
import time 


def topological_sort(genome: Genome) -> List[int]:
    """
    Performs Kahn's algorithm for topological sorting on enabled connections.
    Returns a list of node IDs in activation order.
    Returns None if a cycle is detected.
    """
    if not genome._needs_sort and genome.activation_order is not None:
        return genome.activation_order

    nodes_to_process = set(genome.nodes.keys())
    edges = set()
    successors = {n: set() for n in nodes_to_process}
    in_degree = {n: 0 for n in nodes_to_process}

    # Include input keys conceptually for edge processing
    all_possible_sources = nodes_to_process.union(set(genome.config.input_keys))

    for conn in genome.connections.values():
        if not conn.enabled: continue
        i, o = conn.key
        # ignore edges whose source is a pure input node
        if i not in nodes_to_process:          # input → hidden/output
            continue                         

        # Ensure nodes exist conceptually 
        if i not in all_possible_sources: continue # Skip connections from unknown sources
        if o not in nodes_to_process: continue # Skip connections to unknown targets

        if (i, o) not in edges:
            edges.add((i, o))
            successors.setdefault(i, set()).add(o) 
            in_degree[o] += 1

    # Ready queue
    queue = [n for n in nodes_to_process if in_degree[n] == 0]
    sorted_nodes = []

    while queue:
        u = queue.pop(0)
        sorted_nodes.append(u)

        # For each neighbor v of u, decrease in-degree
        for v in sorted(list(successors.get(u, set()))): 
             if v in in_degree:
                 in_degree[v] -= 1
                 if in_degree[v] == 0:
                     queue.append(v)

    if len(sorted_nodes) != len(nodes_to_process):
        genome.activation_order = None 
        genome._needs_sort = False 
        return None 

    genome.activation_order = sorted_nodes # Cache the result
    genome._needs_sort = False
    return sorted_nodes

def activate(genome: Genome, inputs: np.ndarray) -> np.ndarray:
    """
    Activates the network defined by the genome for a given input vector.
    Assumes a feed-forward network.
    """
    if len(inputs) != genome.config.num_inputs:
        raise ValueError(f"Expected {genome.config.num_inputs} inputs, got {len(inputs)}")

    # Get activation order (handles cycles/caching)
    activation_order = topological_sort(genome)
    if activation_order is None:
        # Handle cycle - return default output (e.g., zeros) or raise error
        return np.zeros(genome.config.num_outputs)

    node_values: Dict[int, float] = {}

    # Set input node values
    for k, v in zip(genome.config.input_keys, inputs):
        node_values[k] = v

    # Process nodes in topological order
    for node_id in activation_order:
        node = genome.nodes[node_id]
        incoming_values = []
        # Gather inputs from connections targeting this node
        for innov, conn in genome.connections.items():
            if conn.enabled and conn.out_node_id == node_id:
                in_node_id = conn.in_node_id
                weight = conn.weight
                incoming_value = node_values.get(in_node_id, 0.0) 
                incoming_values.append(incoming_value * weight)

        # Aggregate inputs
        aggregation_func = node.get_aggregation_function()
        node_sum = aggregation_func(incoming_values)

        # Apply bias, response, and activation function
        activation_func = node.get_activation_function()
        node_values[node_id] = activation_func(node.bias + node.response * node_sum)

    # Get output values
    output_values = np.array([node_values.get(node_id, 0.0) for node_id in genome.config.output_keys])

    return output_values

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    # Subtract max for numerical stability (prevents exp overflow)
    if x.size == 0: return x # Handle empty array
    x_max = np.max(x, axis=0, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=0, keepdims=True)