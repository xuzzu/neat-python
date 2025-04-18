# visualize_genome.py

import sys
import os
import math 
from graphviz import Digraph
from utils import load_genome 
from genome import Genome, NodeGene, ConnectionGene

def is_input_key(key):
    return key < 0

class GenomeConfig: pass
class ReproductionConfig: pass
class StagnationConfig: pass
class SpeciesSetConfig: pass
class MainConfig: pass 

def is_input_key(key):
    return isinstance(key, int) and key < 0

def visualize_genome(filename, output_filename="genome_graph", show_disabled=False, node_names=None, node_colors=None):
    """
    Loads a genome and generates a visualization using Graphviz,
    ignoring input nodes and connections originating from them.
    """
    print(f"Loading Genome: {filename}")
    genome = load_genome(filename)

    if not genome: return
    if not isinstance(genome, Genome): print(f"Error: Not a Genome instance (type: {type(genome)})"); return

    input_keys = None
    output_keys = None
    if hasattr(genome, 'config') and genome.config:
        input_keys = getattr(genome.config, 'input_keys', None)
        output_keys = getattr(genome.config, 'output_keys', None)
    if input_keys is None or output_keys is None:
        print("Warning: Cannot get keys from genome.config, trying config.py")
        try:
            import config as cfg_local
            input_keys = [-i - 1 for i in range(cfg_local.NUM_INPUTS)]
            output_keys = [i for i in range(cfg_local.NUM_OUTPUTS)]
        except (ImportError, AttributeError):
            print("Fatal Error: Cannot determine input/output keys.")
            return

    input_keys_set = set(input_keys)
    output_keys_set = set(output_keys)

    print("\nCreating graph (Hidden & Output Only) ")
    if node_names is None: node_names = {}
    if node_colors is None: node_colors = {}

    node_attrs = {'shape': 'circle', 'fontsize': '9', 'height': '0.2', 'width': '0.2', 'style': 'filled'}
    dot = Digraph(comment=f'Genome {genome.genome_id} Fitness {getattr(genome, "fitness", "N/A"):.4f}',
                  node_attr=node_attrs, graph_attr={'rankdir': 'LR', 'splines': 'spline'})

    # 1. Add ONLY Hidden and Output nodes from genome.nodes
    drawn_nodes = set()
    hidden_node_ids = []
    output_node_ids = []

    for node_key, ng in genome.nodes.items():
        if node_key in input_keys_set: continue # *** SKIP INPUT KEYS ***

        node_graph_id = str(node_key)
        drawn_nodes.add(node_graph_id)
        name = node_names.get(node_key, node_graph_id)
        attrs = {}
        color = node_colors.get(node_key)

        if color is None:
            if node_key in output_keys_set:
                color = 'lightgreen'
                attrs['shape'] = 'doublecircle'
                output_node_ids.append(node_graph_id)
            elif ng.type == 'HIDDEN':
                color = 'lightgrey'
                attrs['shape'] = 'circle'
                hidden_node_ids.append(node_graph_id)
            else:
                print(f"Skipping node {node_key} with type {ng.type}")
                drawn_nodes.remove(node_graph_id)
                continue

        attrs['fillcolor'] = color
        label = f"Node {name}\n{ng.activation}\n{ng.aggregation}"
        dot.node(node_graph_id, label=label, **attrs)

    # 2. Add Edges, ignoring those starting from an input node
    num_enabled, num_disabled = 0, 0

    for conn in genome.connections.values():
        in_key, out_key = conn.key
        weight, enabled = conn.weight, conn.enabled

        # *** SKIP EDGES FROM INPUTS ***
        if in_key in input_keys_set:
            continue

        source_node_graph_id = str(in_key)
        target_node_graph_id = str(out_key)

        # Ensure BOTH source and target nodes were actually drawn
        if source_node_graph_id not in drawn_nodes or target_node_graph_id not in drawn_nodes:
            # print(f"Skipping edge {in_key}->{out_key}: Source or target node not drawn.")
            continue

        # Draw edge if enabled or show_disabled is True
        if enabled or show_disabled:
            style = 'solid' if enabled else 'dashed'
            color = 'grey'
            if enabled:
                 color = 'darkgreen' if weight > 0 else 'red' if weight < 0 else 'grey'
                 penwidth_val = max(0.3, min(abs(weight) * 1.5, 4.0))
                 num_enabled += 1
            else: # Disabled style
                 color = 'lightgrey'
                 penwidth_val = 0.2
                 num_disabled += 1

            penwidth = str(penwidth_val)
            label_str = f'{weight:.2f}'

            dot.edge(source_node_graph_id, target_node_graph_id, label=label_str,
                     style=style, color=color, penwidth=penwidth)


    print(f"Graph Nodes: {len(hidden_node_ids)} Hidden, {len(output_node_ids)} Output (Inputs omitted)")
    print(f"Graph Edges: {num_enabled} Enabled (+{num_disabled} Disabled) shown (Input connections omitted)")

    # Render graph
    try:
        result_dir = os.path.dirname(filename) or '.'
        ensure_dir(result_dir)
        output_path_base = os.path.join(result_dir, output_filename)
        dot.render(output_path_base, format='png', view=False, cleanup=True)
        print(f"Graph saved to {output_path_base}.png")
    except Exception as e:
         print(f"Error rendering graph: {e}")
         dot_source_path = output_path_base + ".gv"
         try: dot.save(dot_source_path); print(f"DOT source saved to {dot_source_path}")
         except Exception as e_save: print(f"Error saving DOT source: {e_save}")

def ensure_dir(directory):
    if not os.path.exists(directory):
        try: os.makedirs(directory); print(f"Created directory: {directory}")
        except OSError as e: print(f"Error creating directory {directory}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_genome.py <path_to_genome.pkl> [output_filename_base]")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        sys.exit(1)

    if len(sys.argv) > 2:
        base_output_name = sys.argv[2]
    else:
        base_output_name = os.path.splitext(os.path.basename(filepath))[0] + "_graph"

    visualize_genome(filepath, base_output_name, show_disabled=True)