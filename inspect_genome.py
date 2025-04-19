# visualize_genome.py
import sys, os, math
from collections import defaultdict
from graphviz import Digraph
from utils import load_genome
from genome import Genome

class GenomeConfig:      pass
class ReproductionConfig: pass
class StagnationConfig:   pass
class SpeciesSetConfig:   pass
class MainConfig:         pass

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def is_input_key(k):        
    return isinstance(k, int) and k < 0


def visualize_genome(
        filename,
        output_basename="genome_graph",
        *,
        show_disabled=False,
        portrait=True,          # NEW  ← True = top→bottom (rankdir="TB")
        squeeze=True,            # NEW  ← tighter spacing / smaller fonts
        node_names=None,
        node_colors=None,
):
    print(f"[viz] loading genome from: {filename}")
    genome = load_genome(filename)
    if not isinstance(genome, Genome):
        print("! error – not a Genome instance");   return

    if getattr(genome, "config", None):
        in_keys  = getattr(genome.config,  "input_keys",  None)
        out_keys = getattr(genome.config, "output_keys", None)
    else:
        in_keys = out_keys = None

    if in_keys is None or out_keys is None:
        print("! cannot infer keys from genome.config – falling back to config.py")
        try:
            import config as cfg
            in_keys  = [-i - 1 for i in range(cfg.NUM_INPUTS)]
            out_keys = list(range(cfg.NUM_OUTPUTS))
        except (ImportError, AttributeError):
            print("!! fatal – could not determine input/output keys");  return

    in_set, out_set = set(in_keys), set(out_keys)

    if node_names is None:   node_names  = {}
    if node_colors is None:  node_colors = {}

    graph_kwargs = dict(rankdir=("TB" if portrait else "LR"), splines="spline")

    if squeeze:
        graph_kwargs.update(
            ranksep="0.25", 
            nodesep="0.15",   
            ratio="compress", 
        )

    dot = Digraph(
        comment=f"Genome {genome.genome_id}  fitness={getattr(genome,'fitness','N/A')}",
        graph_attr=graph_kwargs,
        node_attr={
            "shape": "circle",
            "fontsize": ("7" if squeeze else "9"),
            "height": ("0.18" if squeeze else "0.25"),
            "width":  ("0.18" if squeeze else "0.25"),
            "style":  "filled",
        },
    )

    drawn_nodes = set()

    AGG_NODE_ID = "AGG_INPUT"
    dot.node(AGG_NODE_ID,
             label=f"Inputs\n({len(in_set)})",
             shape="box",
             fillcolor="lightblue")
    drawn_nodes.add(AGG_NODE_ID)

    hidden_ct, output_ct = 0, 0
    for key, ng in genome.nodes.items():
        if key in in_set:           
            continue
        graph_id = str(key)
        drawn_nodes.add(graph_id)

        attrs = {}
        col = node_colors.get(key)
        if col is None:
            if key in out_set:
                col, attrs["shape"] = "lightgreen", "doublecircle"
                output_ct += 1
            else:
                col = "lightgrey"
                hidden_ct += 1
        attrs["fillcolor"] = col

        lbl = f"{node_names.get(key, key)}"
        dot.node(graph_id, label=lbl, **attrs)

    agg_weights_enabled   = defaultdict(list)   
    agg_weights_disabled  = defaultdict(list)

    for conn in genome.connections.values():
        src, dst = conn.key
        if src not in in_set:               
            continue
        (agg_weights_enabled if conn.enabled else agg_weights_disabled)[dst].append(conn.weight)

    def average_edge_style(weights, enabled=True):
        if not weights:
            return None
        w_avg = sum(weights) / len(weights)
        style = "solid" if enabled else "dashed"
        color = ("darkgreen" if w_avg > 0 else
                 "red"       if w_avg < 0 else "grey")
        if not enabled:
            color = "lightgrey"
        penwidth = str(max(0.3, min(abs(w_avg) * 1.5, 4.0)))
        label = f"{w_avg:.2f}\\n(n={len(weights)})"
        return style, color, penwidth, label

    for dst, w_list in agg_weights_enabled.items():
        if str(dst) not in drawn_nodes:   continue
        style, color, penwidth, label = average_edge_style(w_list, enabled=True)
        dot.edge(AGG_NODE_ID, str(dst), label=label,
                 style=style, color=color, penwidth=penwidth)

    if show_disabled:
        for dst, w_list in agg_weights_disabled.items():
            if str(dst) not in drawn_nodes:   continue
            style, color, penwidth, label = average_edge_style(w_list, enabled=False)
            dot.edge(AGG_NODE_ID, str(dst), label=label,
                     style=style, color=color, penwidth=penwidth)

    enabled_ed, disabled_ed = 0, 0
    for conn in genome.connections.values():
        src, dst = conn.key
        if src in in_set:       
            continue
        if str(src) not in drawn_nodes or str(dst) not in drawn_nodes:
            continue

        style = "solid" if conn.enabled else "dashed"
        enabled_flag = conn.enabled
        color = ("darkgreen" if conn.weight > 0 else
                 "red"       if conn.weight < 0 else "grey")
        if not enabled_flag:
            color = "lightgrey"
        penwidth = str(max(0.3, min(abs(conn.weight) * 1.5, 4.0)))
        dot.edge(str(src), str(dst), label=f"{conn.weight:.2f}",
                 style=style, color=color, penwidth=penwidth)
        if enabled_flag:
            enabled_ed += 1
        else:
            disabled_ed += 1

    print(f"[viz] hidden={hidden_ct}  output={output_ct} "
          f"(+ aggregate input node)")
    print(f"[viz] edges: {enabled_ed} enabled  +{disabled_ed} disabled"
          f"  (+ aggregated input edges)")

    out_dir = os.path.dirname(filename) or "."
    ensure_dir(out_dir)
    out_path_base = os.path.join(out_dir, output_basename)
    dot.render(out_path_base, format="png", view=False, cleanup=True)
    print(f"[viz] saved → {out_path_base}.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_genome.py <genome.pkl> [output_name]")
        sys.exit(1)

    fpath   = sys.argv[1]
    outname = sys.argv[2] if len(sys.argv) > 2 else \
              os.path.splitext(os.path.basename(fpath))[0] + "_graph"

    visualize_genome(fpath, outname, show_disabled=True)
