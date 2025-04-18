# Neuroevolution for MNIST Classification

This project is an experimental implementation of the NEAT (NeuroEvolution of Augmenting Topologies) algorithm applied to the classic MNIST handwritten digit classification task. The goal is to evolve both the topology (structure) and weights of a neural network to achieve good classification performance.

## Requirements

*   Python 3.7+
*   NumPy
*   PyTorch & Torchvision (for loading the MNIST dataset)
*   Graphviz (both the system package and the Python library for visualization)

## Running the Experiment

You can run the experiment with already defined or tuned parameters. Important to note that this implementation is quite slow and can take hours to achieve >50% accuracy.

1.  **Configure Parameters (Optional but Important):**
    *   Open `config.py`.
    *   Tune `COMPATIBILITY_THRESHOLD`, which impacts speciation and is key to avoiding stagnation (should be around 1.5 - 3.5)
    *   Adjust `POPULATION_SIZE` and `GENERATIONS` (i.e., number of "models" and "epochs", respectively)
    *   Modify mutation rates (`PROB_ADD_NODE`, `PROB_ADD_CONNECTION`, `PROB_MUTATE_NODE_ATTRS`, `PROB_MUTATE_CONN_ATTRS`, `weight_mutate_power`, etc.) to influence exploration.

2.  **Run the main script:**
    ```bash
    python main.py
    ```
3.  **Monitor Output:**
    *   The script will log progress to the console, showing the current generation, best fitness, number of species, average node/connection counts, and time per generation.
    *   The MNIST dataset will be downloaded to a `./data` directory if it doesn't exist.
    *   Best-performing genomes will be saved as `.pkl` files in the `results/` directory, named like `best_genome_gen_X_fit_Y.pkl`.

4.  **Termination:**
    *   The evolution will run for the number of `GENERATIONS` specified in `config.py`.
    *   It may terminate early if a genome reaches the `fitness_threshold` (if `no_fitness_termination` is `False`).
    *   It might also terminate if complete extinction occurs.
    *   After finishing, the script will evaluate the best genome found overall on the MNIST test set and print the final accuracy.

## Visualization

A script is provided to visualize the structure of saved genomes, focusing on the hidden and output layers (inputs are omitted because of large number of nodes/connections).

1.  **Run the visualization script:**
    ```bash
    python inspect_genome.py results/<name_of_saved_genome>.pkl [optional_output_name]
    ```
    *   Replace `<name_of_saved_genome>.pkl` with the actual filename of a saved genome.

2.  **Options:**
    *   `--show-disabled`: Add this flag to also draw disabled connections (as dashed gray lines).

3.  **Output:**
    *   A PNG image file (e.g., `results/<name_of_saved_genome>_graph_ho.png`) will be created in the same directory as the input `.pkl` file. This image shows the hidden nodes (circles), output nodes (double circles), and the connections between them.
