# main.py
"""Main script to run the NEAT algorithm on MNIST."""

import logging
import time
import os
import copy
import math
import numpy as np
from network import activate, softmax
from utils import setup_logging, load_mnist_data, set_seed, save_genome, load_genome, ensure_dir
import config as cfg_local
from population import Population, CompleteExtinctionException
from genome import Genome


class GenomeConfig:
    def __init__(self):
        self.input_keys = [-i - 1 for i in range(cfg_local.NUM_INPUTS)]
        self.output_keys = [i for i in range(cfg_local.NUM_OUTPUTS)]
        self.num_inputs = cfg_local.NUM_INPUTS
        self.num_outputs = cfg_local.NUM_OUTPUTS
        self.num_hidden = cfg_local.num_hidden
        self.initial_connection = cfg_local.initial_connection
        self.feed_forward = True

        self.activation_default = cfg_local.activation_default
        self.activation_options = cfg_local.activation_options
        self.aggregation_default = cfg_local.aggregation_default
        self.aggregation_options = cfg_local.aggregation_options

        self.bias_init_mean = cfg_local.bias_init_mean
        self.bias_init_stdev = cfg_local.bias_init_stdev
        self.bias_mutate_power = cfg_local.bias_mutate_power
        self.bias_mutate_rate = cfg_local.bias_mutate_rate
        self.bias_replace_rate = cfg_local.bias_replace_rate
        self.bias_min_value = cfg_local.bias_min_value
        self.bias_max_value = cfg_local.bias_max_value

        self.response_init_mean = cfg_local.response_init_mean
        self.response_init_stdev = cfg_local.response_init_stdev
        self.response_mutate_power = cfg_local.response_mutate_power
        self.response_mutate_rate = cfg_local.response_mutate_rate
        self.response_replace_rate = cfg_local.response_replace_rate
        self.response_min_value = cfg_local.response_min_value
        self.response_max_value = cfg_local.response_max_value

        self.weight_init_mean = cfg_local.weight_init_mean
        self.weight_init_stdev = cfg_local.weight_init_stdev
        self.weight_mutate_power = cfg_local.weight_mutate_power
        self.weight_mutate_rate = cfg_local.weight_mutate_rate
        self.weight_replace_rate = cfg_local.weight_replace_rate
        self.weight_min_value = cfg_local.weight_min_value
        self.weight_max_value = cfg_local.weight_max_value

        self.enabled_default = cfg_local.enabled_default
        self.enabled_mutate_rate = cfg_local.enabled_mutate_rate

        self.PROB_ADD_CONNECTION = cfg_local.PROB_ADD_CONNECTION
        self.PROB_ADD_NODE = cfg_local.PROB_ADD_NODE
        self.PROB_MUTATE_NODE_ATTRS = cfg_local.PROB_MUTATE_NODE_ATTRS
        self.PROB_MUTATE_CONN_ATTRS = cfg_local.PROB_MUTATE_CONN_ATTRS

        self.compatibility_disjoint_coefficient = cfg_local.compatibility_disjoint_coefficient
        self.compatibility_weight_coefficient = cfg_local.compatibility_weight_coefficient

        self.genome_type = Genome 

class ReproductionConfig:
     def __init__(self):
         self.elitism = cfg_local.ELITISM
         self.survival_threshold = 0.2
         self.min_species_size = cfg_local.MIN_SPECIES_SIZE
         self.crossover_rate = cfg_local.CROSSOVER_RATE
         self.tournament_size = cfg_local.TOURNAMENT_SIZE

class StagnationConfig:
    def __init__(self):
        self.species_fitness_func = 'max' # Using max fitness for stagnation check
        self.max_stagnation = cfg_local.STALENESS_THRESHOLD
        # Use ELITISM from reproduction config for species protection count
        self.species_elitism = cfg_local.ELITISM

class SpeciesSetConfig:
     def __init__(self):
          self.compatibility_threshold = cfg_local.COMPATIBILITY_THRESHOLD

class MainConfig:
    def __init__(self):
        self.pop_size = cfg_local.POPULATION_SIZE
        self.fitness_criterion = 'max'
        self.fitness_threshold = 0.90
        self.reset_on_extinction = False
        self.no_fitness_termination = False

        self.genome_type = Genome
        self.genome_config = GenomeConfig()
        self.reproduction_config = ReproductionConfig()
        self.species_set_config = SpeciesSetConfig()
        self.stagnation_config = StagnationConfig()

def run_experiment():
    """Runs the NEAT evolution process."""
    setup_logging()
    set_seed(cfg_local.SEED)

    results_dir = "results"
    ensure_dir(results_dir)

    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
    EVAL_SIZE = 50 
    if len(X_val) > EVAL_SIZE:
         indices = np.random.choice(len(X_val), EVAL_SIZE, replace=False)
         X_val_eval = X_val[indices]
         y_val_eval = y_val[indices]
         logging.info(f"Using subset of validation data: {len(X_val_eval)} samples.")
    else:
         X_val_eval = X_val
         y_val_eval = y_val

    # 2. Initialize Population using config
    logging.info("Initializing population...")
    config_obj = MainConfig() # Create the main config container
    pop = Population(config_obj.pop_size, config_obj) # Pass the main config object
    logging.info(f"Initial population created. Pop size: {len(pop.genomes)}")
    logging.info(f"Initial species count: {len(pop.species)}")

    # 3. Evolutionary Loop
    logging.info("Starting evolution...")
    start_time = time.time()
    best_fitness_overall = -math.inf
    best_genome_overall = None

    try:
        for gen in range(cfg_local.GENERATIONS):
            gen_start_time = time.time()

            pop.run_generation(X_val_eval, y_val_eval)

            # Logging and Stats
            best_genome_gen = pop.get_best_genome()
            best_fitness_gen = best_genome_gen.fitness if best_genome_gen and best_genome_gen.fitness is not None else -math.inf
            num_species = len(pop.species)
            active_species_members = sum(len(s.members) for s in pop.species) if pop.species else 0
            genome_values = pop.genomes.values() if isinstance(pop.genomes, dict) else pop.genomes
            avg_nodes = np.mean([len(g.nodes) for g in genome_values]) if genome_values else 0
            avg_conns = np.mean([len(g.connections) for g in genome_values]) if genome_values else 0
            gen_time = time.time() - gen_start_time

            logging.info(
                f"Gen: {gen:3} | Best Fit: {best_fitness_gen:.4f} | Species: {num_species:3} ({active_species_members} genomes) | "
                f"Avg Nodes: {avg_nodes:6.1f} | Avg Conns: {avg_conns:6.1f} | Gen Time: {gen_time:.2f}s"
            )

            if best_genome_gen and best_fitness_gen > best_fitness_overall:
                best_fitness_overall = best_fitness_gen
                best_genome_overall = copy.deepcopy(best_genome_gen)
                logging.info(f"*** New best fitness overall: {best_fitness_overall:.4f} (Genome ID: {best_genome_overall.genome_id}) ***")
                save_filename = os.path.join(results_dir, f"best_genome_gen_{gen}_fit_{best_fitness_overall:.4f}.pkl")
                save_genome(best_genome_overall, save_filename)
                logging.info(f"Saved new best genome to {save_filename}")

            if not config_obj.no_fitness_termination and best_fitness_overall >= config_obj.fitness_threshold:
                 logging.info(f"Solution found! Fitness {best_fitness_overall:.4f} >= threshold {config_obj.fitness_threshold:.4f}")
                 break

            # Check for extinction inside the loop if run_generation doesn't raise it early enough
            if not pop.genomes or not pop.species:
                 logging.error("Population or species extinct during generation loop.")
                 raise CompleteExtinctionException("Extinction detected in main loop.")

    except CompleteExtinctionException:
        logging.error("Evolution ended due to complete extinction.")
        if best_genome_overall:
             logging.info("Saving the best genome found before extinction.")
             save_filename = os.path.join(results_dir, f"best_genome_before_extinction_fit_{best_fitness_overall:.4f}.pkl")
             save_genome(best_genome_overall, save_filename)

    total_time = time.time() - start_time
    logging.info(f"Evolution finished in {total_time:.2f} seconds.")

    # 4. Evaluate Best Genome on Test Set
    if best_genome_overall:
        logging.info(f"Evaluating best genome (ID: {best_genome_overall.genome_id}) on test set...")
        correct_predictions = 0
        total_predictions = len(X_test)
        test_start_time = time.time()

        if not best_genome_overall.config:
            best_genome_overall.config = config_obj.genome_config

        for i in range(total_predictions):
            inputs = X_test[i]
            true_label = y_test[i]
            try:
                outputs = activate(best_genome_overall, inputs)
                probabilities = softmax(outputs)
                predicted_label = np.argmax(probabilities)
                if predicted_label == true_label:
                    correct_predictions += 1
            except Exception as e:
                 logging.error(f"Best genome failed activation on test sample {i}: {e}")
                 continue

        test_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        test_eval_time = time.time() - test_start_time
        logging.info(f"Best Genome Test Accuracy: {test_accuracy:.4f} ({correct_predictions}/{total_predictions})")
        logging.info(f"Test evaluation time: {test_eval_time:.2f}s")
        logging.info(f"Best Genome Details: Nodes={len(best_genome_overall.nodes)}, Conns={len(best_genome_overall.connections)}")
    else:
        logging.warning("No best genome found.")

if __name__ == "__main__":
    run_experiment()