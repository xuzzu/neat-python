# population.py
"""Manages the population of genomes, speciation, and reproduction."""

import copy
import logging
import math
import random as rnd
from functools import partial
from itertools import count
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set, Tuple
import numpy as np
from genome import Genome
from network import activate, softmax

worker_X_val = None
worker_y_val = None

def _next_node_fn(pop, node_dict=None):
    return pop.get_next_node_id()

def _innov_fn(pop, in_id, out_id):
    return pop.get_innovation_number(in_id, out_id)

class GenomeDistanceCache(object):
    def __init__(self, genome_config):
        self.distances = {}
        self.genome_config = genome_config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome1, genome2):
        g1_id = genome1.genome_id
        g2_id = genome2.genome_id
        key = tuple(sorted((g1_id, g2_id)))
        d = self.distances.get(key)
        if d is None:
            if not genome1.config: genome1.config = self.genome_config
            if not genome2.config: genome2.config = self.genome_config
            d = genome1.distance(genome2, self.genome_config)
            self.distances[key] = d
            self.misses += 1
        else:
            self.hits += 1
        return d

class Species:
    """Represents a species within the population."""
    def __init__(self, species_id: int, generation: int):
        self.id: int = species_id
        self.representative: Genome = None 
        self.members: Dict[int, Genome] = {} # {gid: genome}
        self.fitness: float = None 
        self.adjusted_fitness: float = None 
        self.created = generation
        self.last_improved = generation
        self.best_fitness_achieved: float = -math.inf
        self.staleness: int = 0 

    def update(self, representative: Genome, members: Dict[int, Genome]):
        self.representative = representative
        self.members = members

    def sort_members(self):
        valid_members = [(gid, g) for gid, g in self.members.items() if g.fitness is not None and g.fitness > -math.inf]
        invalid_members = [(gid, g) for gid, g in self.members.items() if g.fitness is None or g.fitness <= -math.inf]

        sorted_valid = sorted(valid_members, key=lambda item: item[1].fitness, reverse=True)
        return sorted_valid + invalid_members


    def get_best_genome(self) -> Genome:
        if not self.members: return None
        valid_members = [g for g in self.members.values() if g.fitness is not None and g.fitness > -math.inf]
        return max(valid_members, key=lambda g: g.fitness) if valid_members else None

class DefaultReproduction: 
    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            af = max(0.0, af)
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0: spawn += c
            elif d > 0: spawn += 1
            elif d < 0: spawn -= 1

            # Clamp spawn to be at least min_species_size if possible, but not negative
            spawn_amounts.append(max(0, spawn)) 

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested
        total_spawn = sum(spawn_amounts)
        if total_spawn == 0:
            num_species = len(adjusted_fitness)
            if num_species == 0: return []
            per_species = max(min_species_size, pop_size // num_species)
            spawn_amounts = [per_species] * num_species
            current_total = sum(spawn_amounts)
            diff = pop_size - current_total
            if diff != 0 and spawn_amounts:
                 spawn_amounts[-1] = max(min_species_size, spawn_amounts[-1] + diff)

            return spawn_amounts

        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        # Ensure total spawn doesn't exceed pop_size due to rounding mins
        final_total_spawn = sum(spawn_amounts)
        over_spawn = final_total_spawn - pop_size
        if over_spawn > 0:
             # Reduce spawn amounts, typically from largest spawns first
             spawn_indices = sorted(range(len(spawn_amounts)), key=lambda k: spawn_amounts[k], reverse=True)
             for i in range(over_spawn):
                 idx_to_reduce = spawn_indices[i % len(spawn_indices)]
                 if spawn_amounts[idx_to_reduce] > min_species_size:
                     spawn_amounts[idx_to_reduce] -= 1

        return spawn_amounts

class CompleteExtinctionException(Exception):
    pass


def init_worker(X_val_data, y_val_data):
    """Initializer function for pool workers."""
    global worker_X_val, worker_y_val
    worker_X_val = X_val_data
    worker_y_val = y_val_data

def evaluate_single_genome(genome):
    """
    Evaluates a single genome using worker-local global data.
    Args:
        genome (Genome): The genome object to evaluate.
    Returns:
        tuple: (genome_id, fitness_score)
    """
    global worker_X_val, worker_y_val
    genome_id = genome.genome_id # Get ID early

    if worker_X_val is None or worker_y_val is None:
        logging.error(f"Worker {genome_id}: Validation data not initialized.")
        return genome_id, -math.inf

    correct_predictions = 0
    total_predictions = len(worker_X_val)

    if total_predictions == 0:
        return genome_id, 0.0

    try:
        for i in range(total_predictions):
            inputs = worker_X_val[i]
            true_label = worker_y_val[i]

            outputs = activate(genome, inputs)
            if outputs is None or np.isnan(outputs).any() or np.isinf(outputs).any():
                 correct_predictions = 0
                 break

            probabilities = softmax(outputs)
            if probabilities is None or np.isnan(probabilities).any() or np.isinf(probabilities).any():
                 correct_predictions = 0
                 break

            predicted_label = np.argmax(probabilities)
            if predicted_label == true_label:
                correct_predictions += 1

        fitness = correct_predictions / total_predictions

    except Exception as e:
        logging.error(f"Genome {genome_id} failed during activation/evaluation in worker: {e}", exc_info=False)
        fitness = -math.inf

    return genome_id, fitness

class Population:
    def __init__(self, population_size: int, config_obj):
        self.population_size = population_size
        self.config = config_obj 
        self.genome_config = config_obj.genome_config 
        self.genome_config.get_new_node_key = partial(_next_node_fn, self)
        self.genome_config.get_innovation_number = partial(_innov_fn, self)
        self.genomes: Dict[int, Genome] = {} # {id: genome}
        self.species: List[Species] = [] 
        self.genome_to_species: Dict[int, int] = {} # genome_id -> species_id
        self.generation: int = 0
        self.next_genome_id: int = 0
        self.next_species_id: int = 1 

        # Global Innovation Tracking 
        max_node_key = max(max(self.genome_config.output_keys, default=-1),
                           max((-k for k in self.genome_config.input_keys), default=-1))
        self.node_indexer = count(max_node_key + 1)
        self.current_innovation_number: int = 0
        self.innovation_map: Dict[Tuple[int, int], int] = {}

        # Create initial population
        self.genomes = self.create_new(self.genome_config.genome_type,
                                        self.genome_config,
                                        self.population_size)

        # Initial speciation
        self.speciate()

    def get_next_genome_id(self) -> int:
        gid = self.next_genome_id
        self.next_genome_id += 1
        return gid

    def get_next_species_id(self) -> int:
        sid = self.next_species_id
        self.next_species_id += 1
        return sid

    def get_innovation_number(self, in_node_id: int, out_node_id: int) -> int:
        key = (in_node_id, out_node_id)
        innov = self.innovation_map.get(key)
        if innov is None:
            innov = self.current_innovation_number
            self.innovation_map[key] = innov
            self.current_innovation_number += 1
        return innov

    def get_next_node_id(self, node_dict=None) -> int:
        nid = next(self.node_indexer)
        return nid

    def create_new(self, genome_type, genome_config, num_genomes):
        """Creates the initial population, ensuring innovations are tracked."""
        new_genomes = {}
        if num_genomes == 0:
            return new_genomes

        # Create the first genome 
        key = self.get_next_genome_id()
        g = genome_type(key)
        g.config = genome_config
        g.configure_new(genome_config) 
        new_genomes[key] = g

        for _ in range(num_genomes - 1):
            key = self.get_next_genome_id()
            g_copy = copy.deepcopy(g)
            g_copy.genome_id = key
            g_copy.config = genome_config 
            g_copy.mutate(genome_config)
            new_genomes[key] = g_copy

        return new_genomes


    def evaluate_fitness(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluates the fitness of each genome using multiprocessing with an initializer."""
        num_workers = cpu_count()

        tasks = list(self.genomes.values()) 

        if not tasks:
            logging.warning("No genomes to evaluate.")
            return

        try:
            with Pool(processes=num_workers,
                      initializer=init_worker,
                      initargs=(X_val, y_val)) as pool:
                results = pool.map(evaluate_single_genome, tasks)

        except Exception as e:
            logging.error(f"Multiprocessing pool error during fitness evaluation: {e}", exc_info=True)
            results = [(g.genome_id, -math.inf) for g in tasks]

        # Assign fitness back to the main genome dictionary
        fitness_map = {genome_id: fitness for genome_id, fitness in results}
        evaluated_ids = set()
        for gid, genome in self.genomes.items():
            genome.fitness = fitness_map.get(gid, -math.inf) # Assign fitness or penalty
            evaluated_ids.add(gid)

        result_ids = {res_gid for res_gid, fit in results}
        missing_ids = result_ids - evaluated_ids
        if missing_ids:
             logging.warning(f"Fitness results received for unknown genome IDs: {missing_ids}")

        logging.info("Fitness evaluation complete.")


    def speciate(self):
        """Groups genomes into species based on genomic distance."""
        population = self.genomes
        if not population: # Handle empty population
            self.species = []
            self.genome_to_species = {}
            return

        compatibility_threshold = self.config.species_set_config.compatibility_threshold
        distances = GenomeDistanceCache(self.genome_config)
        new_representatives = {} # sid -> representative_genome_id
        new_members = {} # sid -> list_of_member_ids

        # Use representatives from the previous generation for comparison
        current_representatives = {s.id: s.representative for s in self.species if s.representative is not None}
        species_history = {s.id: (s.created, s.last_improved, s.best_fitness_achieved)
                           for s in self.species}
        temp_species_map = {s.id: s for s in self.species}

        unspeciated_set = set(population.keys())

        # Find new representatives for existing species
        for sid, old_rep in current_representatives.items():
            # Ensure old_rep is still in the population dictionary
            if old_rep.genome_id not in population:
                 continue

            candidates = []
            # Check only against currently unspeciated genomes
            current_unspeciated = unspeciated_set.intersection(population.keys())
            for gid in current_unspeciated:
                g = population[gid]
                # Ensure genomes have config for distance calculation
                if not old_rep.config: old_rep.config = self.genome_config
                if not g.config: g.config = self.genome_config
                d = distances(old_rep, g)
                candidates.append((d, gid))

            if not candidates: continue

            ignored_rdist, new_rep_id = min(candidates, key=lambda x: x[0])
            if new_rep_id in population and new_rep_id in unspeciated_set:
                new_representatives[sid] = new_rep_id
                new_members[sid] = [new_rep_id]
                unspeciated_set.remove(new_rep_id)

        # Partition remaining members
        remaining_unspeciated = list(unspeciated_set)
        rnd.shuffle(remaining_unspeciated) # Process in random order

        for gid in remaining_unspeciated:
            if gid not in population: continue
            g = population[gid]
            best_sid = -1
            min_dist = compatibility_threshold # Initialize with threshold

            # Find the species with the most similar representative
            for sid, rep_id in new_representatives.items():
                 if rep_id not in population: continue
                 rep = population[rep_id]
                 # Ensure configs are set
                 if not rep.config: rep.config = self.genome_config
                 if not g.config: g.config = self.genome_config
                 d = distances(rep, g)
                 if d < min_dist:
                     min_dist = d
                     best_sid = sid

            if best_sid != -1:
                # Assign to the best matching species.
                new_members.setdefault(best_sid, []).append(gid)
            else:
                # No species is similar enough, create a new species.
                sid = self.get_next_species_id()
                new_representatives[sid] = gid
                new_members[sid] = [gid]
                # Create Species object and restore history
                s = Species(sid, self.generation)
                created, last_improved, best_fitness = species_history.get(sid, (self.generation, self.generation, -math.inf))
                s.created = created
                s.last_improved = last_improved
                s.best_fitness_achieved = g.fitness if g.fitness is not None else -math.inf
                temp_species_map[sid] = s


        # Update species list and genome_to_species map
        self.genome_to_species = {}
        updated_species_list = []
        for sid, rep_id in new_representatives.items():
            member_ids = new_members.get(sid, [])
            if not member_ids: continue

            member_dict = {gid: population[gid] for gid in member_ids if gid in population}
            if not member_dict: continue

            representative_genome = population.get(rep_id)
            if not representative_genome: continue

            s = temp_species_map.get(sid)
            if not s:
                 s = Species(sid, self.generation)
                 created, last_improved, best_fitness = species_history.get(sid, (self.generation, self.generation, -math.inf))
                 s.created = created
                 s.last_improved = last_improved
                 s.best_fitness_achieved = representative_genome.fitness if representative_genome.fitness is not None else -math.inf
                 temp_species_map[sid] = s

            s.update(representative_genome, member_dict)

            for gid in member_ids:
                self.genome_to_species[gid] = sid
                if gid in population: population[gid].species_id = sid

            updated_species_list.append(s)

        self.species = updated_species_list


    def reproduce(self, spawns: Dict[int, int]) -> Dict[int, Genome]:
        """Creates the next generation of genomes."""
        next_generation: Dict[int, Genome] = {}
        genome_cfg = self.genome_config

        # Get species eligible for reproduction
        living_species = [s for s in self.species if s.id in spawns and spawns[s.id] > 0 and s.members]
        if not living_species: return {}

        all_parents = []
        for s in living_species:
            num_to_spawn = spawns[s.id]
            if num_to_spawn <= 0: continue

            sorted_members = s.sort_members() # List of (gid, genome)
            if not sorted_members: continue

            num_elites = min(self.config.reproduction_config.elitism, len(sorted_members), num_to_spawn)

            # Add elites
            for i in range(num_elites):
                 gid, elite_genome = sorted_members[i]
                 if gid not in next_generation:
                     next_generation[gid] = copy.deepcopy(elite_genome)

            # --- Crossover and Mutation ---
            num_offspring = num_to_spawn - num_elites
            if num_offspring <= 0: continue

            # Apply survival threshold to select parents
            survival_cutoff = max(2, int(math.ceil(self.config.reproduction_config.survival_threshold * len(sorted_members))))
            potential_parents = [g for gid, g in sorted_members[:survival_cutoff]] # Get genomes

            if not potential_parents: continue

            all_parents.extend(potential_parents)

            s_offspring_count = 0
            while s_offspring_count < num_offspring:
                parent1 = self.tournament_selection(potential_parents) # Use the fixed method
                if not parent1: continue

                child = None
                # Ensure potential_parents has at least 2 for crossover attempt
                if len(potential_parents) > 1 and rnd.random() < self.config.reproduction_config.crossover_rate:
                    parent2 = self.tournament_selection(potential_parents)
                    if parent2 and parent1.genome_id != parent2.genome_id:
                        gid = self.get_next_genome_id()
                        child = genome_cfg.genome_type(gid)
                        # Ensure parents have config before crossover if needed
                        if not parent1.config: parent1.config = genome_cfg
                        if not parent2.config: parent2.config = genome_cfg
                        child.configure_crossover(parent1, parent2, genome_cfg)
                    else: # Fallback to cloning
                        gid = self.get_next_genome_id()
                        child = copy.deepcopy(parent1)
                        child.genome_id = gid
                        child.config = genome_cfg # Ensure config is set
                else: # Clone
                    gid = self.get_next_genome_id()
                    child = copy.deepcopy(parent1)
                    child.genome_id = gid
                    child.config = genome_cfg # Ensure config is set

                if child:
                    child.mutate(genome_cfg)
                    if gid not in next_generation:
                        next_generation[gid] = child
                        s_offspring_count += 1

        # Fill remaining population slots if needed
        current_size = len(next_generation)
        if current_size < self.population_size:
             needed = self.population_size - current_size
             if next_generation:
                  if not all_parents:
                       all_parents = list(next_generation.values())
                  if all_parents:
                       for _ in range(needed):
                           parent = rnd.choice(all_parents)
                           gid = self.get_next_genome_id()
                           child = copy.deepcopy(parent)
                           child.genome_id = gid
                           child.config = genome_cfg # Set config
                           child.mutate(genome_cfg)
                           if gid not in next_generation: # Avoid rare collisions
                               next_generation[gid] = child

        # Ensure exact population size
        while len(next_generation) > self.population_size:
            gid_to_remove = rnd.choice(list(next_generation.keys()))
            del next_generation[gid_to_remove]

        return next_generation

    def tournament_selection(self, participants: List[Genome]) -> Genome:
        """Selects the best genome from a random tournament."""
        if not participants: return None
        # Ensure tournament size is not larger than number of participants
        actual_tournament_size = min(self.config.reproduction_config.tournament_size, len(participants)) # Use config
        if actual_tournament_size <= 0: return None

        # Ensure fitness is not None for participants
        valid_participants = [p for p in participants if p.fitness is not None and p.fitness > -math.inf]
        if not valid_participants: return None # No valid participants to select from

        # Ensure sample size is not larger than population
        sample_size = min(actual_tournament_size, len(valid_participants))
        tournament = rnd.sample(valid_participants, sample_size)

        return max(tournament, key=lambda g: g.fitness)

    def run_generation(self, X_val: np.ndarray, y_val: np.ndarray):
        """Runs one full generation cycle."""
        if isinstance(self.genomes, list): # Ensure it's a dict
            self.genomes = {g.genome_id: g for g in self.genomes}
        if not self.genomes:
             raise CompleteExtinctionException("Population is empty at start of generation.")

        # 1. Evaluate Fitness
        genome_list_tuples = list(self.genomes.items())
        self.evaluate_fitness(X_val, y_val) # Pass only validation data

        # 2. Speciate
        self.speciate()
        if not self.species:
            if self.config.reset_on_extinction:
                logging.warning("Population extinct after speciation, resetting.")
                self.genomes = self.create_new(self.genome_config.genome_type, self.genome_config, self.population_size)
                self.speciate()
                if not self.species: raise CompleteExtinctionException("Extinction after reset.")
            else:
                raise CompleteExtinctionException("Extinction: No species survived speciation.")
            return

        # 3. Stagnation and fitness adjustment
        stale_species_ids = set()
        species_stats = [] 

        all_fitnesses = [g.fitness for g in self.genomes.values() if g.fitness is not None and g.fitness > -math.inf]
        if not all_fitnesses:
             min_fitness, max_fitness, fitness_range = 0.0, 0.0, 1.0
        else:
             min_fitness = min(all_fitnesses)
             max_fitness = max(all_fitnesses)
             fitness_range = max(1.0, max_fitness - min_fitness)

        for s in self.species:
            if not s.members: continue
            member_fitnesses = [g.fitness for g in s.members.values() if g.fitness is not None and g.fitness > -math.inf]
            if not member_fitnesses:
                 s.adjusted_fitness = 0.0
                 s.staleness += 1
                 if s.staleness >= self.config.stagnation_config.max_stagnation:
                     stale_species_ids.add(s.id)
                 species_stats.append({'id': s.id, 'adj_fit': 0.0, 'size': len(s.members)})
                 continue

            current_max_fitness = max(member_fitnesses)
            if current_max_fitness > s.best_fitness_achieved:
                s.best_fitness_achieved = current_max_fitness
                s.last_improved = self.generation
                s.staleness = 0
            else:
                s.staleness += 1

            if s.staleness >= self.config.stagnation_config.max_stagnation:
                stale_species_ids.add(s.id)

            species_avg_fitness = sum(member_fitnesses) / len(member_fitnesses)
            norm_avg_fitness = (species_avg_fitness - min_fitness) / fitness_range
            s.adjusted_fitness = max(0.0, norm_avg_fitness) / len(s.members)

            species_stats.append({'id': s.id, 'adj_fit': s.adjusted_fitness, 'size': len(s.members)})

        # Remove stale species (with elitism protection)
        num_species_total = len(self.species)
        num_stale = len(stale_species_ids)
        num_non_stale = num_species_total - num_stale
        min_species_protected = self.config.reproduction_config.elitism

        final_stale_ids = set()
        if num_non_stale < min_species_protected and stale_species_ids:
             num_to_protect = min_species_protected - num_non_stale
             stale_species_list = [(s.id, s.best_fitness_achieved) for s in self.species if s.id in stale_species_ids]
             stale_species_list.sort(key=lambda x: x[1], reverse=True)
             protected_stale_ids = {sid for sid, fit in stale_species_list[:num_to_protect]}
             final_stale_ids = stale_species_ids - protected_stale_ids
        else:
             final_stale_ids = stale_species_ids

        if final_stale_ids:
             self.species = [s for s in self.species if s.id not in final_stale_ids]
             species_stats = [stats for stats in species_stats if stats['id'] not in final_stale_ids]

        if not self.species:
            if self.config.reset_on_extinction:
                logging.warning("Population extinct after stagnation, resetting.")
                self.genomes = self.create_new(self.genome_config.genome_type, self.genome_config, self.population_size)
                self.speciate()
                if not self.species: raise CompleteExtinctionException("Extinction after reset.")
            else:
                raise CompleteExtinctionException("Extinction: No species survived stagnation.")
            return

        # 4. Calculate spawning sizes
        adj_fitnesses = [max(0.0, s['adj_fit']) for s in species_stats]
        previous_sizes = [s['size'] for s in species_stats]
        min_spawn_size = max(self.config.reproduction_config.min_species_size,
                             self.config.reproduction_config.elitism)

        spawn_amounts = DefaultReproduction.compute_spawn(adj_fitnesses, previous_sizes,
                                                           self.population_size, min_spawn_size)
        spawns = {stats['id']: spawn for stats, spawn in zip(species_stats, spawn_amounts)}

        # 5. Reproduce
        new_population_dict = self.reproduce(spawns)
        self.genomes = new_population_dict

        self.generation += 1


    def get_best_genome(self) -> Genome:
        """Returns the genome with the highest raw fitness."""
        if not self.genomes: return None
        valid_genomes = [g for g in self.genomes.values() if g.fitness is not None and g.fitness > -math.inf]
        return max(valid_genomes, key=lambda g: g.fitness) if valid_genomes else None
