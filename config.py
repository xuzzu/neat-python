# config.py
"""Hyperparameters and configuration settings."""

import math
import random as rnd 
from functools import reduce
from operator import mul

# Data
NUM_INPUTS = 784
NUM_OUTPUTS = 10

# Population
POPULATION_SIZE = 100
GENERATIONS = 150

# Genome Attributes
# Node Attributes
activation_default = 'sigmoid'
activation_options = ['sigmoid', 'tanh', 'relu', 'identity', 'gauss'] # Ensure functions exist in ACTIVATION_FUNCTIONS
aggregation_default = 'sum'
aggregation_options = ['sum', 'product', 'min', 'max', 'mean'] # Ensure functions exist in AGGREGATION_FUNCTIONS

bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7 # Within PROB_MUTATE_NODE_ATTRS
bias_replace_rate = 0.1 # Within PROB_MUTATE_NODE_ATTRS
bias_min_value = -30.0
bias_max_value = 30.0

response_init_mean = 1.0
response_init_stdev = 0.0
response_mutate_power = 0.1
response_mutate_rate = 0.7 # Within PROB_MUTATE_NODE_ATTRS
response_replace_rate = 0.1 # Within PROB_MUTATE_NODE_ATTRS
response_min_value = -30.0
response_max_value = 30.0

# Connection attributes (weights)
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_init_type = 'gaussian'
weight_mutate_power = 0.5
weight_mutate_rate = 0.8 # Within PROB_MUTATE_CONN_ATTRS
weight_replace_rate = 0.1 # Within PROB_MUTATE_CONN_ATTRS
weight_min_value = -30.0
weight_max_value = 30.0

# Connection attributes (Enabled)
enabled_default = True
enabled_mutate_rate = 0.05 

# Structural mutation probabilities
PROB_ADD_CONNECTION = 0.25
PROB_ADD_NODE = 0.25
# Overall probability that some attribute mutation happens if node/conn is chosen
PROB_MUTATE_NODE_ATTRS = 0.5
PROB_MUTATE_CONN_ATTRS = 0.8

# Speciation
COMPATIBILITY_THRESHOLD = 2.5

# Coefficients for genome distance calculation
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5

STALENESS_THRESHOLD = 15
MIN_SPECIES_SIZE = 2 # Min members for a species to be potentially kept (used in spawn calc)

# Reproduction
ELITISM = 2 # Number of best individuals per species to carry over
CROSSOVER_RATE = 0.75 # Probability of crossover vs cloning

# Selection
TOURNAMENT_SIZE = 3

# Other
SEED = 42
initial_connection = 'minimal'
num_hidden = 0 # Start with 0 hidden nodes

# Activation functions
def sigmoid(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))
def tanh(z):
    z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)
def relu(z): return max(0.0, z)
def identity(z): return z
def gauss(z):
    z = max(-3.4, min(3.4, z))
    return math.exp(-5.0 * z ** 2)

ACTIVATION_FUNCTIONS = {
    'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu, 'identity': identity, 'gauss': gauss,
}

# Aggregation functions

def sum_aggregation(x): return sum(x)
def product_aggregation(x): return reduce(mul, x, 1.0)
def min_aggregation(x): return min(x) if x else 0.0
def max_aggregation(x): return max(x) if x else 0.0
def mean_aggregation(x): return sum(x) / len(x) if x else 0.0

AGGREGATION_FUNCTIONS = {
    'sum': sum_aggregation, 'product': product_aggregation, 'min': min_aggregation,
    'max': max_aggregation, 'mean': mean_aggregation,
}

# Helpers
def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)

def mutate_float(value, config, prefix):
    mutate_rate = getattr(config, f"{prefix}_mutate_rate")
    replace_rate = getattr(config, f"{prefix}_replace_rate")
    mutate_power = getattr(config, f"{prefix}_mutate_power")
    min_val = getattr(config, f"{prefix}_min_value")
    max_val = getattr(config, f"{prefix}_max_value")
    init_mean = getattr(config, f"{prefix}_init_mean")
    init_stdev = getattr(config, f"{prefix}_init_stdev")

    r = rnd.random()
    if r < mutate_rate:
        new_value = value + rnd.gauss(0.0, mutate_power)
    elif r < mutate_rate + replace_rate:
        new_value = rnd.gauss(init_mean, init_stdev)
    else:
        return value
    return clamp(new_value, min_val, max_val)

def mutate_string(value, config, prefix):
    mutate_rate = getattr(config, f"{prefix}_mutate_rate", 0.1)
    options = getattr(config, f"{prefix}_options")

    if rnd.random() < mutate_rate:
        available_options = [opt for opt in options if opt != value]
        return rnd.choice(available_options) if available_options else value
    else:
        return value