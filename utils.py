# utils.py
"""Utility functions."""

import logging
import random
import numpy as np
import torch 
from torchvision import datasets, transforms
import config
import pickle 
import os 


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def setup_logging(level=logging.INFO):
    """Configures basic logging."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format)

def load_mnist_data(batch_size=64):
    """Loads MNIST dataset using torchvision."""
    logging.info("Loading MNIST data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Flatten and normalize
    X_train = train_subset.dataset.data[train_subset.indices].numpy().reshape(train_size, -1) / 255.0 #
    y_train = train_subset.dataset.targets[train_subset.indices].numpy()

    X_val = val_subset.dataset.data[val_subset.indices].numpy().reshape(val_size, -1) / 255.0 
    y_val = val_subset.dataset.targets[val_subset.indices].numpy()

    X_test = test_dataset.data.numpy().reshape(len(test_dataset), -1) / 255.0 
    y_test = test_dataset.targets.numpy()

    logging.info(f"Data loaded: Train({X_train.shape}), Val({X_val.shape}), Test({X_test.shape})")
    return X_train, y_train, X_val, y_val, X_test, y_test


def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logging.info(f"Random seed set to {seed}")

def save_genome(genome, filename: str):
    """Saves a Genome object to a file using pickle."""
    try:
        ensure_dir(os.path.dirname(filename))
        with open(filename, 'wb') as f:
            pickle.dump(genome, f)
    except Exception as e:
        logging.error(f"Error saving genome to {filename}: {e}")

def load_genome(filename: str):
    """Loads a Genome object from a file using pickle."""
    try:
        with open(filename, 'rb') as f:
            genome = pickle.load(f)
            logging.info(f"Genome loaded from {filename}")
            return genome
    except FileNotFoundError:
        logging.error(f"Genome file not found: {filename}")
        return None
    except Exception as e:
        logging.error(f"Error loading genome from {filename}: {e}")
        return None