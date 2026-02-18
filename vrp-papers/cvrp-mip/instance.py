"""
CVRP Instance Loader
====================
Wraps vrplib to load CVRP instances and read known optimal values.

Dependencies: vrplib, numpy
"""

import os
import re
import numpy as np
import vrplib


def load_instance(filepath):
    """
    Load a CVRP instance from a .vrp file.

    Returns:
        dist: np.ndarray (n x n) integer distance matrix, n = 1 + n_customers
        demand: list[int], demand[0] = 0 (depot)
        capacity: int
        name: str
        n_customers: int
    """
    inst = vrplib.read_instance(filepath)

    dist = np.array(inst['edge_weight'])
    dist = np.rint(dist).astype(int)

    demand = list(inst['demand'].astype(int))
    capacity = int(inst['capacity'])
    name = os.path.basename(filepath).replace('.vrp', '')
    n_customers = len(demand) - 1  # exclude depot

    return dist, demand, capacity, name, n_customers


def get_optimal(filepath_or_name):
    """
    Look up known optimal/best value from .vrp file COMMENT line.

    Args:
        filepath_or_name: either a .vrp file path or an instance name
    Returns:
        optimal cost (int) or None if not found.
    """
    if filepath_or_name.endswith('.vrp') and os.path.exists(filepath_or_name):
        with open(filepath_or_name, 'r') as f:
            for line in f:
                if line.startswith('COMMENT'):
                    m = re.search(r'(?:Best|Optimal)\s+value:\s*(\d+)', line)
                    if m:
                        return int(m.group(1))
                    break
    return None
