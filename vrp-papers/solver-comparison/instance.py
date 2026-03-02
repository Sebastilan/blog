"""
CVRP Instance Loader
====================
Wraps vrplib to load CVRP instances and read known optimal values.
"""

import os
import re
import json
import math
import numpy as np
import vrplib


# Default data directory (self-contained)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_THIS_DIR, 'data')


def _nint(x):
    """TSPLIB standard rounding: nint(x) = (int)(x + 0.5) for x >= 0."""
    return int(x + 0.5)


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

    # For EUC_2D instances, recompute distances from coordinates using
    # TSPLIB standard nint() to match C++ implementation exactly.
    # vrplib's internal computation can differ (e.g. d[5,8] in P-n16-k8:
    # vrplib returns 25.5 -> np.rint=26, but TSPLIB nint(25.495)=25).
    if 'node_coord' in inst and inst.get('edge_weight_type', '') == 'EUC_2D':
        coords = inst['node_coord']
        n = len(coords)
        dist = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                d = _nint(math.sqrt(dx * dx + dy * dy))
                dist[i][j] = d
                dist[j][i] = d
    else:
        dist = np.array(inst['edge_weight'])
        dist = np.rint(dist).astype(int)

    demand = list(inst['demand'].astype(int))
    capacity = int(inst['capacity'])
    # Use filename (e.g. "E-n13-k4") as canonical name for instances.json lookup
    name = os.path.basename(filepath).replace('.vrp', '')
    n_customers = len(demand) - 1  # exclude depot

    return dist, demand, capacity, name, n_customers


def get_optimal(filepath_or_name):
    """
    Look up known optimal/best value.

    Priority: parse "Best value" / "Optimal value" from .vrp COMMENT line.
    Fallback: instances.json lookup by name.

    Args:
        filepath_or_name: either a .vrp file path or an instance name
    Returns:
        optimal cost (int) or None if not found.
    """
    # Try parsing from .vrp file comment
    if filepath_or_name.endswith('.vrp') and os.path.exists(filepath_or_name):
        with open(filepath_or_name, 'r') as f:
            for line in f:
                if line.startswith('COMMENT'):
                    m = re.search(r'(?:Best|Optimal)\s+value:\s*(\d+)', line)
                    if m:
                        return int(m.group(1))
                    break
        name = os.path.basename(filepath_or_name).replace('.vrp', '')
    else:
        name = filepath_or_name

    # No external instances.json — rely on COMMENT line only
    return None
