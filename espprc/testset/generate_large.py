"""
Generate large ESPPRC instances for performance benchmarking.
Creates n=40, 50, 60 random Euclidean instances.
Oracle RC is null (too large for exact solve).

Usage:
    python generate_large.py
    python generate_large.py --out-dir path/to/instances
"""

import os
import json
import math
import random
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, 'instances')


def euclidean_dist(coords):
    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist[i][j] = int(round(math.sqrt(dx*dx + dy*dy)))
    return dist


def generate_large_instance(n_customers, seed, out_dir):
    rng = random.Random(seed)

    # Random 2D Euclidean coordinates: depot at index 0
    coords = [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n_customers + 1)]
    dist = euclidean_dist(coords)

    # Capacity: random 80~200
    capacity = rng.randint(80, 200)

    # Demands: 5~30 per customer, depot = 0
    demand = [0] + [rng.randint(5, 30) for _ in range(n_customers)]

    # Duals: -20~50 (mix of pos and neg to create interesting pricing subproblems)
    cover_duals = [round(rng.uniform(-20.0, 50.0), 2) for _ in range(n_customers)]

    name = f"large{n_customers}_random"
    inst = {
        "name": name,
        "source": f"Random Euclidean {n_customers}-customer, random duals (seed={seed})",
        "n_customers": n_customers,
        "dist": dist,
        "demand": demand,
        "capacity": capacity,
        "cover_duals": cover_duals,
        "oracle": {
            "optimal_rc": None,
            "optimal_path": None,
            "optimal_cost": None,
            "neg_rc_count": None,
            "method": "null (n too large for exact)"
        }
    }

    out_path = os.path.join(out_dir, f"{name}.json")
    with open(out_path, 'w') as f:
        json.dump(inst, f, indent=2)
    print(f"  -> {name}.json  (n={n_customers}, capacity={capacity})")
    return inst


def main():
    parser = argparse.ArgumentParser(description='Generate large ESPPRC instances')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Generating large ESPPRC instances...")
    sizes = [40, 50, 60]
    seeds = [1001, 1002, 1003]

    for n, seed in zip(sizes, seeds):
        generate_large_instance(n, seed, args.out_dir)

    print(f"\nDone. Instances written to: {args.out_dir}")


if __name__ == '__main__':
    main()
