"""
ESPPRC Test Set Generator
=========================
Generate ESPPRC instances from CVRP benchmarks + hand-crafted cases.
Each instance: graph + demands + capacity + dual vector + verified optimal.

Usage:
    python generate_testset.py              # generate all
    python generate_testset.py --quick      # tiny instances only (fast)
    python generate_testset.py --up-to 15   # instances with n <= 15
"""

import sys
import os
import json
import time
import argparse

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', '..', 'src', 'python', '01_basic_bpc'))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCRIPT_DIR)

from instance import load_instance
from rmp import RMP, make_initial_columns
from pricing import solve_pricing
from espprc_oracle import enumerate_oracle, mip_oracle

BENCH_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', '..', 'benchmarks', 'cvrp'))
OUT_DIR = os.path.join(SCRIPT_DIR, 'instances')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Utilities ───────────────────────────────────────────────────────────────

def save_instance(inst, filename):
    """Save instance dict to JSON."""
    path = os.path.join(OUT_DIR, filename)
    with open(path, 'w') as f:
        json.dump(inst, f, indent=2)
    rc = inst['oracle']['optimal_rc']
    rc_str = f"{rc:.4f}" if rc is not None else "N/A"
    print(f"  -> {filename}  (n={inst['n_customers']}, opt_rc={rc_str})")


def run_oracle(dist, demand, capacity, duals, n_customers,
               oracle_time=120, cross_validate_threshold=20):
    """
    Run appropriate oracle(s) and return result dict.
    For n <= threshold: enumerate + MIP cross-validation.
    For n > threshold: MIP only.
    """
    result = {}

    # Enumeration (with route limit for safety)
    if n_customers <= cross_validate_threshold:
        t0 = time.time()
        erc, epath, ecost, neg_count, complete = enumerate_oracle(
            dist, demand, capacity, duals, n_customers)
        enum_time = time.time() - t0

        if complete:
            result['enumerate'] = {
                'optimal_rc': erc, 'optimal_path': epath,
                'optimal_cost': ecost, 'neg_rc_count': neg_count,
                'time': round(enum_time, 2),
            }

    # MIP
    t0 = time.time()
    mrc, mpath, mcost, mstatus = mip_oracle(
        dist, demand, capacity, duals, n_customers, time_limit=oracle_time)
    mip_time = time.time() - t0
    result['mip'] = {
        'optimal_rc': mrc, 'optimal_path': mpath,
        'optimal_cost': mcost, 'status': mstatus,
        'time': round(mip_time, 2),
    }

    # Cross-validate
    if 'enumerate' in result and mstatus == 'optimal':
        erc_val = result['enumerate']['optimal_rc']
        if erc_val is not None and mrc is not None:
            gap = abs(erc_val - mrc)
            assert gap < 0.1, (
                f"Oracle mismatch: enum={erc_val:.6f}, mip={mrc:.6f}")

    # Build canonical oracle
    if 'enumerate' in result:
        canon = result['enumerate']
        method = 'enumerate+mip' if mstatus == 'optimal' else 'enumerate'
    else:
        canon = result['mip']
        method = f"mip ({mstatus})"

    return {
        'optimal_rc': (round(float(canon['optimal_rc']), 6)
                       if canon['optimal_rc'] is not None else None),
        'optimal_path': canon['optimal_path'],
        'optimal_cost': (int(canon['optimal_cost'])
                         if canon['optimal_cost'] is not None else None),
        'neg_rc_count': canon.get('neg_rc_count'),
        'method': method,
    }


# ── Hand-crafted instances ──────────────────────────────────────────────────

def generate_tiny3():
    """T3: 3 customers, from conftest."""
    print("\n[Tiny3] 3 customers, hand-crafted")
    dist = [[0, 3, 4, 5], [3, 0, 5, 4], [4, 5, 0, 3], [5, 4, 3, 0]]
    demand = [0, 2, 3, 2]
    capacity = 4
    n = 3

    for name, duals, source in [
        ('tiny3_initial',   [6.0, 8.0, 10.0], 'T3 hand-crafted, initial LP duals'),
        ('tiny3_converged', [6.0, 8.0, 6.0],  'T3 hand-crafted, CG converged duals'),
    ]:
        oracle = run_oracle(dist, demand, capacity, duals, n)
        save_instance({
            'name': name, 'source': source,
            'n_customers': n, 'dist': dist,
            'demand': demand, 'capacity': capacity,
            'cover_duals': duals, 'oracle': oracle,
        }, f'{name}.json')


def generate_tiny5():
    """5 customers, hand-crafted Euclidean graph."""
    print("\n[Tiny5] 5 customers, hand-crafted")
    # Depot(0,0), C1(3,0), C2(0,4), C3(3,4), C4(6,0), C5(6,4)
    dist = [
        [0, 3, 4, 5, 6, 7],
        [3, 0, 5, 4, 3, 5],
        [4, 5, 0, 3, 7, 6],
        [5, 4, 3, 0, 5, 3],
        [6, 3, 7, 5, 0, 4],
        [7, 5, 6, 3, 4, 0],
    ]
    demand = [0, 1, 2, 1, 2, 1]
    capacity = 4
    n = 5

    np.random.seed(42)
    duals = np.random.uniform(2, 8, n).round(2).tolist()

    oracle = run_oracle(dist, demand, capacity, duals, n)
    save_instance({
        'name': 'tiny5_random', 'source': 'Hand-crafted 5-customer, random duals (seed=42)',
        'n_customers': n, 'dist': dist,
        'demand': demand, 'capacity': capacity,
        'cover_duals': duals, 'oracle': oracle,
    }, 'tiny5_random.json')


def generate_small8():
    """8 customers, random Euclidean graph."""
    print("\n[Small8] 8 customers, random graph")
    np.random.seed(123)
    coords = np.random.randint(0, 50, (9, 2))  # depot + 8 customers
    N = 9
    dist = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dx = int(coords[i][0]) - int(coords[j][0])
            dy = int(coords[i][1]) - int(coords[j][1])
            dist[i][j] = round((dx**2 + dy**2) ** 0.5)

    demand = [0, 2, 3, 1, 2, 3, 1, 2, 1]
    capacity = 6
    n = 8

    np.random.seed(456)
    duals = np.random.uniform(5, 20, n).round(2).tolist()

    oracle = run_oracle(dist, demand, capacity, duals, n)
    save_instance({
        'name': 'small8_random',
        'source': 'Random Euclidean 8-customer, random duals (seed=456)',
        'n_customers': n, 'dist': dist,
        'demand': demand, 'capacity': capacity,
        'cover_duals': duals, 'oracle': oracle,
    }, 'small8_random.json')


# ── CVRPLIB instances via Column Generation ─────────────────────────────────

def run_cg_snapshots(filepath, iter_snapshots, max_iter=100):
    """
    Run CG on a CVRP instance, capture dual vectors at specified iterations
    and at convergence.

    Args:
        iter_snapshots: list of iteration numbers to capture
        max_iter: max CG iterations

    Returns:
        (dist, demand, capacity, name, n_customers, snapshots_dict)
    """
    dist, demand, capacity, name, n_customers = load_instance(filepath)
    initial_cols = make_initial_columns(dist, n_customers)
    rmp = RMP(n_customers, initial_cols)

    snapshots = {}

    for iteration in range(1, max_iter + 1):
        obj, lambdas, cover_duals, edge_duals, status = rmp.solve()
        if status != 2:
            print(f"  RMP status {status} at iter {iteration}, stopping")
            break

        has_art = rmp.has_artificial()

        if iteration in iter_snapshots:
            tag = f'iter{iteration}'
            snapshots[tag] = {
                'cover_duals': [float(d) for d in cover_duals],
                'rmp_obj': float(obj),
                'has_artificial': has_art,
            }
            print(f"  Snapshot iter {iteration}: obj={obj:.2f}, art={has_art}")

        cols = solve_pricing(dist, demand, capacity, cover_duals, n_customers)
        if not cols:
            snapshots['converged'] = {
                'cover_duals': [float(d) for d in cover_duals],
                'rmp_obj': float(obj),
                'has_artificial': has_art,
            }
            print(f"  Converged at iter {iteration}: obj={obj:.2f}")
            break

        rmp.add_columns([(c, v, p) for c, v, p, r in cols])

    return dist, demand, capacity, name, n_customers, snapshots


def generate_from_cvrp(cvrp_file, name_prefix, iter_snapshots,
                       max_iter=100, oracle_time=120):
    """Generate ESPPRC instances from a CVRP file via CG dual extraction."""
    filepath = os.path.join(BENCH_DIR, cvrp_file)
    if not os.path.exists(filepath):
        print(f"  SKIP: {cvrp_file} not found")
        return

    dist, demand, capacity, inst_name, n_cust = load_instance(filepath)
    dist_list = dist.tolist() if isinstance(dist, np.ndarray) else dist
    demand_list = [int(d) for d in demand]

    # For large instances (n > 15), skip CG — bitmask pricing too slow
    use_cg = len(iter_snapshots) > 0 and n_cust <= 15

    if use_cg:
        print(f"\n[{name_prefix}] {cvrp_file} (CG mode)")
        _, _, _, _, _, snapshots = \
            run_cg_snapshots(filepath, iter_snapshots, max_iter)
    else:
        print(f"\n[{name_prefix}] {cvrp_file} (analytic duals, n={n_cust})")
        snapshots = {}

    # Always add analytic duals from initial columns: pi_i = 2 * dist[0][i]
    initial_duals = [2.0 * float(dist[0][i]) for i in range(1, n_cust + 1)]
    snapshots['initial'] = {
        'cover_duals': initial_duals,
        'rmp_obj': sum(initial_duals),
        'has_artificial': False,
    }

    # Add perturbed duals (random scaling, seed = n_cust)
    np.random.seed(n_cust * 7)
    scale = np.random.uniform(0.3, 1.2, n_cust)
    perturbed_duals = [round(d * s, 2) for d, s in zip(initial_duals, scale)]
    snapshots['perturbed'] = {
        'cover_duals': perturbed_duals,
        'rmp_obj': sum(perturbed_duals),
        'has_artificial': False,
    }

    for tag, snap in snapshots.items():
        name = f"{name_prefix}_{tag}"
        duals = snap['cover_duals']

        print(f"  Oracle for {name} (n={n_cust})...")
        t0 = time.time()
        oracle = run_oracle(dist, demand_list, int(capacity), duals,
                            n_cust, oracle_time=oracle_time)
        elapsed = time.time() - t0
        print(f"    RC={oracle['optimal_rc']}, method={oracle['method']}, "
              f"time={elapsed:.1f}s")

        inst = {
            'name': name,
            'source': f'{inst_name}, {tag} duals',
            'n_customers': n_cust,
            'dist': dist_list,
            'demand': demand_list,
            'capacity': int(capacity),
            'cover_duals': duals,
            'oracle': oracle,
        }
        if 'rmp_obj' in snap:
            inst['cg_info'] = {
                'rmp_obj': snap['rmp_obj'],
                'has_artificial': snap['has_artificial'],
            }
        save_instance(inst, f'{name}.json')


# ── Manifest ────────────────────────────────────────────────────────────────

def write_manifest():
    """Write manifest.json indexing all instances."""
    entries = []
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.json') and f != 'manifest.json':
            with open(os.path.join(OUT_DIR, f)) as fp:
                inst = json.load(fp)
            entries.append({
                'file': f,
                'name': inst['name'],
                'n_customers': inst['n_customers'],
                'oracle_rc': inst['oracle']['optimal_rc'],
                'oracle_method': inst['oracle']['method'],
            })
    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as f:
        json.dump(entries, f, indent=2)
    print(f"\nManifest: {len(entries)} instances")


# ── Main ────────────────────────────────────────────────────────────────────

# (cvrp_file, prefix, iter_snapshots, max_iter, oracle_time_sec)
CVRP_CONFIGS = [
    # Early iters have neg RC (useful for testing); converged has RC>=0
    ('E-n13-k4.vrp',  'en13k4',  [1, 2],     100,  60),
    ('P-n16-k8.vrp',  'pn16k8',  [1],         50,   60),
    # n > 15: bitmask pricing too slow for CG, use analytic duals
    ('P-n19-k2.vrp',  'pn19k2',  [],  1,  120),   # analytic only
    ('P-n22-k2.vrp',  'pn22k2',  [],  1,  180),   # analytic only
    ('A-n32-k5.vrp',  'an32k5',  [],  1,  300),   # analytic only
]


def main():
    parser = argparse.ArgumentParser(description='ESPPRC Test Set Generator')
    parser.add_argument('--quick', action='store_true',
                        help='Tiny instances only (n <= 8)')
    parser.add_argument('--up-to', type=int, default=9999,
                        help='Max n_customers to generate')
    parser.add_argument('--min-n', type=int, default=0,
                        help='Skip CVRP instances with n < this')
    parser.add_argument('--skip-tiny', action='store_true',
                        help='Skip hand-crafted tiny instances')
    args = parser.parse_args()

    print("=" * 60)
    print("ESPPRC Test Set Generator")
    print("=" * 60)

    # Phase 1: Hand-crafted
    if not args.skip_tiny:
        generate_tiny3()
        generate_tiny5()
        generate_small8()

    if args.quick:
        write_manifest()
        return

    # Phase 2: CVRPLIB via CG
    for cvrp_file, prefix, iters, max_iter, oracle_time in CVRP_CONFIGS:
        # Estimate n from filename: X-nNN-kK -> NN-1 customers
        n_est = int(cvrp_file.split('-')[1].replace('n', '')) - 1
        if n_est > args.up_to:
            print(f"\n  SKIP {cvrp_file} (n={n_est} > --up-to {args.up_to})")
            continue
        if n_est < args.min_n:
            print(f"\n  SKIP {cvrp_file} (n={n_est} < --min-n {args.min_n})")
            continue
        generate_from_cvrp(cvrp_file, prefix, iters, max_iter, oracle_time)

    write_manifest()
    print("\nDone.")


if __name__ == '__main__':
    main()
