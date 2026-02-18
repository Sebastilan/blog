"""
CVRP MIP Benchmark Runner
==========================
Runs MIP formulations (MTZ, SCF, 2CF, MCF, DFJ) on selected CVRPLIB instances.
Collects metrics and outputs JSON results.

Usage:
  python benchmark_runner.py                  # run all tiers
  python benchmark_runner.py --tier small     # run only small instances
  python benchmark_runner.py --models MTZ SCF 2CF  # specific models

Dependencies: gurobipy, numpy, vrplib
"""

import sys
import os
import json
import argparse
import time
import re

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from instance import load_instance, get_optimal
from mip_models import MODELS

# ---------------------------------------------------------------------------
# Benchmark instances directory
# Looks for .vrp files in ../dantzig-1959/benchmarks/cvrp/ (blog repo layout)
# or ./benchmarks/ (standalone layout)
# ---------------------------------------------------------------------------
_BENCHMARKS_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, '..', 'dantzig-1959', 'benchmarks', 'cvrp'),
    os.path.join(_SCRIPT_DIR, 'benchmarks'),
]


def _find_benchmarks_dir():
    for d in _BENCHMARKS_CANDIDATES:
        d = os.path.abspath(d)
        if os.path.isdir(d):
            return d
    return None


# ---------------------------------------------------------------------------
# Instance selection
# ---------------------------------------------------------------------------
TIERS = {
    'small': {
        'instances': [
            'E-n13-k4', 'E-n22-k4', 'E-n23-k3',
            'P-n16-k8', 'P-n19-k2', 'P-n20-k2',
            'P-n21-k2', 'P-n22-k2', 'P-n22-k8', 'P-n23-k8',
        ],
        'time_limit': 120,
    },
    'medium': {
        'instances': [
            'A-n32-k5', 'A-n33-k5', 'A-n34-k5',
            'A-n36-k5', 'A-n37-k5', 'A-n38-k5',
            'A-n39-k5', 'B-n31-k5', 'B-n34-k5', 'E-n33-k4',
        ],
        'time_limit': 300,
    },
    'large': {
        'instances': [
            'A-n53-k7', 'A-n60-k9', 'A-n65-k9',
            'B-n50-k7', 'B-n57-k7', 'E-n51-k5', 'P-n50-k7',
        ],
        'time_limit': 180,
    },
}


def parse_k(name):
    """Extract number of vehicles K from instance name like 'X-nNN-kK'."""
    m = re.search(r'k(\d+)', name)
    return int(m.group(1)) if m else None


def run_benchmark(tiers=None, models=None):
    """Run benchmark and return list of result dicts."""
    if tiers is None:
        tiers = list(TIERS.keys())
    if models is None:
        models = list(MODELS.keys())

    benchmarks_dir = _find_benchmarks_dir()
    if benchmarks_dir is None:
        print('ERROR: Cannot find benchmarks directory.')
        print(f'Searched: {_BENCHMARKS_CANDIDATES}')
        print('Place .vrp files in one of these directories.')
        return []

    print(f'Using benchmarks from: {benchmarks_dir}')
    results = []

    for tier_name in tiers:
        tier = TIERS[tier_name]
        time_limit = tier['time_limit']
        inst_names = tier['instances']

        print(f'\n{"="*60}')
        print(f'  Tier: {tier_name.upper()} ({len(inst_names)} instances, '
              f'time_limit={time_limit}s)')
        print(f'{"="*60}')

        for inst_name in inst_names:
            filepath = os.path.join(benchmarks_dir, f'{inst_name}.vrp')
            if not os.path.exists(filepath):
                print(f'  [SKIP] {inst_name}: file not found')
                continue

            dist, demand, capacity, name, n_cust = load_instance(filepath)
            K_name = parse_k(inst_name)
            opt = get_optimal(filepath)
            n_nodes = n_cust + 1

            print(f'\n  {inst_name} (n={n_nodes}, K_name={K_name}, Q={capacity}, opt={opt})')

            for model_name in models:
                solver = MODELS[model_name]
                print(f'    {model_name:4s} ... ', end='', flush=True)

                try:
                    res = solver(dist, demand, capacity,
                                 time_limit=time_limit)
                except Exception as e:
                    print(f'ERROR: {e}')
                    res = {'status': 'error', 'error': str(e)}

                # Augment result with metadata
                res['instance'] = inst_name
                res['model'] = model_name
                res['tier'] = tier_name
                res['n_nodes'] = n_nodes
                res['n_customers'] = n_cust
                res['capacity'] = capacity
                res['K_name'] = K_name
                res['known_optimal'] = opt
                res['time_limit'] = time_limit

                # Compute root LP gap
                if res.get('root_lp') is not None and opt is not None and opt > 0:
                    res['root_lp_gap'] = (opt - res['root_lp']) / opt
                else:
                    res['root_lp_gap'] = None

                results.append(res)

                # Print summary
                status = res.get('status', '?')
                obj = res.get('obj')
                rlp = res.get('root_lp')
                rlp_gap = res.get('root_lp_gap')
                t = res.get('time')
                nodes = res.get('nodes')
                k_min = res.get('K_min')
                k_actual = res.get('K_actual')

                parts = [status]
                if obj is not None:
                    parts.append(f'obj={obj:.0f}')
                if rlp is not None:
                    parts.append(f'LP={rlp:.1f}')
                if rlp_gap is not None:
                    parts.append(f'LP_gap={rlp_gap:.1%}')
                if k_min is not None:
                    parts.append(f'K_min={k_min}')
                if k_actual is not None:
                    parts.append(f'K_act={k_actual}')
                if t is not None:
                    parts.append(f'{t:.1f}s')
                if nodes is not None:
                    parts.append(f'{nodes} nodes')
                print(', '.join(parts))

    return results


def save_results(results, output_path):
    """Save results to JSON, converting numpy types."""
    import numpy as np

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = []
    for r in results:
        clean.append({k: convert(v) for k, v in r.items()})

    with open(output_path, 'w') as f:
        json.dump(clean, f, indent=2, default=str)
    print(f'\nResults saved to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='CVRP MIP Benchmark')
    parser.add_argument('--tier', nargs='+', choices=['small', 'medium', 'large'],
                        default=None, help='Tiers to run (default: all)')
    parser.add_argument('--models', nargs='+',
                        choices=['MTZ', 'SCF', '2CF', 'MCF', 'DFJ', 'SP'],
                        default=None, help='Models to run (default: all)')
    parser.add_argument('--output', default=None,
                        help='Output JSON path (default: results/<tier>_results.json)')
    args = parser.parse_args()

    tiers = args.tier
    results = run_benchmark(tiers=tiers, models=args.models)

    # Save results
    results_dir = os.path.join(_SCRIPT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    if args.output:
        output_path = args.output
    else:
        tier_str = '_'.join(tiers) if tiers else 'all'
        output_path = os.path.join(results_dir, f'{tier_str}_results.json')

    save_results(results, output_path)

    # Print summary table
    print(f'\n{"="*80}')
    print(f'  SUMMARY')
    print(f'{"="*80}')
    print(f'{"Instance":<14} {"Model":<5} {"Status":<9} {"Obj":>7} {"RootLP":>8} '
          f'{"LP Gap":>8} {"Kmin":>4} {"Kact":>4} {"Time":>8} {"Nodes":>8}')
    print('-' * 88)
    for r in results:
        obj_s = f'{r["obj"]:.0f}' if r.get('obj') else '-'
        rlp_s = f'{r["root_lp"]:.1f}' if r.get('root_lp') else '-'
        gap_s = f'{r["root_lp_gap"]:.1%}' if r.get('root_lp_gap') is not None else '-'
        kmin_s = str(r.get('K_min', '-'))
        kact_s = str(r.get('K_actual', '-'))
        time_s = f'{r["time"]:.1f}s' if r.get('time') else '-'
        nodes_s = str(r.get('nodes', '-'))
        print(f'{r["instance"]:<14} {r["model"]:<5} {r.get("status","?"):<9} '
              f'{obj_s:>7} {rlp_s:>8} {gap_s:>8} {kmin_s:>4} {kact_s:>4} '
              f'{time_s:>8} {nodes_s:>8}')


if __name__ == '__main__':
    main()
