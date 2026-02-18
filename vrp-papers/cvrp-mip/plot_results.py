"""
CVRP MIP Results Plotter
========================
Generates comparison charts from benchmark JSON results.

Usage:
  python plot_results.py results/all_results.json
  python plot_results.py results/small_results.json results/medium_results.json

Dependencies: matplotlib, numpy
"""

import sys
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color scheme
COLORS = {'MTZ': '#2196F3', 'SCF': '#FF9800', 'DFJ': '#4CAF50'}


def load_results(*paths):
    """Load and merge results from multiple JSON files."""
    all_results = []
    for p in paths:
        with open(p) as f:
            all_results.extend(json.load(f))
    return all_results


def plot_root_lp_gap(results, output='fig_root_lp_gap.png'):
    """Bar chart: Root LP gap by instance, grouped by model."""
    instances = []
    seen = set()
    for r in results:
        if r['instance'] not in seen:
            instances.append(r['instance'])
            seen.add(r['instance'])

    models = ['MTZ', 'SCF', 'DFJ']
    n_inst = len(instances)

    gaps = {m: [] for m in models}
    for inst in instances:
        for m in models:
            match = [r for r in results
                     if r['instance'] == inst and r['model'] == m]
            if match and match[0].get('root_lp_gap') is not None:
                gaps[m].append(match[0]['root_lp_gap'] * 100)
            else:
                gaps[m].append(0)

    fig, ax = plt.subplots(figsize=(max(12, n_inst * 0.8), 6))
    bar_width = 0.25
    x = np.arange(n_inst)

    for i, m in enumerate(models):
        ax.bar(x + i * bar_width, gaps[m], bar_width,
               label=m, color=COLORS[m], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel('Root LP Gap (%)', fontsize=12)
    ax.set_title('Root LP Relaxation Gap by MIP Formulation', fontsize=14)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(instances, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for m in models:
        valid = [g for g in gaps[m] if g > 0]
        if valid:
            avg = np.mean(valid)
            ax.axhline(y=avg, color=COLORS[m], linestyle='--', alpha=0.5, linewidth=1)
            ax.text(n_inst - 0.5, avg + 0.3, f'{m} avg={avg:.1f}%',
                    fontsize=8, color=COLORS[m], ha='right')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, output)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_solve_time(results, output='fig_solve_time.png'):
    """Bar chart: solve time by instance, grouped by model."""
    instances = []
    seen = set()
    for r in results:
        if r['instance'] not in seen:
            instances.append(r['instance'])
            seen.add(r['instance'])

    models = ['MTZ', 'SCF', 'DFJ']
    n_inst = len(instances)

    times = {m: [] for m in models}
    for inst in instances:
        for m in models:
            match = [r for r in results
                     if r['instance'] == inst and r['model'] == m]
            if match and match[0].get('time') is not None:
                times[m].append(match[0]['time'])
            else:
                times[m].append(0)

    fig, ax = plt.subplots(figsize=(max(12, n_inst * 0.8), 6))
    bar_width = 0.25
    x = np.arange(n_inst)

    for i, m in enumerate(models):
        ax.bar(x + i * bar_width, times[m], bar_width,
               label=m, color=COLORS[m], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel('Solve Time (seconds)', fontsize=12)
    ax.set_title('MIP Solve Time Comparison', fontsize=14)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(instances, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, output)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_nodes(results, output='fig_nodes.png'):
    """Bar chart: B&B nodes explored, grouped by model."""
    instances = []
    seen = set()
    for r in results:
        if r['instance'] not in seen:
            instances.append(r['instance'])
            seen.add(r['instance'])

    models = ['MTZ', 'SCF', 'DFJ']
    n_inst = len(instances)

    nodes = {m: [] for m in models}
    for inst in instances:
        for m in models:
            match = [r for r in results
                     if r['instance'] == inst and r['model'] == m]
            if match and match[0].get('nodes') is not None:
                nodes[m].append(max(1, match[0]['nodes']))
            else:
                nodes[m].append(1)

    fig, ax = plt.subplots(figsize=(max(12, n_inst * 0.8), 6))
    bar_width = 0.25
    x = np.arange(n_inst)

    for i, m in enumerate(models):
        ax.bar(x + i * bar_width, nodes[m], bar_width,
               label=m, color=COLORS[m], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel('B&B Nodes (log scale)', fontsize=12)
    ax.set_title('Branch-and-Bound Nodes Explored', fontsize=14)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(instances, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, output)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_scalability(results, output='fig_scalability.png'):
    """Scatter plot: solve time vs. instance size, by model."""
    models = ['MTZ', 'SCF', 'DFJ']

    fig, ax = plt.subplots(figsize=(10, 6))

    for m in models:
        xs = [r['n_nodes'] for r in results if r['model'] == m and r.get('time')]
        ys = [r['time'] for r in results if r['model'] == m and r.get('time')]
        ax.scatter(xs, ys, label=m, color=COLORS[m], s=60, alpha=0.7,
                   edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Instance Size (nodes)', fontsize=12)
    ax.set_ylabel('Solve Time (seconds, log)', fontsize=12)
    ax.set_title('MIP Solve Time vs. Instance Size', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    time_limits = set(r.get('time_limit', 0) for r in results)
    for tl in time_limits:
        if tl > 0:
            ax.axhline(y=tl, color='red', linestyle=':', alpha=0.5)
            ax.text(ax.get_xlim()[1] * 0.95, tl * 1.1, f'limit={tl}s',
                    fontsize=8, color='red', ha='right')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, output)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def main():
    if len(sys.argv) < 2:
        results_dir = os.path.join(_SCRIPT_DIR, 'results')
        json_files = sorted(
            f for f in os.listdir(results_dir)
            if f.endswith('.json')
        )
        if not json_files:
            print('No result files found. Run benchmark_runner.py first.')
            return
        paths = [os.path.join(results_dir, f) for f in json_files]
    else:
        paths = sys.argv[1:]

    print(f'Loading results from: {paths}')
    results = load_results(*paths)
    print(f'Loaded {len(results)} result entries')

    plot_root_lp_gap(results)
    plot_solve_time(results)
    plot_nodes(results)
    plot_scalability(results)

    print(f'\nAll figures saved to {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
