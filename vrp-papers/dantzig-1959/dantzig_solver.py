"""
Dantzig & Ramser (1959) Layered Bundling Solver for CVRP
=========================================================
General-purpose implementation of the Dantzig (1959) heuristic:
  - Multi-level bundling: auto-compute L = ceil(log2(max_stations_per_vehicle))
  - LP matching at each level (Gurobi)
  - Mini-TSP for aggregate distances (enumeration or Held-Karp DP)
  - Benchmark testing against CVRPLIB instances

Usage:
  python dantzig_solver.py                        # Run benchmarks (n <= 50)
  python dantzig_solver.py path/to/instance.vrp   # Solve single instance
"""

import sys
import json
import math
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Silence Gurobi license message
import os
os.environ.setdefault("GRB_LICENSE_FILE", "")


# ============================================================
# Instance Loading
# ============================================================

def load_instance(filepath):
    """
    Read .vrp file using vrplib.
    Returns: (dist_matrix, demands, capacity, name)
      - dist_matrix: numpy 2D array (n x n), integer distances
      - demands: list of int, index 0 = depot (demand 0)
      - capacity: int
      - name: instance name string
    """
    import vrplib
    inst = vrplib.read_instance(filepath)

    dist_matrix = np.array(inst['edge_weight'])
    # vrplib returns float64; for EUC_2D the CVRPLIB standard is nint()
    # Round to nearest integer for consistency
    dist_matrix = np.rint(dist_matrix).astype(int)

    demands = list(inst['demand'].astype(int))
    capacity = int(inst['capacity'])
    name = inst.get('name', os.path.basename(filepath).replace('.vrp', ''))

    return dist_matrix, demands, capacity, name


# ============================================================
# Level Estimation
# ============================================================

def estimate_levels(demands, capacity):
    """
    Estimate the number of bundling levels.
    Compute max stations per vehicle by greedily packing smallest demands.
    L = ceil(log2(max_stations_per_vehicle)).
    Returns: (L, max_per_vehicle)
    """
    # demands[0] is depot = 0, skip it
    customer_demands = sorted(demands[1:])
    cumsum = 0
    max_per_vehicle = 0
    for d in customer_demands:
        if cumsum + d <= capacity:
            cumsum += d
            max_per_vehicle += 1
        else:
            break

    if max_per_vehicle <= 1:
        return 0, max_per_vehicle

    L = math.ceil(math.log2(max_per_vehicle))
    return L, max_per_vehicle


# ============================================================
# Mini-TSP: LKH via elkai (state-of-the-art)
# ============================================================

def mini_tsp(stations, dist_matrix):
    """
    Solve small TSP: depot(0) -> stations -> depot(0).
    Uses LKH algorithm via elkai for any size.

    Returns: (distance, best_order)
    """
    if len(stations) == 0:
        return 0, []
    if len(stations) == 1:
        s = stations[0]
        return int(dist_matrix[0][s]) + int(dist_matrix[s][0]), [s]

    return _mini_tsp_lkh(stations, dist_matrix)


def _mini_tsp_lkh(stations, dist_matrix):
    """Solve TSP via LKH (elkai). Handles depot + stations."""
    import elkai

    # Build local distance matrix: index 0 = depot, 1..n = stations
    nodes = [0] + list(stations)
    n = len(nodes)
    local_dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            local_dist[i][j] = int(dist_matrix[nodes[i]][nodes[j]])

    # LKH solves symmetric TSP, returns tour starting from node 0
    tour = elkai.solve_int_matrix(local_dist)

    # tour is a permutation of [0, 1, ..., n-1] starting from some node
    # Rotate so depot (local index 0) is first
    depot_pos = tour.index(0)
    tour = tour[depot_pos:] + tour[:depot_pos]

    # Compute distance
    total = 0
    for k in range(len(tour)):
        i = tour[k]
        j = tour[(k + 1) % len(tour)]
        total += local_dist[i][j]

    # Convert back to original node indices (skip depot)
    order = [nodes[t] for t in tour if t != 0]

    return int(total), order


# ============================================================
# LP Matching
# ============================================================

def solve_matching_lp(nodes, dist_func, demand, cap_limit):
    """
    Minimum-cost matching LP.

    min  sum d(i,j) * x(i,j)
    s.t. sum_j x(i,j) = 1   for each i in nodes
         x(i,j) >= 0
         x(i,j) = 0          if demand[i] + demand[j] > cap_limit

    Node 0 is the virtual depot (unconstrained degree).
    Each node in `nodes` must have degree exactly 1.

    Parameters:
        nodes: list of node indices (no depot)
        dist_func: callable(i, j) -> distance
        demand: dict {node: demand_value}
        cap_limit: capacity limit per group

    Returns: (pairs, singles, fractional_groups, obj_value)
    """
    m = gp.Model("matching")
    m.setParam("OutputFlag", 0)

    edges = {}
    # Edges from depot to each node
    for i in nodes:
        key = (0, i)
        edges[key] = m.addVar(lb=0, ub=1, obj=dist_func(0, i), name=f"x_0_{i}")

    # Edges between nodes (if capacity-feasible)
    for idx_a, i in enumerate(nodes):
        for j in nodes[idx_a + 1:]:
            if demand[i] + demand[j] <= cap_limit:
                key = (i, j)
                edges[key] = m.addVar(lb=0, ub=1, obj=dist_func(i, j),
                                      name=f"x_{i}_{j}")

    m.update()

    # Degree constraint: each node exactly 1
    for i in nodes:
        incident = []
        for (a, b), var in edges.items():
            if a == i or b == i:
                incident.append(var)
        m.addConstr(gp.quicksum(incident) == 1, name=f"deg_{i}")

    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"LP solve failed, status={m.status}")

    obj_value = m.ObjVal

    # Extract solution
    sol = {k: v.X for k, v in edges.items() if v.X > 1e-8}

    pairs = []
    singles = []
    frac_edges = []
    paired_nodes = set()

    for (i, j), val in sol.items():
        if i == 0:
            continue
        if abs(val - 1.0) < 1e-6:
            pairs.append((i, j))
            paired_nodes.add(i)
            paired_nodes.add(j)
        elif val > 1e-8:
            frac_edges.append((i, j, val))

    for (i, j), val in sol.items():
        if i == 0 and abs(val - 1.0) < 1e-6 and j not in paired_nodes:
            singles.append(j)

    # Identify fractional groups via BFS connected components
    fractional_groups = []
    if frac_edges:
        frac_nodes = set()
        for (i, j, _) in frac_edges:
            frac_nodes.add(i)
            frac_nodes.add(j)
        remaining = set(frac_nodes)
        while remaining:
            start = remaining.pop()
            component = {start}
            queue = [start]
            while queue:
                node = queue.pop(0)
                for (i, j, _) in frac_edges:
                    other = None
                    if i == node and j in remaining:
                        other = j
                    elif j == node and i in remaining:
                        other = i
                    if other is not None:
                        component.add(other)
                        remaining.discard(other)
                        queue.append(other)
            fractional_groups.append(sorted(component))

    return pairs, singles, fractional_groups, obj_value


# ============================================================
# Aggregate Distance Matrix
# ============================================================

def build_aggregate_distance(groups, dist_matrix, group_demands=None, cap_limit=None):
    """
    Build inter-group distance matrix.
    Distance between two groups = mini-TSP cost of (group_i union group_j) via depot.
    groups[0] -> aggregate node 1, groups[1] -> aggregate node 2, ...
    Skips infeasible pairs (demand > cap_limit) to save time.
    Returns: agg_dist[0..n][0..n] where 0 = depot.
    """
    n = len(groups)
    INF = 1e9
    agg_dist = np.full((n + 1, n + 1), INF, dtype=float)
    np.fill_diagonal(agg_dist, 0)

    for i in range(n):
        d, _ = mini_tsp(groups[i], dist_matrix)
        agg_dist[0][i + 1] = d
        agg_dist[i + 1][0] = d

    for i in range(n):
        for j in range(i + 1, n):
            # Skip infeasible pairs
            if group_demands is not None and cap_limit is not None:
                if group_demands[i] + group_demands[j] > cap_limit:
                    continue
            combined = groups[i] + groups[j]
            d, _ = mini_tsp(combined, dist_matrix)
            agg_dist[i + 1][j + 1] = d
            agg_dist[j + 1][i + 1] = d

    return agg_dist


# ============================================================
# Fractional Resolution
# ============================================================

def resolve_fractional(frac_group, agg_dist):
    """
    Handle LP fractional solutions by enumerating roundings.
    frac_group: list of aggregate node indices (1-based)
    Returns: (best_pairs, best_singles)
      best_pairs: list of (i, j) pairs
      best_singles: list of lone nodes paired with depot
    """
    n = len(frac_group)

    if n == 2:
        # Only two nodes: either pair them or leave both as singles
        i, j = frac_group
        pair_cost = agg_dist[i][j]
        single_cost = agg_dist[0][i] + agg_dist[0][j]
        if pair_cost <= single_cost:
            return [(i, j)], []
        else:
            return [], [i, j]

    if n == 3:
        # Three nodes: try each as the singleton
        best_cost = float('inf')
        best_pairs = None
        best_singles = None
        for lone_idx in range(3):
            lone = frac_group[lone_idx]
            remaining = [frac_group[k] for k in range(3) if k != lone_idx]
            cost = agg_dist[remaining[0]][remaining[1]] + agg_dist[0][lone]
            if cost < best_cost:
                best_cost = cost
                best_pairs = [tuple(remaining)]
                best_singles = [lone]
        return best_pairs, best_singles

    # General case for larger fractional groups:
    # Greedy approach -- pair closest nodes, leave remainder as singletons
    remaining = list(frac_group)
    pairs = []
    while len(remaining) >= 2:
        best_cost = float('inf')
        best_pair = None
        for i_idx in range(len(remaining)):
            for j_idx in range(i_idx + 1, len(remaining)):
                cost = agg_dist[remaining[i_idx]][remaining[j_idx]]
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (i_idx, j_idx)
        if best_pair is None:
            break
        i_idx, j_idx = best_pair
        pairs.append((remaining[i_idx], remaining[j_idx]))
        # Remove in reverse order to preserve indices
        remaining.pop(j_idx)
        remaining.pop(i_idx)
    return pairs, remaining


# ============================================================
# Main Solver
# ============================================================

def _solve_with_levels(dist_matrix, demands, capacity, L, verbose=False):
    """Run layered bundling with a fixed number of levels L."""
    n = len(demands)
    groups = [[i] for i in range(1, n)]
    group_demands = [demands[i] for i in range(1, n)]

    for level in range(1, L + 1):
        cap_limit = capacity / (2 ** (L - level))

        if verbose:
            print(f"\n  Level {level}/{L}: cap_limit = {cap_limit:.0f}, "
                  f"{len(groups)} groups")

        n_groups = len(groups)
        if n_groups <= 1:
            break

        # Build aggregate distance matrix (skip infeasible pairs)
        agg_dist = build_aggregate_distance(
            groups, dist_matrix, group_demands, cap_limit
        )

        # Set up matching LP
        nodes = list(range(1, n_groups + 1))
        demand_dict = {0: 0}
        for i in range(n_groups):
            demand_dict[i + 1] = group_demands[i]

        def dist_func(i, j, _agg=agg_dist):
            return _agg[i][j]

        try:
            pairs, singles, frac_groups, obj = solve_matching_lp(
                nodes, dist_func, demand_dict, cap_limit
            )
        except RuntimeError:
            if verbose:
                print(f"    LP infeasible at level {level}, skipping merge")
            continue

        if verbose:
            print(f"    LP obj = {obj:.1f}, pairs = {len(pairs)}, "
                  f"singles = {len(singles)}, fractional = {len(frac_groups)}")

        # Resolve fractional solutions
        for fg in frac_groups:
            fg_pairs, fg_singles = resolve_fractional(fg, agg_dist)
            pairs.extend(fg_pairs)
            singles.extend(fg_singles)

        # Build new groups by merging pairs
        new_groups = []
        new_demands = []
        merged = set()

        for (ai, aj) in pairs:
            new_group = groups[ai - 1] + groups[aj - 1]
            new_dem = group_demands[ai - 1] + group_demands[aj - 1]
            new_groups.append(new_group)
            new_demands.append(new_dem)
            merged.add(ai)
            merged.add(aj)

        for s in singles:
            new_groups.append(groups[s - 1])
            new_demands.append(group_demands[s - 1])
            merged.add(s)

        # Add any unmatched groups as singletons
        for i in range(1, n_groups + 1):
            if i not in merged:
                new_groups.append(groups[i - 1])
                new_demands.append(group_demands[i - 1])

        groups = new_groups
        group_demands = new_demands

    # Compute final route distances
    routes = groups
    total_distance = 0
    for route in routes:
        d, _ = mini_tsp(route, dist_matrix)
        total_distance += d

    return routes, total_distance


def dantzig_solve(dist_matrix, demands, capacity, verbose=False):
    """
    Dantzig (1959) layered bundling heuristic for CVRP.

    Uses L = ceil(log2(t)) levels as specified in the paper, where t is the
    maximum number of stations per vehicle (computed by greedy packing).

    Returns: (routes, total_distance)
    """
    n = len(demands)
    n_customers = n - 1

    if n_customers == 0:
        return [], 0

    L, max_per_veh = estimate_levels(demands, capacity)

    if verbose:
        print(f"  Customers: {n_customers}, Capacity: {capacity}")
        print(f"  Max stations/vehicle: {max_per_veh}, L: {L}")

    if L == 0:
        routes = [[i] for i in range(1, n)]
        total_dist = sum(dist_matrix[0][i] + dist_matrix[i][0] for i in range(1, n))
        return routes, int(total_dist)

    routes, total_dist = _solve_with_levels(
        dist_matrix, demands, capacity, L, verbose=verbose
    )

    return routes, total_dist


# ============================================================
# Benchmark Runner
# ============================================================

def run_benchmarks(instance_dir, instances_json, max_customers=50):
    """
    Run benchmark tests on CVRPLIB instances.

    Filters instances with n_customers <= max_customers.
    For each: load, solve, compare with known optimal.
    Output: table of (instance_name, n, dantzig_dist, optimal, gap%).
    """
    with open(instances_json, 'r') as f:
        all_instances = json.load(f)

    # Filter by size
    instances = [inst for inst in all_instances if inst['n_customers'] <= max_customers]
    instances.sort(key=lambda x: x['n_customers'])

    print(f"\nDantzig (1959) Layered Bundling Heuristic - Benchmark Results")
    print(f"Max customers: {max_customers}, Instances: {len(instances)}")
    print("=" * 78)
    print(f"{'Instance':<16} {'n':>4} {'k':>3} {'C':>6} "
          f"{'Dantzig':>8} {'Optimal':>8} {'Gap%':>7} {'Time':>6}")
    print("-" * 78)

    gaps = []
    for inst_info in instances:
        name = inst_info['name']
        vrp_path = os.path.join(instance_dir, name + '.vrp')

        if not os.path.exists(vrp_path):
            print(f"{'[SKIP] ' + name:<16} file not found")
            continue

        try:
            t0 = time.time()
            dist_matrix, demands, capacity, _ = load_instance(vrp_path)
            routes, total_dist = dantzig_solve(dist_matrix, demands, capacity)
            elapsed = time.time() - t0

            optimal = inst_info['optimal']
            gap = (total_dist - optimal) / optimal * 100

            # Validate feasibility
            all_customers = set()
            feasible = True
            for route in routes:
                route_dem = sum(demands[c] for c in route)
                if route_dem > capacity:
                    feasible = False
                all_customers.update(route)

            expected = set(range(1, len(demands)))
            if all_customers != expected:
                feasible = False

            feas_mark = "" if feasible else " *INFEASIBLE*"

            print(f"{name:<16} {inst_info['n_customers']:>4} "
                  f"{inst_info['n_vehicles']:>3} {capacity:>6} "
                  f"{total_dist:>8} {optimal:>8} {gap:>6.1f}% "
                  f"{elapsed:>5.1f}s{feas_mark}")

            gaps.append(gap)

        except Exception as e:
            print(f"{name:<16} ERROR: {e}")

    if gaps:
        print("-" * 78)
        avg_gap = sum(gaps) / len(gaps)
        min_gap = min(gaps)
        max_gap = max(gaps)
        print(f"{'Average':<16} {'':>4} {'':>3} {'':>6} "
              f"{'':>8} {'':>8} {avg_gap:>6.1f}%")
        print(f"{'Min/Max gap':<16} {'':>4} {'':>3} {'':>6} "
              f"{'':>8} {'':>8} {min_gap:>5.1f}%-{max_gap:.1f}%")
        print(f"\nInstances solved: {len(gaps)}")


# ============================================================
# Single Instance Mode
# ============================================================

def solve_single(filepath):
    """Solve a single .vrp instance and print detailed results."""
    dist_matrix, demands, capacity, name = load_instance(filepath)
    n_customers = len(demands) - 1

    print(f"\nInstance: {name}")
    print(f"Customers: {n_customers}, Capacity: {capacity}")
    print(f"Total demand: {sum(demands[1:])}")
    print("=" * 60)

    t0 = time.time()
    routes, total_dist = dantzig_solve(dist_matrix, demands, capacity, verbose=True)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Final Routes ({len(routes)} vehicles):")
    print(f"{'=' * 60}")

    for idx, route in enumerate(routes, 1):
        d, order = mini_tsp(route, dist_matrix)
        dem = sum(demands[c] for c in route)
        route_str = " -> ".join(str(s) for s in order)
        print(f"  Route {idx:>2}: 0 -> {route_str} -> 0  "
              f"dist={d}  demand={dem}/{capacity}  "
              f"stations={len(route)}")

    print(f"\nTotal distance: {total_dist}")
    print(f"Solve time: {elapsed:.2f}s")

    # Feasibility check
    all_custs = set()
    for route in routes:
        route_dem = sum(demands[c] for c in route)
        if route_dem > capacity:
            print(f"  WARNING: Route exceeds capacity ({route_dem} > {capacity})")
        all_custs.update(route)

    expected = set(range(1, len(demands)))
    if all_custs != expected:
        missing = expected - all_custs
        extra = all_custs - expected
        if missing:
            print(f"  WARNING: Missing customers: {sorted(missing)}")
        if extra:
            print(f"  WARNING: Extra customers: {sorted(extra)}")

    return routes, total_dist


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), 'benchmarks', 'cvrp')
    INSTANCES_JSON = os.path.join(os.path.dirname(__file__), 'benchmarks', 'instances.json')

    # Normalize paths
    BENCHMARKS_DIR = os.path.normpath(BENCHMARKS_DIR)
    INSTANCES_JSON = os.path.normpath(INSTANCES_JSON)

    if len(sys.argv) > 1:
        # Single instance mode
        solve_single(sys.argv[1])
    else:
        # Benchmark mode
        run_benchmarks(BENCHMARKS_DIR, INSTANCES_JSON, max_customers=50)
