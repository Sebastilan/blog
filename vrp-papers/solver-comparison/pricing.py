"""
ESPPRC Pricing Subproblem (Label-Setting)
==========================================
Elementary Shortest Path Problem with Resource Constraints.

Uses bitmask for visited set — efficient for n_customers <= 25.
Standard dominance: L1 dominates L2 at same node when
  cost1 <= cost2, load1 <= load2, S1 ⊆ S2.
"""

import numpy as np
from collections import deque


def solve_pricing(dist, demand, capacity, cover_duals, n_customers,
                  forbidden_edges=None, edge_duals=None, max_columns=500):
    """
    Solve the ESPPRC pricing subproblem via label-setting.

    Modified edge weights:
        c_bar[i][j] = dist[i][j] - cover_duals[j-1]   (j >= 1)
        c_bar[i][0] = dist[i][0]                        (return to depot)
        c_bar[i][j] -= edge_duals[(min(i,j), max(i,j))] (if forced edge)

    Returns:
        list of (cost, visits, path, rc) with negative reduced cost,
        sorted by rc ascending (most negative first)
    """
    n = n_customers + 1

    if forbidden_edges is None:
        forbidden_edges = set()
    if edge_duals is None:
        edge_duals = {}

    # Normalize forbidden edges
    forbidden = {(min(a, b), max(a, b)) for a, b in forbidden_edges}

    def is_forbidden(i, j):
        return (min(i, j), max(i, j)) in forbidden

    # Modified cost matrix
    c_bar = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            c_bar[i][j] = dist[i][j]
            # Subtract customer dual
            if j >= 1:
                c_bar[i][j] -= cover_duals[j - 1]
            # Subtract forced edge dual
            enorm = (min(i, j), max(i, j))
            if enorm in edge_duals:
                c_bar[i][j] -= edge_duals[enorm]

    # Label-setting with standard dominance
    # labels[node] = list of (rc, load, vmask, path), pairwise non-dominated
    labels = [[] for _ in range(n)]
    labels[0] = [(0.0, 0, 0, [0])]

    def is_dominated(node, rc, load, vmask):
        """Check if (rc, load, vmask) is dominated by any existing label."""
        for erc, eload, evmask, _ in labels[node]:
            if erc <= rc and eload <= load and (evmask & vmask) == evmask:
                return True
        return False

    def add_label(node, rc, load, vmask, path):
        """Add label, removing any it dominates. Returns True if added."""
        if is_dominated(node, rc, load, vmask):
            return False
        labels[node] = [
            lbl for lbl in labels[node]
            if not (rc <= lbl[0] and load <= lbl[1] and (vmask & lbl[2]) == vmask)
        ]
        labels[node].append((rc, load, vmask, path))
        return True

    # BFS queue using deque for O(1) popleft
    queue = deque([(0, 0, 0.0)])
    new_columns = []

    while queue:
        node, vmask, enq_rc = queue.popleft()

        # Check if this queue entry is stale
        current = None
        for lbl in labels[node]:
            if lbl[2] == vmask and abs(lbl[0] - enq_rc) < 1e-10:
                current = lbl
                break
        if current is None:
            continue
        rc, load, _, path = current

        # Extend to each customer
        for j in range(1, n):
            if vmask & (1 << (j - 1)):
                continue  # already visited
            if is_forbidden(node, j):
                continue

            new_load = load + demand[j]
            if new_load > capacity:
                continue

            new_rc = rc + c_bar[node][j]
            new_vmask = vmask | (1 << (j - 1))

            if add_label(j, new_rc, new_load, new_vmask, path + [j]):
                queue.append((j, new_vmask, new_rc))

        # Try completing route (return to depot)
        if node != 0 and not is_forbidden(node, 0):
            final_rc = rc + c_bar[node][0]
            if final_rc < -1e-6:
                full_path = path + [0]
                actual_cost = sum(int(dist[full_path[k]][full_path[k + 1]])
                                  for k in range(len(full_path) - 1))
                visits = [v for v in path if v != 0]
                new_columns.append((actual_cost, visits, full_path, final_rc))

    new_columns.sort(key=lambda x: x[3])
    return new_columns[:max_columns]
