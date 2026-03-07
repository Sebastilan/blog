"""
ESPPRC Oracle — exact optimal reduced cost via DFS enumeration or Gurobi MIP.
===============================================================================
Two independent methods for cross-validation:
  - enumerate_oracle: DFS with capacity pruning (n <= ~15, tight capacity)
  - mip_oracle: Gurobi MIP with MTZ subtour elimination (any n)
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def reduced_cost_matrix(dist, cover_duals, n_customers):
    """c_bar[i][j] = dist[i][j] - pi[j] for j>=1, dist[i][0] for j=0."""
    N = n_customers + 1
    dist_arr = np.array(dist, dtype=float)
    c_bar = dist_arr.copy()
    for j in range(1, N):
        c_bar[:, j] -= cover_duals[j - 1]
    return c_bar


# ── DFS Enumeration Oracle ──────────────────────────────────────────────────

def enumerate_oracle(dist, demand, capacity, cover_duals, n_customers,
                     max_routes=10_000_000):
    """
    Enumerate all feasible elementary routes via DFS with capacity pruning.

    Returns: (opt_rc, opt_path, opt_cost, neg_count, complete)
      complete=False if max_routes exceeded (result is lower bound only).
    """
    N = n_customers + 1
    dist_arr = np.array(dist, dtype=float)
    c_bar = reduced_cost_matrix(dist, cover_duals, n_customers)

    best = [float('inf'), None, None]   # [rc, path, cost]
    neg_count = [0]
    route_count = [0]
    complete = [True]
    path = [0]

    def dfs(node, load, rc, vmask):
        if not complete[0]:
            return

        # Complete: return to depot
        final_rc = rc + c_bar[node][0]
        route_count[0] += 1
        if route_count[0] > max_routes:
            complete[0] = False
            return

        if final_rc < best[0]:
            best[0] = float(final_rc)
            best[1] = list(path) + [0]
            best[2] = (sum(int(dist_arr[path[k]][path[k + 1]])
                           for k in range(len(path) - 1))
                       + int(dist_arr[node][0]))
        if final_rc < -1e-6:
            neg_count[0] += 1

        # Extend to unvisited customers
        for j in range(1, N):
            if vmask & (1 << (j - 1)):
                continue
            new_load = load + demand[j]
            if new_load > capacity:
                continue
            path.append(j)
            dfs(j, new_load, rc + float(c_bar[node][j]),
                vmask | (1 << (j - 1)))
            path.pop()

    for j in range(1, N):
        if demand[j] <= capacity:
            path.append(j)
            dfs(j, demand[j], float(c_bar[0][j]), 1 << (j - 1))
            path.pop()

    return best[0], best[1], best[2], neg_count[0], complete[0]


# ── Gurobi MIP Oracle ──────────────────────────────────────────────────────

def mip_oracle(dist, demand, capacity, cover_duals, n_customers,
               time_limit=120):
    """
    Solve ESPPRC exactly via Gurobi MIP (MTZ subtour elimination).

    Returns: (opt_rc, opt_path, opt_cost, status)
      status: 'optimal' | 'time_limit' | 'infeasible'
    """
    N = n_customers + 1
    dist_arr = np.array(dist, dtype=float)
    c_bar = reduced_cost_matrix(dist, cover_duals, n_customers)

    m = gp.Model("ESPPRC_Oracle")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit

    # Arc variables
    x = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY,
                                    obj=float(c_bar[i][j]),
                                    name=f"x_{i}_{j}")

    # MTZ order variables
    u = {}
    for i in range(1, N):
        u[i] = m.addVar(lb=0, ub=float(capacity), name=f"u_{i}")

    m.update()

    # Depot: leave once, return once
    m.addConstr(gp.quicksum(x[0, j] for j in range(1, N)) == 1, "depot_out")
    m.addConstr(gp.quicksum(x[i, 0] for i in range(1, N)) == 1, "depot_in")

    # Customer flow conservation + visit at most once
    for i in range(1, N):
        inflow = gp.quicksum(x[j, i] for j in range(N) if j != i)
        outflow = gp.quicksum(x[i, j] for j in range(N) if j != i)
        m.addConstr(inflow == outflow, f"flow_{i}")
        m.addConstr(inflow <= 1, f"once_{i}")

    # MTZ subtour elimination
    # Big-M = Q + demand[j] (not just Q) because unvisited nodes have u=0,
    # and the constraint must be trivially satisfied when x[i,j]=0.
    Q = float(capacity)
    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                M_ij = Q + demand[j]
                m.addConstr(u[j] >= u[i] + demand[j] - M_ij * (1 - x[i, j]),
                            f"mtz_{i}_{j}")

    # Demand bounds
    for i in range(1, N):
        yi = gp.quicksum(x[j, i] for j in range(N) if j != i)
        m.addConstr(u[i] >= demand[i] * yi, f"lb_u_{i}")
        m.addConstr(u[i] <= Q * yi, f"ub_u_{i}")

    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        path = _extract_path(x, N)
        cost = sum(int(dist_arr[path[k]][path[k + 1]])
                   for k in range(len(path) - 1))
        status = 'optimal' if m.Status == GRB.OPTIMAL else 'time_limit'
        return float(m.ObjVal), path, cost, status

    return None, None, None, 'infeasible'


def _extract_path(x, N):
    """Extract route path from solved MIP solution."""
    path = [0]
    current = 0
    for _ in range(N):
        for j in range(N):
            if j != current and (current, j) in x and x[current, j].X > 0.5:
                path.append(j)
                current = j
                break
        if current == 0:
            break
    if path[-1] != 0:
        path.append(0)
    return path
