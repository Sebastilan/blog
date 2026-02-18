"""
CVRP MIP Formulations
=====================
Six standard MIP models for the Capacitated Vehicle Routing Problem:
  1. MTZ  — Miller-Tucker-Zemlin subtour elimination
  2. SCF  — Single-Commodity Flow
  3. 2CF  — Two-Commodity Flow (Baldacci et al. 2004)
  4. MCF  — Multi-Commodity Flow (Letchford & Salazar-Gonzalez 2015)
  5. DFJ  — Dantzig-Fulkerson-Johnson (lazy subtour cuts via callback)
  6. SP   — Set Partitioning (column generation) [requires BPC code]

Each function returns a unified result dict.

Dependencies: gurobipy, numpy
"""

import math
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def _result_dict():
    """Empty result template."""
    return {
        'status': None,         # 'optimal', 'feasible', 'infeasible', 'timeout'
        'obj': None,            # best integer objective
        'bound': None,          # best lower bound
        'gap': None,            # MIP gap (fraction)
        'root_lp': None,        # LP relaxation value at root node
        'time': None,           # wall-clock seconds
        'nodes': None,          # B&B nodes explored
        'n_vars': None,         # number of variables
        'n_constrs': None,      # number of constraints (before lazy)
        'K_min': None,          # ceil(sum(demand) / capacity)
        'K_actual': None,       # actual vehicles used in solution
    }


def _status_str(model):
    """Convert Gurobi status code to readable string."""
    s = model.Status
    if s == GRB.OPTIMAL:
        return 'optimal'
    if s == GRB.TIME_LIMIT:
        return 'feasible' if model.SolCount > 0 else 'timeout'
    if s in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return 'infeasible'
    if model.SolCount > 0:
        return 'feasible'
    return 'timeout'


# ---------------------------------------------------------------------------
# 1. MTZ (Miller-Tucker-Zemlin)
# ---------------------------------------------------------------------------
def solve_mtz(dist, demand, capacity, time_limit=300, root_lp_only=False):
    """
    MTZ formulation for CVRP.

    Variables:
      x[i,j] in {0,1}  — arc (i,j) used
      u[i]   in [d_i, Q] — cumulative load upon arriving at customer i

    Subtour elimination: u_j >= u_i + d_j - Q(1 - x_{ij}), for all i,j in C
    """
    n = len(demand)  # 0..n-1, where 0 is depot
    C = range(1, n)  # customers
    V = range(n)     # all nodes
    K_min = math.ceil(sum(demand[i] for i in C) / capacity)

    res = _result_dict()
    res['K_min'] = K_min

    m = gp.Model('CVRP_MTZ')
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit

    # --- Variables ---
    x = {}
    for i in V:
        for j in V:
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
    u = {}
    for i in C:
        u[i] = m.addVar(lb=demand[i], ub=capacity, name=f'u_{i}')

    m.update()

    # --- Objective ---
    m.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for i in V for j in V if i != j),
        GRB.MINIMIZE
    )

    # --- Constraints ---
    # Each customer visited exactly once (in-degree = out-degree = 1)
    for j in C:
        m.addConstr(gp.quicksum(x[i, j] for i in V if i != j) == 1, f'in_{j}')
        m.addConstr(gp.quicksum(x[j, i] for i in V if i != j) == 1, f'out_{j}')

    # Depot: at least K_min vehicles leave and return
    m.addConstr(gp.quicksum(x[0, j] for j in C) >= K_min, 'depot_out')
    m.addConstr(gp.quicksum(x[j, 0] for j in C) >= K_min, 'depot_in')

    # MTZ subtour elimination + capacity
    for i in C:
        for j in C:
            if i != j:
                m.addConstr(
                    u[j] >= u[i] + demand[j] - capacity * (1 - x[i, j]),
                    f'mtz_{i}_{j}'
                )

    m.update()
    res['n_vars'] = m.NumVars
    res['n_constrs'] = m.NumConstrs

    # --- Solve root LP ---
    if root_lp_only:
        m_lp = m.relax()
        m_lp.Params.OutputFlag = 0
        m_lp.optimize()
        res['root_lp'] = m_lp.ObjVal if m_lp.Status == GRB.OPTIMAL else None
        res['time'] = m_lp.Runtime
        return res

    # Get root LP via callback
    root_lp_vals = []

    def _root_cb(model, where):
        if where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
                obj = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                root_lp_vals.append(obj)

    t0 = time.time()
    m.optimize(_root_cb)
    res['time'] = time.time() - t0

    res['status'] = _status_str(m)
    res['obj'] = m.ObjVal if m.SolCount > 0 else None
    res['bound'] = m.ObjBound if hasattr(m, 'ObjBound') else None
    res['gap'] = m.MIPGap if m.SolCount > 0 else None
    res['nodes'] = int(m.NodeCount)
    res['root_lp'] = root_lp_vals[0] if root_lp_vals else None

    # Extract K_actual from solution
    if m.SolCount > 0:
        res['K_actual'] = int(round(sum(x[0, j].X for j in C)))

    # If root_lp not captured by callback, solve LP relaxation explicitly
    if res['root_lp'] is None:
        m_lp = m.relax()
        m_lp.Params.OutputFlag = 0
        m_lp.optimize()
        if m_lp.Status == GRB.OPTIMAL:
            res['root_lp'] = m_lp.ObjVal

    return res


# ---------------------------------------------------------------------------
# 2. SCF (Single-Commodity Flow)
# ---------------------------------------------------------------------------
def solve_scf(dist, demand, capacity, time_limit=300, root_lp_only=False):
    """
    Single-Commodity Flow formulation for CVRP.

    Extra variables:
      f[i,j] >= 0  — flow of "commodity" (total remaining demand) on arc (i,j)

    Flow conservation at each customer: flow_in - flow_out = d_i
    Depot sends out total demand, receives 0 commodity.
    Linking: d_j * x[i,j] <= f[i,j] <= (Q - d_i) * x[i,j]
    """
    n = len(demand)
    C = range(1, n)
    V = range(n)
    Q = capacity
    K_min = math.ceil(sum(demand[i] for i in C) / Q)

    res = _result_dict()
    res['K_min'] = K_min

    m = gp.Model('CVRP_SCF')
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit

    # --- Variables ---
    x = {}
    f = {}
    for i in V:
        for j in V:
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
                f[i, j] = m.addVar(lb=0.0, name=f'f_{i}_{j}')

    m.update()

    # --- Objective ---
    m.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for i in V for j in V if i != j),
        GRB.MINIMIZE
    )

    # --- Routing constraints ---
    for j in C:
        m.addConstr(gp.quicksum(x[i, j] for i in V if i != j) == 1, f'in_{j}')
        m.addConstr(gp.quicksum(x[j, i] for i in V if i != j) == 1, f'out_{j}')

    m.addConstr(gp.quicksum(x[0, j] for j in C) >= K_min, 'depot_out')
    m.addConstr(gp.quicksum(x[j, 0] for j in C) >= K_min, 'depot_in')

    # --- Flow conservation ---
    for j in C:
        m.addConstr(
            gp.quicksum(f[i, j] for i in V if i != j)
            - gp.quicksum(f[j, i] for i in V if i != j) == demand[j],
            f'flow_{j}'
        )

    # --- Linking constraints: f[i,j] <= (Q - d_i) * x[i,j] ---
    for i in V:
        for j in V:
            if i != j:
                ub_ij = Q - demand[i] if i != 0 else Q
                m.addConstr(f[i, j] <= ub_ij * x[i, j], f'link_ub_{i}_{j}')
                # Lower bound linking (optional, strengthens LP)
                if j != 0:
                    m.addConstr(f[i, j] >= demand[j] * x[i, j], f'link_lb_{i}_{j}')

    m.update()
    res['n_vars'] = m.NumVars
    res['n_constrs'] = m.NumConstrs

    if root_lp_only:
        m_lp = m.relax()
        m_lp.Params.OutputFlag = 0
        m_lp.optimize()
        res['root_lp'] = m_lp.ObjVal if m_lp.Status == GRB.OPTIMAL else None
        res['time'] = m_lp.Runtime
        return res

    root_lp_vals = []

    def _root_cb(model, where):
        if where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
                obj = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                root_lp_vals.append(obj)

    t0 = time.time()
    m.optimize(_root_cb)
    res['time'] = time.time() - t0

    res['status'] = _status_str(m)
    res['obj'] = m.ObjVal if m.SolCount > 0 else None
    res['bound'] = m.ObjBound if hasattr(m, 'ObjBound') else None
    res['gap'] = m.MIPGap if m.SolCount > 0 else None
    res['nodes'] = int(m.NodeCount)
    res['root_lp'] = root_lp_vals[0] if root_lp_vals else None

    # Extract K_actual from solution
    if m.SolCount > 0:
        res['K_actual'] = int(round(sum(x[0, j].X for j in C)))

    if res['root_lp'] is None:
        m_lp = m.relax()
        m_lp.Params.OutputFlag = 0
        m_lp.optimize()
        if m_lp.Status == GRB.OPTIMAL:
            res['root_lp'] = m_lp.ObjVal

    return res


# ---------------------------------------------------------------------------
# 3. 2CF (Two-Commodity Flow)
# ---------------------------------------------------------------------------
def solve_2cf(dist, demand, capacity, time_limit=300, root_lp_only=False):
    """
    Two-Commodity Flow formulation (Baldacci et al. 2004).

    Two flow variables per arc:
      g1[i,j] — delivery flow (increases along route)
      g2[i,j] — remaining capacity flow (decreases along route)
    Coupling: g1[i,j] + g2[i,j] = Q * x[i,j]

    Theoretically equivalent LP bound to SCF (Letchford & Salazar-Gonzalez 2006).
    """
    n = len(demand)
    C = range(1, n)
    V = range(n)
    Q = capacity
    K_min = math.ceil(sum(demand[i] for i in C) / Q)

    res = _result_dict()
    res['K_min'] = K_min

    m = gp.Model('CVRP_2CF')
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit

    # --- Variables ---
    x = {}
    g1 = {}
    g2 = {}
    for i in V:
        for j in V:
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
                g1[i, j] = m.addVar(lb=0.0, name=f'g1_{i}_{j}')
                g2[i, j] = m.addVar(lb=0.0, name=f'g2_{i}_{j}')

    m.update()

    # --- Objective ---
    m.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for i in V for j in V if i != j),
        GRB.MINIMIZE
    )

    # --- Routing constraints ---
    for j in C:
        m.addConstr(gp.quicksum(x[i, j] for i in V if i != j) == 1, f'in_{j}')
        m.addConstr(gp.quicksum(x[j, i] for i in V if i != j) == 1, f'out_{j}')

    m.addConstr(gp.quicksum(x[0, j] for j in C) >= K_min, 'depot_out')
    m.addConstr(gp.quicksum(x[j, 0] for j in C) >= K_min, 'depot_in')

    # --- Flow conservation (g1): sum_i g1[i,j] - sum_i g1[j,i] = d_j ---
    for j in C:
        m.addConstr(
            gp.quicksum(g1[i, j] for i in V if i != j)
            - gp.quicksum(g1[j, i] for i in V if i != j) == demand[j],
            f'flow1_{j}'
        )

    # --- Flow conservation (g2): sum_i g2[i,j] - sum_i g2[j,i] = -d_j ---
    for j in C:
        m.addConstr(
            gp.quicksum(g2[i, j] for i in V if i != j)
            - gp.quicksum(g2[j, i] for i in V if i != j) == -demand[j],
            f'flow2_{j}'
        )

    # --- Coupling: g1[i,j] + g2[i,j] = Q * x[i,j] ---
    for i in V:
        for j in V:
            if i != j:
                m.addConstr(g1[i, j] + g2[i, j] == Q * x[i, j],
                            f'couple_{i}_{j}')

    # --- Lower bounds (tightening) ---
    for i in V:
        for j in V:
            if i != j:
                # g1[i,j] >= d_j * x[i,j] (delivery must carry at least d_j)
                if j != 0:
                    m.addConstr(g1[i, j] >= demand[j] * x[i, j],
                                f'lb1_{i}_{j}')
                # g2[i,j] >= d_i * x[i,j] (return direction tightening)
                if i != 0:
                    m.addConstr(g2[i, j] >= demand[i] * x[i, j],
                                f'lb2_{i}_{j}')

    m.update()
    res['n_vars'] = m.NumVars
    res['n_constrs'] = m.NumConstrs

    if root_lp_only:
        m_lp = m.relax()
        m_lp.Params.OutputFlag = 0
        m_lp.optimize()
        res['root_lp'] = m_lp.ObjVal if m_lp.Status == GRB.OPTIMAL else None
        res['time'] = m_lp.Runtime
        return res

    root_lp_vals = []

    def _root_cb(model, where):
        if where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
                obj = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                root_lp_vals.append(obj)

    t0 = time.time()
    m.optimize(_root_cb)
    res['time'] = time.time() - t0

    res['status'] = _status_str(m)
    res['obj'] = m.ObjVal if m.SolCount > 0 else None
    res['bound'] = m.ObjBound if hasattr(m, 'ObjBound') else None
    res['gap'] = m.MIPGap if m.SolCount > 0 else None
    res['nodes'] = int(m.NodeCount)
    res['root_lp'] = root_lp_vals[0] if root_lp_vals else None

    if m.SolCount > 0:
        res['K_actual'] = int(round(sum(x[0, j].X for j in C)))

    if res['root_lp'] is None:
        m_lp = m.relax()
        m_lp.Params.OutputFlag = 0
        m_lp.optimize()
        if m_lp.Status == GRB.OPTIMAL:
            res['root_lp'] = m_lp.ObjVal

    return res


# ---------------------------------------------------------------------------
# 4. MCF (Multi-Commodity Flow)
# ---------------------------------------------------------------------------
def solve_mcf(dist, demand, capacity, time_limit=300, root_lp_only=False):
    """
    Strengthened Multi-Commodity Flow formulation.

    One commodity per customer k: depot is source, customer k is sink.
    Variables: f[k,i,j] — flow of commodity k on arc (i,j).

    Strengthening over basic Gavish-Graves MCF:
      - f[k,k,j] = 0: no outflow from sink k (commodity k is absorbed at k)
      - f[k,j,0] = 0: no commodity flows back to depot (depot is pure source)
    These prevent artificial cycling flows that weaken the LP relaxation.

    LP bound is at least as strong as SCF/2CF.
    O(n^3) variables — feasible for n <= ~50.
    """
    n = len(demand)
    C = range(1, n)
    V = range(n)
    Q = capacity
    K_min = math.ceil(sum(demand[i] for i in C) / Q)

    res = _result_dict()
    res['K_min'] = K_min

    m = gp.Model('CVRP_MCF')
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit

    # --- Variables ---
    x = {}
    for i in V:
        for j in V:
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')

    # f[k, i, j] for each commodity k in C and arc (i,j)
    # Strengthening: f[k,k,j]=0 (no outflow from sink), f[k,j,0]=0 (no return to depot)
    f = {}
    for k in C:
        for i in V:
            for j in V:
                if i != j:
                    ub = 0.0 if (i == k or j == 0) else GRB.INFINITY
                    f[k, i, j] = m.addVar(lb=0.0, ub=ub,
                                           name=f'f_{k}_{i}_{j}')

    m.update()

    # --- Objective ---
    m.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for i in V for j in V if i != j),
        GRB.MINIMIZE
    )

    # --- Routing constraints ---
    for j in C:
        m.addConstr(gp.quicksum(x[i, j] for i in V if i != j) == 1, f'in_{j}')
        m.addConstr(gp.quicksum(x[j, i] for i in V if i != j) == 1, f'out_{j}')

    m.addConstr(gp.quicksum(x[0, j] for j in C) >= K_min, 'depot_out')
    m.addConstr(gp.quicksum(x[j, 0] for j in C) >= K_min, 'depot_in')

    # --- Flow conservation for each commodity k ---
    for k in C:
        for j in V:
            flow_in = gp.quicksum(f[k, i, j] for i in V if i != j)
            flow_out = gp.quicksum(f[k, j, i] for i in V if i != j)
            if j == 0:
                m.addConstr(flow_out == demand[k], f'mcf_depot_{k}')
            elif j == k:
                m.addConstr(flow_in == demand[k], f'mcf_sink_{k}')
            else:
                m.addConstr(flow_in - flow_out == 0,
                            f'mcf_transit_{k}_{j}')

    # --- Capacity coupling: sum_k f[k,i,j] <= Q * x[i,j] ---
    for i in V:
        for j in V:
            if i != j:
                m.addConstr(
                    gp.quicksum(f[k, i, j] for k in C) <= Q * x[i, j],
                    f'cap_{i}_{j}'
                )

    # --- Single-flow upper bound: f[k,i,j] <= d_k * x[i,j] ---
    for k in C:
        for i in V:
            for j in V:
                if i != j and i != k and j != 0:  # skip zero-UB vars
                    m.addConstr(f[k, i, j] <= demand[k] * x[i, j],
                                f'fub_{k}_{i}_{j}')

    # --- Aggregate linking (SCF-equivalent tightening) ---
    for i in V:
        for j in V:
            if i != j:
                agg = gp.quicksum(f[k, i, j] for k in C)
                ub_ij = (Q - demand[i]) if i != 0 else Q
                m.addConstr(agg <= ub_ij * x[i, j], f'agg_ub_{i}_{j}')
                if j != 0:
                    m.addConstr(agg >= demand[j] * x[i, j],
                                f'agg_lb_{i}_{j}')

    m.update()
    res['n_vars'] = m.NumVars
    res['n_constrs'] = m.NumConstrs

    if root_lp_only:
        m_lp = m.relax()
        m_lp.Params.OutputFlag = 0
        m_lp.optimize()
        res['root_lp'] = m_lp.ObjVal if m_lp.Status == GRB.OPTIMAL else None
        res['time'] = m_lp.Runtime
        return res

    root_lp_vals = []

    def _root_cb(model, where):
        if where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
                obj = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                root_lp_vals.append(obj)

    t0 = time.time()
    m.optimize(_root_cb)
    res['time'] = time.time() - t0

    res['status'] = _status_str(m)
    res['obj'] = m.ObjVal if m.SolCount > 0 else None
    res['bound'] = m.ObjBound if hasattr(m, 'ObjBound') else None
    res['gap'] = m.MIPGap if m.SolCount > 0 else None
    res['nodes'] = int(m.NodeCount)
    res['root_lp'] = root_lp_vals[0] if root_lp_vals else None

    if m.SolCount > 0:
        res['K_actual'] = int(round(sum(x[0, j].X for j in C)))

    if res['root_lp'] is None:
        m_lp = m.relax()
        m_lp.Params.OutputFlag = 0
        m_lp.optimize()
        if m_lp.Status == GRB.OPTIMAL:
            res['root_lp'] = m_lp.ObjVal

    return res


# ---------------------------------------------------------------------------
# 5. DFJ (Dantzig-Fulkerson-Johnson) with lazy subtour cuts
# ---------------------------------------------------------------------------
def solve_dfj(dist, demand, capacity, time_limit=300, root_lp_only=False):
    """
    DFJ formulation for CVRP with lazy subtour + capacity elimination constraints.

    For every customer subset S, the Capacitated SEC is:
      sum_{i in S, j in S} x_{ij} <= |S| - r(S)
    where r(S) = ceil(sum_{i in S} d_i / Q).
    """
    n = len(demand)
    C = range(1, n)
    V = range(n)
    Q = capacity
    K_min = math.ceil(sum(demand[i] for i in C) / Q)

    res = _result_dict()
    res['K_min'] = K_min

    m = gp.Model('CVRP_DFJ')
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.LazyConstraints = 1

    # --- Variables ---
    x = {}
    for i in V:
        for j in V:
            if i != j:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')

    m.update()

    # --- Objective ---
    m.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for i in V for j in V if i != j),
        GRB.MINIMIZE
    )

    # --- Degree constraints ---
    for j in C:
        m.addConstr(gp.quicksum(x[i, j] for i in V if i != j) == 1, f'in_{j}')
        m.addConstr(gp.quicksum(x[j, i] for i in V if i != j) == 1, f'out_{j}')

    m.addConstr(gp.quicksum(x[0, j] for j in C) >= K_min, 'depot_out')
    m.addConstr(gp.quicksum(x[j, 0] for j in C) >= K_min, 'depot_in')

    m.update()
    res['n_vars'] = m.NumVars
    res['n_constrs'] = m.NumConstrs

    if root_lp_only:
        m_lp = m.relax()
        m_lp.Params.OutputFlag = 0
        m_lp.optimize()
        res['root_lp'] = m_lp.ObjVal if m_lp.Status == GRB.OPTIMAL else None
        res['time'] = m_lp.Runtime
        return res

    # --- Lazy cut callback ---
    root_lp_vals = []

    def _callback(model, where):
        if where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
                obj = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                root_lp_vals.append(obj)

        if where == GRB.Callback.MIPSOL:
            x_vals = {}
            for (i, j) in x:
                x_vals[i, j] = model.cbGetSolution(x[i, j])

            # Build successor map from integer solution
            succ = {i: None for i in V}
            for (i, j), val in x_vals.items():
                if val > 0.5:
                    succ[i] = j

            # Extract routes: follow arcs from depot
            routes = []
            for j in C:
                if x_vals.get((0, j), 0) > 0.5:
                    route = []
                    node = j
                    while node != 0 and node is not None:
                        route.append(node)
                        node = succ.get(node)
                        if len(route) > n:  # safety
                            break
                    routes.append(route)

            # Check each route for capacity violation
            for route in routes:
                S = set(route)
                demand_S = sum(demand[i] for i in S)
                r_S = math.ceil(demand_S / Q)
                if r_S >= 2:
                    model.cbLazy(
                        gp.quicksum(
                            x[i, j] for i in S for j in S
                            if i != j and (i, j) in x
                        ) <= len(S) - r_S
                    )

            # Check for subtours (components not connected to depot)
            on_route = set()
            for route in routes:
                on_route.update(route)

            remaining = set(C) - on_route
            if remaining:
                visited_r = set()
                for start in remaining:
                    if start in visited_r:
                        continue
                    comp = set()
                    stack = [start]
                    while stack:
                        nd = stack.pop()
                        if nd in visited_r:
                            continue
                        visited_r.add(nd)
                        comp.add(nd)
                        nxt = succ.get(nd)
                        if nxt is not None and nxt != 0 and nxt not in visited_r:
                            stack.append(nxt)
                    if comp:
                        demand_S = sum(demand[i] for i in comp)
                        r_S = max(1, math.ceil(demand_S / Q))
                        model.cbLazy(
                            gp.quicksum(
                                x[i, j] for i in comp for j in comp
                                if i != j and (i, j) in x
                            ) <= len(comp) - r_S
                        )

    t0 = time.time()
    m.optimize(_callback)
    res['time'] = time.time() - t0

    res['status'] = _status_str(m)
    res['obj'] = m.ObjVal if m.SolCount > 0 else None
    res['bound'] = m.ObjBound if hasattr(m, 'ObjBound') else None
    res['gap'] = m.MIPGap if m.SolCount > 0 else None
    res['nodes'] = int(m.NodeCount)
    res['root_lp'] = root_lp_vals[0] if root_lp_vals else None

    if m.SolCount > 0:
        res['K_actual'] = int(round(sum(x[0, j].X for j in C)))

    if res['root_lp'] is None:
        m_lp = m.relax()
        m_lp.Params.OutputFlag = 0
        m_lp.optimize()
        if m_lp.Status == GRB.OPTIMAL:
            res['root_lp'] = m_lp.ObjVal

    return res


# ---------------------------------------------------------------------------
# 6. SP (Set Partitioning via Column Generation)
# ---------------------------------------------------------------------------
def solve_sp(dist, demand, capacity, time_limit=300, root_lp_only=False):
    """
    Set Partitioning formulation solved via column generation.

    Requires BPC code (rmp.py, pricing.py) from the BPC-Lab project.
    If not available, returns status='unavailable'.

    LP bound is the strongest (theoretical upper bound on LP relaxation quality).
    Limited by bitmask pricing: practical for n_customers <= 18.
    """
    try:
        from rmp import RMP, make_initial_columns
        from pricing import solve_pricing
    except ImportError:
        res = _result_dict()
        res['status'] = 'unavailable'
        res['time'] = 0.0
        return res

    n = len(demand)
    n_customers = n - 1
    C = range(1, n)
    K_min = math.ceil(sum(demand[i] for i in C) / capacity)

    res = _result_dict()
    res['K_min'] = K_min

    # Bitmask pricing is O(2^n) — skip for large instances
    if n_customers > 18:
        res['status'] = 'timeout'
        res['time'] = 0.0
        res['n_vars'] = 0
        res['n_constrs'] = n_customers
        return res

    t0 = time.time()

    # --- Column generation ---
    init_cols = make_initial_columns(dist, n_customers)
    rmp = RMP(n_customers, init_cols)

    max_iter = 500
    total_cols = len(init_cols)
    cg_converged = False
    cg_time_limit = time_limit * 0.95

    for _ in range(max_iter):
        elapsed = time.time() - t0
        if elapsed > cg_time_limit:
            break

        obj, lambdas, cover_duals, edge_duals, status = rmp.solve()
        if obj is None:
            break

        new_cols = solve_pricing(
            dist, demand, capacity, cover_duals, n_customers,
            forbidden_edges=set(), edge_duals={}
        )

        if not new_cols:
            cg_converged = True
            break

        added = [(c, v, p) for c, v, p, rc in new_cols]
        rmp.add_columns(added)
        total_cols += len(added)

    # Final LP solve
    obj, lambdas, cover_duals, edge_duals, status = rmp.solve()

    if obj is not None and rmp.has_artificial():
        obj = None

    root_lp = obj if cg_converged else None
    res['root_lp'] = root_lp
    res['n_vars'] = total_cols
    res['n_constrs'] = n_customers

    if root_lp_only:
        res['time'] = time.time() - t0
        if not cg_converged:
            res['status'] = 'timeout'
        return res

    if obj is None and not cg_converged:
        res['time'] = time.time() - t0
        res['status'] = 'timeout'
        return res
    elif obj is None:
        res['time'] = time.time() - t0
        res['status'] = 'infeasible'
        return res

    # --- Restricted MIP for integer solution ---
    ip = gp.Model("SP_RestrictedMIP")
    ip.Params.OutputFlag = 0
    remaining_time = max(1.0, time_limit - (time.time() - t0))
    ip.Params.TimeLimit = remaining_time

    ip_vars = []
    for idx, (cost, visits, path) in enumerate(rmp.columns):
        v = ip.addVar(obj=cost, vtype=GRB.BINARY, name=f"r{idx}")
        ip_vars.append(v)
    ip.update()

    for i in range(1, n_customers + 1):
        expr = gp.LinExpr()
        for idx, (cost, visits, path) in enumerate(rmp.columns):
            if i in visits:
                expr.addTerms(1.0, ip_vars[idx])
        ip.addConstr(expr >= 1, name=f"cover_{i}")

    ip.update()
    ip.optimize()

    res['time'] = time.time() - t0

    if ip.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or ip.SolCount > 0:
        res['status'] = 'optimal' if ip.Status == GRB.OPTIMAL else 'feasible'
        res['obj'] = round(ip.ObjVal)
        res['bound'] = ip.ObjBound if hasattr(ip, 'ObjBound') else root_lp
        res['gap'] = ip.MIPGap if ip.SolCount > 0 else None
        res['nodes'] = int(ip.NodeCount)
        k_actual = sum(1 for v in ip_vars if v.X > 0.5)
        res['K_actual'] = k_actual
    elif ip.Status == GRB.TIME_LIMIT:
        res['status'] = 'timeout'
    else:
        res['status'] = 'infeasible'

    return res


# ---------------------------------------------------------------------------
# Convenience: solve all models
# ---------------------------------------------------------------------------
MODELS = {
    'MTZ': solve_mtz,
    'SCF': solve_scf,
    '2CF': solve_2cf,
    'MCF': solve_mcf,
    'DFJ': solve_dfj,
    'SP': solve_sp,
}
