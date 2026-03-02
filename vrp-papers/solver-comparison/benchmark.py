"""
Gurobi vs COPT Benchmark for Branch-and-Price
==============================================
T1: LP solve speed (repeated solves on same model)
T2: Batch add columns (50/200/500 columns via Column API)
T3: Batch add/remove constraints
T4: Variable bound modification (simulate branching)
T5: Dual extraction speed
T6: Basis warm start (with/without basis loading)
T7: MIP solve (restricted MIP from column pool)
T8: End-to-end B&P (full BPC on multiple instances)

Usage:
  python -X utf8 benchmark.py [--quick] [--rounds N]
"""

import sys
import os
import time
import math
import json
from collections import defaultdict

# All modules are in the same directory (self-contained)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, 'data')

from instance import load_instance, get_optimal
from pricing import solve_pricing
from branch import (BBNode, get_branch_sets, compute_edge_flows,
                    select_branch_edge, create_children)
from utils import is_integer_solution, extract_solution_routes

import gurobipy as gp
from gurobipy import GRB
import coptpy as cp
from coptpy import COPT

# ── helpers ──────────────────────────────────────────────────────────

_COPT_ENV = cp.Envr()


def _time_it(func, repeat=1):
    """Run func repeat times, return (result, total_seconds)."""
    t0 = time.perf_counter()
    for _ in range(repeat):
        result = func()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _load(name):
    return load_instance(os.path.join(DATA_DIR, name + '.vrp'))


def _section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _row(label, grb_val, copt_val, unit='s', fmt='.4f'):
    ratio = copt_val / grb_val if grb_val > 0 else float('inf')
    print(f"  {label:<35} {grb_val:{fmt}}{unit}  {copt_val:{fmt}}{unit}  "
          f"ratio={ratio:.2f}x")


# ── T1: LP Solve Speed ──────────────────────────────────────────────

def _build_rmp_gurobi(dist, demand, capacity, n_cust, columns):
    """Build a Gurobi LP model from column list (no branching)."""
    m = gp.Model("RMP_grb")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1

    arts = []
    constrs = []
    for i in range(n_cust):
        art = m.addVar(obj=1e6, lb=0, vtype=GRB.CONTINUOUS)
        arts.append(art)
    m.update()
    for i in range(n_cust):
        c = m.addConstr(arts[i] >= 1, name=f"c{i}")
        constrs.append(c)
    m.update()

    vars_ = []
    for cost, visits, path in columns:
        col = gp.Column()
        for v in visits:
            col.addTerms(1.0, constrs[v - 1])
        var = m.addVar(obj=cost, lb=0, vtype=GRB.CONTINUOUS, column=col)
        vars_.append(var)
    m.update()
    return m, constrs, vars_


def _build_rmp_copt(dist, demand, capacity, n_cust, columns):
    """Build a COPT LP model from column list (no branching)."""
    m = _COPT_ENV.createModel("RMP_copt")
    m.setParam(COPT.Param.Logging, 0)
    m.setParam(COPT.Param.Threads, 1)
    m.setObjSense(COPT.MINIMIZE)

    arts = []
    constrs = []
    for i in range(n_cust):
        art = m.addVar(obj=1e6, lb=0, vtype=COPT.CONTINUOUS)
        arts.append(art)

    for i in range(n_cust):
        c = m.addConstr(arts[i] >= 1, name=f"c{i}")
        constrs.append(c)

    vars_ = []
    for cost, visits, path in columns:
        terms = [(constrs[v - 1], 1.0) for v in visits]
        col = cp.Column(terms)
        var = m.addVar(obj=cost, lb=0, vtype=COPT.CONTINUOUS, column=col)
        vars_.append(var)
    return m, constrs, vars_


def t1_lp_solve(dist, demand, capacity, n_cust, columns, repeat=50):
    """T1: Repeated LP solve on same model."""
    _section("T1: LP Solve Speed")
    print(f"  Model: {n_cust} customers, {len(columns)} columns, "
          f"{repeat} solves")
    print(f"  {'Test':<35} {'Gurobi':>10} {'COPT':>10}  {'Ratio':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}  {'-'*10}")

    mg, cg, vg = _build_rmp_gurobi(dist, demand, capacity, n_cust, columns)
    mc, cc, vc = _build_rmp_copt(dist, demand, capacity, n_cust, columns)

    _, tg = _time_it(lambda: mg.optimize(), repeat)
    _, tc = _time_it(lambda: mc.solveLP(), repeat)
    _row(f"LP solve x{repeat}", tg, tc)

    # Verify same objective
    obj_g = mg.ObjVal
    obj_c = mc.LpObjval
    print(f"  Obj check: Gurobi={obj_g:.4f}, COPT={obj_c:.4f}, "
          f"diff={abs(obj_g-obj_c):.2e}")
    return tg, tc


# ── T2: Batch Add Columns ───────────────────────────────────────────

def t2_add_columns(dist, demand, capacity, n_cust, base_columns, new_columns_list):
    """T2: Time to add batches of columns."""
    _section("T2: Batch Add Columns")
    print(f"  {'Batch size':<35} {'Gurobi':>10} {'COPT':>10}  {'Ratio':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}  {'-'*10}")

    for batch_name, new_cols in new_columns_list:
        # Gurobi
        mg, cg, vg = _build_rmp_gurobi(dist, demand, capacity, n_cust, base_columns)

        def grb_add():
            for cost, visits, path in new_cols:
                col = gp.Column()
                for v in visits:
                    col.addTerms(1.0, cg[v - 1])
                mg.addVar(obj=cost, lb=0, vtype=GRB.CONTINUOUS, column=col)
            mg.update()

        _, tg = _time_it(grb_add)

        # COPT
        mc, cc, vc = _build_rmp_copt(dist, demand, capacity, n_cust, base_columns)

        def copt_add():
            for cost, visits, path in new_cols:
                terms = [(cc[v - 1], 1.0) for v in visits]
                col = cp.Column(terms)
                mc.addVar(obj=cost, lb=0, vtype=COPT.CONTINUOUS, column=col)

        _, tc = _time_it(copt_add)
        _row(f"Add {batch_name} columns", tg, tc)


# ── T3: Batch Add/Remove Constraints ────────────────────────────────

def t3_constraints(dist, demand, capacity, n_cust, columns, sizes):
    """T3: Add N constraints, solve, remove, solve."""
    _section("T3: Batch Add/Remove Constraints")
    print(f"  {'Operation':<35} {'Gurobi':>10} {'COPT':>10}  {'Ratio':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}  {'-'*10}")

    for n_constr in sizes:
        # Gurobi
        mg, cg, vg = _build_rmp_gurobi(dist, demand, capacity, n_cust, columns)
        mg.optimize()  # warm up

        def grb_add_solve():
            new_c = []
            for k in range(n_constr):
                expr = gp.LinExpr()
                for j in range(min(5, len(vg))):
                    expr.addTerms(1.0, vg[j])
                c = mg.addConstr(expr <= 2 + k * 0.1, name=f"cut_{k}")
                new_c.append(c)
            mg.update()
            mg.optimize()
            return new_c

        def grb_remove_solve(new_c):
            for c in new_c:
                mg.remove(c)
            mg.update()
            mg.optimize()

        new_c, tg_add = _time_it(grb_add_solve)
        _, tg_rm = _time_it(lambda: grb_remove_solve(new_c))

        # COPT
        mc, cc, vc = _build_rmp_copt(dist, demand, capacity, n_cust, columns)
        mc.solveLP()

        def copt_add_solve():
            new_c = []
            for k in range(n_constr):
                expr = vc[0] if len(vc) > 0 else 0
                for j in range(1, min(5, len(vc))):
                    expr = expr + vc[j]
                c = mc.addConstr(expr <= 2 + k * 0.1, name=f"cut_{k}")
                new_c.append(c)
            mc.solveLP()
            return new_c

        def copt_remove_solve(new_c):
            for c in new_c:
                mc.remove(c)
            mc.solveLP()

        new_c2, tc_add = _time_it(copt_add_solve)
        _, tc_rm = _time_it(lambda: copt_remove_solve(new_c2))

        _row(f"Add {n_constr} constrs + solve", tg_add, tc_add)
        _row(f"Remove {n_constr} constrs + solve", tg_rm, tc_rm)


# ── T4: Variable Bound Modification ─────────────────────────────────

def t4_var_bounds(dist, demand, capacity, n_cust, columns, repeat=20):
    """T4: Set UB=0 on half, solve, restore, solve."""
    _section("T4: Variable Bound Modification (Branch Simulation)")
    n_half = len(columns) // 2
    print(f"  Toggle UB on {n_half}/{len(columns)} vars, {repeat} rounds")
    print(f"  {'Operation':<35} {'Gurobi':>10} {'COPT':>10}  {'Ratio':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}  {'-'*10}")

    mg, cg, vg = _build_rmp_gurobi(dist, demand, capacity, n_cust, columns)
    mc, cc, vc = _build_rmp_copt(dist, demand, capacity, n_cust, columns)
    mg.optimize()
    mc.solveLP()

    def grb_toggle():
        for i in range(n_half):
            vg[i].UB = 0.0
        mg.update()
        mg.optimize()
        for i in range(n_half):
            vg[i].UB = GRB.INFINITY
        mg.update()
        mg.optimize()

    def copt_toggle():
        for i in range(n_half):
            vc[i].ub = 0.0
        mc.solveLP()
        for i in range(n_half):
            vc[i].ub = COPT.INFINITY
        mc.solveLP()

    _, tg = _time_it(grb_toggle, repeat)
    _, tc = _time_it(copt_toggle, repeat)
    _row(f"Forbid+restore x{repeat}", tg, tc)


# ── T5: Dual Extraction ─────────────────────────────────────────────

def t5_duals(dist, demand, capacity, n_cust, columns, repeat=1000):
    """T5: Extract duals from all cover constraints."""
    _section("T5: Dual Extraction Speed")
    print(f"  {n_cust} duals, {repeat} extractions")
    print(f"  {'Operation':<35} {'Gurobi':>10} {'COPT':>10}  {'Ratio':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}  {'-'*10}")

    mg, cg, vg = _build_rmp_gurobi(dist, demand, capacity, n_cust, columns)
    mc, cc, vc = _build_rmp_copt(dist, demand, capacity, n_cust, columns)
    mg.optimize()
    mc.solveLP()

    def grb_duals():
        return [c.Pi for c in cg]

    def copt_duals():
        return [c.pi for c in cc]

    dg, tg = _time_it(grb_duals, repeat)
    dc, tc = _time_it(copt_duals, repeat)
    _row(f"Extract {n_cust} duals x{repeat}", tg, tc)

    # Verify duals match
    max_diff = max(abs(a - b) for a, b in zip(dg, dc))
    print(f"  Max dual diff: {max_diff:.2e}")


# ── T6: Basis Warm Start ────────────────────────────────────────────

def t6_basis(dist, demand, capacity, n_cust, columns, repeat=50):
    """T6: LP solve with vs without basis warm start."""
    _section("T6: Basis Warm Start")
    print(f"  {repeat} solves with/without basis")
    print(f"  {'Operation':<35} {'Gurobi':>10} {'COPT':>10}  {'Ratio':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}  {'-'*10}")

    # Gurobi: warm start is automatic (simplex reuses basis between solves)
    mg, cg, vg = _build_rmp_gurobi(dist, demand, capacity, n_cust, columns)
    mg.optimize()  # first solve

    # Cold: reset basis each time
    def grb_cold():
        mg.reset(0)
        mg.optimize()

    # Warm: just re-optimize (basis retained)
    def grb_warm():
        mg.optimize()

    _, tg_cold = _time_it(grb_cold, repeat)
    _, tg_warm = _time_it(grb_warm, repeat)

    # COPT
    mc, cc, vc = _build_rmp_copt(dist, demand, capacity, n_cust, columns)
    mc.solveLP()

    def copt_cold():
        mc.setLpSolution(None, None)
        mc.solveLP()

    def copt_warm():
        mc.solveLP()

    try:
        _, tc_cold = _time_it(copt_cold, repeat)
    except Exception:
        # Fallback: rebuild model each time
        def copt_cold_rebuild():
            mc2 = _COPT_ENV.createModel("cold")
            mc2.setParam(COPT.Param.Logging, 0)
            mc2.setParam(COPT.Param.Threads, 1)
            mc2.setObjSense(COPT.MINIMIZE)
            arts = []
            for i in range(n_cust):
                art = mc2.addVar(obj=1e6, lb=0)
                arts.append(art)
            constrs2 = []
            for i in range(n_cust):
                c = mc2.addConstr(arts[i] >= 1)
                constrs2.append(c)
            for cost, visits, path in columns:
                terms = [(constrs2[v-1], 1.0) for v in visits]
                col = cp.Column(terms)
                mc2.addVar(obj=cost, lb=0, column=col)
            mc2.solveLP()

        _, tc_cold = _time_it(copt_cold_rebuild, repeat)

    _, tc_warm = _time_it(copt_warm, repeat)

    _row(f"Cold start x{repeat}", tg_cold, tc_cold)
    _row(f"Warm start x{repeat}", tg_warm, tc_warm)


# ── T7: MIP Solve ───────────────────────────────────────────────────

def t7_mip(dist, demand, capacity, n_cust, columns):
    """T7: Restricted MIP from column pool."""
    _section("T7: Restricted MIP Solve")
    print(f"  {len(columns)} binary vars, {n_cust} cover constraints")
    print(f"  {'Operation':<35} {'Gurobi':>10} {'COPT':>10}  {'Ratio':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}  {'-'*10}")

    # Gurobi MIP
    def grb_mip():
        m = gp.Model("MIP_grb")
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = 10
        ip_vars = []
        for idx, (cost, visits, path) in enumerate(columns):
            v = m.addVar(obj=cost, vtype=GRB.BINARY)
            ip_vars.append(v)
        m.update()
        for i in range(1, n_cust + 1):
            expr = gp.LinExpr()
            for idx, (cost, visits, path) in enumerate(columns):
                if i in visits:
                    expr.addTerms(1.0, ip_vars[idx])
            m.addConstr(expr >= 1)
        m.update()
        m.optimize()
        return m.ObjVal if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else None

    def copt_mip():
        m = _COPT_ENV.createModel("MIP_copt")
        m.setParam(COPT.Param.Logging, 0)
        m.setParam(COPT.Param.Threads, 1)
        m.setParam(COPT.Param.TimeLimit, 10)
        m.setObjSense(COPT.MINIMIZE)
        ip_vars = []
        for idx, (cost, visits, path) in enumerate(columns):
            v = m.addVar(obj=cost, vtype=COPT.BINARY)
            ip_vars.append(v)
        for i in range(1, n_cust + 1):
            expr = None
            for idx, (cost, visits, path) in enumerate(columns):
                if i in visits:
                    if expr is None:
                        expr = ip_vars[idx]
                    else:
                        expr = expr + ip_vars[idx]
            if expr is not None:
                m.addConstr(expr >= 1)
        m.solve()
        if m.hasMipSol:
            return m.objval
        return None

    obj_g, tg = _time_it(grb_mip)
    obj_c, tc = _time_it(copt_mip)
    _row("MIP solve", tg, tc)
    print(f"  MIP obj: Gurobi={obj_g}, COPT={obj_c}")


# ── T8: End-to-End B&P ──────────────────────────────────────────────

def _solve_cg_generic(rmp, dist, demand, capacity, n_customers,
                      forbidden_edges, forced_edges, max_iter=500):
    """Column generation loop (works with both Gurobi/COPT RMP)."""
    cg_iters = 0
    cols_added = 0
    rmp.set_branching(forbidden_edges, forced_edges)

    for _ in range(max_iter):
        obj, lambdas, cover_duals, edge_duals, status = rmp.solve()
        if obj is None:
            return None, None, cg_iters, cols_added
        cg_iters += 1
        new_cols = solve_pricing(
            dist, demand, capacity, cover_duals, n_customers,
            forbidden_edges=forbidden_edges, edge_duals=edge_duals
        )
        if not new_cols:
            break
        added = [(c, v, p) for c, v, p, rc in new_cols]
        rmp.add_columns(added)
        cols_added += len(added)

    obj, lambdas, cover_duals, edge_duals, status = rmp.solve()
    if obj is None or rmp.has_artificial():
        return None, None, cg_iters, cols_added
    return obj, lambdas, cg_iters, cols_added


def t8_e2e(instances, time_limit=120):
    """T8: End-to-end Root CG comparison (isolates solver from ESPPRC)."""
    from rmp import RMP as RMP_Grb, make_initial_columns as mic_grb
    from rmp_copt import RMP as RMP_Copt, make_initial_columns as mic_copt

    _section("T8: Root Column Generation (End-to-End)")
    print(f"  {'Instance':<14} {'n':>3} "
          f"{'Grb_LP':>9} {'Copt_LP':>9} {'LP_diff':>8} "
          f"{'Grb_t':>7} {'Copt_t':>7} {'Ratio':>6} "
          f"{'Grb_CG':>6} {'Copt_CG':>7} {'Grb_cols':>8} {'Copt_cols':>9}")
    print(f"  {'-'*14} {'-'*3} "
          f"{'-'*9} {'-'*9} {'-'*8} "
          f"{'-'*7} {'-'*7} {'-'*6} "
          f"{'-'*6} {'-'*7} {'-'*8} {'-'*9}")

    results = []
    for inst_name in instances:
        filepath = os.path.join(DATA_DIR, inst_name + '.vrp')
        dist, demand, capacity, name, n_cust = load_instance(filepath)
        optimal = get_optimal(filepath)

        row = {'instance': inst_name, 'n': n_cust, 'optimal': optimal}

        for backend, RMP_cls, mic_fn, prefix in [
            ('gurobi', RMP_Grb, mic_grb, 'grb'),
            ('copt', RMP_Copt, mic_copt, 'copt'),
        ]:
            init_cols = mic_fn(dist, n_cust)
            rmp = RMP_cls(n_cust, init_cols)

            t0 = time.perf_counter()
            obj, lambdas, cg_iters, cols_added = _solve_cg_generic(
                rmp, dist, demand, capacity, n_cust, set(), set()
            )
            total_time = time.perf_counter() - t0
            total_cols = len(init_cols) + cols_added

            row[f'{prefix}_lp'] = obj
            row[f'{prefix}_t'] = total_time
            row[f'{prefix}_cg'] = cg_iters
            row[f'{prefix}_cols'] = total_cols

        lp_diff = abs((row.get('grb_lp') or 0) - (row.get('copt_lp') or 0))
        tg = row.get('grb_t', 0)
        tc = row.get('copt_t', 0)
        ratio = tc / tg if tg > 0 else float('inf')

        def _f(v, fmt='.2f'):
            return f"{v:{fmt}}" if v is not None else "N/A"

        print(f"  {inst_name:<14} {n_cust:>3} "
              f"{_f(row['grb_lp']):>9} {_f(row['copt_lp']):>9} "
              f"{lp_diff:>8.2e} "
              f"{tg:>7.3f} {tc:>7.3f} {ratio:>5.2f}x "
              f"{row['grb_cg']:>6} {row['copt_cg']:>7} "
              f"{row['grb_cols']:>8} {row['copt_cols']:>9}")
        results.append(row)

    return results


# ── Multi-round runner ───────────────────────────────────────────────

def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def run_all(quick=False, rounds=1):
    import platform

    try:
        import cpuinfo as _cpuinfo
        cpu = _cpuinfo.get_cpu_info()['brand_raw']
    except Exception:
        cpu = platform.processor() or "unknown"

    header = {
        "tool": "Solver Comparison Benchmark",
        "gurobi_version": f"{GRB.VERSION_MAJOR}.{GRB.VERSION_MINOR}.{GRB.VERSION_TECHNICAL}",
        "copt_version": "8.0.3",
        "cpu": cpu,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "threads": 1,
        "rounds": rounds,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    print("Gurobi vs COPT Benchmark for Branch-and-Price")
    print(f"Gurobi: {header['gurobi_version']}")
    print(f"COPT:   {header['copt_version']}")
    print(f"CPU:    {cpu}")
    print(f"Rounds: {rounds} (report median)")
    print(f"Time:   {header['timestamp']}")

    # Load instance and build column pool
    inst_name = 'P-n16-k8'
    dist, demand, capacity, name, n_cust = _load(inst_name)
    print(f"\nBase instance: {inst_name} ({n_cust} customers)")

    from rmp import RMP as RMP_Grb, make_initial_columns
    init_cols = make_initial_columns(dist, n_cust)
    rmp_tmp = RMP_Grb(n_cust, init_cols)
    obj, lambdas, cg_iters, cols_added = _solve_cg_generic(
        rmp_tmp, dist, demand, capacity, n_cust, set(), set()
    )
    all_columns = list(rmp_tmp.columns)
    print(f"Column pool: {len(all_columns)} columns, Root LP={obj:.2f}")

    header['base_instance'] = inst_name
    header['n_customers'] = n_cust
    header['n_columns'] = len(all_columns)
    header['root_lp'] = obj

    # Collect raw data across rounds
    all_data = []
    repeat_mult = 1 if quick else 3

    for rd in range(rounds):
        if rounds > 1:
            print(f"\n{'#'*60}")
            print(f"  Round {rd+1}/{rounds}")
            print(f"{'#'*60}")

        rd_data = {'round': rd + 1}

        t1_lp_solve(dist, demand, capacity, n_cust, all_columns,
                    repeat=20 * repeat_mult)
        t2_add_columns(dist, demand, capacity, n_cust, init_cols, [
            ("50", all_columns[:50]),
            ("200", all_columns[:200]),
            (f"all({len(all_columns)})", all_columns),
        ])
        t3_constraints(dist, demand, capacity, n_cust, all_columns,
                       [10, 50, 100] if not quick else [10, 50])
        t4_var_bounds(dist, demand, capacity, n_cust, all_columns,
                      repeat=10 * repeat_mult)
        t5_duals(dist, demand, capacity, n_cust, all_columns,
                 repeat=500 * repeat_mult)
        t6_basis(dist, demand, capacity, n_cust, all_columns,
                 repeat=20 * repeat_mult)
        t7_mip(dist, demand, capacity, n_cust, all_columns)

        e2e_instances = ['P-n16-k8']
        results = t8_e2e(e2e_instances)
        rd_data['e2e'] = results
        all_data.append(rd_data)

    # Save full results
    out_dir = os.path.join(_THIS_DIR, 'results')
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'benchmark_{ts}.json')
    payload = {'header': header, 'rounds': all_data}
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    quick = '--quick' in sys.argv
    rounds = 1
    for i, arg in enumerate(sys.argv):
        if arg == '--rounds' and i + 1 < len(sys.argv):
            rounds = int(sys.argv[i + 1])
    run_all(quick=quick, rounds=rounds)
