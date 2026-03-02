"""
COPT-based RMP — API-compatible drop-in replacement for rmp.py (Gurobi).

Identical interface: RMP(n_customers, initial_columns),
    add_columns, set_branching, solve, has_artificial, artificial_cost.
"""

import coptpy as cp
from coptpy import COPT

BIG_M = 1e6

# Shared COPT environment (one per process)
_ENV = cp.Envr()


def _route_edges(path):
    edges = set()
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        edges.add((min(a, b), max(a, b)))
    return edges


class RMP:
    """Restricted Master Problem for CVRP — COPT backend."""

    def __init__(self, n_customers, initial_columns):
        self.n = n_customers
        self.model = _ENV.createModel("RMP")
        self.model.setParam(COPT.Param.Logging, 0)

        self.columns = []
        self.vars = []
        self._col_edges = []

        # Covering constraints: art_i >= 1  (will add column terms later)
        self.constrs = []
        self.cover_art = []
        for i in range(self.n):
            art = self.model.addVar(obj=BIG_M, lb=0,
                                    vtype=COPT.CONTINUOUS,
                                    name=f"art_cover_{i+1}")
            self.cover_art.append(art)

        for i in range(self.n):
            c = self.model.addConstr(self.cover_art[i] >= 1,
                                     name=f"cover_{i+1}")
            self.constrs.append(c)

        # Edge branching state
        self.edge_constrs = {}
        self.edge_art = {}
        self.forbidden_edges = set()
        self._cur_forbidden = set()
        self._cur_forced = set()

        self.model.setObjSense(COPT.MINIMIZE)

        for cost, visits, path in initial_columns:
            self._add_column_internal(cost, visits, path)

    def _add_column_internal(self, cost, visits, path):
        terms = []
        for v in visits:
            terms.append((self.constrs[v - 1], 1.0))

        edges = _route_edges(path)
        for edge, constr in self.edge_constrs.items():
            if edge in edges:
                terms.append((constr, 1.0))

        uses_forbidden = bool(edges & self.forbidden_edges)
        ub = 0.0 if uses_forbidden else COPT.INFINITY

        col = cp.Column(terms) if terms else cp.Column()
        idx = len(self.vars)
        var = self.model.addVar(
            obj=cost, lb=0.0, ub=ub,
            vtype=COPT.CONTINUOUS,
            name=f"r{idx}",
            column=col
        )
        self.columns.append((cost, visits, path))
        self.vars.append(var)
        self._col_edges.append(edges)

    def add_columns(self, new_cols):
        for cost, visits, path in new_cols:
            self._add_column_internal(cost, visits, path)

    def set_branching(self, forbidden_edges, forced_edges):
        new_forbidden = {(min(a, b), max(a, b)) for a, b in forbidden_edges}
        new_forced = {(min(a, b), max(a, b)) for a, b in forced_edges}

        to_unforbid = self._cur_forbidden - new_forbidden
        to_forbid = new_forbidden - self._cur_forbidden

        if to_unforbid:
            for idx, edges_set in enumerate(self._col_edges):
                if edges_set & to_unforbid:
                    if not (edges_set & new_forbidden):
                        self.vars[idx].ub = COPT.INFINITY

        if to_forbid:
            for idx, edges_set in enumerate(self._col_edges):
                if edges_set & to_forbid:
                    self.vars[idx].ub = 0.0

        for edge in self._cur_forced - new_forced:
            self.model.remove(self.edge_constrs.pop(edge))
            self.model.remove(self.edge_art.pop(edge))

        edges_to_add = new_forced - self._cur_forced
        if edges_to_add:
            for edge in edges_to_add:
                art = self.model.addVar(
                    obj=BIG_M, lb=0, vtype=COPT.CONTINUOUS,
                    name=f"art_force_{edge[0]}_{edge[1]}")
                self.edge_art[edge] = art

            for edge in edges_to_add:
                expr = self.edge_art[edge]
                for idx, edges_set in enumerate(self._col_edges):
                    if self.vars[idx].ub > 0 and edge in edges_set:
                        expr = expr + self.vars[idx]
                constr = self.model.addConstr(expr >= 1,
                                              name=f"force_{edge[0]}_{edge[1]}")
                self.edge_constrs[edge] = constr

        self.forbidden_edges = new_forbidden
        self._cur_forbidden = new_forbidden
        self._cur_forced = new_forced

    def solve(self):
        self.model.solveLP()

        if self.model.LpStatus == 1:  # OPTIMAL
            obj = self.model.LpObjval
            lambdas = [v.x for v in self.vars]
            cover_duals = [c.pi for c in self.constrs]
            edge_duals = {e: c.pi for e, c in self.edge_constrs.items()}
            return obj, lambdas, cover_duals, edge_duals, 1
        else:
            return None, None, None, None, self.model.LpStatus

    def has_artificial(self, tol=1e-6):
        for art in self.cover_art:
            if art.x > tol:
                return True
        for art in self.edge_art.values():
            if art.x > tol:
                return True
        return False

    def artificial_cost(self):
        total = 0.0
        for art in self.cover_art:
            total += art.x * BIG_M
        for art in self.edge_art.values():
            total += art.x * BIG_M
        return total


def make_initial_columns(dist, n_customers):
    columns = []
    for i in range(1, n_customers + 1):
        cost = int(dist[0][i] + dist[i][0])
        columns.append((cost, [i], [0, i, 0]))
    return columns
