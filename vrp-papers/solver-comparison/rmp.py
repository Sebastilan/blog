"""
Restricted Master Problem (RMP)
===============================
Gurobi-based set-covering formulation for CVRP column generation.

    min  sum_r  c_r * lambda_r  +  M * sum artificial
    s.t. sum_r  a_{ir} * lambda_r  + art_i  >= 1   for each customer i
         sum_r  b_{er} * lambda_r  + art_e  >= 1   for each forced edge e
         lambda_r >= 0
         lambda_r = 0                    for routes using forbidden edges

Artificial variables (cost M=1e6) ensure the RMP is always feasible,
so CG can always extract duals. If any artificial > 0 after CG converges,
the node is truly infeasible.
"""

import gurobipy as gp
from gurobipy import GRB

BIG_M = 1e6


def _route_edges(path):
    """Extract normalized edge set {(min(a,b), max(a,b))} from a path."""
    edges = set()
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        edges.add((min(a, b), max(a, b)))
    return edges


class RMP:
    """Restricted Master Problem for CVRP."""

    def __init__(self, n_customers, initial_columns):
        self.n = n_customers
        self.model = gp.Model("RMP")
        self.model.Params.OutputFlag = 0

        self.columns = []   # (cost, visits, path)
        self.vars = []

        # Precomputed edge sets per column (avoid recomputing)
        self._col_edges = []  # [set of normalized edges] per column

        # Covering constraints: sum_r a_{ir} * lambda_r + art_i >= 1
        self.constrs = []
        self.cover_art = []  # artificial vars for covering
        for i in range(self.n):
            art = self.model.addVar(obj=BIG_M, lb=0, vtype=GRB.CONTINUOUS,
                                    name=f"art_cover_{i+1}")
            self.cover_art.append(art)
        self.model.update()

        for i in range(self.n):
            c = self.model.addConstr(
                self.cover_art[i] >= 1, name=f"cover_{i+1}")
            self.constrs.append(c)

        # Edge branching state (incremental)
        self.edge_constrs = {}     # edge -> Gurobi Constr
        self.edge_art = {}         # edge -> artificial var
        self.forbidden_edges = set()
        self._cur_forbidden = set()   # currently applied forbidden
        self._cur_forced = set()      # currently applied forced

        # Add initial columns
        for cost, visits, path in initial_columns:
            self._add_column_internal(cost, visits, path)

        self.model.update()

    def _add_column_internal(self, cost, visits, path):
        """Add a column with coefficients for all active constraints."""
        col = gp.Column()
        for v in visits:
            col.addTerms(1.0, self.constrs[v - 1])

        edges = _route_edges(path)
        for edge, constr in self.edge_constrs.items():
            if edge in edges:
                col.addTerms(1.0, constr)

        uses_forbidden = bool(edges & self.forbidden_edges)

        idx = len(self.vars)
        var = self.model.addVar(
            obj=cost, lb=0.0,
            ub=0.0 if uses_forbidden else GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"r{idx}",
            column=col
        )
        self.columns.append((cost, visits, path))
        self.vars.append(var)
        self._col_edges.append(edges)


    def add_columns(self, new_cols):
        """Add multiple columns. Each: (cost, visits, path)."""
        for cost, visits, path in new_cols:
            self._add_column_internal(cost, visits, path)
        self.model.update()

    def set_branching(self, forbidden_edges, forced_edges):
        """Configure branching incrementally — only process diff."""
        new_forbidden = {(min(a, b), max(a, b)) for a, b in forbidden_edges}
        new_forced = {(min(a, b), max(a, b)) for a, b in forced_edges}

        to_unforbid = self._cur_forbidden - new_forbidden
        to_forbid = new_forbidden - self._cur_forbidden

        # 1. Undo removed forbidden (restore UB=∞ if no other forbidden)
        if to_unforbid:
            for idx, edges_set in enumerate(self._col_edges):
                if edges_set & to_unforbid:
                    if not (edges_set & new_forbidden):
                        self.vars[idx].UB = GRB.INFINITY

        # 2. Apply new forbidden (set UB=0)
        if to_forbid:
            for idx, edges_set in enumerate(self._col_edges):
                if edges_set & to_forbid:
                    self.vars[idx].UB = 0.0

        # 3. Undo removed forced (remove constraint + art)
        for edge in self._cur_forced - new_forced:
            self.model.remove(self.edge_constrs.pop(edge))
            self.model.remove(self.edge_art.pop(edge))

        # 4. Apply new forced (add art + constraint)
        edges_to_add = new_forced - self._cur_forced
        if edges_to_add:
            for edge in edges_to_add:
                art = self.model.addVar(
                    obj=BIG_M, lb=0, vtype=GRB.CONTINUOUS,
                    name=f"art_force_{edge[0]}_{edge[1]}")
                self.edge_art[edge] = art
            self.model.update()

            for edge in edges_to_add:
                expr = gp.LinExpr()
                expr.addTerms(1.0, self.edge_art[edge])
                for idx, edges_set in enumerate(self._col_edges):
                    if self.vars[idx].UB > 0 and edge in edges_set:
                        expr.addTerms(1.0, self.vars[idx])
                constr = self.model.addConstr(expr >= 1,
                                              name=f"force_{edge[0]}_{edge[1]}")
                self.edge_constrs[edge] = constr

        self.forbidden_edges = new_forbidden
        self._cur_forbidden = new_forbidden
        self._cur_forced = new_forced
        self.model.update()

    def solve(self):
        """
        Solve LP relaxation.

        Returns:
            obj, lambdas, cover_duals, edge_duals, status
        """
        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            obj = self.model.ObjVal
            lambdas = [v.X for v in self.vars]
            cover_duals = [c.Pi for c in self.constrs]
            edge_duals = {e: c.Pi for e, c in self.edge_constrs.items()}
            return obj, lambdas, cover_duals, edge_duals, self.model.Status
        else:
            return None, None, None, None, self.model.Status

    def has_artificial(self, tol=1e-6):
        """Check if any artificial variable is used in the current solution."""
        for art in self.cover_art:
            if art.X > tol:
                return True
        for art in self.edge_art.values():
            if art.X > tol:
                return True
        return False

    def artificial_cost(self):
        """Total cost of artificial variables in current solution."""
        total = 0.0
        for art in self.cover_art:
            total += art.X * BIG_M
        for art in self.edge_art.values():
            total += art.X * BIG_M
        return total


def make_initial_columns(dist, n_customers):
    """Create trivial initial columns: one customer per vehicle."""
    columns = []
    for i in range(1, n_customers + 1):
        cost = int(dist[0][i] + dist[i][0])
        columns.append((cost, [i], [0, i, 0]))
    return columns
