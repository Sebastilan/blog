"""
Edge Branching for Branch-and-Bound
====================================
Aggregates edge flows from LP solution, selects the most fractional edge,
creates two children: forbid (x_ij=0) / force (x_ij>=1).
"""

import math


class BranchConstraint:
    """A single branching decision: forbid or force an edge."""
    __slots__ = ('edge', 'kind')

    def __init__(self, edge, kind):
        """
        Args:
            edge: (i, j) with i < j (normalized)
            kind: 'forbid' or 'force'
        """
        self.edge = edge
        self.kind = kind

    def __repr__(self):
        return f"Branch({self.edge}, {self.kind})"


class BBNode:
    """A node in the Branch-and-Bound tree."""
    __slots__ = ('constraints', 'lb', 'depth', 'id')
    _counter = 0

    def __init__(self, constraints, lb=-float('inf'), depth=0):
        self.constraints = list(constraints)
        self.lb = lb
        self.depth = depth
        BBNode._counter += 1
        self.id = BBNode._counter

    def __lt__(self, other):
        return self.lb < other.lb

    def __repr__(self):
        return f"BBNode(id={self.id}, lb={self.lb:.2f}, d={self.depth})"


def get_branch_sets(node):
    """Extract forbidden and forced edge sets from a BBNode."""
    forbidden = set()
    forced = set()
    for bc in node.constraints:
        if bc.kind == 'forbid':
            forbidden.add(bc.edge)
        else:
            forced.add(bc.edge)
    return forbidden, forced


def compute_edge_flows(lambdas, columns):
    """
    Aggregate edge flows: x[e] = sum_r lambda_r * [route r uses edge e].

    Returns: dict (i,j) with i<j -> flow value
    """
    flows = {}
    for idx, (cost, visits, path) in enumerate(columns):
        lam = lambdas[idx]
        if lam < 1e-8:
            continue
        for k in range(len(path) - 1):
            a, b = path[k], path[k + 1]
            edge = (min(a, b), max(a, b))
            flows[edge] = flows.get(edge, 0.0) + lam
    return flows


def select_branch_edge(flows):
    """
    Select the most fractional edge (closest to 0.5).

    Returns: (i,j) with i<j, or None if all integer.
    """
    best_edge = None
    best_score = -1.0

    for edge, flow in flows.items():
        frac = flow - math.floor(flow)
        if frac < 1e-6:
            continue  # integer
        # Score: distance from 0.5, closer to 0.5 = higher score
        score = 0.5 - abs(frac - 0.5)
        if score > best_score:
            best_score = score
            best_edge = edge

    return best_edge


def create_children(parent, branch_edge, lb):
    """
    Create two child nodes: forbid edge (left) / force edge (right).
    """
    left = BBNode(
        parent.constraints + [BranchConstraint(branch_edge, 'forbid')],
        lb=lb, depth=parent.depth + 1
    )
    right = BBNode(
        parent.constraints + [BranchConstraint(branch_edge, 'force')],
        lb=lb, depth=parent.depth + 1
    )
    return left, right
