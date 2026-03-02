"""
Utility functions for the BPC solver.
"""

import time


class Timer:
    """Simple context-manager timer."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def is_integer_solution(lambdas, tol=1e-6):
    """Check if all lambda values are integer (0 or 1)."""
    for lam in lambdas:
        frac = abs(lam - round(lam))
        if frac > tol:
            return False
    return True


def extract_solution_routes(lambdas, columns, tol=1e-6):
    """
    Extract active routes from an integer LP solution.

    Returns: list of (cost, path) for routes with lambda ≈ 1
    """
    routes = []
    for idx, lam in enumerate(lambdas):
        if lam > 1 - tol:
            cost, visits, path = columns[idx]
            routes.append((cost, path))
    return routes
