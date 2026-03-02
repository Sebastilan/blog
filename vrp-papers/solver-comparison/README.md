# Gurobi vs COPT: Branch-and-Price Solver Benchmark

Benchmark comparing **Gurobi 13.0** and **COPT 8.0.3** on fine-grained operations used in Branch-and-Price (B&P) for CVRP.

## Tests

| # | Test | B&P Scenario |
|---|------|-------------|
| T1 | LP solve speed | CG core loop |
| T2 | Batch add columns | CG add negative RC columns |
| T3 | Batch add/remove constraints | Branching cuts |
| T4 | Variable bound modification | Forbidden edge branching |
| T5 | Dual extraction | CG pricing input |
| T6 | Basis warm start | CG adjacent iterations |
| T7 | MIP solve | Restricted MIP heuristic |
| T8 | End-to-end CG | Full root column generation |

## Requirements

```
Python 3.8+
gurobipy   (requires Gurobi license)
coptpy     (requires COPT license)
vrplib
numpy
```

## Usage

```bash
# Quick run (~10s)
python -X utf8 benchmark.py --quick

# Full run (~30s)
python -X utf8 benchmark.py

# 3 rounds, report median (~90s)
python -X utf8 benchmark.py --rounds 3
```

## Structure

```
solver-comparison/
├── benchmark.py      # Main benchmark (T1-T8)
├── rmp.py            # Gurobi RMP backend
├── rmp_copt.py       # COPT RMP backend
├── instance.py       # CVRP instance loader
├── pricing.py        # ESPPRC label-setting
├── branch.py         # Edge branching
├── utils.py          # Timer, solution helpers
├── blog.md           # Full analysis article
├── data/
│   └── P-n16-k8.vrp  # Benchmark instance (CVRPLIB)
└── results/
    └── *.json         # Experiment data
```

## Blog

See [blog.md](blog.md) for detailed analysis and results.
