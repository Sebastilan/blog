"""
Benchmark: MIP (Gurobi) solve time for ESPPRC instances.
Runs mip_oracle on all testset instances, records wall-clock time.
"""
import json, time, os, sys, glob

sys.path.insert(0, os.path.dirname(__file__))
from espprc_oracle import mip_oracle

INSTANCE_DIR = os.path.join(os.path.dirname(__file__), "instances")
TIME_LIMIT = 120  # seconds

def run_benchmark():
    files = sorted(glob.glob(os.path.join(INSTANCE_DIR, "*.json")))
    results = []

    for fpath in files:
        name = os.path.basename(fpath).replace(".json", "")
        if name == "manifest":
            continue

        with open(fpath) as f:
            data = json.load(f)

        n = data["n_customers"]
        dist = data["dist"]
        demand = data["demand"]
        capacity = data["capacity"]
        duals = data["cover_duals"]

        print(f"Solving {name} (n={n})...", end=" ", flush=True)

        t0 = time.perf_counter()
        opt_rc, opt_path, opt_cost, status = mip_oracle(
            dist, demand, capacity, duals, n, time_limit=TIME_LIMIT
        )
        elapsed = time.perf_counter() - t0

        oracle_rc = data.get("oracle", {}).get("optimal_rc")
        match = ""
        if opt_rc is not None and oracle_rc is not None:
            match = "MATCH" if abs(opt_rc - oracle_rc) < 0.5 else f"DIFF(oracle={oracle_rc:.1f})"

        results.append({
            "name": name, "n": n,
            "mip_rc": opt_rc, "mip_cost": opt_cost,
            "status": status, "time_s": elapsed, "match": match
        })
        print(f"{status} in {elapsed:.3f}s, rc={opt_rc}, {match}")

    # Summary table
    print("\n" + "="*80)
    print(f"{'Instance':<25} {'n':>3} {'Status':<12} {'Time(s)':>10} {'MIP RC':>12} {'Match':>8}")
    print("-"*80)
    for r in results:
        rc_str = f"{r['mip_rc']:.2f}" if r['mip_rc'] is not None else "N/A"
        print(f"{r['name']:<25} {r['n']:>3} {r['status']:<12} {r['time_s']:>10.3f} {rc_str:>12} {r['match']:>8}")

if __name__ == "__main__":
    run_benchmark()
