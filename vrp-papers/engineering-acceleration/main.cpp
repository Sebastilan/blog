// BPC Solver for CVRP — engineering acceleration strategies.
// Strategies: warm-start, batch update, Farkas pricing, multi-pricing + dedup, dual simplex.
// Usage: bpc_solver <instance.vrp> [time_limit] [--no-dedup] [--no-farkas] [--no-warmstart]

#include "instance.h"
#include "rmp.h"
#include "pricing.h"
#include "branch.h"
#include "column_pool.h"
#include <iostream>
#include <queue>
#include <chrono>
#include <cmath>
#include <functional>
#include <cstring>

using Clock = std::chrono::steady_clock;

struct AccelFlags {
    bool dedup      = true;   // column dedup via path hash
    bool farkas     = true;   // Farkas pricing for infeasible RMP
    bool warmstart  = true;   // basis warm-start
    bool presolve0  = false;  // Presolve=0 (auto is usually better)
    bool dualsimplex = true;  // Method=1 dual simplex
};

// Column generation loop
struct CGResult {
    bool feasible;
    double obj;
    std::vector<double> lambdas;
    int cg_iters;
    int cols_added;
    int pricing_calls;
};

CGResult solve_cg(RMP& rmp, const Instance& inst,
                  const std::set<int>& forbidden_keys,
                  const std::set<int>& forced_keys,
                  ColumnPool& pool,
                  const AccelFlags& flags,
                  int max_iter = 500, bool verbose = false)
{
    CGResult res{false, 0.0, {}, 0, 0, 0};

    // Batch update: one update() call for all branching constraints
    rmp.setBranching(forbidden_keys, forced_keys);

    std::vector<double> c_bar(inst.n * inst.n);
    std::vector<uint64_t> fmask;
    build_forbidden_mask(forbidden_keys, inst.n, fmask);

    for (int iter = 0; iter < max_iter; iter++) {
        // Warm-start: if disabled, clear basis before each LP solve
        if (!flags.warmstart) rmp.clearBasis();
        auto sol = rmp.solve();
        if (!sol.feasible) return res;

        res.cg_iters++;

        // Farkas pricing: when artificials are active, use Farkas duals
        if (flags.farkas && rmp.hasArtificial()) {
            auto fk = rmp.getFarkasDuals();
            if (fk.has_duals) {
                std::vector<double> fk_edge_flat;
                if (!fk.edge_duals.empty())
                    build_edge_duals_flat(fk.edge_duals, inst.n, fk_edge_flat);
                build_modified_cost(c_bar.data(), inst.n, inst.dist.data(),
                                   fk.cover_duals.data(),
                                   fk_edge_flat.empty() ? nullptr : fk_edge_flat.data(),
                                   inst.n_customers);
                res.pricing_calls++;
                auto new_cols = solve_pricing(
                    inst.dist.data(), inst.n,
                    inst.demand.data(), inst.capacity,
                    fk.cover_duals.data(), inst.n_customers,
                    c_bar.data(),
                    fmask.empty() ? nullptr : fmask.data());
                if (new_cols.empty()) return res;  // truly infeasible
                for (auto& col : new_cols) {
                    pool.add(col.cost, col.visits, col.path, inst.n);
                    rmp.addColumn(col.cost, col.visits, col.path);
                    res.cols_added++;
                }
                continue;
            }
        }

        // Build modified cost with exact duals
        std::vector<double> edge_duals_flat;
        if (!sol.edge_duals.empty())
            build_edge_duals_flat(sol.edge_duals, inst.n, edge_duals_flat);

        build_modified_cost(c_bar.data(), inst.n, inst.dist.data(),
                           sol.cover_duals.data(),
                           edge_duals_flat.empty() ? nullptr : edge_duals_flat.data(),
                           inst.n_customers);

        // Multi-pricing: add ALL negative-rc columns from ESPPRC
        res.pricing_calls++;
        auto new_cols = solve_pricing(
            inst.dist.data(), inst.n,
            inst.demand.data(), inst.capacity,
            sol.cover_duals.data(), inst.n_customers,
            c_bar.data(),
            fmask.empty() ? nullptr : fmask.data());

        if (new_cols.empty()) break;  // CG converged

        bool any_new = false;
        for (auto& col : new_cols) {
            pool.add(col.cost, col.visits, col.path, inst.n);
            // Dedup: skip columns already in RMP (same path hash)
            if (rmp.addColumn(col.cost, col.visits, col.path, flags.dedup)) {
                res.cols_added++;
                any_new = true;
            }
        }
        if (flags.dedup && !any_new) break;  // all columns already in RMP

        if (verbose && (iter < 3 || (iter + 1) % 20 == 0)) {
            std::cout << "    CG iter " << (iter + 1)
                      << ": obj=" << sol.obj
                      << ", +" << new_cols.size() << " cols\n";
        }
    }

    // Final solve
    if (!flags.warmstart) rmp.clearBasis();
    auto sol = rmp.solve();
    if (!sol.feasible) return res;
    if (rmp.hasArtificial()) return res;

    res.feasible = true;
    res.obj = sol.obj;
    res.lambdas = std::move(sol.lambdas);
    return res;
}

int main(int argc, char* argv[]) {
    std::string filepath;
    double time_limit = 300.0;
    AccelFlags flags;

    // Parse arguments
    int positional = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-dedup") == 0)        { flags.dedup = false; }
        else if (strcmp(argv[i], "--no-farkas") == 0)   { flags.farkas = false; }
        else if (strcmp(argv[i], "--no-warmstart") == 0){ flags.warmstart = false; }
        else if (strcmp(argv[i], "--presolve0") == 0)    { flags.presolve0 = true; }
        else if (strcmp(argv[i], "--no-dualsimplex") == 0){ flags.dualsimplex = false; }
        else {
            if (positional == 0) filepath = argv[i];
            else if (positional == 1) time_limit = std::stod(argv[i]);
            positional++;
        }
    }
    if (filepath.empty()) filepath = "../../../benchmarks/cvrp/E-n13-k4.vrp";

    // Load instance
    Instance inst;
    try {
        inst = load_instance(filepath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Instance: " << inst.name << "\n";
    std::cout << "  Customers: " << inst.n_customers
              << ", Capacity: " << inst.capacity << "\n";
    if (inst.optimal > 0)
        std::cout << "  Known optimal: " << inst.optimal << "\n";
    std::cout << "  Accel: dedup=" << flags.dedup
              << " farkas=" << flags.farkas
              << " warmstart=" << flags.warmstart
              << " presolve0=" << flags.presolve0
              << " dualsimplex=" << flags.dualsimplex << "\n";

    auto t_start = Clock::now();

    // Heuristic upper bound
    auto [heur_cost, heur_routes] = greedy_upper_bound(
        inst.dist.data(), inst.demand.data(), inst.capacity,
        inst.n, inst.n_customers);
    int UB = heur_cost;
    auto best_routes = heur_routes;
    std::cout << "  Heuristic UB: " << UB << "\n";

    // Build RMP with initial columns
    RMP rmp(inst.n_customers, inst.n);
    if (flags.presolve0)    rmp.setPresolve(0);
    if (!flags.dualsimplex) rmp.setMethod(-1);
    auto init_cols = make_initial_columns(inst.dist.data(), inst.n, inst.n_customers);
    rmp.addColumns(init_cols);

    // Global column pool
    ColumnPool pool;
    for (auto& [cost, visits, path] : init_cols)
        pool.add(cost, visits, path, inst.n);

    // Root column generation
    std::cout << "\n  Root column generation...\n";
    std::set<int> empty_set;
    auto root_cg = solve_cg(rmp, inst, empty_set, empty_set, pool, flags, 500, true);
    int total_cg_iters = root_cg.cg_iters;
    int total_cols = (int)init_cols.size() + root_cg.cols_added;
    int total_pricing_calls = root_cg.pricing_calls;

    if (!root_cg.feasible) {
        std::cout << "  Root LP infeasible!\n";
        return 1;
    }
    std::cout << "  Root LP: " << root_cg.obj
              << " (" << root_cg.cg_iters << " CG iters, "
              << total_cols << " cols)\n";

    // Restricted MIP heuristic
    auto [mip_cost, mip_routes] = rmp.solveRestrictedMIP(10.0);
    if (mip_cost > 0 && mip_cost < UB) {
        UB = mip_cost;
        best_routes = mip_routes;
        std::cout << "  Restricted MIP UB: " << UB << "\n";
    }

    // Check root integrality
    if (is_integer_solution(root_cg.lambdas)) {
        int cost = (int)std::round(root_cg.obj);
        if (cost < UB) {
            UB = cost;
            best_routes = extract_solution_routes(root_cg.lambdas, rmp.columns());
        }
        std::cout << "  Root LP is integer! cost=" << cost << "\n";
    } else {
        std::cout << "  Root gap: "
                  << ((UB - root_cg.obj) / UB * 100.0) << "%\n";
    }

    // Integer-cost pruning
    auto can_prune = [&](double lb) -> bool {
        return (int)std::ceil(lb - 1e-6) >= UB;
    };

    // B&B with DFS priority
    struct HeapEntry {
        int neg_depth;
        double lb;
        BBNode node;
        bool operator>(const HeapEntry& o) const {
            if (neg_depth != o.neg_depth) return neg_depth > o.neg_depth;
            return lb > o.lb;
        }
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<>> heap;

    int nodes_explored = 0;
    int nodes_pruned = 0;
    int nodes_infeasible = 0;

    // Branch from root
    if (!is_integer_solution(root_cg.lambdas) && !can_prune(root_cg.obj)) {
        auto flows = compute_edge_flows(root_cg.lambdas, rmp.columns(), inst.n);
        int bek = select_branch_edge(flows);
        if (bek >= 0) {
            BBNode root_node;
            root_node.lb = root_cg.obj;
            root_node.depth = 0;
            root_node.id = next_node_id();

            auto [left, right] = create_children(root_node, bek, root_cg.obj);
            heap.push({-left.depth, left.lb, left});
            heap.push({-right.depth, right.lb, right});
            nodes_explored = 1;
        }
    } else {
        nodes_explored = 1;
    }

    std::cout << "\n";

    while (!heap.empty()) {
        auto elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        if (elapsed > time_limit) {
            std::cout << "  Time limit (" << time_limit << "s)\n";
            break;
        }

        auto entry = heap.top();
        heap.pop();
        auto& node = entry.node;

        if (can_prune(node.lb)) {
            nodes_pruned++;
            continue;
        }

        nodes_explored++;
        std::set<int> forbidden, forced;
        get_branch_sets(node, inst.n, forbidden, forced);

        auto cg = solve_cg(rmp, inst, forbidden, forced, pool, flags);
        total_cg_iters += cg.cg_iters;
        total_cols += cg.cols_added;
        total_pricing_calls += cg.pricing_calls;

        if (!cg.feasible) {
            nodes_infeasible++;
            continue;
        }

        double LB = cg.obj;
        node.lb = LB;

        if (can_prune(LB)) {
            nodes_pruned++;
            continue;
        }

        if (nodes_explored <= 10 || nodes_explored % 100 == 0) {
            auto elapsed2 = std::chrono::duration<double>(Clock::now() - t_start).count();
            std::cout << "  [" << elapsed2 << "s] Node " << node.id
                      << ": LB=" << LB << " UB=" << UB
                      << " d=" << node.depth << " cols=" << total_cols << "\n";
        }

        if (is_integer_solution(cg.lambdas)) {
            int cost = (int)std::round(cg.obj);
            if (cost < UB) {
                UB = cost;
                best_routes = extract_solution_routes(cg.lambdas, rmp.columns());
                auto elapsed2 = std::chrono::duration<double>(Clock::now() - t_start).count();
                std::cout << "  [" << elapsed2 << "s] *** INTEGER cost=" << cost
                          << " (node=" << node.id << ", d=" << node.depth << ")\n";
            }
            continue;
        }

        auto flows = compute_edge_flows(cg.lambdas, rmp.columns(), inst.n);
        int bek = select_branch_edge(flows);
        if (bek < 0) continue;

        auto [left, right] = create_children(node, bek, LB);
        heap.push({-left.depth, left.lb, left});
        heap.push({-right.depth, right.lb, right});
    }

    auto elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();

    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "Instance: " << inst.name << "\n";
    std::cout << "Best cost: " << UB << "\n";
    if (inst.optimal > 0) {
        double gap = std::abs((double)UB - inst.optimal) / inst.optimal * 100.0;
        std::cout << "Known optimal: " << inst.optimal << ", Gap: " << gap << "%\n";
    }
    std::cout << "Routes (" << best_routes.size() << "):\n";
    for (auto& [cost, path] : best_routes) {
        std::cout << "  [";
        for (int k = 0; k < (int)path.size(); k++) {
            if (k > 0) std::cout << ",";
            std::cout << path[k];
        }
        std::cout << "]  cost=" << cost << "\n";
    }
    std::cout << "B&B nodes explored: " << nodes_explored << "\n";
    std::cout << "  pruned by bound: " << nodes_pruned << "\n";
    std::cout << "  infeasible: " << nodes_infeasible << "\n";
    std::cout << "CG iterations: " << total_cg_iters << "\n";
    std::cout << "Total columns: " << total_cols << "\n";
    std::cout << "ESPPRC calls: " << total_pricing_calls << "\n";
    std::cout << "LP solves: " << rmp.totalLPCalls() << "\n";
    std::cout << "LP simplex iters: " << rmp.totalLPIters() << "\n";
    double avg_iters = rmp.totalLPCalls() > 0
        ? (double)rmp.totalLPIters() / rmp.totalLPCalls() : 0;
    std::cout << "Avg iters/LP: " << avg_iters << "\n";
    std::cout << "Time: " << elapsed << "s\n";
    std::cout << std::string(50, '=') << "\n";

    return 0;
}
