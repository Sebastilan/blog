#pragma once
// Restricted Master Problem — set-covering with incremental branching.
// Uses Solver abstraction (COPT or Gurobi backend).

#include "solver.h"
#include "types.h"
#include <vector>
#include <set>
#include <map>
#include <cstdint>
#include <cmath>
#include <memory>
#include <unordered_map>

static constexpr double BIG_M = 1e6;

class RMP {
public:
    RMP(int n_customers, int n_nodes)
        : nc_(n_customers), n_(n_nodes) {
        solver_ = std::make_unique<Solver>("RMP");

        // Create artificial variables for covering constraints
        art_cover_.resize(nc_);
        for (int i = 0; i < nc_; i++)
            art_cover_[i] = solver_->addVar(BIG_M, 0.0, 1e30);

        // Covering constraints: art_i >= 1  (columns will be added to these)
        cover_constrs_.resize(nc_);
        for (int i = 0; i < nc_; i++)
            cover_constrs_[i] = solver_->addConstrSingleGe(art_cover_[i], 1.0);

        solver_->update();
    }

    // Add a route column (returns false if duplicate path when dedup=true)
    bool addColumn(int cost, const std::vector<int>& visits, const std::vector<int>& path,
                   bool dedup = true) {
        size_t h = path_hash(path);
        if (dedup && column_hash_map_.count(h)) return false;

        RouteColumn col;
        col.cost = cost;
        col.visits = visits;
        col.path = path;
        col.edge_bits = 0;

        // Compute edge set
        std::set<int> edges;
        for (int k = 0; k + 1 < (int)path.size(); k++) {
            edges.insert(edge_key(path[k], path[k+1], n_));
        }
        col_edges_.push_back(edges);

        // Build column coefficients
        std::vector<int> ci;
        std::vector<double> cv;

        for (int v : visits) {
            ci.push_back(cover_constrs_[v - 1]);
            cv.push_back(1.0);
        }

        for (auto& [ek, info] : forced_edges_) {
            if (edges.count(ek)) {
                ci.push_back(info.constr_id);
                cv.push_back(1.0);
            }
        }

        // Check if route uses any forbidden edge
        bool uses_forbidden = false;
        for (int ek : edges) {
            if (forbidden_keys_.count(ek)) { uses_forbidden = true; break; }
        }

        double ub = uses_forbidden ? 0.0 : 1e30;
        int var_id = solver_->addVarCol((double)cost, 0.0, ub,
                                         ci.data(), cv.data(), (int)ci.size());
        int col_idx = (int)columns_.size();
        columns_.push_back(std::move(col));
        var_ids_.push_back(var_id);
        column_hash_map_[h] = col_idx;
        return true;
    }

    void addColumns(const std::vector<std::tuple<int, std::vector<int>, std::vector<int>>>& cols) {
        for (auto& [cost, visits, path] : cols)
            addColumn(cost, visits, path);
    }

    // Incremental branching — batch update: one update() for all constraint changes
    void setBranching(const std::set<int>& new_forbidden, const std::set<int>& new_forced) {
        // 1. Undo removed forbidden (restore UB)
        for (int ek : forbidden_keys_) {
            if (!new_forbidden.count(ek)) {
                for (int idx = 0; idx < (int)col_edges_.size(); idx++) {
                    if (col_edges_[idx].count(ek)) {
                        bool still_blocked = false;
                        for (int fk : new_forbidden) {
                            if (col_edges_[idx].count(fk)) { still_blocked = true; break; }
                        }
                        if (!still_blocked)
                            solver_->setVarUB(var_ids_[idx], 1e30);
                    }
                }
            }
        }

        // 2. Apply new forbidden (set UB=0)
        for (int ek : new_forbidden) {
            if (!forbidden_keys_.count(ek)) {
                for (int idx = 0; idx < (int)col_edges_.size(); idx++) {
                    if (col_edges_[idx].count(ek))
                        solver_->setVarUB(var_ids_[idx], 0.0);
                }
            }
        }

        // 3. Remove old forced edge constraints no longer needed
        for (auto it = forced_edges_.begin(); it != forced_edges_.end(); ) {
            if (!new_forced.count(it->first)) {
                solver_->removeConstr(it->second.constr_id);
                solver_->setVarUB(it->second.art_id, 0.0);
                it = forced_edges_.erase(it);
            } else {
                ++it;
            }
        }

        // 4. Add new forced edge constraints
        for (int ek : new_forced) {
            if (!forced_edges_.count(ek)) {
                int art = solver_->addVar(BIG_M, 0.0, 1e30);

                std::vector<int> vi;
                std::vector<double> vc;
                vi.push_back(art);
                vc.push_back(1.0);
                for (int idx = 0; idx < (int)col_edges_.size(); idx++) {
                    if (col_edges_[idx].count(ek) && !isForbiddenColumn(idx, new_forbidden)) {
                        vi.push_back(var_ids_[idx]);
                        vc.push_back(1.0);
                    }
                }
                int cid = solver_->addConstrGe(vi.data(), vc.data(), (int)vi.size(), 1.0);
                forced_edges_[ek] = {cid, art};
            }
        }

        forbidden_keys_ = new_forbidden;

        // Batch flush: one update() for all changes above
        solver_->update();
    }

    struct SolveResult {
        bool feasible;
        double obj;
        std::vector<double> lambdas;
        std::vector<double> cover_duals;
        std::map<int, double> edge_duals;
    };

    // Force cold start (for A/B testing warm-start benefit)
    void clearBasis() { solver_->clearBasis(); }

    // Solver parameter tuning
    void setPresolve(int val) { solver_->setPresolve(val); }
    void setMethod(int val) { solver_->setMethod(val); }

    SolveResult solve() {
        SolveResult res;
        LPStatus st = solver_->solveLP();
        if (st != LPStatus::OPTIMAL) {
            res.feasible = false;
            return res;
        }

        res.feasible = true;
        res.obj = solver_->getObjVal();

        res.lambdas.resize(var_ids_.size());
        for (int i = 0; i < (int)var_ids_.size(); i++)
            res.lambdas[i] = solver_->getVarVal(var_ids_[i]);

        res.cover_duals.resize(nc_);
        for (int i = 0; i < nc_; i++)
            res.cover_duals[i] = solver_->getDual(cover_constrs_[i]);

        for (auto& [ek, info] : forced_edges_)
            res.edge_duals[ek] = solver_->getDual(info.constr_id);

        return res;
    }

    bool hasArtificial(double tol = 1e-6) {
        for (int art : art_cover_)
            if (solver_->getVarVal(art) > tol) return true;
        for (auto& [ek, info] : forced_edges_)
            if (solver_->getVarVal(info.art_id) > tol) return true;
        return false;
    }

    // Farkas dual extraction for pricing to restore feasibility
    struct FarkasResult {
        bool has_duals;
        std::vector<double> cover_duals;
        std::map<int, double> edge_duals;
    };

    FarkasResult getFarkasDuals() {
        FarkasResult res{false, {}, {}};

        for (int art : art_cover_)
            solver_->setVarUB(art, 0.0);
        for (auto& [ek, info] : forced_edges_)
            solver_->setVarUB(info.art_id, 0.0);

        LPStatus st = solver_->solveLP();

        if (st == LPStatus::INFEASIBLE) {
            res.has_duals = true;
            res.cover_duals.resize(nc_);
            for (int i = 0; i < nc_; i++)
                res.cover_duals[i] = solver_->getFarkasDual(cover_constrs_[i]);
            for (auto& [ek, info] : forced_edges_)
                res.edge_duals[ek] = solver_->getFarkasDual(info.constr_id);
        }

        for (int art : art_cover_)
            solver_->setVarUB(art, 1e30);
        for (auto& [ek, info] : forced_edges_)
            solver_->setVarUB(info.art_id, 1e30);

        return res;
    }

    const std::vector<RouteColumn>& columns() const { return columns_; }
    int numColumns() const { return (int)columns_.size(); }

    int totalLPIters() const { return solver_->totalLPIters(); }
    int totalLPCalls() const { return solver_->totalLPCalls(); }
    int lastLPIters() const { return solver_->lastIterCount(); }

    // For restricted MIP
    std::pair<int, std::vector<std::pair<int, std::vector<int>>>>
    solveRestrictedMIP(double timeLimit = 10.0) {
        Solver mip("RestrictedMIP");

        std::vector<int> mip_vars(columns_.size());
        for (int i = 0; i < (int)columns_.size(); i++) {
            mip_vars[i] = mip.addVar((double)columns_[i].cost, 0.0, 1.0);
            mip.setVarBinary(mip_vars[i]);
        }

        for (int c = 1; c <= nc_; c++) {
            std::vector<int> vi;
            std::vector<double> vc;
            for (int i = 0; i < (int)columns_.size(); i++) {
                for (int v : columns_[i].visits) {
                    if (v == c) { vi.push_back(mip_vars[i]); vc.push_back(1.0); break; }
                }
            }
            mip.addConstrGe(vi.data(), vc.data(), (int)vi.size(), 1.0);
        }

        LPStatus st = mip.solveMIP(timeLimit);
        if (st != LPStatus::OPTIMAL)
            return {-1, {}};

        int cost = (int)std::round(mip.getMIPObjVal());
        std::vector<std::pair<int, std::vector<int>>> routes;
        for (int i = 0; i < (int)columns_.size(); i++) {
            if (mip.getVarVal(mip_vars[i]) > 0.5)
                routes.push_back({columns_[i].cost, columns_[i].path});
        }
        return {cost, routes};
    }

private:
    bool isForbiddenColumn(int idx, const std::set<int>& forbidden) const {
        for (int ek : col_edges_[idx])
            if (forbidden.count(ek)) return true;
        return false;
    }

    struct ForcedEdgeInfo {
        int constr_id;
        int art_id;
    };

    int nc_;
    int n_;

    std::unique_ptr<Solver> solver_;

    std::vector<int> art_cover_;
    std::vector<int> cover_constrs_;

    std::vector<RouteColumn> columns_;
    std::vector<int> var_ids_;
    std::vector<std::set<int>> col_edges_;

    std::set<int> forbidden_keys_;
    std::map<int, ForcedEdgeInfo> forced_edges_;

    std::unordered_map<size_t, int> column_hash_map_;
};

// Create trivial initial columns: one customer per vehicle
inline std::vector<std::tuple<int, std::vector<int>, std::vector<int>>>
make_initial_columns(const int* dist, int n, int n_customers) {
    std::vector<std::tuple<int, std::vector<int>, std::vector<int>>> cols;
    for (int i = 1; i <= n_customers; i++) {
        int cost = dist[0 * n + i] + dist[i * n + 0];
        cols.push_back({cost, {i}, {0, i, 0}});
    }
    return cols;
}
