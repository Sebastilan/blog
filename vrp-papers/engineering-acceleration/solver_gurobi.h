#pragma once
// Gurobi backend — placeholder until Gurobi C++ SDK is installed.
// Same Solver interface as solver_copt.h.

#include "gurobi_c++.h"
#include <vector>
#include <string>
#include <stdexcept>

enum class LPStatus { OPTIMAL, INFEASIBLE, UNBOUNDED, ERROR };

class Solver {
public:
    Solver(const std::string& name = "model")
        : env_(), model_(env_) {
        model_.set(GRB_IntParam_OutputFlag, 0);
        model_.set(GRB_StringAttr_ModelName, name);
        model_.set(GRB_IntParam_Presolve, -1);  // auto (tested: auto >= presolve=0)
        model_.set(GRB_IntParam_Method, 1);      // dual simplex (best for CG)
        model_.set(GRB_IntParam_InfUnbdInfo, 1);  // enable Farkas dual extraction
    }

    void setPresolve(int val) { model_.set(GRB_IntParam_Presolve, val); }
    void setMethod(int val) { model_.set(GRB_IntParam_Method, val); }

    int addVar(double obj, double lb = 0.0, double ub = 1e30) {
        int idx = (int)vars_.size();
        vars_.push_back(model_.addVar(lb, ub, obj, GRB_CONTINUOUS));
        // Pending var: can be used in addConstr / set* before update().
        // optimize() will implicitly flush all pending changes.
        return idx;
    }

    int addVarCol(double obj, double lb, double ub,
                  const int* constr_indices, const double* coeffs, int nterms) {
        GRBColumn col;
        for (int k = 0; k < nterms; k++)
            col.addTerm(coeffs[k], constrs_[constr_indices[k]]);
        int idx = (int)vars_.size();
        vars_.push_back(model_.addVar(lb, ub, obj, GRB_CONTINUOUS, col));
        // No update() — batch variable additions, optimize() will flush.
        return idx;
    }

    void setVarUB(int var, double ub) { vars_[var].set(GRB_DoubleAttr_UB, ub); }
    void setVarLB(int var, double lb) { vars_[var].set(GRB_DoubleAttr_LB, lb); }
    void setVarObj(int var, double obj) { vars_[var].set(GRB_DoubleAttr_Obj, obj); }
    void setVarBinary(int var) { vars_[var].set(GRB_CharAttr_VType, GRB_BINARY); }
    void setVarContinuous(int var) { vars_[var].set(GRB_CharAttr_VType, GRB_CONTINUOUS); }

    int addConstrGe(const int* var_indices, const double* coeffs, int nterms, double rhs) {
        GRBLinExpr expr;
        for (int k = 0; k < nterms; k++)
            expr += coeffs[k] * vars_[var_indices[k]];
        int idx = (int)constrs_.size();
        constrs_.push_back(model_.addConstr(expr >= rhs));
        // No per-op update() — batch with explicit update() calls.
        // optimize() will flush all pending changes.
        return idx;
    }

    int addConstrSingleGe(int var, double rhs) {
        GRBLinExpr expr = vars_[var];
        int idx = (int)constrs_.size();
        constrs_.push_back(model_.addConstr(expr >= rhs));
        return idx;
    }

    void removeConstr(int constr) {
        model_.remove(constrs_[constr]);
    }

    // Flush all pending changes (constraint add/remove, var add, bound changes).
    // Call after a batch of modifications, before the next attribute query.
    void update() { model_.update(); }

    LPStatus solveLP() {
        model_.optimize();
        total_lp_iters_ += (int)model_.get(GRB_DoubleAttr_IterCount);
        total_lp_calls_++;
        int st = model_.get(GRB_IntAttr_Status);
        if (st == GRB_OPTIMAL) return LPStatus::OPTIMAL;
        if (st == GRB_INFEASIBLE) return LPStatus::INFEASIBLE;
        if (st == GRB_UNBOUNDED) return LPStatus::UNBOUNDED;
        return LPStatus::ERROR;
    }

    int lastIterCount() const { return (int)model_.get(GRB_DoubleAttr_IterCount); }
    int totalLPIters() const { return total_lp_iters_; }
    int totalLPCalls() const { return total_lp_calls_; }

    LPStatus solveMIP(double timeLimit = 10.0) {
        model_.set(GRB_DoubleParam_TimeLimit, timeLimit);
        model_.optimize();
        int st = model_.get(GRB_IntAttr_Status);
        if (st == GRB_OPTIMAL || st == GRB_SUBOPTIMAL)
            return LPStatus::OPTIMAL;
        if (st == GRB_INFEASIBLE) return LPStatus::INFEASIBLE;
        if (st == GRB_UNBOUNDED) return LPStatus::UNBOUNDED;
        return LPStatus::ERROR;
    }

    double getObjVal() const { return model_.get(GRB_DoubleAttr_ObjVal); }
    double getMIPObjVal() const { return model_.get(GRB_DoubleAttr_ObjVal); }
    double getVarVal(int var) const { return vars_[var].get(GRB_DoubleAttr_X); }
    double getDual(int constr) const { return constrs_[constr].get(GRB_DoubleAttr_Pi); }

    void getDuals(const int* constr_indices, double* out, int n) const {
        for (int k = 0; k < n; k++)
            out[k] = constrs_[constr_indices[k]].get(GRB_DoubleAttr_Pi);
    }

    double getFarkasDual(int constr) const {
        return constrs_[constr].get(GRB_DoubleAttr_FarkasDual);
    }

    struct Basis {
        std::vector<int> col_basis;
        std::vector<int> row_basis;
    };

    Basis saveBasis() const {
        Basis b;
        int nc = (int)vars_.size();
        int nr = (int)constrs_.size();
        b.col_basis.resize(nc);
        b.row_basis.resize(nr);
        for (int i = 0; i < nc; i++)
            b.col_basis[i] = vars_[i].get(GRB_IntAttr_VBasis);
        for (int i = 0; i < nr; i++)
            b.row_basis[i] = constrs_[i].get(GRB_IntAttr_CBasis);
        return b;
    }

    void loadBasis(const Basis& b) {
        int nc = std::min((int)b.col_basis.size(), (int)vars_.size());
        int nr = std::min((int)b.row_basis.size(), (int)constrs_.size());
        for (int i = 0; i < nc; i++)
            vars_[i].set(GRB_IntAttr_VBasis, b.col_basis[i]);
        for (int i = 0; i < nr; i++)
            constrs_[i].set(GRB_IntAttr_CBasis, b.row_basis[i]);
    }

    // Force cold start: discard all solution/basis info
    void clearBasis() {
        model_.update();  // flush pending changes first
        model_.reset();   // discard all solution info (forces cold start)
    }

    int numVars() const { return (int)vars_.size(); }
    int numConstrs() const { return (int)constrs_.size(); }

private:
    GRBEnv env_;
    GRBModel model_;
    std::vector<GRBVar> vars_;
    std::vector<GRBConstr> constrs_;
    int total_lp_iters_ = 0;
    int total_lp_calls_ = 0;
};
