#pragma once
// Solver abstraction — compile-time backend selection via USE_COPT / USE_GUROBI.
// Both backends expose the same Solver class interface.

#if defined(USE_COPT)
#include "solver_copt.h"
#elif defined(USE_GUROBI)
#include "solver_gurobi.h"
#else
#error "Define USE_COPT or USE_GUROBI"
#endif
