#pragma once
// Shared types used across BPC modules.

#include <vector>
#include <cstdint>
#include <algorithm>
#include <functional>

struct RouteColumn {
    int cost;
    std::vector<int> visits;    // customer indices, 1-based
    std::vector<int> path;      // 0 ... 0
    uint64_t edge_bits;         // reserved for fast intersection
};

// Encode edge (i,j) normalized as min<max into flat index
inline int edge_key(int a, int b, int n) {
    int i = std::min(a, b), j = std::max(a, b);
    return i * n + j;
}

// Hash a route path for dedup (boost::hash_combine style)
inline size_t path_hash(const std::vector<int>& path) {
    size_t h = path.size();
    for (int v : path)
        h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}
