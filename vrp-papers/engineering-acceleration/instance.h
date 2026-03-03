#pragma once
// CVRP Instance Loader — parses .vrp files (EUC_2D / EXPLICIT)
// No external dependencies.

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <regex>
#include <stdexcept>

struct Instance {
    std::string name;
    int n;              // total nodes (depot + customers)
    int n_customers;    // n - 1
    int capacity;
    int optimal;        // known optimal cost, or -1

    std::vector<int> dist;      // flat n*n, access: dist[i*n+j]
    std::vector<int> demand;    // size n, demand[0]=0

    int d(int i, int j) const { return dist[i * n + j]; }
};

namespace detail {

inline std::string trim(const std::string& s) {
    auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

inline std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}

} // namespace detail

inline Instance load_instance(const std::string& filepath) {
    Instance inst;
    inst.optimal = -1;

    std::ifstream fin(filepath);
    if (!fin.is_open())
        throw std::runtime_error("Cannot open: " + filepath);

    // Extract name from filename
    {
        auto pos = filepath.find_last_of("/\\");
        std::string fname = (pos != std::string::npos) ? filepath.substr(pos + 1) : filepath;
        auto dot = fname.rfind('.');
        inst.name = (dot != std::string::npos) ? fname.substr(0, dot) : fname;
    }

    std::string edge_weight_type;
    std::string edge_weight_format;
    std::string comment;
    int dimension = 0;
    std::vector<std::pair<double,double>> coords;

    std::string line;
    while (std::getline(fin, line)) {
        line = detail::trim(line);
        if (line.empty()) continue;

        // Parse key : value
        auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = detail::trim(detail::to_upper(line.substr(0, colon)));
            std::string val = detail::trim(line.substr(colon + 1));

            if (key == "DIMENSION") {
                dimension = std::stoi(val);
            } else if (key == "CAPACITY") {
                inst.capacity = std::stoi(val);
            } else if (key == "EDGE_WEIGHT_TYPE") {
                edge_weight_type = detail::to_upper(val);
            } else if (key == "EDGE_WEIGHT_FORMAT") {
                edge_weight_format = detail::to_upper(val);
            } else if (key == "COMMENT") {
                comment = val;
                // Try parse optimal from comment
                std::regex re(R"((?:Best|Optimal)\s+value:\s*(\d+))");
                std::smatch m;
                if (std::regex_search(val, m, re)) {
                    inst.optimal = std::stoi(m[1].str());
                }
            }
            continue;
        }

        std::string upper = detail::to_upper(line);

        if (upper == "NODE_COORD_SECTION") {
            coords.resize(dimension);
            for (int i = 0; i < dimension; i++) {
                int idx; double x, y;
                fin >> idx >> x >> y;
                coords[idx - 1] = {x, y};
            }
            continue;
        }

        if (upper == "EDGE_WEIGHT_SECTION") {
            inst.n = dimension;
            inst.dist.resize(dimension * dimension, 0);

            // Eilon instances (CVRPLIB bug): LOWER_ROW label but data is
            // actually upper-triangular column-wise. See vrplib issue #40.
            bool is_eilon = (comment.find("Eilon") != std::string::npos ||
                             comment.find("eilon") != std::string::npos);

            if (edge_weight_format == "FULL_MATRIX") {
                for (int i = 0; i < dimension; i++)
                    for (int j = 0; j < dimension; j++)
                        fin >> inst.dist[i * dimension + j];
            } else if (edge_weight_format == "LOWER_ROW" && is_eilon) {
                // Read as sorted pairs (i,j) with i<j — upper triangular
                for (int i = 0; i < dimension; i++) {
                    for (int j = i + 1; j < dimension; j++) {
                        int v; fin >> v;
                        inst.dist[i * dimension + j] = v;
                        inst.dist[j * dimension + i] = v;
                    }
                }
            } else if (edge_weight_format == "LOWER_ROW") {
                for (int i = 1; i < dimension; i++) {
                    for (int j = 0; j < i; j++) {
                        int v; fin >> v;
                        inst.dist[i * dimension + j] = v;
                        inst.dist[j * dimension + i] = v;
                    }
                }
            } else if (edge_weight_format == "UPPER_ROW") {
                for (int i = 0; i < dimension - 1; i++) {
                    for (int j = i + 1; j < dimension; j++) {
                        int v; fin >> v;
                        inst.dist[i * dimension + j] = v;
                        inst.dist[j * dimension + i] = v;
                    }
                }
            } else if (edge_weight_format == "LOWER_DIAG_ROW") {
                for (int i = 0; i < dimension; i++) {
                    for (int j = 0; j <= i; j++) {
                        int v; fin >> v;
                        inst.dist[i * dimension + j] = v;
                        inst.dist[j * dimension + i] = v;
                    }
                }
            } else if (edge_weight_format == "UPPER_DIAG_ROW") {
                for (int i = 0; i < dimension; i++) {
                    for (int j = i; j < dimension; j++) {
                        int v; fin >> v;
                        inst.dist[i * dimension + j] = v;
                        inst.dist[j * dimension + i] = v;
                    }
                }
            } else {
                throw std::runtime_error("Unsupported EDGE_WEIGHT_FORMAT: " + edge_weight_format);
            }
            continue;
        }

        if (upper == "DEMAND_SECTION") {
            inst.demand.resize(dimension);
            for (int i = 0; i < dimension; i++) {
                int idx, d;
                fin >> idx >> d;
                inst.demand[idx - 1] = d;
            }
            continue;
        }

        if (upper == "EOF" || upper == "DEPOT_SECTION") break;
    }

    // Compute EUC_2D distances if coordinates given
    if (edge_weight_type == "EUC_2D" && !coords.empty()) {
        inst.n = dimension;
        inst.dist.resize(dimension * dimension, 0);
        for (int i = 0; i < dimension; i++) {
            for (int j = i + 1; j < dimension; j++) {
                double dx = coords[i].first - coords[j].first;
                double dy = coords[i].second - coords[j].second;
                int d = (int)(std::sqrt(dx*dx + dy*dy) + 0.5);
                inst.dist[i * dimension + j] = d;
                inst.dist[j * dimension + i] = d;
            }
        }
    }

    inst.n_customers = inst.n - 1;
    return inst;
}
