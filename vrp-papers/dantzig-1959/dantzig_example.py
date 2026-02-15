"""
Dantzig & Ramser (1959) — The Truck Dispatching Problem
========================================================
逐层捆绑算法：严格复现论文中 12 站数值示例。

距离矩阵从 Table 1 提取并经 Table 3 交叉验证。
求解器：Gurobi
"""

import numpy as np
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB

# ============================================================
# 数据：论文 Table 1
# ============================================================

D = np.array([
    # P0  P1  P2  P3  P4  P5  P6  P7  P8  P9  P10 P11 P12
    [  0,  9, 14, 21, 23, 22, 25, 32, 36, 38, 42, 50, 52],  # P0
    [  9,  0,  5, 12, 22, 21, 24, 31, 35, 37, 41, 49, 51],  # P1
    [ 14,  5,  0,  7, 17, 16, 23, 26, 30, 36, 36, 44, 46],  # P2
    [ 21, 12,  7,  0, 10, 21, 30, 27, 37, 43, 31, 37, 39],  # P3
    [ 23, 22, 17, 10,  0, 19, 28, 25, 35, 41, 29, 31, 29],  # P4
    [ 22, 21, 16, 21, 19,  0,  9, 11, 16, 22, 20, 28, 30],  # P5
    [ 25, 24, 23, 30, 28,  9,  0,  7, 11, 13, 17, 25, 27],  # P6
    [ 32, 31, 26, 27, 25, 11,  7,  0, 10, 16, 10, 22, 20],  # P7
    [ 36, 35, 30, 37, 35, 16, 11, 10,  0,  6,  6, 14, 16],  # P8
    [ 38, 37, 36, 43, 41, 22, 13, 16,  6,  0, 12, 20, 10],  # P9
    [ 42, 41, 36, 31, 29, 20, 17, 10,  6, 12,  0, 12, 10],  # P10
    [ 50, 49, 44, 37, 31, 28, 25, 22, 14, 20, 12,  0, 10],  # P11
    [ 52, 51, 46, 39, 29, 30, 27, 20, 16, 10, 10, 10,  0],  # P12
])

Q = [0, 1200, 1700, 1500, 1400, 1700, 1400, 1200, 1900, 1800, 1600, 1700, 1100]
CAP = 6000


# ============================================================
# 工具函数
# ============================================================

def mini_tsp(stations, dist_matrix=D):
    """
    小规模 TSP：穷举所有排列，返回经过 depot(0) + stations 的最短路线距离。
    返回: (distance, best_order)
    """
    if len(stations) == 0:
        return 0, []
    if len(stations) == 1:
        return 2 * dist_matrix[0][stations[0]], list(stations)
    best = float('inf')
    best_order = None
    for perm in permutations(stations):
        dist = dist_matrix[0][perm[0]]
        for k in range(len(perm) - 1):
            dist += dist_matrix[perm[k]][perm[k + 1]]
        dist += dist_matrix[perm[-1]][0]
        if dist < best:
            best = dist
            best_order = list(perm)
    return best, best_order


def solve_matching_lp(nodes, dist_func, demand, cap_limit):
    """
    通用匹配 LP：在 nodes 之间求最小代价匹配。

    min  sum d(i,j) * x(i,j)
    s.t. sum_{j} x(i,j) = 1   for each node i in nodes
         x(i,j) >= 0
         x(i,j) = 0            if demand[i] + demand[j] > cap_limit

    节点 0 是虚拟 depot（度数不限），nodes 中的每个节点度 = 1。

    参数:
        nodes: 节点编号列表（不含 depot 0）
        dist_func: dist_func(i, j) -> 距离
        demand: dict {node: demand_value}，demand[0] = 0
        cap_limit: 每组需求上限

    返回: (pairs, singles, fractional_groups, obj_value)
        pairs: [(i, j), ...] 整数配对
        singles: [i, ...] 与 depot 配对的单独节点
        fractional_groups: [[i, j, k, ...], ...] 小数解涉及的节点组
        obj_value: LP 最优值
    """
    m = gp.Model("matching")
    m.setParam("OutputFlag", 0)

    # 边集：(0, i) for i in nodes + (i, j) for i < j in nodes if feasible
    edges = {}
    for i in nodes:
        key = (0, i)
        edges[key] = m.addVar(lb=0, ub=1, obj=dist_func(0, i), name=f"x_{0}_{i}")
    for idx_a, i in enumerate(nodes):
        for j in nodes[idx_a + 1:]:
            if demand[i] + demand[j] <= cap_limit:
                key = (i, j)
                edges[key] = m.addVar(lb=0, ub=1, obj=dist_func(i, j),
                                      name=f"x_{i}_{j}")

    m.update()

    # 约束：每个节点度 = 1
    for i in nodes:
        incident = []
        for (a, b), var in edges.items():
            if a == i or b == i:
                incident.append(var)
        m.addConstr(gp.quicksum(incident) == 1, name=f"deg_{i}")

    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"LP 求解失败, status={m.status}")

    obj_value = m.ObjVal

    # 提取解
    sol = {k: v.X for k, v in edges.items() if v.X > 1e-8}

    # 分类
    pairs = []
    singles = []
    frac_edges = []
    paired_nodes = set()

    for (i, j), val in sol.items():
        if i == 0:
            continue
        if abs(val - 1.0) < 1e-6:
            pairs.append((i, j))
            paired_nodes.add(i)
            paired_nodes.add(j)
        elif val > 1e-8:
            frac_edges.append((i, j, val))

    for (i, j), val in sol.items():
        if i == 0 and abs(val - 1.0) < 1e-6 and j not in paired_nodes:
            singles.append(j)

    # 识别小数组：找连通分量
    fractional_groups = []
    if frac_edges:
        frac_nodes = set()
        for (i, j, _) in frac_edges:
            frac_nodes.add(i)
            frac_nodes.add(j)
        # 简单 BFS 找连通分量
        remaining = set(frac_nodes)
        while remaining:
            start = remaining.pop()
            component = {start}
            queue = [start]
            while queue:
                node = queue.pop(0)
                for (i, j, _) in frac_edges:
                    other = None
                    if i == node and j in remaining:
                        other = j
                    elif j == node and i in remaining:
                        other = i
                    if other is not None:
                        component.add(other)
                        remaining.discard(other)
                        queue.append(other)
            fractional_groups.append(sorted(component))

    return pairs, singles, fractional_groups, obj_value


def build_aggregate_distance(groups, dist_matrix=D):
    """
    构建组间距离矩阵：每对组的距离 = mini_tsp(组1 ∪ 组2)。
    groups[0] 对应 A1, groups[1] 对应 A2, ...
    返回矩阵 agg_dist[0..n][0..n]，其中 0 = depot(A0)。
    """
    n = len(groups)
    agg_dist = np.zeros((n + 1, n + 1))

    for i in range(n):
        d, _ = mini_tsp(groups[i], dist_matrix)
        agg_dist[0][i + 1] = d
        agg_dist[i + 1][0] = d

    for i in range(n):
        for j in range(i + 1, n):
            combined = groups[i] + groups[j]
            d, _ = mini_tsp(combined, dist_matrix)
            agg_dist[i + 1][j + 1] = d
            agg_dist[j + 1][i + 1] = d

    return agg_dist


def resolve_fractional(frac_group, agg_dist, group_names):
    """
    处理 LP 小数解：枚举取整方案，选最优。
    frac_group: [A_i, A_j, A_k] (aggregate 编号，含 depot 偏移)
    返回: (paired, lone) — paired = (i, j) 配对, lone = 单独与 A0 配对
    """
    best_cost = float('inf')
    best_pair = None
    best_lone = None
    results = []

    n = len(frac_group)
    for lone_idx in range(n):
        lone = frac_group[lone_idx]
        remaining = [frac_group[k] for k in range(n) if k != lone_idx]

        if len(remaining) == 2:
            pair_cost = agg_dist[remaining[0]][remaining[1]]
            depot_cost = agg_dist[0][lone]
            total = pair_cost + depot_cost
            results.append((remaining, lone, total))
            if total < best_cost:
                best_cost = total
                best_pair = tuple(remaining)
                best_lone = lone

    return best_pair, best_lone, results


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  Dantzig & Ramser (1959) 逐层捆绑算法")
    print("=" * 60)

    # --- 数据验证 ---
    assert np.allclose(D, D.T), "距离矩阵不对称!"
    depot_sum = sum(D[0][i] for i in range(1, 13))
    assert depot_sum == 364, f"depot 距离和 = {depot_sum}, 期望 364"
    total_demand = sum(Q[1:])
    min_vehicles = -(-total_demand // CAP)  # 上取整
    print(f"\n总需求: {total_demand}, 容量: {CAP}, 至少 {min_vehicles} 辆车")

    # 计算 t: 最多几站一辆车
    sorted_q = sorted(Q[1:])
    cumsum, t = 0, 0
    for q in sorted_q:
        if cumsum + q <= CAP:
            cumsum += q
            t += 1
        else:
            break
    print(f"每车最多 {t} 站 -> 需要 2 层合并")

    # ================================================================
    # Level 0: 初始解
    # ================================================================
    print("\n" + "=" * 60)
    print("Level 0: 初始解")
    print("=" * 60)
    init_dist = 2 * depot_sum
    print(f"  12 条路线，总距离 {init_dist}")

    # ================================================================
    # Level 1: 站点配对（每组 <= 2 站，需求 <= C/2 = 3000）
    # ================================================================
    print("\n" + "=" * 60)
    print("Level 1: 站点配对（每组 <= 2 站，需求 <= 3000）")
    print("=" * 60)

    cap1 = CAP // 2  # 3000
    nodes1 = list(range(1, 13))
    demand1 = {i: Q[i] for i in range(13)}

    pairs1, singles1, frac1, obj1 = solve_matching_lp(
        nodes1, lambda i, j: D[i][j], demand1, cap1
    )

    print(f"\n  LP 最优 D = {obj1:.0f}")
    print(f"  总距离 = 364 + {obj1:.0f} = {364 + obj1:.0f}")
    print(f"\n  配对结果:")
    for (i, j) in sorted(pairs1):
        print(f"    {{P{i}, P{j}}}  需求={Q[i]+Q[j]}, d={D[i][j]}")
    if singles1:
        print(f"  单独: {', '.join(f'P{s}' for s in sorted(singles1))}")
    if frac1:
        print(f"  小数组: {frac1}")

    # 构建 aggregates
    groups = []
    group_names = []
    group_demands = []

    for (i, j) in sorted(pairs1):
        groups.append([i, j])
        group_names.append(f"A{len(groups)}={{P{i},P{j}}}")
        group_demands.append(Q[i] + Q[j])

    for s in sorted(singles1):
        groups.append([s])
        group_names.append(f"A{len(groups)}={{P{s}}}")
        group_demands.append(Q[s])

    n_groups = len(groups)
    print(f"\n  {n_groups} 个小组:")
    for idx, (g, name, dem) in enumerate(zip(groups, group_names, group_demands)):
        d_trip, _ = mini_tsp(g)
        print(f"    {name}  需求={dem}  路线距离={d_trip}")

    # ================================================================
    # 组间距离（mini-TSP）
    # ================================================================
    print("\n" + "=" * 60)
    print("组间距离矩阵（mini-TSP）")
    print("=" * 60)

    agg_dist = build_aggregate_distance(groups)

    # 打印矩阵
    header = "       A0"
    for i in range(n_groups):
        header += f"   A{i+1}"
    print(header)
    for i in range(n_groups + 1):
        row = f"  A{i:1d}"
        for j in range(n_groups + 1):
            row += f"  {agg_dist[i][j]:4.0f}"
        print(row)

    # ================================================================
    # Level 2: 小组合并（需求 <= C = 6000）
    # ================================================================
    print("\n" + "=" * 60)
    print("Level 2: 小组合并（每条路线需求 <= 6000）")
    print("=" * 60)

    nodes2 = list(range(1, n_groups + 1))
    demand2 = {0: 0}
    for i in range(n_groups):
        demand2[i + 1] = group_demands[i]

    pairs2, singles2, frac2, obj2 = solve_matching_lp(
        nodes2, lambda i, j: agg_dist[i][j], demand2, CAP
    )

    print(f"\n  LP 最优值 = {obj2:.2f}")

    if pairs2:
        print("  整数配对:")
        for (i, j) in pairs2:
            print(f"    {group_names[i-1]} - {group_names[j-1]}  "
                  f"距离={agg_dist[i][j]:.0f}")
    if singles2:
        print("  单独:")
        for s in singles2:
            print(f"    A0 - {group_names[s-1]}  距离={agg_dist[0][s]:.0f}")

    if frac2:
        print(f"\n  出现小数解！涉及小组: {frac2}")
        for fg in frac2:
            print(f"\n  枚举取整方案:")
            pair, lone, results = resolve_fractional(fg, agg_dist, group_names)
            for idx, (rem, ln, total) in enumerate(results):
                name_pair = f"{group_names[rem[0]-1]}-{group_names[rem[1]-1]}"
                name_lone = f"A0-{group_names[ln-1]}"
                marker = " <-- 最优" if (rem[0], rem[1]) == (pair[0], pair[1]) and ln == lone else ""
                print(f"    方案{chr(9312+idx)}: {name_pair} + {name_lone} = "
                      f"{agg_dist[rem[0]][rem[1]]:.0f} + {agg_dist[0][ln]:.0f} = "
                      f"{total:.0f}{marker}")

            pairs2.append(pair)
            singles2.append(lone)

    # ================================================================
    # 最终结果
    # ================================================================
    print("\n" + "=" * 60)
    print("最终路线方案")
    print("=" * 60)

    total_distance = 0
    route_num = 0

    for (ai, aj) in pairs2:
        route_num += 1
        stations = groups[ai - 1] + groups[aj - 1]
        dist, order = mini_tsp(stations)
        total_distance += dist
        demand = sum(Q[s] for s in stations)
        route_str = " -> ".join([f"P{s}" for s in order])
        print(f"  路线 {route_num}: P0 -> {route_str} -> P0  "
              f"距离={dist}  需求={demand}")

    for ai in singles2:
        route_num += 1
        stations = groups[ai - 1]
        dist, order = mini_tsp(stations)
        total_distance += dist
        demand = sum(Q[s] for s in stations)
        route_str = " -> ".join([f"P{s}" for s in order])
        print(f"  路线 {route_num}: P0 -> {route_str} -> P0  "
              f"距离={dist}  需求={demand}")

    print(f"\n  总距离: {total_distance}")
    print(f"  论文 best solution: 294")
    print(f"  论文 conjectured optimal: 290")

    # ================================================================
    # 对照验证
    # ================================================================
    print("\n" + "=" * 60)
    print("对照验证")
    print("=" * 60)

    for label, routes in [("论文 best solution (294)",
                           [[1, 2, 3, 4], [7, 12, 11, 9], [6, 10, 8], [5]]),
                          ("论文 conjectured optimal (290)",
                           [[1, 2, 3, 4], [7, 12, 11, 10], [6, 8, 9], [5]])]:
        print(f"\n{label}:")
        total = 0
        for r in routes:
            dist, order = mini_tsp(r)
            total += dist
            demand = sum(Q[s] for s in r)
            route_str = " -> ".join([f"P{s}" for s in order])
            print(f"  P0 -> {route_str} -> P0  距离={dist}  需求={demand}")
        print(f"  总距离: {total}")


if __name__ == "__main__":
    main()
