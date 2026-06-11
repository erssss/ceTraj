import math
import numpy as np
import networkx as nx
from collections import defaultdict
from simgen import *


class ShortestPathCache:
    def __init__(self, G):
        self.G = G
        self.cache = {}

    def distances_from(self, src):
        if src in self.cache:
            return self.cache[src]
        dist = nx.single_source_dijkstra_path_length(self.G, src, weight="length")
        self.cache[src] = dist
        return dist

    def dist(self, a, b):
        da = self.distances_from(a)
        return da.get(b, math.inf)


def cov_per_time(candidate_coords, tower_positions, observations, coverage_radius):
    n = len(observations)
    C = len(candidate_coords)
    cov_scores = [None] * n
    dists = np.linalg.norm(candidate_coords[:, None, :] - tower_positions[None, :, :], axis=2)  # (C, m)
    for i, (tidx, tower_list) in enumerate(observations):
        mask = np.zeros(dists.shape[1], dtype=bool)
        if tower_list:
            mask[tower_list] = True
        within = (dists <= coverage_radius)
        cov = np.sum(within[:, mask], axis=1) if mask.any() else np.zeros(C, dtype=int)
        cov_scores[i] = cov
    return cov_scores


def run_TREC(
    G,
    pos,
    candidate_nodes,
    candidate_coords,
    tower_positions,
    coverage_radius,
    timestamps,
    observations,
    max_speed,
    filter_candidates_by_observation=False,
    dist_matrix=None,
):
    n = len(observations)
    C = len(candidate_coords)

    if dist_matrix is None:
        dist_matrix = compute_candidate_distance_matrix(G, list(G.nodes()))
    
    cov_scores = cov_per_time(candidate_coords, tower_positions, observations, coverage_radius)

    feasible_idxs_by_time = []
    for i in range(n):
        if filter_candidates_by_observation:
            feas = [j for j in range(C) if cov_scores[i][j] > 0]
            if len(feas) == 0:
                feas = list(range(C))
        else:
            feas = list(range(C))
        feasible_idxs_by_time.append(feas)

    dts = [timestamps[i+1] - timestamps[i] for i in range(n-1)]
    max_dists = [max_speed * dt for dt in dts]

    from collections import defaultdict
    Prev = [defaultdict(list) for _ in range(n-1)]

    if dist_matrix is not None:
        for i in range(n-1):
            allowed = max_dists[i]
            feas_i = feasible_idxs_by_time[i]
            feas_ip1 = feasible_idxs_by_time[i+1]
            for k in feas_ip1:
                prev_list = [j for j in feas_i if dist_matrix[j, k] <= allowed + 1e-8]
                Prev[i][k] = prev_list
    else:
        spcache = ShortestPathCache(G)
        for i in range(n-1):
            allowed = max_dists[i]
            feas_i = feasible_idxs_by_time[i]
            feas_ip1 = feasible_idxs_by_time[i+1]
            for k in feas_ip1:
                prev_list = []
                for j in feas_i:
                    node_j = candidate_nodes[j]
                    node_k = candidate_nodes[k]
                    dist = spcache.dist(node_j, node_k)
                    if dist <= allowed + 1e-8:
                        prev_list.append(j)
                Prev[i][k] = prev_list

    NEG_INF = -10**18
    dp_prev = {j: int(cov_scores[0][j]) for j in feasible_idxs_by_time[0]}
    backptrs = [dict() for _ in range(n)]

    i=0
    for i in range(1, n):
        dp_curr = {}
        feas_i = feasible_idxs_by_time[i]
        for k in feas_i:
            preds = Prev[i-1].get(k, [])
            best_val = NEG_INF
            best_pred = None
            if preds:
                for j in preds:
                    valj = dp_prev.get(j, NEG_INF)
                    if valj > NEG_INF/2:
                        cand_val = valj + int(cov_scores[i][k])
                        if cand_val > best_val:
                            best_val = cand_val
                            best_pred = j
            if best_pred is not None:
                dp_curr[k] = best_val
                backptrs[i][k] = best_pred
                
        dp_prev = dp_curr
        if not dp_prev:
            return None, None, None, None

    last_time = n-1
    best_k, best_score = max(dp_prev.items(), key=lambda x: x[1])
    sel_indices = [None] * n
    sel_indices[last_time] = best_k
    for t in range(last_time, 0, -1):
        sel_indices[t-1] = backptrs[t][sel_indices[t]]
    sel_coords = [tuple(candidate_coords[idx]) for idx in sel_indices]
    sel_covs = [int(cov_scores[i][sel_indices[i]]) for i in range(n)]
    
    return best_score, sel_indices, sel_coords, sel_covs