import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Any
from simgen import *
from TREC import *
import heapq
from collections import defaultdict


def map_compressed_traj_to_original(
    sel_indices,
    rep_cidx_of_cluster,
    candidate_coords,
):
    orig_indices = []
    orig_coords = []

    for cid in sel_indices:
        rep_idx = rep_cidx_of_cluster[cid]
        orig_indices.append(rep_idx)
        orig_coords.append(candidate_coords[rep_idx])

    orig_coords = np.array(orig_coords)
    return orig_indices, orig_coords


def pick_representative(indices, coords, cid):
    pts = coords[indices]
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    rep_local = int(np.argmin(dists))
    return indices[rep_local]


def clu(
    new_pos: dict,
    candidates: list,
    delta: float,
    lazy_dist_matrix,
):
    n_candidates = len(candidates)
    visited = set()
    final_cluster = []

    D_internal = np.zeros((n_candidates, n_candidates))

    def get_distance(i, j):
        if D_internal[i, j] == 0 and i != j:
            D_internal[i, j] = lazy_dist_matrix[i, j]
            D_internal[j, i] = lazy_dist_matrix[j, i]
        return D_internal[i, j]

    densities = np.zeros(n_candidates)
    for i in range(n_candidates):
        for j in range(n_candidates):
            if i == j:
                continue
            dist = get_distance(i, j)
            if dist < delta / 2:
                densities[i] += 1

    sorted_indices = np.argsort(-densities)

    for idx in sorted_indices:
        if idx in visited:
            continue

        current_cluster = [idx]
        visited.add(idx)

        for j in range(n_candidates):
            if j in visited:
                continue

            can_add = True
            for k in current_cluster:
                dist_jk = get_distance(j, k)
                if dist_jk >= delta:
                    can_add = False
                    break

            if can_add:
                current_cluster.append(j)
                visited.add(j)

        final_cluster.append(current_cluster)

    return final_cluster


def compress(
    G,
    new_pos,
    candidates,
    candidate_coords,
    timestamps,
    speed,
    delta,
    D,
):

    n = len(candidate_coords)

    final_clusters = clu(new_pos, candidates, delta, D)

    cluster_id_of_cidx = {}
    rep_cidx_of_cluster = {}
    reps = []
    for cid, idxs in enumerate(final_clusters):
        for i in idxs:
            cluster_id_of_cidx[i] = cid
        rep = pick_representative(idxs, candidate_coords, cid)
        reps.append(rep)
        rep_cidx_of_cluster[cid] = rep

    Gc = nx.MultiDiGraph()
    new_pos_c = {}
    for cid, rep_cidx in rep_cidx_of_cluster.items():
        Gc.add_node(cid)
        new_pos_c[cid] = tuple(candidate_coords[rep_cidx])

    rep_node_to_cid = {rep_idx: cid for cid, rep_idx in rep_cidx_of_cluster.items()}
    kept_nodes = set(reps)
    all_nodes_in_G = set(G.nodes())
    removed_nodes = all_nodes_in_G - kept_nodes

    def dijkstra_restricted(start):
        dist = {}
        seen = set()
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            dist[u] = d
            if (u in kept_nodes) and (u != start):
                continue
            for v, keydict in G[u].items():
                for _, attr in keydict.items():
                    L = float(attr.get("length", 0.0))
                    if v in seen:
                        continue
                    if (v in removed_nodes) or (v in kept_nodes):
                        heapq.heappush(pq, (d + L, v))
        return dist

    edge_len_map = {}
    for start in kept_nodes:
        dist = dijkstra_restricted(start)
        cu = rep_node_to_cid[start]
        for node_v, length in dist.items():
            if node_v == start:
                continue
            if node_v not in rep_node_to_cid:
                continue
            cv = rep_node_to_cid[node_v]
            if cu == cv:
                continue
            key = (cu, cv)
            if key not in edge_len_map or length < edge_len_map[key]:
                edge_len_map[key] = length

    for (cu, cv), L in edge_len_map.items():
        if cu == cv:
            continue
        Gc.add_edge(cu, cv, length=float(L))

    candidates_c = []
    coords_c = []
    for cid in range(len(final_clusters)):
        rep_cidx = rep_cidx_of_cluster[cid]
        try:
            orig_tuple = candidates[rep_cidx]
            if isinstance(orig_tuple, tuple) and len(orig_tuple) == 4:
                candidates_c.append(orig_tuple)
            else:
                loc = tuple(candidate_coords[rep_cidx])
                candidates_c.append((None, None, None, loc))
        except Exception:
            loc = tuple(candidate_coords[rep_cidx])
            candidates_c.append((None, None, None, loc))
        coords_c.append(candidate_coords[rep_cidx])
    candidate_coords_c = np.array(coords_c, dtype=float)

    node_compression_ratio = Gc.number_of_nodes() / max(1, G.number_of_nodes())
    edge_compression_ratio = Gc.number_of_edges() / max(1, G.number_of_edges())

    return (
        Gc,
        new_pos_c,
        candidates_c,
        candidate_coords_c,
        cluster_id_of_cidx,
        rep_cidx_of_cluster,
        node_compression_ratio,
        edge_compression_ratio,
        final_clusters,
    )


def compute_Dc_from_D(D, rep_cidx_of_cluster):
    rep_indices = list(rep_cidx_of_cluster.values())
    k = len(rep_indices)
    Dc = np.zeros((k, k))
    for i, orig_i in enumerate(rep_indices):
        for j, orig_j in enumerate(rep_indices):
            Dc[i, j] = D[orig_i, orig_j]

    return Dc


global_D = None


def run_TRRP(
    G,
    new_pos,
    candidates,
    candidate_coords,
    timestamps,
    speed,
    delta,
    tower_positions,
    coverage_radius,
    observations,
    max_speed,
    cal=False,
    dist_matrix=None,
):
    global global_D
    candidate_nodes = list(range(len(candidate_coords)))

    if dist_matrix is None:
        D = compute_dis(G, candidate_nodes)
    else:
        D = dist_matrix

    (
        Gc,
        new_pos_c,
        candidates_c,
        candidate_coords_c,
        cluster_id_of_cidx,
        rep_cidx_of_cluster,
        node_compression_ratio,
        edge_compression_ratio,
        final_clusters,
    ) = compress(
        G, new_pos, candidates, candidate_coords, timestamps, speed, delta, D
    )

    m = len(final_clusters)
    if m == 0:
        return None, None, None, None, 0, 0, None, None, None, None

    if global_D is None:
        if cal:
            Dc = compute_Dc_from_D(D, rep_cidx_of_cluster)
        else:
            Dc = compute_dis(Gc, list(Gc.nodes()))
        global_D = Dc
    else:
        Dc = global_D
    best_score, sel_indices, sel_coords, sel_covs = run_TREC(
        Gc,
        new_pos_c,
        list(Gc.nodes()),
        candidate_coords_c,
        tower_positions,
        coverage_radius,
        timestamps,
        observations,
        max_speed,
        filter_candidates_by_observation=False,
        dist_matrix=Dc,
    )

    return (
        best_score,
        sel_indices,
        sel_coords,
        sel_covs,
        node_compression_ratio,
        edge_compression_ratio,
        Gc,
        new_pos_c,
        candidates_c,
        candidate_coords_c,
        final_clusters,
        rep_cidx_of_cluster,
        cluster_id_of_cidx,
    )
