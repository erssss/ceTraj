from scipy.spatial import KDTree
import numpy as np
import networkx as nx
from math import inf
from simgen import *
import numpy as np
from TRRP import *
from collections import deque
from scipy.stats import norm
import numpy as np
import networkx as nx


def run_TRHA(
    G,
    pos,
    candidate_nodes,
    candidate_coords,
    tower_positions,
    coverage_radius,
    timestamps,
    observations,
    max_speed,
    k=10,
    alpha=0.6,
    dist_matrix=None
):
    num_timestamps = len(timestamps)
    num_candidates = len(candidate_coords)
    
    has_out_edge = np.zeros(num_candidates, dtype=bool)
    for i, node_id in enumerate(candidate_nodes):
        for _, neighbor in G.out_edges(node_id[0]):
            if neighbor != node_id[0]:
                has_out_edge[i] = True
                break

    if dist_matrix is None:
        dist_matrix = compute_dis(G, list(range(num_candidates)))
    
    max_time_diff = np.array([timestamps[i] - timestamps[i-1] for i in range(1, num_timestamps)])
    max_possible_move = max_speed * max_time_diff
    

    cover_scores = [None] * num_timestamps
    dists = np.linalg.norm(candidate_coords[:, None, :] - tower_positions[None, :, :], axis=2)  # (C, m)
    for i, (tidx, tower_list) in enumerate(observations):
        mask = np.zeros(dists.shape[1], dtype=bool)
        if tower_list:
            mask[tower_list] = True
        within = (dists <= coverage_radius)
        cov = np.sum(within[:, mask], axis=1) if mask.any() else np.zeros(num_candidates, dtype=int)
        cover_scores[i] = cov

    selected_indices = np.full(num_timestamps, -1, dtype=int)
    traj_coords = [None] * num_timestamps
    cover_values = np.zeros(num_timestamps)
    
    for t in range(num_timestamps):
        current_cover = cover_scores[t]
        move_scores = np.zeros(num_candidates)
        if  t + k < num_timestamps:
            for kk in range(k,k+1):
                future_cover = cover_scores[t + kk]
                best_future_mask = (future_cover == future_cover.max())
                if best_future_mask.any():
                    candidate_indices = np.where(best_future_mask)[0]
                    min_distance = float('inf')
                    best_candidate_id = None
                    future_coords = candidate_coords[best_future_mask].mean(axis=0)
                    centroid_idx = np.argmin(np.linalg.norm(candidate_coords - future_coords, axis=1))

                    for idx in candidate_indices:
                        distance = dist_matrix[idx, centroid_idx]
                        if min_distance>distance:
                            min_distance = distance
                            best_candidate_id = idx
                    
                    if best_candidate_id is None:
                        break

                    dists = dist_matrix[:,best_candidate_id]

                    dists_clean = np.where(np.isfinite(dists), dists, np.nan)
                    max_dist = np.nanmax(dists_clean)
                    
                    if max_dist == 0:
                        normalized_dists = np.zeros_like(dists)
                    else:
                        normalized_dists = np.clip(dists / max_dist, 0, 1)
                    move_scores += (k - kk + 1) * (1.0 - normalized_dists)
        if t == 0:
            valid_candidates = np.arange(num_candidates)
        else:
            prev_idx = selected_indices[t - 1]
            
            valid_candidates = []
            for j in range(num_candidates):
                dist = dist_matrix[prev_idx,j]
                if dist <= max_possible_move[t-1]:
                    valid_candidates.append(j)
            
            valid_candidates = np.array(valid_candidates)

        if t < num_timestamps - 1:
            valid_with_out_edge = []
            for cand in valid_candidates:
                if has_out_edge[cand]:
                    valid_with_out_edge.append(cand)
            
            valid_candidates = np.array(valid_with_out_edge)
        
        if valid_candidates.size == 0:
            print(f"============== Warning: No valid candidates at timestamp {t} ==============")
        else:
            total_scores = (
                alpha * current_cover[valid_candidates]/np.max(cover_scores) +
                (1 - alpha) * (move_scores[valid_candidates])
            )
            cover_score = current_cover[valid_candidates]/np.max(cover_scores)

            max_score = np.max(total_scores)
            best_candidates = valid_candidates[total_scores == max_score]

            if len(best_candidates) > 1:
                if t + k < num_timestamps:
                    prev_idx = selected_indices[t-1]
                    future_cover = cover_scores[t + k]
                    best_future_mask = (future_cover == future_cover.max())
                    if best_future_mask.any():
                        future_coords = candidate_coords[best_future_mask].mean(axis=0)
                        
                        candidate_directions = candidate_coords[best_candidates] - candidate_coords[prev_idx]
                        future_directions = future_coords - candidate_coords[prev_idx]
                        
                        dot_products = np.dot(candidate_directions, future_directions)
                        
                        best_idx = best_candidates[np.argmax(dot_products)]
                    else:
                        best_idx = best_candidates[0]
                else:
                    if t > 1:
                        prev_coord2 = candidate_coords[selected_indices[t-2]]
                        direction = candidate_coords[prev_idx] - prev_coord2
                        candidate_vectors = candidate_coords[best_candidates] - candidate_coords[prev_idx]
                        dot_products = np.dot(candidate_vectors, direction)
                        best_idx = best_candidates[np.argmax(dot_products)]
                    else:
                        distances = np.linalg.norm(candidate_coords[best_candidates] - candidate_coords[prev_idx], axis=1)
                        best_idx = best_candidates[np.argmax(distances)]
            else:
                best_idx = best_candidates[0]

        selected_indices[t] = best_idx
        traj_coords[t] = candidate_coords[best_idx].tolist()
        cover_values[t] = current_cover[best_idx]

    max_coverage = cover_values.sum()
    return max_coverage, selected_indices.tolist(), traj_coords, cover_values.tolist()

