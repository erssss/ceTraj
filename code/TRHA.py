"""Horizon-anchored recovery (Algorithm 3)."""
import numpy as np
from simgen import coverage_counts, get_distances, validate_inputs, package_result

def trha(coverage, times, coords, distances, speed, alpha, k,
         reachability=None, trace=None):
    """Algorithm 3: O(n P) time with precomputed coverage and distances.

    Anchor scores use road distance. Optional reachability uses speed's units
    (minutes with speed=1 for per-edge). Ties use the anchor-direction dot
    product, previous movement, then candidate index.
    """
    coverage, times = validate_inputs(coverage, times, speed)
    if not 0 < alpha <= 1 or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('require 0 < alpha <= 1 and positive integer k')
    n, count = coverage.shape
    coords = np.asarray(coords)
    if np.shape(distances) != (count, count) or coords.shape != (count, 2):
        raise ValueError('candidate coordinates/table shape mismatch')
    travel = distances if reachability is None else reachability
    if np.shape(travel) != (count, count):
        raise ValueError('reachability table shape mismatch')
    cmax = coverage.max()  # Compute once in O(n P).
    selected = np.empty(n, dtype=np.int64)
    for i in range(n):
        anchor = None
        score = coverage[i] / cmax if cmax > 0 else np.zeros(count)
        if i + k < n:
            future = coverage[i+k]
            J = np.flatnonzero(future == future.max())
            centroid = coords[J].mean(axis=0)
            c = int(np.argmin(np.linalg.norm(coords-centroid, axis=1)))
            anchor = int(J[np.argmin(distances[J, c])])
            to_anchor = distances[:, anchor]
            finite = np.isfinite(to_anchor)
            dmax = np.max(to_anchor[finite]) if finite.any() else 0.
            proximity = np.zeros(count)
            if dmax > 0:
                proximity[finite] = 1 - np.minimum(1, to_anchor[finite]/dmax)
            else:
                # Zero-distance candidates have proximity one.
                proximity[finite] = 1.
            score = alpha * score + (1-alpha) * proximity
        valid = np.arange(count) if i == 0 else np.flatnonzero(
            travel[selected[i-1]] <= speed*(times[i]-times[i-1]) + 1e-8)
        if not len(valid):
            raise ValueError('no reachable candidate')
        ties = valid[score[valid] == np.max(score[valid])]
        chosen = int(ties[0])
        direction = None
        if i > 0 and anchor is not None:
            direction = coords[anchor]-coords[selected[i-1]]
        if (direction is None or not np.any(direction)) and i > 1:
            direction = coords[selected[i-1]]-coords[selected[i-2]]
        if len(ties) > 1 and direction is not None and np.any(direction):
            chosen = int(ties[np.argmax((coords[ties]-coords[selected[i-1]]) @ direction)])
        selected[i] = chosen
        if trace is not None:
            trace.append({'i':i, 'selected':chosen, 'anchor':anchor,
                          'score':float(score[chosen]), 'ties':ties.tolist()})
    return selected



def run_TRHA(G, pos, candidate_nodes, candidate_coords, tower_positions,
             coverage_radius, timestamps, observations, max_speed,
             k=10, alpha=0.6, dist_matrix=None, *, coverage=None,
             reachability_matrix=None, candidate_node_ids=None, trace=None):
    D = get_distances(G, candidate_coords, dist_matrix, candidate_node_ids)
    cov = coverage_counts(candidate_coords, tower_positions, observations, coverage_radius) if coverage is None else coverage
    selected = trha(cov, timestamps, candidate_coords, D, max_speed, alpha, k,
                    reachability=reachability_matrix, trace=trace)
    return package_result(selected, cov, candidate_coords)
