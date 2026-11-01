"""Exact trajectory recovery (Algorithm 1)."""
import numpy as np
from simgen import coverage_counts, get_distances, validate_inputs, package_result

def trec(coverage, times, distances, speed, tolerance=1e-8):
    """Algorithm 1: O(n P^2) time and O(n P) extra space."""
    coverage, times = validate_inputs(coverage, times, speed)
    n, count = coverage.shape
    if np.shape(distances) != (count, count):
        raise ValueError('distance table shape mismatch')
    previous = coverage[0].astype(float)
    parent = np.full((n, count), -1, dtype=np.int64)
    for i in range(1, n):
        current = np.full(count, -np.inf)
        budget = speed * (times[i] - times[i-1])
        for j in range(count):
            values = np.where(distances[:, j] <= budget + tolerance, previous, -np.inf)
            h = int(np.argmax(values))
            if np.isfinite(values[h]):
                current[j] = values[h] + coverage[i, j]
                parent[i, j] = h
        previous = current
    if not np.isfinite(previous).any():
        raise ValueError('no feasible trajectory')
    selected = np.empty(n, dtype=np.int64)
    selected[-1] = np.argmax(previous)
    for i in range(n-1, 0, -1):
        selected[i-1] = parent[i, selected[i]]
    return selected


cov_per_time = coverage_counts


def run_TREC(G, pos, candidate_nodes, candidate_coords, tower_positions,
             coverage_radius, timestamps, observations, max_speed,
             filter_candidates_by_observation=False, dist_matrix=None,
             *, coverage=None, candidate_node_ids=None):
    if filter_candidates_by_observation:
        raise ValueError('Algorithm 1 retains zero-coverage candidates; filtering is not supported')
    D = get_distances(G, candidate_coords, dist_matrix, candidate_node_ids)
    cov = coverage_counts(candidate_coords, tower_positions, observations, coverage_radius) if coverage is None else coverage
    selected = trec(cov, timestamps, D, max_speed)
    return package_result(selected, cov, candidate_coords)
