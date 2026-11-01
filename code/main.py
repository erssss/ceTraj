"""Recover a trajectory from cellular records."""
import argparse
import json
from pathlib import Path

import numpy as np

from load_real import load_realdata
from simgen import candidate_distances, coverage_counts, package_result
from TREC import trec
from TRHA import trha
from TRRP import clu, pick_representative
from TRSN import build_subnets


def recover(G, pos, coords, towers, timestamps, observations, *,
            radius, speed, delta, alpha, k, algorithms, localize=True):
    """Return original candidate indices and coordinates for one sequence."""
    original_nodes = list(G)
    if localize:
        group = build_subnets([observations], towers, radius, G, pos,
                             candidate_coords=coords)[0]
        graph = group['subG']
    else:
        graph = G
    indices = np.array([i for i, v in enumerate(original_nodes) if v in graph])
    nodes = [original_nodes[i] for i in indices]
    xy = np.asarray(coords)[indices]
    distances = candidate_distances(graph, nodes)
    coverage = coverage_counts(xy, towers, observations, radius)
    results = {}
    for algorithm in algorithms:
        if algorithm == 'TREC':
            selected = trec(coverage, timestamps, distances, speed)
        elif algorithm == 'TRHA':
            selected = trha(coverage, timestamps, xy, distances, speed, alpha, k)
        elif algorithm == 'TRRP':
            groups = clu({}, nodes, delta, distances, coords=xy)
            reps = np.array([pick_representative(group, xy) for group in groups])
            selected = reps[trec(coverage[:, reps], timestamps,
                                 distances[np.ix_(reps, reps)], speed)]
        else:
            raise ValueError(f'Unknown algorithm: {algorithm}')
        total, _, coordinates, scores = package_result(selected, coverage, xy)
        results[algorithm] = {
            'candidate_indices': indices[selected].tolist(),
            'coordinates': coordinates,
            'coverage': total,
            'coverage_by_timestamp': scores,
        }
    return results


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--obs', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True,
                        help='Road graph path prefix, without _graph.pkl')
    parser.add_argument('--cells', type=Path, default=root/'data/GZ/input_cells.csv')
    parser.add_argument('--algs', nargs='+', choices=['TREC', 'TRRP', 'TRHA'],
                        default=['TREC', 'TRRP', 'TRHA'])
    parser.add_argument('--seg', type=float, required=True)
    parser.add_argument('--radius', type=float, required=True)
    parser.add_argument('--speed', type=float, required=True)
    parser.add_argument('--delta', type=float, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--no-trsn', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    G, pos, _, coords, towers, _, times, _, obs, _ = load_realdata(
        args.seg, args.obs, str(args.graph), cell_path=args.cells)
    result = recover(G, pos, coords, towers, times, obs, radius=args.radius,
                     speed=args.speed, delta=args.delta, alpha=args.alpha,
                     k=args.k, algorithms=args.algs, localize=not args.no_trsn)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
