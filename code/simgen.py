"""Candidate sampling, coverage, and road-network distance utilities."""
import numpy as np
import networkx as nx
from shapely.geometry import LineString


def candidate_distances(graph, nodes, weight='length'):
    """Full-graph distances in the supplied node order."""
    nodes = list(nodes)
    if len(set(nodes)) != len(nodes) or any(v not in graph for v in nodes):
        raise ValueError('candidate node IDs must be unique graph nodes')
    index = {v: i for i, v in enumerate(nodes)}
    result = np.full((len(nodes), len(nodes)), np.inf)
    # Validate weights once during preprocessing.
    for _, _, d in graph.edges(data=True):
        if weight not in d or not np.isfinite(d[weight]) or d[weight] < 0:
            raise ValueError('each edge requires a finite nonnegative weight')
    for i, node in enumerate(nodes):
        for target, value in nx.single_source_dijkstra_path_length(graph, node, weight=weight).items():
            if target in index:
                result[i, index[target]] = value
    return result


def get_distances(graph, coords, table=None, node_ids=None):
    size = len(coords)
    if table is None:
        nodes = list(graph.nodes) if node_ids is None else list(node_ids)
        if len(nodes) != size:
            raise ValueError('supply candidate_node_ids matching coordinate order')
        table = candidate_distances(graph, nodes)
    # Shape-only validation keeps recovery linear.
    if np.shape(table) != (size, size):
        raise ValueError('distance table must match candidate order and count')
    return table


def validate_inputs(coverage, times, speed):
    coverage = np.asarray(coverage)
    times = np.asarray(times, dtype=float)
    if coverage.ndim != 2 or min(coverage.shape) == 0:
        raise ValueError('coverage must be a nonempty timestamp-by-candidate array')
    if len(times) != len(coverage) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('timestamps must be finite and strictly increasing')
    if not np.isfinite(speed) or speed < 0:
        raise ValueError('speed must be finite and nonnegative')
    if not np.all(np.isfinite(coverage)) or np.any(coverage < 0):
        raise ValueError('coverage must be finite and nonnegative')
    return coverage, times


def coverage_counts(coords, towers, observations, radius):
    """Disk coverage preprocessing; duplicate observed IDs count only once.

    O(P*m + P*sum_i |C_i|) time, O(P*m+n*P) memory. This is NOT
    included in the O(n*P) TRHA recovery kernel claim.
    """
    coords, towers = np.asarray(coords), np.asarray(towers)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('coverage radius must be finite and nonnegative')
    within = np.linalg.norm(coords[:, None, :] - towers[None, :, :], axis=2) <= radius
    out = np.zeros((len(observations), len(coords)), dtype=np.int64)
    for i, (_, station_ids) in enumerate(observations):
        ids = sorted(set(station_ids))
        if any(j < 0 or j >= len(towers) for j in ids):
            raise ValueError('unknown station index')
        if ids:
            out[i] = within[:, ids].sum(axis=1)
    return out


def package_result(selected, coverage, coords):
    cov = np.asarray(coverage)[np.arange(len(selected)), selected]
    return float(cov.sum()), list(map(int, selected)), np.asarray(coords)[selected].tolist(), cov.tolist()


def compute_dis(G, candidate_nodes):
    return candidate_distances(G, list(candidate_nodes))


def gen_cans(G, pos, seg):
    if not np.isfinite(seg) or seg <= 0:
        raise ValueError('candidate spacing must be positive and finite')
    candidates = []
    candidate_coords = []
    next_node_id = max(G.nodes) + 1 if len(G.nodes) > 0 else 0
    
    for i in G.nodes:
        loc = pos[i]
        candidate_coords.append(loc)
        candidates.append((i, i, 0, loc))
    
    new_pos = dict(pos)
    edges_to_process = list(G.edges(keys=True, data=True))

    for u, v, k, data in edges_to_process:
        # Preserve road metadata on sampled edges.
        attrs = {key:value for key,value in data.items() if key not in ('length','geometry')}
        attrs['source_edge'] = (u,v,k)
        if 'geometry' in data:
            line = data['geometry']
            if not isinstance(line, LineString):
                continue
            edge_length = line.length
        else:
            u_pos, v_pos = np.array(pos[u]), np.array(pos[v])
            edge_length = np.linalg.norm(u_pos - v_pos)
            line = LineString([pos[u], pos[v]])

        if u == v or edge_length == 0:
            continue

        num_segments = max(1, int(edge_length // seg))
        step_size = edge_length / (num_segments + 1)

        prev_node = u
        prev_pos = np.array(pos[u])

        for i in range(1, num_segments + 1):
            distance_along_line = i * step_size
            point_on_line = line.interpolate(distance_along_line)
            loc = np.array([point_on_line.x, point_on_line.y])

            cand_node = next_node_id
            next_node_id += 1

            new_pos[cand_node] = tuple(loc)
            candidates.append((u, v, distance_along_line/edge_length, tuple(loc)))
            candidate_coords.append(tuple(loc))
            segment_line = LineString([prev_pos, loc])
            seg_len = segment_line.length
            G.add_edge(prev_node, cand_node, length=seg_len, geometry=segment_line, **attrs)
            
            prev_node = cand_node
            prev_pos = loc

        segment_line = LineString([prev_pos, pos[v]])
        seg_len = segment_line.length
        G.add_edge(prev_node, v, length=seg_len, geometry=segment_line, **attrs)

        if G.has_edge(u, v, k):
            G.remove_edge(u, v, k)

    return candidates, np.array(candidate_coords), new_pos
