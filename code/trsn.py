"""TRSN: trajectory-restricted subnetwork construction (Algorithm 4 of the paper).

Two preservation mechanisms are provided:

* ``mode="boundary"`` implements Algorithm 4 exactly: the induced subnetwork of
  R = bbox(U) plus, for every pair of boundary nodes u, v of R, the shortest
  path pi_M(u, v) in the full network.  This is the construction covered by the
  solution-preservation proof (Proposition 7).
* ``mode="buffer"`` is the configuration used by the scalability experiments:
  the induced subnetwork of the bbox buffered by B = s * max_dt, without
  boundary-path augmentation.  Preservation holds for this variant because any
  feasible hop between consecutive recovered points has network length at most
  s * dt <= B and therefore stays inside the buffered box.

Both modes return the subnetwork together with the candidate lists restricted
to it, so TREC/TRRP/TRHA can run unchanged on the subnetwork.
"""
from __future__ import annotations

import numpy as np
import networkx as nx


def seq_bbox(observations, tower_positions, radius):
    """bbox of the coverage disks of all stations observed in one sequence."""
    towers = set()
    for _, tower_list in observations:
        towers.update(tower_list)
    pts = tower_positions[sorted(towers)]
    return [pts[:, 0].min() - radius, pts[:, 1].min() - radius,
            pts[:, 0].max() + radius, pts[:, 1].max() + radius]


def group_by_grid(centers, cell_size):
    """group(F) of Algorithm 4: grid grouping of sequence centers."""
    groups = {}
    for idx, (cx, cy) in enumerate(centers):
        key = (int(np.floor(cx / cell_size)), int(np.floor(cy / cell_size)))
        groups.setdefault(key, []).append(idx)
    return list(groups.values())


def _induced(G, pos, bbox, buffer):
    xmin, ymin, xmax, ymax = bbox
    xmin -= buffer
    ymin -= buffer
    xmax += buffer
    ymax += buffer
    inside = [v for v, (x, y) in pos.items()
              if xmin <= x <= xmax and ymin <= y <= ymax]
    inside_set = set(inside)
    subG = nx.MultiDiGraph()
    subG.add_nodes_from(inside)
    for u, v, k, data in G.edges(keys=True, data=True):
        if u in inside_set and v in inside_set:
            subG.add_edge(u, v, key=k, **data)
    return subG, inside, inside_set


def build_subnetwork(G, pos, bbox, buffer=0.0, mode="boundary",
                     max_dt=None, speed=None, cell_size=None):
    """Construct the localized subnetwork for one group of sequences.

    mode="buffer"  : induced subgraph of bbox buffered by ``buffer``.
    mode="boundary": induced subgraph of bbox (buffer=0) augmented with the
                     full-network shortest path for every boundary pair.
    """
    if mode == "buffer":
        if buffer <= 0:
            raise ValueError("buffer mode requires buffer > 0")
        subG, inside, _ = _induced(G, pos, bbox, buffer)
        return subG, inside
    subG, inside, inside_set = _induced(G, pos, bbox, 0.0)
    boundary = sorted({n for n in inside
                       if any(nb not in inside_set for _, nb in G.edges(n))})
    for i, u in enumerate(boundary):
        for v in boundary[i + 1:]:
            try:
                path = nx.shortest_path(G, u, v, weight="length")
            except nx.NetworkXNoPath:
                continue
            for a, b in zip(path, path[1:]):
                if not subG.has_edge(a, b):
                    data = G.get_edge_data(a, b)
                    key = next(iter(data.values())) if data else {"length": 1.0}
                    subG.add_edge(a, b, **key)
            for n in path:
                if n not in subG:
                    subG.add_node(n)
    return subG, inside


def build_subnets(sequences, tower_positions, radius, G, pos,
                  mode="boundary", cell_size=5000.0, buffer=None,
                  speed=None, timestamps=None):
    """Algorithm 4: per-sequence bbox, grid grouping, per-group subnetwork."""
    centers = []
    boxes = []
    for obs in sequences:
        b = seq_bbox(obs, tower_positions, radius)
        boxes.append(b)
        centers.append(((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0))
    groups = group_by_grid(centers, cell_size)
    out = []
    for group in groups:
        xmin = min(boxes[i][0] for i in group)
        ymin = min(boxes[i][1] for i in group)
        xmax = max(boxes[i][2] for i in group)
        ymax = max(boxes[i][3] for i in group)
        if mode == "buffer":
            if buffer is None and speed is not None and timestamps is not None:
                buffer = speed * max(timestamps[i + 1] - timestamps[i]
                                    for i in range(len(timestamps) - 1))
            subG, inside = build_subnetwork(G, pos, [xmin, ymin, xmax, ymax],
                                            buffer=buffer, mode="buffer")
        else:
            subG, inside = build_subnetwork(G, pos, [xmin, ymin, xmax, ymax],
                                            mode="boundary")
        out.append({"group": group, "subG": subG, "inside": inside})
    return out
