"""Algorithm 4 on a directed graph subdivided at candidate locations.

Boundary nodes are inside endpoints of crossing edges in either direction.
Retained nodes include added shortest-path nodes.
"""
import heapq
from itertools import count
import numpy as np
import networkx as nx


def seq_bbox(observations, tower_positions, radius, candidate_coords=None):
    ids = sorted({j for _, stations in observations for j in stations})
    if not ids:
        return None
    towers = np.asarray(tower_positions)[ids]
    if candidate_coords is not None:
        coords = np.asarray(candidate_coords)
        supported = np.zeros(len(coords), dtype=bool)
        for tower in towers:
            supported |= np.linalg.norm(coords-tower, axis=1) <= radius
        points = coords[supported]
        if not len(points):
            return None
        lo, hi = points.min(axis=0), points.max(axis=0)
    else:
        # Use the disk envelope when candidates are unavailable.
        lo, hi = towers.min(axis=0)-radius, towers.max(axis=0)+radius
    return [float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])]


def group_by_grid(centers, cell_size):
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError('cell_size must be positive, in coordinate units')
    groups = {}
    for i,(x,y) in enumerate(centers):
        groups.setdefault((int(np.floor(x/cell_size)),int(np.floor(y/cell_size))),[]).append(i)
    return list(groups.values())


def _induced(G, pos, bbox, buffer):
    xmin,ymin,xmax,ymax = bbox
    inside = [v for v in G if xmin-buffer <= pos[v][0] <= xmax+buffer
              and ymin-buffer <= pos[v][1] <= ymax+buffer]
    return G.subgraph(inside).copy(), inside, set(inside)


def _shortest_tree(graph, source):
    # Store one shortest-path parent per vertex.
    parent = {}; best = {source:0.}; done=set(); serial=count()
    queue=[(0.,next(serial),source)]
    while queue:
        distance,_,u=heapq.heappop(queue)
        if u in done: continue
        done.add(u)
        for v,d in graph[u].items():
            candidate=distance+d['length']
            if v not in done and candidate < best.get(v,float('inf')):
                best[v]=candidate;parent[v]=u
                heapq.heappush(queue,(candidate,next(serial),v))
    return parent,done


def build_subnetwork(G, pos, bbox, buffer=0., mode='boundary', **unused):
    if not G.is_directed() or not G.is_multigraph():
        raise ValueError('TRSN requires a directed MultiDiGraph')
    if mode not in ('boundary','buffer'):
        raise ValueError('unknown TRSN mode')
    if mode=='buffer':
        if buffer <= 0:
            raise ValueError('buffer mode requires buffer > 0')
        sub,inside,_ = _induced(G,pos,bbox,buffer)
        return sub,inside
    sub,inside,inside_set = _induced(G,pos,bbox,0.)
    boundary_set = set()
    # Include incoming and outgoing cut edges.
    for u,v in G.edges():
        if (u in inside_set) != (v in inside_set):
            boundary_set.add(u if u in inside_set else v)
    boundary = [v for v in G if v in boundary_set]
    simple = nx.DiGraph()
    simple.add_nodes_from(G)
    for u,v,k,d in G.edges(keys=True,data=True):
        w = d.get('length')
        if w is None or not np.isfinite(w) or w < 0:
            raise ValueError('every road edge requires a finite nonnegative length')
        if not simple.has_edge(u,v) or w < simple[u][v]['length']:
            simple.add_edge(u,v,length=w,original_key=k)
    for source in boundary:
        # One search per source covers all ordered boundary pairs.
        parent,reached = _shortest_tree(simple,source)
        for target in boundary:
            if target==source or target not in reached:
                continue
            route = [target]
            while route[-1] != source:
                route.append(parent[route[-1]])
            route.reverse()
            for node in route:
                if node not in sub:
                    sub.add_node(node,**G.nodes[node])
            for u,v in zip(route,route[1:]):
                k = simple[u][v]['original_key']
                sub.add_edge(u,v,key=k,**G[u][v][k])
    return sub,[v for v in G if v in sub]


def build_subnets(sequences,tower_positions,radius,G,pos,mode='boundary',
                  cell_size=5.,buffer=None,speed=None,timestamps=None,
                  candidate_coords=None):
    """Coordinates/radius/cell_size in km for released GZ data."""
    boxes = [seq_bbox(obs,tower_positions,radius,candidate_coords) for obs in sequences]
    if any(b is None for b in boxes):
        # Retain the full network for empty coverage support.
        full = np.asarray([pos[v] for v in G])
        lo,hi=full.min(axis=0),full.max(axis=0)
        boxes=[b if b is not None else [lo[0],lo[1],hi[0],hi[1]] for b in boxes]
    groups=group_by_grid([((b[0]+b[2])/2,(b[1]+b[3])/2) for b in boxes],cell_size)
    out=[]
    for group in groups:
        b=[min(boxes[i][0] for i in group),min(boxes[i][1] for i in group),
           max(boxes[i][2] for i in group),max(boxes[i][3] for i in group)]
        actual_buffer=buffer
        if mode=='buffer' and actual_buffer is None:
            if speed is None or timestamps is None:
                raise ValueError('buffer requires explicit buffer or speed and timestamps')
            actual_buffer=speed*max(np.diff(timestamps))
        sub,retained=build_subnetwork(G,pos,b,buffer=actual_buffer or 0.,mode=mode)
        _,inside,_=_induced(G,pos,b,0.)
        out.append({'group':group,'subG':sub,'inside':inside,'retained':retained,'bbox':b})
    return out
