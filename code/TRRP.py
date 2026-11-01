"""Representative grouping (Algorithm 2), using the original network table."""
import numpy as np
from scipy.spatial import cKDTree
from simgen import get_distances, coverage_counts, package_result
from TREC import trec


def map_compressed_traj_to_original(sel_indices, rep_cidx_of_cluster, candidate_coords):
    indices = [rep_cidx_of_cluster[int(i)] for i in sel_indices]
    return indices, np.asarray(candidate_coords)[indices]


def pick_representative(indices, coords, cid=None):
    points = np.asarray(coords)[indices]
    return indices[int(np.argmin(np.linalg.norm(points-points.mean(axis=0), axis=1)))]


def clu(new_pos, candidates, delta, lazy_dist_matrix, kdtree=None, coords=None):
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('group diameter must be positive and finite')
    if coords is None:
        coords = np.asarray([new_pos[i] for i in range(len(candidates))])
    if kdtree is None:
        kdtree = cKDTree(coords)
    count = len(candidates)
    D = lazy_dist_matrix
    if np.shape(D) != (count, count):
        raise ValueError('distance table shape mismatch')
    def ball(i, radius):
        return sorted(j for j in kdtree.query_ball_point(coords[i], radius) if j != i)
    density = np.empty(count, dtype=np.int64)
    for i in range(count):
        density[i] = sum(D[i,j] < delta/2 for j in ball(i, delta/2))
    visited = np.zeros(count, dtype=bool)
    groups = []
    for i in np.argsort(-density, kind='stable'):
        if visited[i]:
            continue
        group = [int(i)]
        visited[i] = True
        for j in ball(i, delta):
            if not visited[j] and all(max(D[j,h], D[h,j]) < delta for h in group):
                group.append(j)
                visited[j] = True
        groups.append(group)
    return groups


def compress(G, new_pos, candidates, candidate_coords, timestamps, speed, delta, D):
    """Return representatives and membership; Gc and edge ratio are None.

    Representative distances use the original network table D[R,R].
    """
    groups = clu(new_pos, candidates, delta, D, coords=np.asarray(candidate_coords))
    reps = {i:pick_representative(group, candidate_coords) for i,group in enumerate(groups)}
    membership = {j:i for i,group in enumerate(groups) for j in group}
    indices = list(reps.values())
    coords = np.asarray(candidate_coords)[indices]
    return (None, dict(enumerate(map(tuple, coords))), [candidates[j] for j in indices],
            coords, membership, reps, len(indices)/len(candidates), None, groups)


def compute_Dc_from_D(D, rep_cidx_of_cluster):
    indices = list(rep_cidx_of_cluster.values())
    return D[np.ix_(indices, indices)]


def run_TRRP(G, new_pos, candidates, candidate_coords, timestamps, speed, delta,
             tower_positions, coverage_radius, observations, max_speed, cal=False,
             dist_matrix=None, *, coverage=None, candidate_node_ids=None,
             reachability_matrix=None):
    if speed != max_speed:
        raise ValueError('legacy speed and max_speed arguments must agree')
    D = get_distances(G, candidate_coords, dist_matrix, candidate_node_ids)
    Gc, pc, cans, cc, membership, reps, ratio, edge_ratio, groups = compress(
        G,new_pos,candidates,candidate_coords,timestamps,speed,delta,D)
    indices = list(reps.values())
    Dc = compute_Dc_from_D(D if reachability_matrix is None else reachability_matrix, reps)
    cov = coverage_counts(cc,tower_positions,observations,coverage_radius) if coverage is None else np.asarray(coverage)[:,indices]
    selected = trec(cov,timestamps,Dc,max_speed)
    score,sel,xy,values = package_result(selected,cov,cc)
    return score,sel,xy,values,ratio,edge_ratio,Gc,pc,cans,cc,groups,reps,membership
