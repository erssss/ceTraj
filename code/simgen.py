import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import networkx as nx
import time
from scipy.spatial import KDTree
from tqdm import tqdm
import random
from scipy.spatial import cKDTree
import math
from sklearn.metrics import precision_score, recall_score, f1_score
import subprocess
import os
import sumolib
from shapely.geometry import LineString, Point
import numpy as np


ROADSCALE = 10
KILOMETER = 1000
p_tx_dbm = 30

def estimate_coverage_radius(
    p_tx_dbm=43.0,
    pl0_db=32.4,
    d0=1.0,
    n_pathloss=3.0,
    detect_rsrp_thresh_dbm=-120.0
):
    exponent = (p_tx_dbm - pl0_db - detect_rsrp_thresh_dbm) / (10.0 * n_pathloss)
    d_max = d0 * (10 ** exponent)
    return d_max

def dbm_to_mw(dbm):
    return 10 ** (dbm / 10.0)

def mw_to_dbm(mw):
    return 10 * np.log10(np.maximum(mw, 1e-15))

def simulate_signal_strengths(
    user_pos,
    tower_positions,
    p_tx_dbm=20,
    n_pathloss=3.0,
    d0=1.0,
    pl0_db=32.4,
    shadow_sigma_db=4.0,
    fast_fading_db=0.0,
    noise_floor_dbm=-104.0,
    n_prb=50,
):
    dists = np.linalg.norm(tower_positions - user_pos, axis=1) + 1e-6
    pl_db = pl0_db + 10.0 * n_pathloss * np.log10(dists / d0)
    shadow = np.random.normal(0, shadow_sigma_db, size=len(dists))
    fast = np.random.normal(0, fast_fading_db, size=len(dists)) if fast_fading_db > 0 else 0.0
    rsrp_dbm = p_tx_dbm - pl_db + shadow + fast
    rsrp_mw = dbm_to_mw(rsrp_dbm)
    noise_mw = dbm_to_mw(noise_floor_dbm)
    rssi_mw = np.sum(rsrp_mw) + noise_mw
    rssi_dbm = mw_to_dbm(rssi_mw)
    rsrq_linear = n_prb * rsrp_mw / rssi_mw
    rsrq_db = 10.0 * np.log10(rsrq_linear + 1e-15)

    return rsrp_dbm, rssi_dbm, rsrq_db, dists


def compute_dis(G, candidate_nodes):

    n = len(candidate_nodes)
    node_to_idx = {node: i for i, node in enumerate(candidate_nodes)}
    D = np.full((n, n), np.inf, dtype=float)

    subG = G.subgraph(candidate_nodes).copy()
    for i, src in enumerate(candidate_nodes):
        lengths = nx.single_source_dijkstra_path_length(subG, src, weight="length")
        for tgt, dist in lengths.items():
            j = node_to_idx.get(tgt)
            if j is not None:
                D[i, j] = dist
    np.fill_diagonal(D, 0.0)
    return D



def eval(
    path_nodes,
    predicted_idx,
    candidates,
    candidate_coords,
    G
):
    gt_coords = np.array([node[0] for node in path_nodes])
    pred_coords = np.array([candidate_coords[idx] for idx in predicted_idx])
    rmse = np.sqrt(np.mean(np.sum((gt_coords - pred_coords)**2, axis=1)))
    mae = np.mean(np.linalg.norm(gt_coords - pred_coords, axis=1))
    
    def get_edge_id(candidate_idx):
        cand = candidates[candidate_idx]
        if cand[2] == 0:
            u = cand[0]
            return set((u,))
        else:
            u, v = cand[0], cand[1]
            return set((u, v))
    hits = 0
    for i in range(len(path_nodes)):
        gt_curr_id = set(path_nodes[i][1])
        pred_curr_id = get_edge_id(predicted_idx[i])
        if gt_curr_id == pred_curr_id:
            hits+=1
        elif len(gt_curr_id & pred_curr_id) > 0 and (len(pred_curr_id)==1 or len(gt_curr_id)==1):
            hits+=1
    
    return {
        'rmse': rmse,
        'mae': mae,
        'precision': hits / len(path_nodes)
    }
    
    
def place_towers_fully_random(num_towers, x_range=(0, 1), y_range=(0, 1), noise_level=1):
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    grid_size = int(np.ceil(np.sqrt(num_towers)))
    
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    
    if len(grid_points) > num_towers:
        indices = np.random.choice(len(grid_points), num_towers, replace=False)
        tower_locs = grid_points[indices]
    else:
        tower_locs = grid_points
        remaining = num_towers - len(grid_points)
        if remaining > 0:
            random_points = np.column_stack((
                np.random.uniform(x_min, x_max, remaining),
                np.random.uniform(y_min, y_max, remaining)
            ))
            tower_locs = np.vstack((tower_locs, random_points))
    
    x_spacing = (x_max - x_min) / (grid_size - 1) if grid_size > 1 else (x_max - x_min)
    y_spacing = (y_max - y_min) / (grid_size - 1) if grid_size > 1 else (y_max - y_min)
    
    max_noise_x = noise_level * x_spacing
    max_noise_y = noise_level * y_spacing
    
    noise_x = np.random.uniform(-max_noise_x, max_noise_x, num_towers)
    noise_y = np.random.uniform(-max_noise_y, max_noise_y, num_towers)
    noise = np.column_stack((noise_x, noise_y))
    
    tower_locs += noise
    
    tower_locs[:, 0] = np.clip(tower_locs[:, 0], x_min, x_max)
    tower_locs[:, 1] = np.clip(tower_locs[:, 1], y_min, y_max)
    
    return tower_locs


def place_towers_simple(num_towers, candidate_coords, x_range=(0, 1), y_range=(0, 1)):
    x_min, x_max = x_range
    y_min, y_max = y_range

    candidate_x = candidate_coords[:, 0]
    candidate_y = candidate_coords[:, 1]
    in_range_mask = (
        (candidate_x >= x_min)
        & (candidate_x <= x_max)
        & (candidate_y >= y_min)
        & (candidate_y <= y_max)
    )
    valid_candidate_idxs = np.where(in_range_mask)[0]

    selected_idxs = np.random.choice(
        valid_candidate_idxs,
        size=num_towers,
        # replace=False
    )

    tower_locs = candidate_coords[selected_idxs] + (x_max*0.2)*np.random.normal(
        scale=0.2, size=(num_towers, 2)
    )

    return tower_locs

def gentrajectory(
    timestamps,
    G,
    pos,
    start_node=None,
    edge_selection="random",
    speed_range=(0, 35),
):
    num_timestamps = len(timestamps)
    trajectory = []
    path_nodes = []

    if start_node is None:
        start_node = random.choice(list(G.nodes()))
    current_node = start_node
    current_edge = None
    current_r = 0.0
    current_speed = random.uniform(*speed_range)
    
    start_coord = np.array(pos[start_node])
    trajectory.append(start_coord.tolist())
    path_nodes.append((start_coord,(start_node,start_node)))

    for i in range(1, num_timestamps):
        dt = timestamps[i] - timestamps[i-1]
        distance_to_travel = current_speed * dt
        
        while distance_to_travel > 0:
            if current_edge is None:
                out_edges = []
                for neighbor in G.neighbors(current_node):
                    for key in G[current_node][neighbor]:
                        edge_len = G[current_node][neighbor][key]["length"]
                        out_edges.append(((current_node, neighbor, key), edge_len))
                
                if not out_edges:
                    distance_to_travel = 0
                    continue
                
                if edge_selection == "random":
                    current_edge_data = random.choice(out_edges)
                else:
                    current_edge_data = min(out_edges, key=lambda x: x[1])
                
                current_edge, edge_len = current_edge_data
                current_r = 0.0
            
            remaining_on_edge = edge_len * (1 - current_r)
            
            if distance_to_travel <= remaining_on_edge:
                current_r += distance_to_travel / edge_len
                distance_to_travel = 0
            else:
                distance_to_travel -= remaining_on_edge
                current_r = 1.0
                
                u, v, key = current_edge
                current_node = v
                current_edge = None
        
        if current_edge is not None:
            u, v, key = current_edge
            u_pos = np.array(pos[u])
            v_pos = np.array(pos[v])
            current_coord = (1 - current_r) * u_pos + current_r * v_pos
            path_nodes.append((current_coord,(u, v)))
        else:
            current_coord = np.array(pos[current_node])
            path_nodes.append((current_coord,(current_node,current_node)))
        
        current_speed = random.uniform(*speed_range)
        trajectory.append(current_coord.tolist())
    
    return trajectory, path_nodes


def gen_timestamps(frequency, sigma, num_timestamps, start_time=0.0):
    timestamps = [start_time]
    current_time = start_time
    for _ in range(num_timestamps - 1):
        while True:
            interval = np.random.normal(loc=frequency, scale=sigma)
            if interval >= 0:
                break

        next_time = current_time + interval
        timestamps.append(next_time)
        current_time = next_time

    return timestamps


def gen_cans(G, pos, seg):
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
        if 'geometry' in data:
            line = data['geometry']
            if not isinstance(line, LineString):
                continue
            edge_length = line.length
        else:
            u_pos, v_pos = np.array(pos[u]), np.array(pos[v])
            edge_length = np.linalg.norm(u_pos - v_pos)
            line = LineString([pos[u], pos[v]])

        if u == v:
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
            G.add_edge(prev_node, cand_node, length=seg_len)
            
            prev_node = cand_node
            prev_pos = loc

        segment_line = LineString([prev_pos, pos[v]])
        seg_len = segment_line.length
        G.add_edge(prev_node, v, length=seg_len)

        if G.has_edge(u, v, k):
            G.remove_edge(u, v, k)

    return candidates, np.array(candidate_coords), new_pos

def generate_observations(
    traj,
    tower_positions,
    candidate_coords,
    a2_thresh_db=-12.0,
    a4_thresh_db=-10.0,
    hysteresis_db=1.0,
    ttt_steps=2,
    p_tx_dbm=p_tx_dbm,
    n_pathloss=4,
    shadow_sigma_db=3.0,
    fast_fading_db=0.5,
    noise_floor_dbm=-104.0,
    n_prb=50,
    detect_rsrp_thresh_dbm=-120.0
):
    num_timestamps = len(traj)
    observations = []

    serving = None
    a2_counter = 0
    a4_counters = None

    for t in range(num_timestamps):
        user_pos = np.array(traj[t])

        rsrp_dbm, rssi_dbm, rsrq_db, dists = simulate_signal_strengths(
            user_pos*KILOMETER,
            tower_positions*KILOMETER,
            p_tx_dbm=p_tx_dbm,
            n_pathloss=n_pathloss,
            shadow_sigma_db=shadow_sigma_db,
            fast_fading_db=fast_fading_db,
            noise_floor_dbm=noise_floor_dbm,
            n_prb=n_prb,
        )

        if serving is None:
            serving = int(np.argmax(rsrp_dbm))
            a4_counters = {i: 0 for i in range(len(tower_positions)) if i != serving}

        serving_rsrq = rsrq_db[serving]

        if serving_rsrq <= (a2_thresh_db - hysteresis_db):
            a2_counter += 1
        else:
            a2_counter = 0

        a4_ready = []
        for i in range(len(tower_positions)):
            if i == serving:
                continue
            if rsrq_db[i] >= (a4_thresh_db + hysteresis_db):
                a4_counters[i] = a4_counters.get(i, 0) + 1
            else:
                a4_counters[i] = 0
            if a4_counters[i] >= ttt_steps:
                a4_ready.append(i)

        if a2_counter >= ttt_steps and len(a4_ready) > 0:
            best_nb = max(a4_ready, key=lambda i: rsrq_db[i])
            if best_nb != serving:
                serving = best_nb
                a2_counter = 0
                a4_counters = {i: 0 for i in range(len(tower_positions)) if i != serving}

        detectable = set(np.where(rsrp_dbm >= detect_rsrp_thresh_dbm)[0].tolist())
        detectable.add(serving)
        current_obs = sorted(detectable)
        observations.append((t, current_obs))
        
    r= estimate_coverage_radius(p_tx_dbm,32.4,1,n_pathloss,detect_rsrp_thresh_dbm)
    return observations,r


            
def augment_trajectory_data(obs, path_nodes, trajectory, timestamps, min_length=5, max_length=None, overlap_ratio=0.5, max_time_gap=30):
    if max_length is None:
        max_length = len(obs)
    else:
        max_length = min(max_length, len(obs))
    
    augmented_data = []
    n = len(obs)
    
    step = max(1, int((1 - overlap_ratio) * max_length))
    
    for start in range(0, n - min_length + 1, step):
        for length in range(min_length, min(max_length + 1, n - start + 1)):
            end = start + length
            
            valid = True
            for i in range(start, end - 1):
                if timestamps[i+1] - timestamps[i] > max_time_gap:
                    valid = False
                    break
                    
            if valid:
                augmented_data.append({
                    'obs': obs[start:end],
                    'path_nodes': path_nodes[start:end],
                    'trajectory': trajectory[start:end],
                    'timestamps': timestamps[start:end],
                    "start_idx":start,
                })
    
    return augmented_data


def gen_G(
    net_file="random.net.xml",
    iterations=200,
    min_dist=30,
    max_dist=80,
    connectivity=0.8,
    max_x=1000,
    max_y=1000
):
    cmd = [
        "netgenerate", "--rand",
        f"--rand.iterations={iterations}",
        f"--rand.min-distance={min_dist}",
        f"--rand.max-distance={max_dist}",
        f"--rand.connectivity={connectivity}",
        "-o", net_file
    ]
    subprocess.run(cmd, check=True)
    net = sumolib.net.readNet(net_file)

    raw_ids = [node.getID() for node in net.getNodes()]
    id_map = {rid: i for i, rid in enumerate(raw_ids)}

    G_und = nx.Graph()
    for node in net.getNodes():
        G_und.add_node(id_map[node.getID()])

    for edge in net.getEdges():
        u = id_map[edge.getFromNode().getID()]
        v = id_map[edge.getToNode().getID()]
        G_und.add_edge(u, v)

    pos = {}
    for node in net.getNodes():
        node_id = id_map[node.getID()]
        pos[node_id] = node.getCoord()

    xs, ys = zip(*pos.values())
    min_x_raw, max_x_raw = min(xs), max(xs)
    min_y_raw, max_y_raw = min(ys), max(ys)
    scale_x = max_x / (max_x_raw - min_x_raw) if max_x_raw > min_x_raw else 1
    scale_y = max_y / (max_y_raw - min_y_raw) if max_y_raw > min_y_raw else 1
    pos = {node: ((x - min_x_raw) * scale_x, (y - min_y_raw) * scale_y)
           for node, (x, y) in pos.items()}

    G = nx.MultiDiGraph()
    G.add_nodes_from(G_und.nodes())
    for u, v in G_und.edges():
        G.add_edge(u, v)
        G.add_edge(v, u)

    for u, v, k in G.edges(keys=True):
        G.edges[u, v, k]["length"] = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))

    return G, pos

def simulate_mul(
    num_towers,
    max_len,
    min_len,
    seq_number,
    seg,
    speed,
    seed = None,
    iterations = 100,
    min_prefix_length=10
):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G, pos = gen_G(
        iterations=iterations+seed//100,
        min_dist=300,
        max_dist=500,
        connectivity=0.9,
        max_x=ROADSCALE,
        max_y=ROADSCALE
    )
    
    tower_positions = place_towers_fully_random(num_towers,x_range=(0, ROADSCALE),y_range=(0, ROADSCALE))
    G_ori=G.copy()
    candidates, candidate_coords, new_pos = gen_cans(G, pos, seg)
    
    data_segments = []
    for i in range(seq_number):
        num_timestamps = random.randint(min_len, max_len)
        if i>=0.9*seq_number:
            num_timestamps+=min_prefix_length
        timestamps = gen_timestamps(1, 0.1, num_timestamps) 
        trajectory, path_nodes = gentrajectory(timestamps,G_ori,pos,edge_selection="random",speed_range=[0, speed])

        obs,r = generate_observations(trajectory, tower_positions, candidate_coords)
        if i>=0.9*seq_number:
            data_segments.append({
                    'obs': obs[min_prefix_length:],
                    'path_nodes': path_nodes[min_prefix_length:],
                    'trajectory': trajectory[min_prefix_length:],
                    'timestamps': timestamps[min_prefix_length:]
                })
        else:
            data_segments.append({
                    'obs': obs,
                    'path_nodes': path_nodes,
                    'trajectory': trajectory,
                    'timestamps': timestamps
                })
    r /= KILOMETER
    
    return (
        G,
        new_pos,
        candidates,
        candidate_coords,
        tower_positions,
        data_segments,
        r
    )

