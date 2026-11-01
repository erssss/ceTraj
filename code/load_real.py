"""Read cellular records, road networks, and candidate locations."""
import ast
import csv
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import utm

from simgen import gen_cans

METER2KILO = 1000


def load_cell_data(file_path,zone_info):
    tower_positions = []
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            longitude = float(row['longitude'])
            latitude = float(row['latitude'])
            easting, northing, _, _ = utm.from_latlon(latitude, longitude,force_zone_number=zone_info[0],force_zone_letter=zone_info[1])
            tower_positions.append([easting/METER2KILO, northing/METER2KILO])
    
    return np.array(tower_positions)


def load_road_network(load_path):
    try:
        with open(f"{load_path}_graph.pkl", 'rb') as f:
            data = pickle.load(f)
            return data['graph'], data['utm_positions'], data['zone_info']
    except FileNotFoundError:
        return None, None, None


def load_observation_data(file_path):
    observations = []
    positions = []
    utm_positions = []
    zone_info = None
    
    min_lon = float('inf')
    max_lon = float('-inf')
    min_lat = float('inf')
    max_lat = float('-inf')
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = float(row['timestamp'])
            longitude = float(row['longitude'])
            latitude = float(row['latitude'])
            
            if zone_info is None:
                easting, northing, zone_number, zone_letter = utm.from_latlon(latitude, longitude)
                zone_info = (zone_number, zone_letter)
            else:
                easting, northing, _, _ = utm.from_latlon(latitude, longitude,force_zone_number=zone_info[0],force_zone_letter=zone_info[1])
                
            utm_positions.append((easting/METER2KILO, northing/METER2KILO))
            try:
                eci_ids = ast.literal_eval(row['eci_ids'])
            except (SyntaxError, ValueError) as e:
                print(f"'{row['eci_ids']}' - {e}")
                continue
                
            observations.append((timestamp, [id -1 for id in eci_ids]))
            positions.append((longitude, latitude))
            min_lon = min(min_lon, longitude)
            max_lon = max(max_lon, longitude)
            min_lat = min(min_lat, latitude)
            max_lat = max(max_lat, latitude)
    
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    min_lon = min_lon - lon_range * 0.05
    max_lon = max_lon + lon_range * 0.05
    min_lat = min_lat - lat_range * 0.05
    max_lat = max_lat + lat_range * 0.05
    
    bbox = [min_lon, min_lat, max_lon, max_lat]
    
    return observations, positions, utm_positions, bbox, zone_info


def get_gt(truth_utm, gt_edge):
    path_nodes = []
    for coord,edge in zip(truth_utm, gt_edge):
        path_nodes.append((coord,(edge[0],edge[1])))
    return path_nodes


def load_edge_gt(gtedge_path,old_to_new):
    edges = []
    reader = pd.read_csv(gtedge_path,index_col=None)
    for _,row in reader.iterrows():
        u = int(row[0])
        v = int(row[1])
        key = int(row[2])
        edges.append((u, v, key))

    gt_edge = []
    for u, v, key in edges:
        new_u = old_to_new[u]
        new_v = old_to_new[v]
        gt_edge.append((new_u, new_v, key))
    return gt_edge


def load_realdata(seg,obs_path,graph_path,gtedge_path=None,cell_path=None):
    print("load_observation_data")
    obs_data, _, truth_utm, bbox, obs_zone_info = load_observation_data(obs_path)
    print("load_cell_data")
    if cell_path is None:
        cell_path = Path(__file__).resolve().parents[1] / 'data/GZ/input_cells.csv'
    tower_positions = load_cell_data(cell_path,obs_zone_info)
    print("tower_positions number:",len(tower_positions))
    print("get_road_network")
    G, pos, zone_info = load_road_network(graph_path)
    
    old_nodes = sorted(G.nodes())
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(old_nodes)}
    G = nx.relabel_nodes(G, old_to_new, copy=True)
    
    new_pos = {old_to_new[old_id]: pos[old_id] for old_id in old_nodes}
    
    for u, v, k in G.edges(keys=True):
        if 'geometry' in G.edges[u, v, k]:
            dist = G.edges[u, v, k]['geometry'].length
        elif 'length' in G.edges[u, v, k]:
            dist = G.edges[u, v, k]['length']
        else:
            dist = np.linalg.norm(np.array(new_pos[u]) - np.array(new_pos[v]))
        
        G.edges[u, v, k]["length"] = dist
    
    timestamps = np.array([obs[0] for obs in obs_data])
    timestamps = timestamps - timestamps[0]
    
    if gtedge_path is None:
        path_nodes = None
    else:
        gt_edge = load_edge_gt(gtedge_path,old_to_new)
        path_nodes = get_gt(truth_utm, gt_edge)

    candidates, candidate_coords, new_pos = gen_cans(G, new_pos, seg)
    print("len(G.nodes())",len(G.nodes()))
    

    return (
        G,
        new_pos,
        candidates,
        candidate_coords,
        tower_positions,
        path_nodes,
        timestamps,
        truth_utm,
        obs_data,
        obs_zone_info
    )
