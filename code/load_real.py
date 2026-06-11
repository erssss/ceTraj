import csv
import math
from collections import defaultdict
import osmnx as ox
import networkx as nx
import numpy as np
import time
import ast
import matplotlib.pyplot as plt
import utm
from simgen import *
import pickle
import json
import os
from shapely.geometry import LineString, Point


METER2KILO = 1000
TOWERR = 1


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


def get_road_network(bbox, zone_info):
    ox.settings.user_agent = "my_road_network_app"
    max_retries = 3
    retry_delay = 5 
    G_osm = None
    
    for attempt in range(max_retries):
        try:
            G_osm = ox.graph_from_bbox(
                bbox,
                network_type='drive'
            )
            break
        except (ConnectionError, ConnectionResetError) as e:
            print(f"Try {attempt + 1}/{max_retries} Fail: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("Max attempt")
                raise
    
    if not isinstance(G_osm, nx.MultiDiGraph):
        G_osm = ox.utils_graph.get_digraph(G_osm, weight='length')
    
    for node, data in G_osm.nodes(data=True):
        if 'x' not in data or 'y' not in data:
            if 'geometry' in data:
                point = data['geometry'].centroid
                data['x'] = point.x
                data['y'] = point.y
            else:
                data['x'] = 0.0
                data['y'] = 0.0
    
    for node, data in G_osm.nodes(data=True):
        lon = data["x"]
        lat = data["y"]
        # easting, northing, _, _ = utm.from_latlon(lat, lon)
        easting, northing, _, _ = utm.from_latlon(lat, lon,force_zone_number=zone_info[0],force_zone_letter=zone_info[1])
        data["x_utm"] = easting/KILOMETER
        data["y_utm"] = northing/KILOMETER
    
    utm_pos = {node: (data["x_utm"], data["y_utm"]) for node, data in G_osm.nodes(data=True)}
    
    for u, v, key, data in G_osm.edges(keys=True, data=True):
        if 'geometry' in data:
            coords = list(data['geometry'].coords)
            utm_coords = []
            for lon, lat in coords:
                # easting, northing, _, _ = utm.from_latlon(lat, lon)
                easting, northing, _, _ = utm.from_latlon(lat, lon,force_zone_number=zone_info[0],force_zone_letter=zone_info[1])
                utm_coords.append((easting/KILOMETER, northing/KILOMETER))
            data['geometry'] = LineString(utm_coords)
        else:
            data['geometry'] = LineString([utm_pos[u], utm_pos[v]])
    
    for u, v, key, data in G_osm.edges(keys=True, data=True):
        if 'geometry' in data:
            data['length'] = data['geometry'].length
        else:
            dist = np.linalg.norm(np.array(utm_pos[u]) - np.array(utm_pos[v]))
            data['length'] = dist
    
    save_path = "../data/GZ/realroad5"
    save_road_network(G_osm, utm_pos, zone_info, save_path)
    print("save_road_network DONE")
    
    return G_osm, utm_pos


def save_road_network(G, utm_pos, zone_info, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(f"{save_path}_graph.pkl", 'wb') as f:
        pickle.dump({
            'graph': G,
            'utm_positions': utm_pos,
            'zone_info': zone_info
        }, f)
    
    graph_data = {
        'nodes': [],
        'edges': [],
        'zone_info': zone_info
    }
    
    for node, data in G.nodes(data=True):
        graph_data['nodes'].append({
            'id': node,
            'utm_x': utm_pos[node][0],
            'utm_y': utm_pos[node][1],
            'original_data': data
        })
    
    for u, v, data in G.edges(data=True):
        graph_data['edges'].append({
            'from': u,
            'to': v,
            'length': data.get('length', 0),
            'geometry': str(data.get('geometry', ''))
        })
    
    with open(f"{save_path}_graph.json", 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    


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
                # easting, northing, zone_number, zone_letter = latlon_to_utm(latitude, longitude)
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



def load_realdata(seg,obs_path,graph_path,gtedge_path):
    print("load_observation_data")
    obs_data, _, truth_utm, bbox, obs_zone_info = load_observation_data(obs_path)
    print("load_cell_data")
    tower_positions = load_cell_data('../data/GZ/input_cells.csv',obs_zone_info)
    print("tower_positions number:",len(tower_positions))
    print("get_road_network")
    # G, pos = get_road_network(bbox, obs_zone_info)
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
    
    print("map_coords")
    gt_edge = load_edge_gt(gtedge_path,old_to_new)
    path_nodes = get_gt(truth_utm, gt_edge)
    
    truth_x, truth_y = [], []
    for i in range(len(path_nodes)):
        u, v = path_nodes[i][1]
        edge_data = G.get_edge_data(u, v, 0) or G.get_edge_data(v, u, 0)
        geom = G.edges[u, v, k]['geometry']

        xs, ys = geom.xy
        truth_x.append(xs)
        truth_y.append(ys)
    
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
        obs_zone_info,
        (truth_x,truth_y)
    )
    
