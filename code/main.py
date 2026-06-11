from TREC import *
from simgen import *
from TRRP import *
from TRHA import *
from load_real import *
import argparse
import bisect


def convert_to_latlon(coords, zone_number, zone_letter):
    latlon_coords = []
    for coord in coords:
        lat, lon = utm.to_latlon(
            coord[0] * 1000, coord[1] * 1000, zone_number, zone_letter
        )
        latlon_coords.append([lon, lat])
    return latlon_coords


def sample_by_time_interval(timestamps, obs, path_node, interval=60):

    sampled_indices = []
    sampled_obs = []
    sampled_pn = []
    current_target = timestamps[0]

    while current_target <= timestamps[-1]:
        idx = bisect.bisect_left(timestamps, current_target)

        if idx < len(timestamps):
            sampled_indices.append(idx)
            sampled_obs.append(obs[idx])
            sampled_pn.append(path_node[idx])
        else:
            sampled_indices.append(len(timestamps) - 1)
            sampled_obs.append(obs[-1])
            sampled_pn.append(path_node[-1])

        current_target += interval

    return sampled_indices, sampled_obs, sampled_pn


def split_datasets(long_sequence, test_id=0, min_prefix_length=0, seed=0):
    import random

    random.seed(seed)

    n = len(long_sequence)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    n_test = n - n_train - n_val

    valid_indices = [
        i for i in range(n) if long_sequence[i]["start_idx"] >= min_prefix_length
    ]

    random.shuffle(valid_indices)
    test_indices = valid_indices

    if len(test_indices) < n_test:
        n_test = len(test_indices)

    rest_indices = list(list(set(range(n)) - set(valid_indices)))
    random.shuffle(rest_indices)

    n_train = int(len(rest_indices) / 9 * 7)
    train_indices = rest_indices[:n_train]
    val_indices = rest_indices[n_train:]

    if len(val_indices) < n_val:
        n_val = len(val_indices)

    return train_indices, val_indices, test_indices


def evaluate_on_data(
    G,
    pos,
    candidates,
    candidate_coords,
    tower_positions,
    coverage_radius,
    timestamps,
    obs,
    path_nodes,
    trajectory,
    speed,
    min_length=10,
    max_length=30,
    overlap_ratio=0.3,
    k=5,
    alpha=0.5,
    delta=0.1,
    algorithms=["TRHA", "TRRP", "TREC"],
    augment=True,
    data_segments=None,
    visualize=True,
    output_prefix="result",
    min_prefix_length=5,
    data_idx=0,
    zone_info=None,
):

    if data_segments == None and augment:
        data_segments = augment_trajectory_data(
            obs,
            path_nodes,
            trajectory,
            timestamps,
            min_length=min_length,
            max_length=max_length,
            overlap_ratio=overlap_ratio,
        )
        train_indices, val_indices, test_indices = split_datasets(
            data_segments, min_prefix_length=min_prefix_length
        )
        data_segments = [data_segments[i] for i in test_indices]
        # total = sum([len(seq["obs"]) for seq in data_segments])
    elif data_segments == None:
        data_segments = [
            {
                "obs": obs,
                "path_nodes": path_nodes,
                "trajectory": trajectory,
                "timestamps": timestamps,
            }
        ]

    dist_matrix = compute_dis(G, list(G.nodes()))

    all_results = []
    full_traj_results = {}

    for i, sub_data in enumerate(tqdm(data_segments, desc="Processing")):
        sub_results = {"segment_id": i, "length": len(sub_data["obs"])}

        sub_obs = sub_data["obs"]
        sub_path_nodes = sub_data["path_nodes"]
        sub_traj = sub_data["trajectory"]
        sub_times = sub_data["timestamps"]

        if "TRHA" in algorithms:
            TRHA_start = time.time()
            _, sel_indices_TRHA, sel_coords_TRHA, _ = run_TRHA(
                G,
                pos,
                candidates,
                candidate_coords,
                tower_positions,
                coverage_radius,
                sub_times,
                sub_obs,
                speed,
                k=k,
                alpha=alpha,
                dist_matrix=dist_matrix,
            )
            TRHA_time = time.time() - TRHA_start

            metrics = eval(
                path_nodes=sub_path_nodes,
                predicted_idx=sel_indices_TRHA,
                candidates=candidates,
                candidate_coords=candidate_coords,
                G=G,
            )
            sub_results.update(
                {
                    "TRHA_time": TRHA_time,
                    "TRHA_rmse": metrics["rmse"],
                    "TRHA_precision": metrics["precision"],
                }
            )

            full_traj_results["TRHA"] = {
                "coords": sel_coords_TRHA,
                "indices": sel_indices_TRHA,
            }

        if "TRRP" in algorithms:
            TRRP_start = time.time()
            _, sel_indices, _, _, _, _, _, _, _, _, _, rep_cidx_of_cluster, _ = (
                run_TRRP(
                    G,
                    pos,
                    candidates,
                    candidate_coords,
                    sub_times,
                    speed,
                    delta,
                    tower_positions,
                    coverage_radius,
                    sub_obs,
                    speed,
                    dist_matrix=dist_matrix,
                )
            )
            TRRP_time = time.time() - TRRP_start

            orig_indices, orig_coords = map_compressed_traj_to_original(
                sel_indices, rep_cidx_of_cluster, candidate_coords
            )

            metrics = eval(
                path_nodes=sub_path_nodes,
                predicted_idx=orig_indices,
                candidates=candidates,
                candidate_coords=candidate_coords,
                G=G,
            )
            sub_results.update(
                {
                    "TRRP_time": TRRP_time,
                    "TRRP_rmse": metrics["rmse"],
                    "TRRP_precision": metrics["precision"],
                }
            )

            full_traj_results["TRRP"] = {"coords": orig_coords, "indices": orig_indices}

        if "TREC" in algorithms:
            TREC_start = time.time()
            _, sel_indices_exact, sel_coords_exact, _ = run_TREC(
                G,
                pos,
                list(G.nodes),
                candidate_coords,
                tower_positions,
                coverage_radius,
                sub_times,
                sub_obs,
                speed,
                filter_candidates_by_observation=False,
                dist_matrix=dist_matrix,
            )
            TREC_time = time.time() - TREC_start

            metrics = eval(
                path_nodes=sub_path_nodes,
                predicted_idx=sel_indices_exact,
                candidates=candidates,
                candidate_coords=candidate_coords,
                G=G,
            )
            sub_results.update(
                {
                    "TREC_time": TREC_time,
                    "TREC_rmse": metrics["rmse"],
                    "TREC_precision": metrics["precision"],
                }
            )

            full_traj_results["TREC"] = {
                "coords": sel_coords_exact,
                "indices": sel_indices_exact,
            }
        all_results.append(sub_results)

    detailed_df = pd.DataFrame(all_results)
    summary_data = []
    for algo in algorithms:
        rmse_col = f"{algo}_rmse"
        precision_col = f"{algo}_precision"
        time_col = f"{algo}_time"
        valid_results = detailed_df[rmse_col].notna()
        summary_data.append(
            {
                "algorithm": algo,
                "overall_rmse": detailed_df.loc[valid_results, rmse_col].mean(),
                "overall_precision": detailed_df.loc[
                    valid_results, precision_col
                ].mean(),
                "total_time": detailed_df.loc[valid_results, time_col].sum(),
            }
        )

    summary_df = pd.DataFrame(summary_data)

    return summary_df, detailed_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_towers", type=int, default=150)
    parser.add_argument("--num_nodes", type=int, default=15)
    parser.add_argument("--num_timestamps", type=int, default=1000)
    parser.add_argument("--coverage_radius", type=float, default=1)
    parser.add_argument("--seg", type=float, default=0.4)
    parser.add_argument("--speed", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--overlap_ratio", type=float, default=0.9)
    parser.add_argument("--data_idx", type=int, default=0)
    parser.add_argument("--min_prefix_length", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="sim")
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--seq_number", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument('--algs', nargs='+', default=['TRHA', 'TRRP', 'TREC'],
                        choices=['TRHA', 'TRRP', 'TREC'])

    args = parser.parse_args()
    if args.delta is None:
        delta = args.seg * 1.5
    else:
        delta = args.delta
    coverage_radius = args.coverage_radius
    if args.seed is None:
        args.seed = random.randint(1, 10000)

    test_id = 0
    datasets = [
        # {'obs': '../data/GZ/obs/input1.csv', 'graph': '../data/GZ/graph/realroad1', 'edgegt': '../data/GZ/obs/gt_edge1.csv'},
        # {'obs': '../data/GZ/obs/input2.csv', 'graph': '../data/GZ/graph/realroad2', 'edgegt': '../data/GZ/obs/gt_edge2.csv'},
        # {'obs': '../data/GZ/obs/input3.csv', 'graph': '../data/GZ/graph/realroad3', 'edgegt': '../data/GZ/obs/gt_edge3.csv'},
        {
            "obs": "../data/GZ/obs/input4.csv",
            "graph": "../data/GZ/graph/realroad4",
            "edgegt": "../data/GZ/obs/gt_edge4.csv",
        },
        {
            "obs": "../data/GZ/obs/input5.csv",
            "graph": "../data/GZ/graph/realroad5",
            "edgegt": "../data/GZ/obs/gt_edge5.csv",
        },
    ]

    all_summary_dfs = []
    all_detailed_dfs = []
    augment = False
    for i, dataset in enumerate(datasets):
        zone_info = None
        data_segments = None
        timestamps = None
        obs = None
        path_nodes = None
        trajectory = None
        if args.dataset == "real":
            dataset = datasets[i]

            obs_path = dataset["obs"]
            graph_path = dataset["graph"]
            gtedge_path = dataset["edgegt"]
            (
                G,
                pos,
                candidates,
                candidate_coords,
                tower_positions,
                path_nodes,
                timestamps,
                trajectory,
                obs,
                zone_info,
                _,
            ) = load_realdata(
                args.seg, obs_path, graph_path, gtedge_path
            )

            # sampled_indices, sampled_obs, sampled_pn = sample_by_time_interval(timestamps, obs,path_nodes)
            # timestamps = timestamps[sampled_indices]/60
            # print("len(timestamps)",len(timestamps))
            # obs = sampled_obs
            # path_nodes = sampled_pn
            # trajectory = np.array(trajectory)
            # trajectory = trajectory[sampled_indices]
            augment = True

        elif args.dataset == "sim":
            (
                G,
                pos,
                candidates,
                candidate_coords,
                tower_positions,
                data_segments,
                coverage_radius,
            ) = simulate_mul(
                num_towers=args.num_towers,
                max_len=args.max_len,
                min_len=args.min_len,
                seq_number=args.seq_number,
                seg=args.seg,
                speed=args.speed,
                seed=args.seed,
                iterations=args.iter,
                min_prefix_length=args.min_prefix_length,
            )

            augment = False

        summary_df, detailed_df = evaluate_on_data(
            G,
            pos,
            candidates,
            candidate_coords,
            tower_positions,
            coverage_radius,
            timestamps,
            obs,
            path_nodes,
            trajectory,
            args.speed,
            min_length=args.min_length,
            max_length=args.max_length,
            overlap_ratio=args.overlap_ratio,
            k=args.k,
            alpha=args.alpha,
            delta=delta,
            data_segments=data_segments,
            algorithms=args.algs,
            augment=augment,
            min_prefix_length=args.min_prefix_length,
            data_idx=i,
            zone_info=zone_info,
        )

        summary_df["dataset"] = f"dataset_{i+1}"
        detailed_df["dataset"] = f"dataset_{i+1}"

        all_summary_dfs.append(summary_df)
        all_detailed_dfs.append(detailed_df)

        if args.dataset != "real":
            break

    combined_summary = pd.concat(all_summary_dfs, ignore_index=True)
    combined_detailed = pd.concat(all_detailed_dfs, ignore_index=True)

    overall_avg_list = []
    algorithms = [
        col.replace("_rmse", "")
        for col in combined_detailed.columns
        if col.endswith("_rmse")
    ]
    algorithms = list(dict.fromkeys(algorithms))

    for algo in algorithms:
        rmse_col = f"{algo}_rmse"
        precision_col = f"{algo}_precision"
        time_col = f"{algo}_time"

        if rmse_col in combined_detailed.columns:
            valid_rows = combined_detailed[rmse_col].notna()
            overall_avg_list.append(
                {
                    "algorithm": algo,
                    "overall_rmse": combined_detailed.loc[valid_rows, rmse_col].mean(),
                    "overall_precision": combined_detailed.loc[
                        valid_rows, precision_col
                    ].mean(),
                    "total_time": combined_detailed.loc[valid_rows, time_col].sum(),
                }
            )

    overall_avg = pd.DataFrame(overall_avg_list)

    print(overall_avg)
