
# Tracing the Unseen: Trajectory Recovery from Cellular Records

This repository accompanies the paper accepted at ICDE 2027. It contains the core algorithms, the released GZ data, and the online appendix.

## File Structure

- code: source code of algorithms.
  - code/trsn.py: TRSN localized-subnetwork construction (Algorithm 4). `mode="boundary"` implements the boundary-path-augmented construction covered by the solution-preservation proof (Proposition 7); `mode="buffer"` reproduces the buffered-bbox configuration used by the scalability experiments, where preservation holds because any feasible hop of network length at most s*dt stays inside the box buffered by s*dt_max.
- data: released GZ dataset source files.
- Appendix.pdf: the camera-ready online appendix (proofs, additional sensitivity analyses, stage-timing table).

## Dataset

- GZ: Real-world data stored in data/GZ. The repository contains five observation subsets with matched ground truth (input1-5 / gt_edge1-5: 292 timestamp rows and 944 station entries in the observed-station lists), the combined GYC view of those subsets, the base-station table (1,899 stations), and road-network graphs. Table I reports 31,965 records in the originally collected carrier corpus; the released subsets constitute the complete data used by all experiments in the paper, whose experimental setup evaluates exactly these subsets.
  - Manual labeling of ground truth.
  - data/GZ/obs/ contains the observed cellular records of each input subset (input1-5.csv), the corresponding ground-truth trajectories (gt_edge1-5.csv), and the GYC pair; data/GZ/graph/ contains the road-network graphs; data/GZ/input_cells.csv lists the cellular tower locations.
- Cellular: Synthetic data generated at runtime by code/simgen.py (invoked through main.py with --dataset sim).
  - Pass --seed 42 for a seeded run. simgen.py seeds Python's random module and NumPy; main.py draws a random seed when --seed is omitted. The SUMO netgenerate version and settings must also be controlled to reproduce the generated road network.
  - Generation requires the SUMO netgenerate binary on PATH: simgen.py invokes it via subprocess to build the random road network.
- Beijing road-network simulation: the reported Beijing-scale trajectories and cellular observations were generated on a cached OpenStreetMap driving graph. The evaluated runs do not read raw T-Drive taxi GPS. The graph cache, experiment drivers, and result CSVs are not distributed here.

## Dependencies

```
matplotlib==3.7.2
networkx==3.2.1
numpy==1.24.3
osmnx==2.0.6
pandas==1.4.2
pillow==9.5.0
scikit_learn==1.0.2
scipy==1.8.1
Shapely==2.1.2
sumolib==1.24.0
tqdm==4.67.1
utm==0.8.1

```

matplotlib is imported by load_real.py and TRRP.py. Pillow is listed as a dependency of the plotting environment, but is not directly imported by the released Python sources.

## Instruction

To run the program, use the following command-line arguments:

- `--dataset`: Dataset mode (`sim` or `real`). The released `real` path runs the GZ-4 and GZ-5 subsets.

- `--algs`: Method name. This parameter allows you to select the algorithm to use.

- `--k`: Window size. This parameter controls the size of the window used in TRHA.

- `--delta`: Clustering diameter. This parameter sets the diameter used in TRRP.

- `--seg`: Candidate granularity. This parameter controls the granularity of candidate point generation.

- `--speed`: Speed constraint, supplied as one numeric value per run.

- `--seed`: Random seed.

You can run the program using the following command line example:

Use a Python 3.10 environment with the dependencies listed above installed. Synthetic mode also requires a compatible SUMO `netgenerate` executable on `PATH`.

```bash
cd code
python3 main.py --dataset sim --algs TRHA TRRP TREC --seed 42
```

## Reproducibility

- This repository currently contains the core algorithm code (code/), the released GZ files (data/), and an online appendix snapshot (Appendix.pdf).
- Experiment driver scripts and plotting scripts are not included in this repository.
- Per-segment result CSVs and instrumented timing logs used for the paper's aggregate tables are not included; the calibration notes below describe their aggregation and cannot alone reproduce every reported number from this release.
- Consult the paper and online appendix for the reported aggregate results and their definitions. The result CSVs needed to recompute every aggregate are not part of this release.

## Citation

If you use this artifact, please cite the paper. Publication details can be added after the proceedings are released.

```bibtex
@inproceedings{zhu2027tracing,
  author    = {Jingyu Zhu and Yu Sun and Shaoxu Song and Dewang Ren and Junhui Liu and Xiaojie Yuan},
  title     = {Tracing the Unseen: Trajectory Recovery from Cellular Records},
  booktitle = {IEEE International Conference on Data Engineering (ICDE)},
  year      = {2027},
  note      = {Accepted paper; final proceedings details forthcoming}
}
```

## Calibration

This section documents the exact aggregation calibrations behind the numbers reported in the camera-ready paper.

### Pooled GZ metrics (GZ-4 + GZ-5)

GZ is a single real-world dataset with two subsets, GZ-4 (input4) and GZ-5 (input5). After augmentation and train/test splitting, the test set contains 30 segments from GZ-4 (395 ground-truth points) and 10 segments from GZ-5 (110 ground-truth points), pooled into 40 segments / 505 points. Pooled metrics are point-weighted over segments:

- pooled RMSE = sqrt( sum_i (rmse_i^2 * n_i) / sum_i n_i ), where rmse_i is the per-segment RMSE and n_i is the number of ground-truth points in segment i (the length column of the per-segment result CSVs); equivalently, per-segment squared errors are recombined into a single sum of squared errors before taking the square root.
- pooled route accuracy = sum_i (acc_i * n_i) / sum_i n_i.
- pooled running time = sum_i time_i (per-segment times add; they are not averaged).
- Only valid segments (those with a non-empty per-segment RMSE) enter the aggregation.

Note: the overall_* summary printed by code/main.py is the unweighted mean over per-segment values; the pooled numbers in the paper follow the point-weighted definitions above.

### Serving-only ablation: two calibrations

The serving-only ablation keeps, for each timestamp, only its serving-station record. The resulting degradation is reported under two calibrations:

1. Point-weighted pooled relative degradation: (pooled_rmse_serving - pooled_rmse_full) / pooled_rmse_full, where each pooled RMSE uses the point-weighted definition above. This is the calibration used for the percentages in the main text.
2. Mean per-segment relative degradation: the arithmetic mean, over valid segments, of (rmse_serving,i - rmse_full,i) / rmse_full,i. This calibration weights every segment equally regardless of its length.

The two calibrations can differ substantially for the same runs, so each reported percentage must state which calibration it uses.

### Timing amortization

In the GZ stage-timing experiment, the full-network distance table is computed once per dataset and shared across query segments; this construction is excluded from per-segment query times. Candidate/coverage construction is included in the measured Build stage of the GZ stage-timing table. In that table, the 16.4 s GZ-4 table costs 0.55 s per segment when allocated over 30 segments, and the 20.1 s GZ-5 table costs 2.01 s per segment over 10 segments. These benchmark-specific allocations are outside the reported Build, Recover, and Total columns. The internal result CSVs track preprocessing time and candidate counts, but those files are not distributed here. For TRRP, the instrumented build time assumes no cache reuse across segments.
