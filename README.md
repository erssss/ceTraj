# Tracing the Unseen: Trajectory Recovery from Cellular Records

Core algorithms, GZ data, and the online appendix for the paper.

## File Structure

- `code/`: source code of the algorithms.
  - `TREC.py`: exact recovery (Algorithm 1).
  - `TRRP.py`: representative-point recovery (Algorithm 2).
  - `TRHA.py`: horizon-anchored recovery (Algorithm 3).
  - `TRSN.py`: localized subnetwork construction (Algorithm 4).
  - `simgen.py`: candidate sampling, coverage, and distance utilities.
  - `load_real.py`: cellular record and road-network loading.
  - `main.py`: single-trajectory recovery entry point.
- `data/`: released GZ data.
- `Appendix.pdf`: online appendix.

## Dataset

- **GZ**: Downsampled and manually denoised real-world cellular records and corresponding road networks in `data/GZ/`, with manually labeled ground truth.
- **Cellular**: Synthetic cellular data generated using SUMO.
- **T-Drive**: [Dataset download](https://www.kaggle.com/datasets/arashnic/tdriver).

## Dependencies

Python 3.10+, NumPy, SciPy, NetworkX, pandas, Shapely, and utm.

## Instruction

Run `python code/main.py --help` for the recovery interface.

- `--obs`: cellular-record CSV.
- `--graph`: road-network path prefix, without `_graph.pkl`.
- `--cells`: base-station CSV; defaults to `data/GZ/input_cells.csv`.
- `--algs`: recovery methods (`TREC`, `TRRP`, `TRHA`).
- `--seg`: candidate granularity in km.
- `--radius`: coverage radius in km.
- `--speed`: maximum speed in km/min.
- `--delta`: TRRP group diameter in km.
- `--alpha`: TRHA coverage weight.
- `--k`: TRHA look-ahead window.
- `--no-trsn`: use the full road network.
- `--output`: output JSON containing recovered positions and coverage scores.

Supply the algorithm parameters explicitly. Timestamps use minutes. TRSN is
applied before recovery unless `--no-trsn` is specified.

The modules can also be called directly. `run_TREC` and `run_TRHA` return total
coverage, selected candidate indices, coordinates, and per-timestamp coverage.
`run_TRRP` also returns grouping and representative mappings; use
`map_compressed_traj_to_original` to map its selected representatives to original
candidate indices. Precomputed coverage and distance tables can be reused.
