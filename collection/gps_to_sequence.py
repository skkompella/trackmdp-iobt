"""
Convert a GPS CSV (lat/lon, ~1 Hz) to a sequence of cell indices (0–5)
by assigning each timestep to the nearest of the 6 ReSpeaker nodes.
"""
import argparse
import numpy as np
from pathlib import Path

INTENSITY_NODE_IDS = [11, 12, 13, 14, 15, 16]


def gps_to_cell_sequence(gps_csv, nodes_txt, node_ids=None) -> np.ndarray:
    """
    Returns (M,) int32 array of cell indices 0–(N-1), one per GPS sample.
    node_ids defaults to INTENSITY_NODE_IDS [11..16].
    """
    from collection.flac_to_trackmdp import load_gps_track, load_node_positions_ordered

    if node_ids is None:
        node_ids = INTENSITY_NODE_IDS

    node_xy = load_node_positions_ordered(Path(nodes_txt), node_ids)
    gps_df  = load_gps_track(Path(gps_csv))

    sequence = np.zeros(len(gps_df), dtype=np.int32)
    for idx, row in gps_df.iterrows():
        dx = row['x'] - node_xy[:, 0]
        dy = row['y'] - node_xy[:, 1]
        sequence[idx] = int(np.argmin(dx ** 2 + dy ** 2))

    return sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert GPS CSV → nearest-node cell index sequence")
    parser.add_argument("--gps-csv",   required=True)
    parser.add_argument("--nodes-txt", required=True)
    args = parser.parse_args()

    seq = gps_to_cell_sequence(args.gps_csv, args.nodes_txt)
    cells_seen = sorted(set(seq.tolist()))
    print(f"Sequence length : {len(seq)}")
    print(f"Cells seen      : {cells_seen}  "
          f"(nodes {[INTENSITY_NODE_IDS[c] for c in cells_seen]})")
    for c in cells_seen:
        pct = (seq == c).mean() * 100
        print(f"  cell {c} (node {INTENSITY_NODE_IDS[c]}): {pct:.1f}%")
