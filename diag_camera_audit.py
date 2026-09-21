"""E-cam-1: camera filter audit for session 20250812_091600.

For each node, during the bins where the GT object is AT that node, tally raw YOLO
detections by class/conf/depth and compare GT-bin coverage under:
  - current filter: class=='car', conf>=0.5, depth<=35
  - relaxed vehicle filter: class in VEHICLES, conf>=0.25, depth<=40
  - any detection at all
Quantifies how much camera coverage the class/conf filter is discarding.

Usage: track_mdp_env/bin/python diag_camera_audit.py [session]
"""
import sys, os, json
sys.path.insert(0, os.getcwd())
import numpy as np
from pathlib import Path
from collections import Counter

from collection.train_node_classifiers_10 import _parse_session_start_10
from examples.finetune_deterministic import build_iobt_gt_sequence

VEHICLES = {"car", "truck", "bus", "train", "boat", "airplane", "motorcycle"}
BIN = 0.5

session = sys.argv[1] if len(sys.argv) > 1 else "20250812_091600"
sd = Path("iobt_data_10") / session
gt_seq, gps_mask = build_iobt_gt_sequence(str(sd), step_s=BIN)
full_idx = np.where(gps_mask)[0]
start = _parse_session_start_10(session).timestamp()
T_full = len(gps_mask)

print(f"SESSION {session}: {len(gt_seq)} GT steps")
print(f"{'node':>5} {'gt_bins':>7} | {'any_det':>7} {'cur_filt':>8} {'relaxed':>8} | top classes at GT (n, conf p50/p90)")

from datetime import datetime
def parse_t(ts):
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    except Exception:
        return None

tot = {"gt": 0, "any": 0, "cur": 0, "rel": 0}
for k in range(10):
    nid = k + 1
    gt_bins = set(full_idx[gt_seq == k].tolist())
    if not gt_bins:
        continue
    jp = sd / f"node{nid}_zed_yolo.json"
    rows = []
    if jp.exists():
        try:
            frames = json.load(open(jp))
            for fr in frames:
                for d in fr or []:
                    tu = parse_t(d.get("t", ""))
                    if tu is None:
                        continue
                    b = int((tu - start) // BIN)
                    if b in gt_bins:
                        rows.append((d["class"], float(d["conf"]),
                                     float(d.get("depth", 99)), b))
        except Exception as e:
            print(f"{nid:>5}  JSON error: {e}")
            continue
    bins_any = {b for _, _, _, b in rows}
    bins_cur = {b for c, cf, dp, b in rows if c == "car" and cf >= 0.5 and dp <= 35}
    bins_rel = {b for c, cf, dp, b in rows
                if c in VEHICLES and cf >= 0.25 and dp <= 40}
    cls = Counter(c for c, _, _, _ in rows)
    top = []
    for cname, n in cls.most_common(3):
        confs = [cf for c, cf, _, _ in rows if c == cname]
        top.append(f"{cname}({n}, {np.median(confs):.2f}/{np.percentile(confs, 90):.2f})")
    n = len(gt_bins)
    tot["gt"] += n; tot["any"] += len(bins_any)
    tot["cur"] += len(bins_cur); tot["rel"] += len(bins_rel)
    print(f"{nid:>5} {n:>7} | {len(bins_any):>7} {len(bins_cur):>8} {len(bins_rel):>8} | "
          + "  ".join(top))

print(f"\nTOTALS over {tot['gt']} GT bins: any={tot['any']} ({tot['any']/tot['gt']:.1%})  "
      f"current filter={tot['cur']} ({tot['cur']/tot['gt']:.1%})  "
      f"relaxed={tot['rel']} ({tot['rel']/tot['gt']:.1%})")
print("(current-filter number ~ cam lock coverage before smoothing; "
      "'relaxed' shows recoverable headroom)")
