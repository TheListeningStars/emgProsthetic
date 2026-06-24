"""score_model.py — load a saved bundle, pick a model, score it on session CSVs.

No training, no servo, no camera.  Just:
  1) load a bundle from models/ (default: most recent)
  2) pick which model in the bundle to evaluate (CLI flag or interactive prompt)
  3) build (window, label) pairs from the given session CSVs at the bundle's
     horizon (or an override)
  4) run that model on every pair using the bundle's saved normalizer stats
  5) print MAE / RMSE / R²  (and optionally plot pred vs true)

Usage:
  python score_model.py SESSION_CSV [SESSION_CSV ...]            # interactive
  python score_model.py SESSION_CSV --model RF
  python score_model.py SESSION_CSV --bundle models/bundle_offline_<TS>.pkl --model ENS
  python score_model.py SESSION_CSV --val-only --plot

Notes:
- The same `live_common` contract that trained the bundle must be importable;
  feature shape and EMG window length must match.
- `--val-only` keeps only the last `--val-frac` of each session (matches
  the temporal split in train_offline.py) — useful for honest held-out
  scoring when you fed in the same sessions you trained on.
"""

import argparse
import csv
import glob
import os
from datetime import datetime

import numpy as np

import live_common as lc
from live_common import (
    features_from_arr, load_bundle, get_pred,
    N_CH, EMG_WINDOW, PRED_HORIZON_S,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it=None, **kw):
        return it if it is not None else iter(())


# ====== CSV LOADING (mirrors train_offline.py) ==============================
def _raw_path_for(main_csv):
    base, ext = os.path.splitext(main_csv)
    p = f"{base}_emg_raw{ext}"
    if not os.path.exists(p):
        raise SystemExit(f"matching raw CSV not found: {p}")
    return p


def _iso_to_unix(s):
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def load_session(main_csv):
    raw_csv = _raw_path_for(main_csv)
    samples = []
    with open(raw_csv) as f:
        for row in csv.DictReader(f):
            try:
                samples.append((float(row["t_mono"]),
                                float(row["m1"]), float(row["m2"]),
                                float(row["m3"]), float(row["grav_env"])))
            except (ValueError, KeyError):
                continue
    samples = np.array(samples, dtype=float)

    angles = []
    with open(main_csv) as f:
        reader = csv.DictReader(f)
        has_t = "t_mono" in (reader.fieldnames or [])
        for row in reader:
            try:
                ang = row.get("angle_deg")
                if ang in (None, ""):
                    continue
                ang = float(ang)
                if not np.isfinite(ang):
                    continue
                t = float(row["t_mono"]) if (has_t and row.get("t_mono")) \
                    else _iso_to_unix(row["timestamp"])
                if t is None:
                    continue
                angles.append((t, ang))
            except (ValueError, KeyError):
                continue
    angles = np.array(angles, dtype=float)
    return samples, angles


def build_pairs(samples, angles, horizon_s, tol_s=0.05, step=1):
    if len(samples) < EMG_WINDOW or len(angles) == 0:
        return (np.zeros((0, 4 * N_CH)),
                np.zeros((0, EMG_WINDOW, N_CH)),
                np.zeros(0))
    angles = angles[angles[:, 0].argsort()]
    ang_ts = angles[:, 0]
    feats, raws, ys = [], [], []
    ends = range(EMG_WINDOW, len(samples) + 1, step)
    for end in tqdm(ends, desc="pairing", unit="win", leave=False):
        win = samples[end - EMG_WINDOW:end]
        t_end = float(win[-1, 0])
        target_t = t_end + horizon_s
        idx = np.searchsorted(ang_ts, target_t)
        cand = []
        if idx > 0:           cand.append(idx - 1)
        if idx < len(ang_ts): cand.append(idx)
        if not cand:
            continue
        best = min(cand, key=lambda j: abs(ang_ts[j] - target_t))
        if abs(ang_ts[best] - target_t) > tol_s:
            continue
        feats.append(features_from_arr(win))
        raws.append(win[:, 1:1 + N_CH].copy())
        ys.append(float(angles[best, 1]))
    return (np.array(feats, dtype=float),
            np.array(raws, dtype=float),
            np.array(ys, dtype=float))


# ====== METRICS =============================================================
def regression_metrics(yt, yp):
    yt = np.asarray(yt, dtype=float)
    yp = np.asarray(yp, dtype=float)
    e = yp - yt
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e * e)))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - float(np.sum(e * e)) / ss_tot if ss_tot > 1e-9 else 0.0
    return mae, rmse, r2


# ====== MODEL PICKER ========================================================
def pick_model_interactively(display):
    print("\nModels in this bundle:")
    for i, m in enumerate(display, 1):
        tag = "trained" if m.trained else "NOT trained"
        print(f"  [{i}] {m.name:<6} ({m.kind:<4}, {tag})")
    while True:
        choice = input("Pick (number or name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(display):
            return display[int(choice) - 1]
        for m in display:
            if m.name.lower() == choice.lower():
                return m
        print("  invalid — try again.")


def _pick_latest_bundle():
    paths = sorted(glob.glob("models/bundle_*.pkl"))
    return paths[-1] if paths else None


# ====== MAIN ================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    p.add_argument("sessions", nargs="+",
                   help="paths to live_train_<TS>.csv files to score on")
    p.add_argument("--bundle", default=None,
                   help="bundle path (default: most recent in models/)")
    p.add_argument("--model", default=None,
                   help="model to score: RLS / RF / LSTM / GRU / ENS. "
                        "Prompts interactively if omitted.")
    p.add_argument("--horizon", type=float, default=None,
                   help="override prediction horizon (default: bundle meta)")
    p.add_argument("--tol-s", type=float, default=0.05)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--val-only", action="store_true",
                   help="score only the last --val-frac of each session")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--plot", action="store_true",
                   help="show pred vs true plot (requires matplotlib)")
    return p.parse_args()


def main():
    args = parse_args()

    bundle_path = args.bundle or _pick_latest_bundle()
    if not bundle_path or not os.path.exists(bundle_path):
        raise SystemExit("No bundle found. Pass --bundle PATH or save one with "
                         "train_offline.py / live_train.py 'S' key first.")
    print(f"Loading bundle: {bundle_path}")
    b = load_bundle(bundle_path)
    norms    = b["norms"]
    models   = b["models"]
    ensemble = b["ensemble"]
    display  = b["display"]
    meta     = b["meta"]
    print(f"  meta: {meta}")

    # Horizon: prefer CLI override, then bundle meta, then live_common default.
    horizon = (args.horizon if args.horizon is not None
               else float(meta.get("pred_horizon_s", PRED_HORIZON_S)))
    print(f"  scoring at horizon = {horizon:.3f}s")

    # Pick the model
    if args.model:
        selected = next((m for m in display if m.name.upper() == args.model.upper()),
                        None)
        if selected is None:
            raise SystemExit(f"Unknown model {args.model!r}. "
                             f"Available: {[m.name for m in display]}")
    else:
        selected = pick_model_interactively(display)

    if selected.kind != "ens" and not selected.trained:
        raise SystemExit(f"Model {selected.name} is not trained in this bundle.")
    print(f"Selected: {selected.name}")

    # Load + pair each session
    all_f, all_r, all_y = [], [], []
    for sess in args.sessions:
        print(f"\nLoading: {sess}")
        samples, angles = load_session(sess)
        print(f"  raw samples: {len(samples)}   angles: {len(angles)}")
        f, r, y = build_pairs(samples, angles, horizon_s=horizon,
                              tol_s=args.tol_s, step=args.step)
        print(f"  pairs: {len(y)}")
        if args.val_only and len(y) >= 2:
            n_val = max(1, int(len(y) * args.val_frac))
            f, r, y = f[-n_val:], r[-n_val:], y[-n_val:]
            print(f"  --val-only: kept last {len(y)} (val_frac={args.val_frac})")
        if len(f) > 0:
            all_f.append(f); all_r.append(r); all_y.append(y)

    if not all_f:
        raise SystemExit("No usable pairs across the given sessions.")

    feats = np.concatenate(all_f)
    raws  = np.concatenate(all_r)
    ys    = np.concatenate(all_y)
    print(f"\nTotal pairs to score: {len(ys)}")

    # Normalize with the bundle's stats (do NOT update them).
    xn = np.array([norms.norm_feat(f) for f in feats], dtype=np.float32)
    rn = np.array([norms.norm_raw(r)  for r in raws],  dtype=np.float32)

    # Predict
    preds = []
    for i in tqdm(range(len(ys)), desc=f"scoring {selected.name}", unit="sample"):
        p = get_pred(selected, xn[i], rn[i], all_models=models)
        preds.append(p if p is not None else np.nan)
    preds = np.array(preds, dtype=float)
    valid = np.isfinite(preds)
    print(f"valid predictions: {valid.sum()} / {len(ys)}")
    if not valid.any():
        raise SystemExit("Model returned no valid predictions.")

    mae, rmse, r2 = regression_metrics(ys[valid], preds[valid])
    print(f"\n=== {selected.name} on {len(args.sessions)} session(s) ===")
    print(f"  MAE   = {mae:.2f}°")
    print(f"  RMSE  = {rmse:.2f}°")
    print(f"  R^2   = {r2:.3f}")
    print(f"  N     = {valid.sum()}")
    print(f"  y range: [{ys.min():.1f}, {ys.max():.1f}]   "
          f"pred range: [{np.nanmin(preds[valid]):.1f}, "
          f"{np.nanmax(preds[valid]):.1f}]")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(ys,    label="true",            color="black",     lw=1.0)
            ax.plot(preds, label=f"pred {selected.name}",
                    color="tab:blue", alpha=0.8, lw=0.9)
            ax.set_xlabel("sample index (time order)")
            ax.set_ylabel("ankle angle (deg)")
            ax.set_title(f"{selected.name} — "
                         f"MAE={mae:.2f}°, RMSE={rmse:.2f}°, R²={r2:.3f}, "
                         f"H={horizon:.3f}s")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
