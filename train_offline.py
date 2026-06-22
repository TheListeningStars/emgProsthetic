"""train_offline.py — train all models from scratch on logged session data.

Takes one or more session main CSVs (logs/liveTrain/live_train_<TS>.csv) and
their matching _emg_raw.csv files (auto-resolved by filename), pairs EMG
windows with camera angle labels at PRED_HORIZON_S, trains the full model
lineup from scratch, and saves a bundle to models/.

Usage:
  python train_offline.py SESSION_CSV [SESSION_CSV ...]
                          [--out PATH] [--epochs N]
                          [--tol-s 0.05] [--no-seq]

Each SESSION_CSV is the per-frame log (with angle labels). The matching
*_emg_raw.csv is read from the same directory automatically.

The output bundle is loadable by live_deploy.py.
"""

import argparse
import csv
import glob
import os
import time
from datetime import datetime

import numpy as np

import live_common as lc
from live_common import (
    Normalizers, features_from_arr, make_models, save_bundle,
    set_active_norms, HAS_TORCH,
    N_CH, EMG_WINDOW, PRED_HORIZON_S,
    SEQ_BATCH,
    SGDModel, RLSModel, RFModel,
)

if HAS_TORCH:
    import torch


# ====== CSV LOADING =========================================================
def _raw_path_for(main_csv):
    base, ext = os.path.splitext(main_csv)
    cand = f"{base}_emg_raw{ext}"
    if not os.path.exists(cand):
        raise SystemExit(f"matching raw CSV not found: {cand}")
    return cand


def _parse_iso_to_unix(s):
    """The main CSV stores datetime.now().isoformat(). Convert back to Unix."""
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def load_session(main_csv):
    """Returns (samples Nx4, angles Mx2) for one session.

    samples columns: [t_mono, m1, m2, grav_env]
    angles  columns: [t_mono, angle_deg]
    """
    raw_csv = _raw_path_for(main_csv)
    print(f"  raw: {raw_csv}")

    samples = []
    with open(raw_csv) as f:
        for row in csv.DictReader(f):
            try:
                samples.append((
                    float(row["t_mono"]),
                    float(row["m1"]),
                    float(row["m2"]),
                    float(row["grav_env"]),
                ))
            except (ValueError, KeyError):
                continue
    samples = np.array(samples, dtype=float)
    print(f"  raw samples: {len(samples)}")

    angles = []
    with open(main_csv) as f:
        reader = csv.DictReader(f)
        # main CSV may or may not have a t_mono column depending on version
        has_t_mono = "t_mono" in (reader.fieldnames or [])
        for row in reader:
            try:
                ang = row.get("angle_deg")
                if ang in (None, ""):
                    continue
                ang = float(ang)
                if not np.isfinite(ang):
                    continue
                if has_t_mono and row.get("t_mono"):
                    t = float(row["t_mono"])
                else:
                    t = _parse_iso_to_unix(row["timestamp"])
                    if t is None:
                        continue
                angles.append((t, ang))
            except (ValueError, KeyError):
                continue
    angles = np.array(angles, dtype=float)
    print(f"  angles: {len(angles)}")
    return samples, angles


def build_pairs(samples, angles, horizon_s=PRED_HORIZON_S, tol_s=0.05,
                step=1):
    """Slide an EMG_WINDOW over `samples`, pair each window end with the
    angle closest to (t_end + horizon_s) within tol_s.

    Returns (feats Nx12, raws NxTxN_CH, ys N).  step=1 means snapshot at
    every sample (heavy overlap = data augmentation for sequence nets).
    """
    if len(samples) < EMG_WINDOW or len(angles) == 0:
        return (np.zeros((0, 12)), np.zeros((0, EMG_WINDOW, N_CH)), np.zeros(0))
    angles = angles[angles[:, 0].argsort()]
    ang_ts = angles[:, 0]

    feats = []
    raws = []
    ys = []
    for end in range(EMG_WINDOW, len(samples) + 1, step):
        win = samples[end - EMG_WINDOW:end]   # (T, 4)
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


# ====== TRAINING ===========================================================
def fit_norms(feats, raws, ys):
    """Sweep the data once to seed the EMA normalizers (so the first model
    predictions during eval aren't on un-normalized inputs)."""
    norms = Normalizers()
    for i in range(len(feats)):
        norms.update_feat(feats[i])
        norms.update_raw(raws[i])
        norms.update_tgt(ys[i])
    set_active_norms(norms)
    return norms


def normalize_all(feats, raws, norms):
    xn = np.array([norms.norm_feat(f) for f in feats], dtype=np.float32)
    rn = np.array([norms.norm_raw(r) for r in raws], dtype=np.float32)
    return xn, rn


def train_sgd(model, xn, ys, epochs):
    print(f"  SGD: {epochs} epochs over {len(xn)} samples")
    for _ in range(epochs):
        idx = np.random.permutation(len(xn))
        model.reg.partial_fit(xn[idx], ys[idx])
    model.trained = True


def train_rls(model, xn, ys):
    print(f"  RLS: single pass over {len(xn)} samples")
    for i in range(len(xn)):
        model.update(xn[i], float(ys[i]))


def train_rf(model, xn, ys):
    n = min(len(xn), lc.RF_REPLAY_MAX)
    print(f"  RF: fit on last {n} samples (cap={lc.RF_REPLAY_MAX})")
    model.fit_now(xn[-n:], ys[-n:])


def train_seq(model, rn, ys, norms, epochs, batch_size=SEQ_BATCH):
    n = len(rn)
    print(f"  {model.name}: {epochs} epochs over {n} samples, batch={batch_size}")
    mean = norms.t_mean
    std = norms.t_std
    ys_norm = ((ys - mean) / std).astype(np.float32)
    rn = rn.astype(np.float32)
    ys_t = torch.from_numpy(ys_norm)
    rn_t = torch.from_numpy(rn)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        losses = []
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            xb = rn_t[b]
            yb = ys_t[b]
            model.opt.zero_grad()
            pred = model.net(xb)
            loss = model.loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), 5.0)
            model.opt.step()
            losses.append(loss.item())
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"     ep {ep+1}/{epochs}  loss={np.mean(losses):.4f}")
    model.trained = True
    model.extra = f"loss={np.mean(losses):.3f}"


def per_model_mae(model, xn, rn, ys, norms):
    preds = []
    for i in range(len(ys)):
        inp = xn[i] if model.kind == "feat" else rn[i]
        p = model.predict(inp)
        preds.append(p if p is not None else np.nan)
    preds = np.array(preds)
    valid = np.isfinite(preds)
    if not valid.any():
        return float("nan")
    mae = float(np.mean(np.abs(preds[valid] - ys[valid])))
    # populate roll so EnsembleModel.serialize picks it up
    for p, y in zip(preds[valid][-lc.ROLL_WINDOW:], ys[valid][-lc.ROLL_WINDOW:]):
        model.roll.update(float(y), float(p))
    return mae


# ====== MAIN ================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("sessions", nargs="+",
                   help="paths to live_train_<TS>.csv files (one per session)")
    p.add_argument("--out", default=None,
                   help="output bundle path (default models/bundle_offline_<TS>.pkl)")
    p.add_argument("--epochs", type=int, default=20,
                   help="epochs for SGD / LSTM / GRU")
    p.add_argument("--tol-s", type=float, default=0.05,
                   help="max |t_window+H - t_angle| to accept a pair")
    p.add_argument("--no-seq", action="store_true",
                   help="skip LSTM/GRU even if torch is available")
    p.add_argument("--step", type=int, default=1,
                   help="window stride in samples (default 1 = every sample)")
    return p.parse_args()


def main():
    args = parse_args()
    all_feats = []
    all_raws = []
    all_ys = []
    for sess in args.sessions:
        print(f"Loading session: {sess}")
        samples, angles = load_session(sess)
        if len(samples) < EMG_WINDOW or len(angles) == 0:
            print("  (skipping: too little data)")
            continue
        f, r, y = build_pairs(samples, angles, horizon_s=PRED_HORIZON_S,
                              tol_s=args.tol_s, step=args.step)
        print(f"  pairs: {len(f)}")
        if len(f) == 0:
            continue
        all_feats.append(f); all_raws.append(r); all_ys.append(y)

    if not all_feats:
        raise SystemExit("No usable pairs across all sessions.")

    feats = np.concatenate(all_feats, axis=0)
    raws  = np.concatenate(all_raws,  axis=0)
    ys    = np.concatenate(all_ys,    axis=0)
    print(f"\nTotal pairs: {len(ys)}  (feats {feats.shape}, raws {raws.shape})")
    print(f"  y range: [{ys.min():.1f}, {ys.max():.1f}]  mean {ys.mean():.1f}")

    norms = fit_norms(feats, raws, ys)
    xn, rn = normalize_all(feats, raws, norms)

    include_seq = HAS_TORCH and not args.no_seq
    models, ensemble, display = make_models(include_seq=include_seq)

    sgd = next(m for m in models if isinstance(m, SGDModel))
    rls = next(m for m in models if isinstance(m, RLSModel))
    rf  = next(m for m in models if isinstance(m, RFModel))

    t0 = time.time()
    print("\nTraining feature models...")
    train_sgd(sgd, xn, ys, epochs=args.epochs)
    train_rls(rls, xn, ys)
    train_rf(rf, xn, ys)

    if include_seq:
        print("\nTraining sequence nets...")
        for m in models:
            if m.kind == "seq":
                train_seq(m, rn, ys, norms, epochs=args.epochs)

    # populate per-model training MAE so the ensemble has sane weights
    print("\nScoring on training set (for ensemble weights):")
    for m in models:
        mae = per_model_mae(m, xn, rn, ys, norms)
        print(f"  {m.name}: train MAE = {mae:.2f}°")

    out_path = args.out
    if out_path is None:
        os.makedirs("models", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"models/bundle_offline_{ts}.pkl"

    save_bundle(out_path, models, ensemble, norms, extra_meta={
        "trained_offline": True,
        "n_pairs": int(len(ys)),
        "n_sessions": len(args.sessions),
        "epochs": int(args.epochs),
        "step": int(args.step),
    })
    print(f"\nSaved bundle -> {out_path}  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
