"""train_offline.py — train all models from scratch on logged session data.

Takes one or more session main CSVs (logs/liveTrain/live_train_<TS>.csv) and
their matching _emg_raw.csv files, pairs EMG windows with camera angle
labels at PRED_HORIZON_S, splits each session's pairs temporally (last
--val-frac for validation), trains the full model lineup from scratch, and
reports per-model train + val MAE.

Modes:
  Single training (default):  trains at one horizon, prints metrics, and
    saves a bundle to models/.
  Horizon sweep:  pass --horizons "0.05,0.10,0.15,0.20"  to re-pair and
    re-train at each horizon; no bundle saved, just a comparison table.

Usage:
  python train_offline.py SESSION_CSV [SESSION_CSV ...]
      [--out PATH] [--epochs N] [--tol-s 0.05] [--no-seq]
      [--val-frac 0.2] [--horizon 0.10] [--horizons "0.05,0.10,0.15,0.20"]

Each SESSION_CSV is the per-frame log (with angle labels).  Its matching
*_emg_raw.csv is found in the same directory automatically.

The output bundle is loadable by live_deploy.py.
"""

import argparse
import csv
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
    RLSModel, RFModel,
)

if HAS_TORCH:
    import torch


# ====== DEVICE SELECTION ====================================================
def pick_device(name="auto"):
    """Returns a torch.device. Falls back to CPU if requested device is unavailable."""
    if not HAS_TORCH:
        return None
    name = (name or "auto").lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[device] cuda requested but unavailable -> cpu")
        return torch.device("cpu")
    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[device] mps requested but unavailable -> cpu")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ====== CSV LOADING =========================================================
def _raw_path_for(main_csv):
    base, ext = os.path.splitext(main_csv)
    cand = f"{base}_emg_raw{ext}"
    if not os.path.exists(cand):
        raise SystemExit(f"matching raw CSV not found: {cand}")
    return cand


def _parse_iso_to_unix(s):
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def load_session(main_csv):
    """Returns (samples Nx5, angles Mx2) for one session.

    samples columns: [t_mono, m1, m2, m3, grav_env]
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
                    float(row["m3"]),
                    float(row["grav_env"]),
                ))
            except (ValueError, KeyError):
                continue
    samples = np.array(samples, dtype=float)
    print(f"  raw samples: {len(samples)}")

    angles = []
    with open(main_csv) as f:
        reader = csv.DictReader(f)
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


def build_pairs(samples, angles, horizon_s, tol_s=0.05, step=1):
    """Slide an EMG_WINDOW over `samples`, pair each window end with the
    angle closest to (t_end + horizon_s) within tol_s.

    Returns (feats Nx(4*N_CH), raws NxTxN_CH, ys N).
    """
    if len(samples) < EMG_WINDOW or len(angles) == 0:
        return (np.zeros((0, 4 * N_CH)),
                np.zeros((0, EMG_WINDOW, N_CH)),
                np.zeros(0))
    angles = angles[angles[:, 0].argsort()]
    ang_ts = angles[:, 0]

    feats, raws, ys = [], [], []
    for end in range(EMG_WINDOW, len(samples) + 1, step):
        win = samples[end - EMG_WINDOW:end]   # (T, 1+N_CH)
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


# ====== TRAIN / VAL SPLIT ===================================================
def temporal_split(per_session, val_frac):
    """For each session's (feats, raws, ys), take the LAST val_frac as val.

    per_session: list of (feats, raws, ys) tuples.
    Returns (train_feats, train_raws, train_ys, val_feats, val_raws, val_ys).
    """
    tr_f, tr_r, tr_y = [], [], []
    va_f, va_r, va_y = [], [], []
    for f, r, y in per_session:
        n = len(y)
        if n < 2:
            tr_f.append(f); tr_r.append(r); tr_y.append(y)
            continue
        n_val = max(1, int(n * val_frac))
        n_val = min(n_val, n - 1)
        tr_f.append(f[:-n_val]); tr_r.append(r[:-n_val]); tr_y.append(y[:-n_val])
        va_f.append(f[-n_val:]); va_r.append(r[-n_val:]); va_y.append(y[-n_val:])
    cat = lambda L, default_shape: (np.concatenate(L) if L else
                                    np.zeros(default_shape))
    return (
        cat(tr_f, (0, 4 * N_CH)), cat(tr_r, (0, EMG_WINDOW, N_CH)), cat(tr_y, (0,)),
        cat(va_f, (0, 4 * N_CH)), cat(va_r, (0, EMG_WINDOW, N_CH)), cat(va_y, (0,)),
    )


# ====== NORMALIZATION / TRAINING ===========================================
def fit_norms(feats, raws, ys):
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


def train_rls(model, xn, ys):
    print(f"  RLS: single pass over {len(xn)} samples")
    for i in range(len(xn)):
        model.update(xn[i], float(ys[i]))


def train_rf(model, xn, ys):
    n = min(len(xn), lc.RF_REPLAY_MAX)
    print(f"  RF: fit on last {n} samples (cap={lc.RF_REPLAY_MAX}, "
          f"trees={lc.RF_TREES}, depth={lc.RF_MAX_DEPTH})")
    model.fit_now(xn[-n:], ys[-n:])


def train_seq(model, rn, ys, norms, epochs, batch_size=SEQ_BATCH, device=None):
    n = len(rn)
    device = device or torch.device("cpu")
    print(f"  {model.name}: {epochs} epochs over {n} samples on {device}, "
          f"hidden={lc.SEQ_HIDDEN} layers={lc.SEQ_LAYERS} batch={batch_size}")

    # move net to device + recreate optimizer (so Adam state lives on the same device)
    model.net.to(device)
    model.opt = torch.optim.Adam(model.net.parameters(),
                                 lr=lc.SEQ_LR, weight_decay=lc.SEQ_WD)

    mean = norms.t_mean
    std = norms.t_std
    ys_norm = ((ys - mean) / std).astype(np.float32)
    rn_t = torch.from_numpy(rn.astype(np.float32)).to(device)
    ys_t = torch.from_numpy(ys_norm).to(device)

    last_loss = float("nan")
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
        last_loss = float(np.mean(losses))
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"     ep {ep+1}/{epochs}  loss={last_loss:.4f}")

    # move back to CPU so model.predict() works under the existing CPU-only
    # contract used by eval, save_bundle, live_deploy, and live_train.
    if device.type != "cpu":
        model.net.to(torch.device("cpu"))
        model.opt = torch.optim.Adam(model.net.parameters(),
                                     lr=lc.SEQ_LR, weight_decay=lc.SEQ_WD)
    model.trained = True
    model.extra = f"loss={last_loss:.3f}"


# ====== EVAL ===============================================================
def model_mae(model, xn, rn, ys):
    if len(ys) == 0:
        return float("nan")
    preds = []
    for i in range(len(ys)):
        inp = xn[i] if model.kind == "feat" else rn[i]
        p = model.predict(inp)
        preds.append(p if p is not None else np.nan)
    preds = np.array(preds)
    valid = np.isfinite(preds)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(preds[valid] - ys[valid])))


def ensemble_mae(ensemble, members, xn, rn, ys, use_train_weights=True):
    """Score the ensemble. Members must already have roll populated if you
    want it to use rolling weights; here we'd already-set static_mae."""
    if len(ys) == 0:
        return float("nan")
    preds = []
    for i in range(len(ys)):
        per = {}
        for m in members:
            inp = xn[i] if m.kind == "feat" else rn[i]
            per[m.name] = m.predict(inp)
        preds.append(ensemble.blend(per))
    preds = np.array([p if p is not None else np.nan for p in preds])
    valid = np.isfinite(preds)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(preds[valid] - ys[valid])))


# ====== TRAINING PIPELINE PER HORIZON =======================================
def train_at_horizon(loaded_sessions, horizon_s, args):
    """Pair + split + train + score at one horizon.

    loaded_sessions: list of (samples, angles) — pre-loaded so the sweep
    doesn't re-parse CSVs.
    Returns (models, ensemble, norms, metrics) where metrics is a dict
    {model_name: {"train_mae": ..., "val_mae": ...}, "n_train": ..., "n_val": ...}.
    """
    per_sess = []
    for samples, angles in loaded_sessions:
        f, r, y = build_pairs(samples, angles, horizon_s=horizon_s,
                              tol_s=args.tol_s, step=args.step)
        if len(f) > 0:
            per_sess.append((f, r, y))
    if not per_sess:
        raise SystemExit(f"horizon={horizon_s}: no usable pairs.")

    tr_f, tr_r, tr_y, va_f, va_r, va_y = temporal_split(per_sess, args.val_frac)
    print(f"  pairs:  train={len(tr_y)}  val={len(va_y)}  "
          f"(val_frac={args.val_frac})")
    if len(tr_y) < 10:
        raise SystemExit(f"horizon={horizon_s}: too few training pairs.")

    norms = fit_norms(tr_f, tr_r, tr_y)
    tr_xn, tr_rn = normalize_all(tr_f, tr_r, norms)
    va_xn, va_rn = (normalize_all(va_f, va_r, norms)
                    if len(va_y) else (np.zeros((0, tr_xn.shape[1])),
                                       np.zeros((0,) + tr_rn.shape[1:])))

    include_seq = HAS_TORCH and not args.no_seq
    models, ensemble, display = make_models(include_seq=include_seq)

    rls = next(m for m in models if isinstance(m, RLSModel))
    rf  = next(m for m in models if isinstance(m, RFModel))

    print("  training feature models...")
    train_rls(rls, tr_xn, tr_y)
    train_rf(rf, tr_xn, tr_y)
    if include_seq:
        print("  training sequence nets...")
        for m in models:
            if m.kind == "seq":
                train_seq(m, tr_rn, tr_y, norms, epochs=args.epochs,
                          device=args.device_obj)

    # set ensemble static weights based on VAL MAE (more honest than train).
    # Falls back to train MAE if there's no val data.
    metrics = {}
    member_mae_for_blend = {}
    for m in models:
        tr_mae = model_mae(m, tr_xn, tr_rn, tr_y)
        va_mae = model_mae(m, va_xn, va_rn, va_y) if len(va_y) else float("nan")
        metrics[m.name] = {"train_mae": tr_mae, "val_mae": va_mae}
        weight_basis = va_mae if np.isfinite(va_mae) else tr_mae
        member_mae_for_blend[m.name] = weight_basis if np.isfinite(weight_basis) else 1.0
    ensemble.static_mae = member_mae_for_blend
    ens_tr = ensemble_mae(ensemble, models, tr_xn, tr_rn, tr_y)
    ens_va = (ensemble_mae(ensemble, models, va_xn, va_rn, va_y)
              if len(va_y) else float("nan"))
    metrics["ENS"] = {"train_mae": ens_tr, "val_mae": ens_va}
    metrics["_n_train"] = int(len(tr_y))
    metrics["_n_val"]   = int(len(va_y))

    return models, ensemble, norms, metrics


def print_metrics_block(metrics, indent="    "):
    n_tr = metrics["_n_train"]
    n_va = metrics["_n_val"]
    names = [k for k in metrics if not k.startswith("_")]
    width = max(len(n) for n in names)
    print(f"{indent}{'model':<{width}}   train MAE     val MAE")
    for n in names:
        tr = metrics[n]["train_mae"]
        va = metrics[n]["val_mae"]
        print(f"{indent}{n:<{width}}   {tr:7.2f}°     {va:7.2f}°"
              f"{'   (no val)' if not np.isfinite(va) else ''}")
    print(f"{indent}(n_train={n_tr}, n_val={n_va})")


# ====== MAIN ===============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("sessions", nargs="+",
                   help="paths to live_train_<TS>.csv files (one per session)")
    p.add_argument("--out", default=None,
                   help="output bundle path (default models/bundle_offline_<TS>.pkl)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--tol-s", type=float, default=0.05)
    p.add_argument("--no-seq", action="store_true",
                   help="skip LSTM/GRU even if torch is available")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="fraction of each session's tail to hold out for val")
    p.add_argument("--horizon", type=float, default=PRED_HORIZON_S,
                   help="prediction horizon in seconds (default %(default)s)")
    p.add_argument("--horizons", default=None,
                   help='comma-separated horizon sweep, e.g. "0.05,0.10,0.15,0.20" '
                        "— in sweep mode no bundle is saved")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="device for LSTM/GRU training. Default auto "
                        "(cuda > mps > cpu). Model is moved back to CPU "
                        "before save so deploy stays CPU-only.")
    return p.parse_args()


def main():
    args = parse_args()
    args.device_obj = pick_device(args.device) if HAS_TORCH else None
    if HAS_TORCH:
        # If we're on CPU, let torch use all cores (live_common pinned it to 1
        # to keep the camera loop snappy, but for offline training we want speed).
        if args.device_obj.type == "cpu":
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        print(f"[device] using {args.device_obj} "
              f"(torch threads={torch.get_num_threads()})")

    print("Loading sessions...")
    loaded = []
    for sess in args.sessions:
        print(f"Loading: {sess}")
        samples, angles = load_session(sess)
        if len(samples) < EMG_WINDOW or len(angles) == 0:
            print("  (skipping: too little data)")
            continue
        loaded.append((samples, angles))
    if not loaded:
        raise SystemExit("No usable sessions.")

    # ---- sweep mode ----
    if args.horizons:
        horizons = [float(x) for x in args.horizons.split(",") if x.strip()]
        print(f"\n=== Horizon sweep: {horizons} ===")
        summary = {}
        for h in horizons:
            print(f"\n--- horizon = {h:.3f}s ---")
            t0 = time.time()
            _, _, _, metrics = train_at_horizon(loaded, h, args)
            print_metrics_block(metrics)
            print(f"    ({time.time()-t0:.1f}s)")
            summary[h] = metrics

        # final comparison table
        print("\n=== Sweep summary (val MAE in degrees) ===")
        model_names = [k for k in summary[horizons[0]] if not k.startswith("_")]
        head = ["horizon", "pairs(val)"] + model_names
        widths = [max(len(h), 8) for h in head]
        print("  ".join(f"{h:<{w}}" for h, w in zip(head, widths)))
        for h in horizons:
            row = [f"{h:.3f}", str(summary[h]["_n_val"])]
            for n in model_names:
                v = summary[h][n]["val_mae"]
                row.append(f"{v:.2f}" if np.isfinite(v) else "  --")
            print("  ".join(f"{c:<{w}}" for c, w in zip(row, widths)))

        # best horizon per model
        print("\n=== Best horizon per model ===")
        for n in model_names:
            best = min(horizons,
                       key=lambda h: (summary[h][n]["val_mae"]
                                      if np.isfinite(summary[h][n]["val_mae"])
                                      else 1e9))
            print(f"  {n:<6} horizon={best:.3f}  val MAE="
                  f"{summary[best][n]['val_mae']:.2f}°")
        print("\nSweep mode: no bundle saved. Re-run without --horizons to save.")
        return

    # ---- single training mode ----
    print(f"\n=== Training at horizon = {args.horizon:.3f}s ===")
    t0 = time.time()
    models, ensemble, norms, metrics = train_at_horizon(loaded, args.horizon, args)
    print()
    print_metrics_block(metrics, indent="  ")

    out_path = args.out
    if out_path is None:
        os.makedirs("models", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"models/bundle_offline_{ts}.pkl"

    save_bundle(out_path, models, ensemble, norms, extra_meta={
        "trained_offline": True,
        "n_train_pairs":   int(metrics["_n_train"]),
        "n_val_pairs":     int(metrics["_n_val"]),
        "n_sessions":      len(loaded),
        "epochs":          int(args.epochs),
        "step":            int(args.step),
        "pred_horizon_s":  float(args.horizon),
        "val_frac":        float(args.val_frac),
    })
    print(f"\nSaved bundle -> {out_path}  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
