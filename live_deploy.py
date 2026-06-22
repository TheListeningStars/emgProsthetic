"""live_deploy.py — load a trained model bundle and drive the servo.

Reads raw EMG from the ESP32 (running esp32_online.ino), runs the gravity
filter chain + features, predicts with one model from the bundle, and sends
the resulting angle to the ESP32. No camera, no training, no dashboard.

Usage:
  python live_deploy.py [--bundle PATH] [--model NAME] [--port PORT]
                        [--no-invert] [--print-every 1.0]

  --bundle    path to models/bundle_*.pkl. Default: most recent in models/.
  --model     which model to use. Default: ENS (or first trained one if ENS
              has no static weights). Options: SGD, RLS, RF, LSTM, GRU, ENS.
  --port      serial device. Default: SERIAL_PORT below.
  --no-invert disable the servo-direction flip (default is flip).
  --print-every  seconds between prediction prints (default 0.5; 0 = silent).
"""

import argparse
import csv
import glob
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import serial

import live_common as lc
from live_common import (
    GravityEMGProcessor, features_from_arr, load_bundle, get_pred,
    set_active_norms,
    N_CH, EMG_WINDOW, ESP32_FS_HZ,
)

# ====== DEFAULTS ============================================================
SERIAL_PORT  = "/dev/cu.usbserial-0001"
SERIAL_BAUD  = 921600

ANGLE_MIN_DEG = 90.0
ANGLE_MAX_DEG = 180.0

PRED_EVERY_S = 0.02   # how often to predict + send (~50 Hz)

# ====== CLI =================================================================
def _pick_latest_bundle():
    paths = sorted(glob.glob("models/bundle_*.pkl"))
    return paths[-1] if paths else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default=None)
    p.add_argument("--model", default="ENS")
    p.add_argument("--port", default=SERIAL_PORT)
    p.add_argument("--baud", type=int, default=SERIAL_BAUD)
    p.add_argument("--no-invert", action="store_true",
                   help="disable the default servo direction flip")
    p.add_argument("--print-every", type=float, default=0.5,
                   help="seconds between predicted-angle prints (0 = silent)")
    p.add_argument("--log", default=None,
                   help="optional CSV path to log predictions (timestamp,t_mono,pred)")
    return p.parse_args()


# ====== MAIN ================================================================
def main():
    args = parse_args()
    bundle_path = args.bundle or _pick_latest_bundle()
    if bundle_path is None or not os.path.exists(bundle_path):
        raise SystemExit("No bundle found. Train and save one first (live_train.py "
                         "press S), or pass --bundle explicitly.")
    print(f"Loading bundle: {bundle_path}")
    b = load_bundle(bundle_path)
    norms   = b["norms"]
    models  = b["models"]
    ensemble = b["ensemble"]
    display = b["display"]
    set_active_norms(norms)
    meta = b["meta"]
    print(f"  meta: {meta}")
    trained_names = [m.name for m in models if m.trained]
    print(f"  trained: {trained_names}  + ENS({'static' if ensemble.static_mae else 'flat'})")

    # pick the model
    name = args.model.upper()
    selected = next((m for m in display if m.name == name), None)
    if selected is None:
        raise SystemExit(f"Unknown model '{name}'. Available: {[m.name for m in display]}")
    if selected.kind != "ens" and not selected.trained:
        raise SystemExit(f"Model {name} is not trained in this bundle. "
                         f"Trained: {trained_names}")
    print(f"Using model: {selected.name}")
    invert = not args.no_invert

    # serial
    ser = serial.Serial(args.port, args.baud, timeout=0.05)
    time.sleep(2.0)
    ser.reset_input_buffer()
    print(f"Connected {args.port} @ {args.baud}")

    def send_angle(deg):
        if deg is None or not np.isfinite(deg):
            return
        deg = float(max(ANGLE_MIN_DEG, min(ANGLE_MAX_DEG, float(deg))))
        if invert:
            deg = ANGLE_MIN_DEG + ANGLE_MAX_DEG - deg
        try:
            ser.write(f"A,{deg:.2f}\n".encode("ascii"))
        except Exception as e:
            print(f"send_angle: {e}")

    # filter + window
    grav_proc = GravityEMGProcessor()
    grav_proc.set_fs(ESP32_FS_HZ)
    emg_buf = deque(maxlen=EMG_WINDOW)

    log_csv = None
    log_writer = None
    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        log_csv = open(args.log, "w", newline="")
        log_writer = csv.writer(log_csv)
        log_writer.writerow(["timestamp", "t_mono", f"{selected.name}_pred"])

    last_pred_t = 0.0
    last_print_t = 0.0
    last_pred_val = None
    rdbuf = bytearray()

    try:
        while True:
            chunk = ser.read(4096)
            if not chunk:
                continue
            rdbuf.extend(chunk)
            while b'\n' in rdbuf:
                line, _, rest = rdbuf.partition(b'\n')
                rdbuf = bytearray(rest)
                s = line.decode("ascii", errors="ignore").strip()
                if not s:
                    continue
                if s.startswith("S,"):
                    print(f"[esp32] {s[2:]}")
                    continue
                if not s.startswith("R,"):
                    continue
                parts = s[2:].split(",")
                if len(parts) != 4:
                    continue
                try:
                    grav_raw = int(parts[1])
                    m1 = int(parts[2])
                    m2 = int(parts[3])
                except ValueError:
                    continue
                now = time.time()
                grav_env = grav_proc.push(grav_raw)
                emg_buf.append((now, m1, m2, grav_env, grav_raw))
                if len(emg_buf) < EMG_WINDOW:
                    continue
                if now - last_pred_t < PRED_EVERY_S:
                    continue
                last_pred_t = now

                arr = np.array(emg_buf, dtype=float)
                feat = features_from_arr(arr)
                raw = arr[:, 1:1 + N_CH]
                feat_n = norms.norm_feat(feat)
                raw_n = norms.norm_raw(raw)

                pred = get_pred(selected, feat_n, raw_n, all_models=models)
                if pred is None:
                    continue
                last_pred_val = pred
                send_angle(pred)

                if log_writer is not None:
                    log_writer.writerow([datetime.now().isoformat(), now, pred])

                if args.print_every > 0 and (now - last_print_t) >= args.print_every:
                    last_print_t = now
                    print(f"[{selected.name}] {pred:6.2f}°")
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        try:
            send_angle(135.0)
            time.sleep(0.05)
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
        if log_csv is not None:
            log_csv.close()
        print("done.")


if __name__ == "__main__":
    main()
