"""live_train.py — closed-loop online training, all models live on the laptop.

The ESP32 (running esp32_online.ino) is a dumb sensor/actuator node: it
streams raw 3-channel ADC samples and waits for servo angle commands.
Everything else — gravity-EMG filter chain, features, training, prediction —
runs here.

Models, features, filter, and save/load all live in live_common.py.

Controls (camera window):
  Click shin -> ankle -> foot on the Mask window to initialize.
  Trackbar "Servo src"  pick which model drives the servo.
  Keys 1..N             same.
  Key S                 save all currently-trained models to models/*.pkl
  Key R                 reset clicks (only before init)
  Key Q                 quit.
"""

import os
import csv
import math
import time
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import serial

import live_common as lc
from live_common import (
    GravityEMGProcessor, Normalizers, features_from_arr, get_pred,
    make_models, save_bundle, set_active_norms,
    HAS_TORCH, COL_TRUE, COL_LABEL,
    N_CH, EMG_WINDOW, ESP32_FS_HZ, PRED_HORIZON_S, ROLL_WINDOW,
    REPLAY_MAX,
)

# ====== SETTINGS  ===========================================================
SERIAL_PORT = "/dev/cu.usbserial-0001"
SERIAL_BAUD = 921600
CAMERA_INDEX = 1

HSV_DEFAULT_LO = (35, 30, 25)
HSV_DEFAULT_HI = (90, 255, 255)
MIN_BLOB_AREA = 100

POSITION_WEIGHT = 1.0
LENGTH_WEIGHT = 2.0

WARMUP_SECONDS = 8.0
PRED_SMOOTH = 5

# Trainer cadence.
TRAIN_TICK_S = 0.004
DISP_PRED_EVERY_S = 0.02
MATCH_TOL_S = 0.05
WINDOW_HIST_MAX = 600

# Servo
ANGLE_MIN_DEG = 90.0
ANGLE_MAX_DEG = 180.0
SERVO_INVERT = True   # flip if your servo horn is mounted reversed

# Anti-jitter for the servo command (live-tunable trackbars below override these).
SERVO_SMOOTH_DEFAULT = 5    # EMA window N: alpha = 1/N. 1 = off, larger = smoother.
SERVO_SLEW_DEFAULT   = 0    # max deg change per command. 0 = disabled.

# ====== LOGGING / MODELS DIR ================================================
os.makedirs("logs/liveTrain", exist_ok=True)
os.makedirs("models", exist_ok=True)
session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"logs/liveTrain/live_train_{session_time}.csv"
raw_log_path = f"logs/liveTrain/live_train_{session_time}_emg_raw.csv"

# ====== SHARED STATE ========================================================
emg_buf = deque(maxlen=EMG_WINDOW)   # (t, m1, m2, m3, grav_env, grav_raw)
stop_flag = threading.Event()

grav_proc = GravityEMGProcessor()
grav_proc.set_fs(ESP32_FS_HZ)

norms = Normalizers()
set_active_norms(norms)
MODELS, ensemble, DISPLAY = make_models()
replay = deque(maxlen=REPLAY_MAX)    # (raw_n [T,N_CH], target)

# ====== SERIAL ==============================================================
ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.05)
time.sleep(2.0)
ser.reset_input_buffer()
print(f"Connected {SERIAL_PORT} @ {SERIAL_BAUD}")

ser_lock = threading.Lock()


def send_angle(deg):
    if deg is None or not np.isfinite(deg):
        return
    deg = float(max(ANGLE_MIN_DEG, min(ANGLE_MAX_DEG, float(deg))))
    if SERVO_INVERT:
        deg = ANGLE_MIN_DEG + ANGLE_MAX_DEG - deg
    msg = f"A,{deg:.2f}\n".encode("ascii")
    with ser_lock:
        try:
            ser.write(msg)
        except Exception as e:
            print(f"send_angle: {e}")


_fs_times = deque(maxlen=600)

raw_csv_file = open(raw_log_path, "w", newline="")
raw_writer = csv.writer(raw_csv_file)
raw_writer.writerow(["timestamp", "t_mono", "m1", "m2", "m3", "grav_raw", "grav_env"])
_raw_count = [0]


def emg_reader():
    rdbuf = bytearray()
    while not stop_flag.is_set():
        try:
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
                if len(parts) != 5:
                    continue
                try:
                    grav_raw = int(parts[1])
                    m1 = int(parts[2])
                    m2 = int(parts[3])
                    m3 = int(parts[4])
                except ValueError:
                    continue
                now = time.time()
                _fs_times.append(now)
                grav_env = grav_proc.push(grav_raw)
                emg_buf.append((now, m1, m2, m3, grav_env, grav_raw))
                raw_writer.writerow(
                    [datetime.now().isoformat(), now, m1, m2, m3, grav_raw, grav_env])
                _raw_count[0] += 1
                if _raw_count[0] % 250 == 0:
                    raw_csv_file.flush()
        except Exception:
            continue


# ====== CAMERA / ANGLE ======================================================
def distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def calculate_angle(shin, ankle, foot):
    v1 = np.array(shin) - np.array(ankle)
    v2 = np.array(foot) - np.array(ankle)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-6:
        return np.nan
    c = np.dot(v1, v2) / denom
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


MASK_WIN = "Mask (HSV tuner)"


def _setup_hsv_trackbars():
    cv2.namedWindow(MASK_WIN)
    n = lambda _x: None
    cv2.createTrackbar("H min", MASK_WIN, HSV_DEFAULT_LO[0], 179, n)
    cv2.createTrackbar("H max", MASK_WIN, HSV_DEFAULT_HI[0], 179, n)
    cv2.createTrackbar("S min", MASK_WIN, HSV_DEFAULT_LO[1], 255, n)
    cv2.createTrackbar("S max", MASK_WIN, HSV_DEFAULT_HI[1], 255, n)
    cv2.createTrackbar("V min", MASK_WIN, HSV_DEFAULT_LO[2], 255, n)
    cv2.createTrackbar("V max", MASK_WIN, HSV_DEFAULT_HI[2], 255, n)
    cv2.createTrackbar("Min area", MASK_WIN, MIN_BLOB_AREA, 2000, n)


def _get_hsv_bounds():
    lo = np.array([cv2.getTrackbarPos("H min", MASK_WIN),
                   cv2.getTrackbarPos("S min", MASK_WIN),
                   cv2.getTrackbarPos("V min", MASK_WIN)], dtype=np.uint8)
    hi = np.array([cv2.getTrackbarPos("H max", MASK_WIN),
                   cv2.getTrackbarPos("S max", MASK_WIN),
                   cv2.getTrackbarPos("V max", MASK_WIN)], dtype=np.uint8)
    min_area = max(1, cv2.getTrackbarPos("Min area", MASK_WIN))
    return lo, hi, min_area


def detect_markers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo, hi, min_area = _get_hsv_bounds()
    mask = cv2.inRange(hsv, lo, hi)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        blobs.append(((cx, cy), area))
    blobs.sort(key=lambda x: x[1], reverse=True)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for (cx, cy), _ in blobs[:6]:
        cv2.circle(mask_bgr, (cx, cy), 8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        mask_bgr,
        f"H {lo[0]}-{hi[0]}  S {lo[1]}-{hi[1]}  V {lo[2]}-{hi[2]}  "
        f"area>={min_area}  blobs={len(blobs)}",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0, 255, 255), 1, cv2.LINE_AA,
    )
    return [b[0] for b in blobs[:3]], mask_bgr


def choose_ankle_foot(shin, a, b, prev_ankle, prev_foot, exp_shin, exp_foot):
    def score(ankle, foot):
        return (POSITION_WEIGHT * distance(ankle, prev_ankle)
                + POSITION_WEIGHT * distance(foot, prev_foot)
                + LENGTH_WEIGHT * abs(distance(shin, ankle) - exp_shin)
                + LENGTH_WEIGHT * abs(distance(ankle, foot) - exp_foot))
    return (a, b) if score(a, b) <= score(b, a) else (b, a)


# ====== WINDOW SNAPSHOT =====================================================
def snapshot_emg():
    if len(emg_buf) < EMG_WINDOW:
        return None
    try:
        arr = np.array(emg_buf, dtype=float)
    except (RuntimeError, ValueError):
        return None
    if arr.shape[0] < EMG_WINDOW:
        return None
    t_end = float(arr[-1, 0])
    feat = features_from_arr(arr)
    raw = arr[:, 1:1 + N_CH].copy()
    return t_end, feat, raw


# ====== SERVO SELECTION + ANTI-JITTER STATE =================================
servo_select = {"idx": len(DISPLAY) - 1, "name": DISPLAY[-1].name}
servo_state  = {"smooth_window": SERVO_SMOOTH_DEFAULT,
                "slew_max":      SERVO_SLEW_DEFAULT,
                "smoothed":      None,
                "last_sent":     None,
                "sel_tracked":   None}


def servo_command(raw_pred, sel_name):
    """Apply EMA + optional slew limit to the raw prediction before sending."""
    if raw_pred is None or not np.isfinite(raw_pred):
        return
    # if model changed, drop the smoother state so we don't carry old values.
    if servo_state["sel_tracked"] != sel_name:
        servo_state["smoothed"] = None
        servo_state["last_sent"] = None
        servo_state["sel_tracked"] = sel_name

    win = max(1, int(servo_state["smooth_window"]))
    alpha = 1.0 / win
    prev = servo_state["smoothed"]
    sm = float(raw_pred) if prev is None else (alpha * raw_pred + (1 - alpha) * prev)
    servo_state["smoothed"] = sm

    slew_max = float(servo_state["slew_max"])
    if slew_max > 0 and servo_state["last_sent"] is not None:
        delta = sm - servo_state["last_sent"]
        if delta > slew_max:
            sm = servo_state["last_sent"] + slew_max
        elif delta < -slew_max:
            sm = servo_state["last_sent"] - slew_max
    servo_state["last_sent"] = sm
    send_angle(sm)

# ====== TRAINER THREAD ======================================================
window_hist = deque(maxlen=WINDOW_HIST_MAX)
angle_samples = deque(maxlen=600)
angle_lock = threading.Lock()

angle_min_obs = None
angle_max_obs = None


def trainer():
    last_label_t = 0.0
    last_disp_t = 0.0
    last_servo_t = 0.0
    while not stop_flag.is_set():
        snap = snapshot_emg()
        if snap is None:
            time.sleep(TRAIN_TICK_S)
            continue
        t_end, feat, raw = snap
        window_hist.append((t_end, feat, raw))

        with angle_lock:
            new_labels = [s for s in angle_samples if s[0] > last_label_t]

        for (t_a, ang) in new_labels:
            last_label_t = max(last_label_t, t_a)
            target_time = t_a - PRED_HORIZON_S
            best = None
            best_d = 1e9
            for (tw, fw, rw) in window_hist:
                d = abs(tw - target_time)
                if d < best_d:
                    best_d = d
                    best = (fw, rw)
            if best is None or best_d > MATCH_TOL_S:
                continue
            fw, rw = best
            norms.update_feat(fw)
            norms.update_raw(rw)
            norms.update_tgt(ang)
            xn = norms.norm_feat(fw)
            raw_n = norms.norm_raw(rw)
            warm = (time.time() - start) > WARMUP_SECONDS

            # prequential eval
            eval_preds = {}
            for m in MODELS:
                inp = xn if m.kind == "feat" else raw_n
                p = m.predict(inp)
                eval_preds[m.name] = p
                if warm:
                    if p is not None:
                        m.metrics.update(ang, p)
                        m.roll.update(ang, p)
                        m.err_hist.append(p - ang)
                    else:
                        m.err_hist.append(np.nan)
            ens_eval = ensemble.blend(eval_preds)
            if warm:
                if ens_eval is not None:
                    ensemble.metrics.update(ang, ens_eval)
                    ensemble.roll.update(ang, ens_eval)
                    ensemble.err_hist.append(ens_eval - ang)
                else:
                    ensemble.err_hist.append(np.nan)

            replay.append((raw_n.copy(), float(ang)))
            for m in MODELS:
                if m.kind == "feat":
                    m.update(xn, ang)

        for m in MODELS:
            if m.kind == "seq":
                m.train_steps(replay, norms)

        now = time.time()
        if norms.feat_mean is not None and (now - last_disp_t) >= DISP_PRED_EVERY_S:
            last_disp_t = now
            xn_cur = norms.norm_feat(feat)
            raw_cur = norms.norm_raw(raw)
            disp = {}
            for m in MODELS:
                inp = xn_cur if m.kind == "feat" else raw_cur
                p = m.predict(inp)
                disp[m.name] = p
                m.last_pred = p
            ensemble.last_pred = ensemble.blend(disp)

            if (now - last_servo_t) >= DISP_PRED_EVERY_S:
                last_servo_t = now
                sel_name = servo_select["name"]
                for m in DISPLAY:
                    if m.name == sel_name:
                        servo_command(m.last_pred, sel_name)
                        break

        time.sleep(TRAIN_TICK_S)


# ====== SAVE  ===============================================================
def save_models_now():
    """Snapshot the current models + normalizers to models/bundle_<TS>.pkl."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"models/bundle_{ts}.pkl"
    try:
        save_bundle(path, MODELS, ensemble, norms,
                    extra_meta={"session_ts": session_time,
                                "saved_at": ts,
                                "labels_seen": norms.feat_n})
        sizes = []
        for m in MODELS:
            sizes.append(f"{m.name}={'Y' if m.trained else '-'}")
        print(f"[save] -> {path}  ({'  '.join(sizes)})  labels={norms.feat_n}")
    except Exception as e:
        print(f"[save] failed: {e}")


# ====== DASHBOARD ===========================================================
DASH_W, DASH_H = 880, 880
HIST_LEN = 300
angle_hist = deque(maxlen=HIST_LEN)
ch_hist = [deque(maxlen=HIST_LEN) for _ in range(N_CH)]

fps_t = time.time()
fps_count = 0
fps_value = 0.0


def sparkline(img, values, x, y, w, h, color, vmin=None, vmax=None,
              overlays=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), 1)
    if len(values) < 2:
        return
    arr = np.array(values, dtype=float)
    series = [arr]
    if overlays:
        for ov, _ in overlays:
            series.append(np.array(ov, dtype=float))
    finite_concat = np.concatenate([s[np.isfinite(s)] for s in series
                                    if np.any(np.isfinite(s))]) \
        if any(np.any(np.isfinite(s)) for s in series) else None
    if finite_concat is None or len(finite_concat) < 2:
        return
    lo = float(np.min(finite_concat)) if vmin is None else vmin
    hi = float(np.max(finite_concat)) if vmax is None else vmax
    if hi - lo < 1e-6:
        hi = lo + 1.0

    def to_pts(values_seq):
        n = len(values_seq)
        pts = []
        for i, v in enumerate(values_seq):
            if not np.isfinite(v):
                pts.append(None); continue
            px = x + int(i * (w - 1) / max(1, n - 1))
            py = y + h - 1 - int((v - lo) / (hi - lo) * (h - 1))
            pts.append((px, py))
        return pts

    def draw_pts(pts, col):
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None: continue
            cv2.line(img, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)

    draw_pts(to_pts(arr), color)
    if overlays:
        for ov, col in overlays:
            draw_pts(to_pts(np.array(ov, dtype=float)), col)


def draw_dashboard(angle_val):
    img = np.full((DASH_H, DASH_W, 3), 18, dtype=np.uint8)
    F = cv2.FONT_HERSHEY_SIMPLEX

    def text(s, x, y, scale=0.55, color=(220, 220, 220), thick=1):
        cv2.putText(img, s, (x, y), F, scale, color, thick, cv2.LINE_AA)

    text("LIVE TRAIN — laptop owns all models (servo from selected one)",
         16, 28, 0.65, (255, 255, 255), 2)
    elapsed = time.time() - start
    fs_txt = (f"{1.0/np.median(np.diff(np.array(_fs_times))):.0f}Hz"
              if len(_fs_times) > 10 else "?")
    text(f"t={elapsed:6.1f}s  fps={fps_value:4.1f}  labels={norms.feat_n}  "
         f"replay={len(replay)}  emg~{fs_txt}  horizon={PRED_HORIZON_S*1000:.0f}ms  "
         f"SERVO={servo_select['name']}  [S=save]",
         16, 54, 0.5, (170, 170, 170))

    ty = 86
    text(f"model  (metrics = rolling last {ROLL_WINDOW})", 16, ty, 0.5,
         COL_LABEL, 1)
    text("MAE", 240, ty, 0.5, COL_LABEL, 1)
    text("RMSE", 320, ty, 0.5, COL_LABEL, 1)
    text("R^2", 410, ty, 0.5, COL_LABEL, 1)
    text("cumMAE", 480, ty, 0.5, COL_LABEL, 1)
    text("N", 580, ty, 0.5, COL_LABEL, 1)
    text("status", 640, ty, 0.5, COL_LABEL, 1)
    for j, m in enumerate(DISPLAY):
        ry = ty + 24 + j * 23
        mae, rmse, r2 = m.roll.get()
        cmae = m.metrics.get()[0]
        tag = "*" if m.name == servo_select["name"] else " "
        text(f"{tag}{m.name}", 16, ry, 0.55, m.color, 2)
        text(f"{mae:6.2f}", 240, ry, 0.5)
        text(f"{rmse:6.2f}", 320, ry, 0.5)
        text(f"{r2:6.3f}", 410, ry, 0.5)
        text(f"{cmae:6.2f}", 480, ry, 0.5, (170, 170, 170))
        text(f"{m.metrics.n}", 580, ry, 0.5, (170, 170, 170))
        text(m.extra, 640, ry, 0.45, (170, 170, 170))

    gauge_top = ty + 24 + len(DISPLAY) * 23 + 16
    text("ANGLE  (true + each model)", 16, gauge_top, 0.55, COL_LABEL, 2)
    if angle_min_obs is not None and angle_max_obs is not None:
        lo, hi = angle_min_obs, angle_max_obs
    else:
        lo, hi = 60.0, 120.0
    if hi - lo < 5.0:
        hi = lo + 5.0
    gx, gw = 16, DASH_W - 32
    rows = [("true", angle_val, COL_TRUE)] + \
        [(m.name, m.smoothed, m.color) for m in DISPLAY]
    for j, (lbl, val, col) in enumerate(rows):
        ry = gauge_top + 12 + j * 24
        cv2.rectangle(img, (gx, ry), (gx + gw - 90, ry + 17), (50, 50, 50), 1)
        if val is not None and np.isfinite(val):
            t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
            cv2.rectangle(img, (gx, ry),
                          (gx + int((gw - 90) * t), ry + 17), col, -1)
            text(f"{lbl}: {val:6.1f}", gx + gw - 85, ry + 13, 0.5, col)
        else:
            text(f"{lbl}:   --", gx + gw - 85, ry + 13, 0.5, col)
    sp_top = gauge_top + 12 + len(rows) * 24 + 18
    text(f"range used: [{lo:.0f}, {hi:.0f}] deg", gx, sp_top - 6,
         0.45, (160, 160, 160))

    spx = 16
    text("ANGLE  true (cyan) + model predictions", spx, sp_top + 16,
         0.55, COL_LABEL, 2)
    sparkline(img, list(angle_hist), spx, sp_top + 24, DASH_W - 32, 80,
              COL_TRUE,
              overlays=[(list(m.hist), m.color) for m in DISPLAY])

    err_top = sp_top + 24 + 80 + 22
    text("PREDICTION ERROR  (per model)", spx, err_top, 0.55, COL_LABEL, 2)
    if DISPLAY:
        sparkline(img, list(DISPLAY[0].err_hist), spx, err_top + 8,
                  DASH_W - 32, 50, DISPLAY[0].color,
                  overlays=[(list(m.err_hist), m.color) for m in DISPLAY[1:]])

    ch_top = err_top + 8 + 50 + 26
    text("EMG  m1 | m2 | m3 | gravity (envelope)", spx, ch_top, 0.5, COL_LABEL, 2)
    cw = (DASH_W - 32 - (N_CH - 1) * 8) // N_CH
    ch_cols = [(90, 230, 110), (90, 180, 255), (200, 200, 90), (200, 120, 255)]
    for ci in range(N_CH):
        cx = spx + ci * (cw + 8)
        sparkline(img, list(ch_hist[ci]), cx, ch_top + 8, cw, 50, ch_cols[ci])

    return img


# ====== NEEDLE OVERLAY ======================================================
def draw_needles(frame, ankle, shin, foot, model_vals, selected_name):
    if ankle is None or shin is None or foot is None:
        return
    ax, ay = ankle
    shin_dx, shin_dy = shin[0] - ax, shin[1] - ay
    base_len = math.hypot(shin_dx, shin_dy)
    if base_len < 1e-3:
        return
    base_ang = math.atan2(shin_dy, shin_dx)
    foot_dx, foot_dy = foot[0] - ax, foot[1] - ay
    cross = shin_dx * foot_dy - shin_dy * foot_dx
    side = 1.0 if cross >= 0 else -1.0

    radius = int(base_len * 0.6)
    base_deg = math.degrees(base_ang)
    a0, a1 = (base_deg, base_deg + 180) if side > 0 else (base_deg - 180, base_deg)
    cv2.ellipse(frame, (ax, ay), (radius, radius), 0, a0, a1,
                (80, 80, 80), 2, cv2.LINE_AA)
    for d in range(0, 181, 30):
        a = base_ang + side * math.radians(d)
        x1 = int(ax + (radius - 6) * math.cos(a))
        y1 = int(ay + (radius - 6) * math.sin(a))
        x2 = int(ax + (radius + 6) * math.cos(a))
        y2 = int(ay + (radius + 6) * math.sin(a))
        cv2.line(frame, (x1, y1), (x2, y2), (120, 120, 120), 1, cv2.LINE_AA)
        lx = int(ax + (radius + 22) * math.cos(a))
        ly = int(ay + (radius + 22) * math.sin(a))
        cv2.putText(frame, f"{d}", (lx - 10, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140),
                    1, cv2.LINE_AA)

    def one_needle(val, color, thick, label, is_sel):
        if val is None or not np.isfinite(val):
            return
        ang = base_ang + side * math.radians(float(val))
        nx = int(ax + base_len * math.cos(ang))
        ny = int(ay + base_len * math.sin(ang))
        t = thick + (2 if is_sel else 0)
        cv2.line(frame, (ax, ay), (nx, ny), color, t, cv2.LINE_AA)
        cv2.circle(frame, (nx, ny), 4, color, -1, cv2.LINE_AA)
        tag = f"{label}{' [SRV]' if is_sel else ''} {val:.1f}"
        cv2.putText(frame, tag, (nx + 8, ny - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    for val, color, thick, label in model_vals:
        one_needle(val, color, thick, label, label == selected_name)
    cv2.circle(frame, (ax, ay), 8, (200, 200, 200), -1, cv2.LINE_AA)


# ====== CAMERA SETUP ========================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

initialized = False
shin = ankle = foot = None
exp_shin_len = exp_foot_len = None

WIN_NAME = "Live Train (camera)"
cv2.namedWindow(WIN_NAME)
_setup_hsv_trackbars()


def _on_servo_src(idx):
    idx = max(0, min(len(DISPLAY) - 1, int(idx)))
    servo_select["idx"] = idx
    servo_select["name"] = DISPLAY[idx].name


cv2.createTrackbar("Servo src", WIN_NAME,
                   servo_select["idx"], len(DISPLAY) - 1, _on_servo_src)


def _on_smooth(v):
    servo_state["smooth_window"] = max(1, int(v))


def _on_slew(v):
    servo_state["slew_max"] = float(v)


# Anti-jitter knobs.  Smooth = EMA window N (1 = off, larger = smoother).
# Slew = max degrees per command (0 = disabled).
cv2.createTrackbar("Smooth", WIN_NAME, SERVO_SMOOTH_DEFAULT, 30, _on_smooth)
cv2.createTrackbar("Slew",   WIN_NAME, SERVO_SLEW_DEFAULT,   60, _on_slew)

click_points = []
CLICK_LABELS = ["shin", "ankle", "foot"]
CLICK_SNAP_PX = 40


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(click_points) < 3:
        click_points.append((x, y))


cv2.setMouseCallback(MASK_WIN, on_click)

# ====== CSV =================================================================
csv_file = open(log_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(
    ["timestamp", "t_mono", "angle_deg"]
    + [f"{m.name}_pred" for m in DISPLAY]
    + ["m1", "m2", "m3", "grav_raw", "grav_env", "servo_src"]
)

print(f"Logging to {log_path}")
print(f"Full-rate EMG log: {raw_log_path}")
print(f"Models: {[m.name for m in DISPLAY]}")
print("Click shin -> ankle -> foot on the Mask window to initialize.")
print(f"Warmup {WARMUP_SECONDS:.0f}s before metrics start counting.")
print(f"Pick servo source with the 'Servo src' trackbar or number keys 1..{len(DISPLAY)}.")
print("Press S to save the current models to models/bundle_<TS>.pkl.")

start = time.time()

# ====== THREADS =============================================================
threading.Thread(target=emg_reader, daemon=True).start()
threading.Thread(target=trainer, daemon=True).start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        points, mask_bgr = detect_markers(frame)
        angle_val = float("nan")

        if not initialized:
            for i, p in enumerate(click_points):
                cv2.circle(mask_bgr, p, 8, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(mask_bgr, CLICK_LABELS[i], (p[0] + 12, p[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2, cv2.LINE_AA)
            if len(click_points) < 3:
                msg = (f"Click {CLICK_LABELS[len(click_points)]} on Mask "
                       f"window  ({len(click_points)}/3)")
                cv2.putText(mask_bgr, msg, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(mask_bgr, "R reset, Q quit", (10, 74),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(frame, "click on Mask window to set shin/ankle/foot",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2, cv2.LINE_AA)
            else:
                def snap(p):
                    if not points:
                        return p
                    nearest = min(points, key=lambda c: distance(c, p))
                    return nearest if distance(nearest, p) <= CLICK_SNAP_PX else p

                shin = snap(click_points[0])
                ankle = snap(click_points[1])
                foot = snap(click_points[2])
                exp_shin_len = distance(shin, ankle)
                exp_foot_len = distance(ankle, foot)
                initialized = True
                print(f"Initialized: shin={shin} ankle={ankle} foot={foot}")
        else:
            if len(points) == 3:
                points.sort(key=lambda p: p[1])
                shin = points[0]
                ankle, foot = choose_ankle_foot(
                    shin, points[1], points[2], ankle, foot,
                    exp_shin_len, exp_foot_len)
                angle_val = calculate_angle(shin, ankle, foot)
                if np.isfinite(angle_val):
                    if angle_min_obs is None or angle_val < angle_min_obs:
                        angle_min_obs = angle_val
                    if angle_max_obs is None or angle_val > angle_max_obs:
                        angle_max_obs = angle_val
                    with angle_lock:
                        angle_samples.append((time.time(), float(angle_val)))

            for m in DISPLAY:
                p = m.last_pred
                if p is not None and np.isfinite(p):
                    if m.smoothed is None:
                        m.smoothed = p
                    else:
                        a = 1.0 / PRED_SMOOTH
                        m.smoothed = a * p + (1 - a) * m.smoothed

            if len(emg_buf) > 0:
                last = emg_buf[-1]
                ch_vals = [last[1], last[2], last[3], last[4]]  # m1, m2, m3, grav_env
                grav_raw_last = last[5]
            else:
                ch_vals = [float("nan")] * N_CH
                grav_raw_last = float("nan")

            angle_hist.append(angle_val if np.isfinite(angle_val) else np.nan)
            for m in DISPLAY:
                m.hist.append(m.smoothed if m.smoothed is not None else np.nan)
            for ci in range(N_CH):
                v = ch_vals[ci]
                ch_hist[ci].append(float(v) if v == v else 0.0)

            writer.writerow(
                [datetime.now().isoformat(), time.time(), angle_val]
                + [(m.last_pred if m.last_pred is not None else "") for m in DISPLAY]
                + [ch_vals[0], ch_vals[1], ch_vals[2], grav_raw_last, ch_vals[3],
                   servo_select["name"]]
            )
            csv_file.flush()

            for p, name in ((shin, "shin"), (ankle, "ankle"), (foot, "foot")):
                if p is not None:
                    cv2.circle(frame, p, 6, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.putText(frame, name, (p[0] + 8, p[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1, cv2.LINE_AA)
            if shin is not None and ankle is not None:
                cv2.line(frame, shin, ankle, (0, 180, 180), 2, cv2.LINE_AA)
            if ankle is not None and foot is not None:
                cv2.line(frame, ankle, foot, (0, 180, 180), 2, cv2.LINE_AA)

            draw_needles(frame, ankle, shin, foot,
                         [(m.smoothed, m.color, m.thick, m.name) for m in DISPLAY],
                         servo_select["name"])

            cv2.putText(frame, f"true: {angle_val:6.1f}", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_TRUE, 2)
            for j, m in enumerate(DISPLAY):
                txt = (f"{m.smoothed:6.1f}" if m.smoothed is not None else "  --  ")
                tag = " *" if m.name == servo_select["name"] else "  "
                cv2.putText(frame, f"{tag}{m.name:4s}: {txt}",
                            (20, 56 + j * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, m.color, 2)
            any_trained = any(m.trained for m in MODELS)
            base_y = 56 + len(DISPLAY) * 24
            if not any_trained:
                cv2.putText(frame, "TRAINING...", (20, base_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            elif time.time() - start < WARMUP_SECONDS:
                remain = WARMUP_SECONDS - (time.time() - start)
                cv2.putText(frame, f"warmup {remain:.1f}s", (20, base_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.putText(frame,
                        f"SERVO src: {servo_select['name']}   "
                        f"smooth={servo_state['smooth_window']}   "
                        f"slew={servo_state['slew_max']:.0f}deg   S=save",
                        (20, base_y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)

        fps_count += 1
        if time.time() - fps_t >= 0.5:
            fps_value = fps_count / (time.time() - fps_t)
            fps_count = 0
            fps_t = time.time()

        cv2.imshow(WIN_NAME, frame)
        cv2.imshow(MASK_WIN, mask_bgr)
        cv2.imshow("Dashboard", draw_dashboard(
            angle_val if initialized else float("nan")))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r") and not initialized:
            click_points.clear()
            print("Clicks reset.")
        if key == ord("s"):
            # save in a background thread so the GUI stays responsive
            threading.Thread(target=save_models_now, daemon=True).start()
        if ord("1") <= key <= ord("9"):
            i = key - ord("1")
            if i < len(DISPLAY):
                servo_select["idx"] = i
                servo_select["name"] = DISPLAY[i].name
                cv2.setTrackbarPos("Servo src", WIN_NAME, i)
                print(f"servo source -> {DISPLAY[i].name}")

finally:
    stop_flag.set()
    time.sleep(0.2)
    send_angle(135.0)
    time.sleep(0.05)
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    raw_csv_file.flush()
    raw_csv_file.close()
    try:
        ser.close()
    except Exception:
        pass
    print("Stopped.")
