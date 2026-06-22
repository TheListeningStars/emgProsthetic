# emgProsthetic — closed-loop online training, ESP32 servo target

Predict an ankle angle in real time from 4-channel surface EMG and drive a
servo with the prediction.  Five regression models (SGD, RLS, RF, LSTM, GRU)
+ an error-weighted ensemble train **continuously and in parallel** while the
camera supplies ground-truth angle labels.  You pick which model drives the
servo with a trackbar.

The laptop does all of the work — filtering, feature extraction, training,
prediction.  The ESP32 is a dumb sensor / actuator node: it streams raw ADC
samples and writes whatever angle the laptop tells it to.

## What runs where

| | Laptop (`live_train.py`) | ESP32 (`esp32_online.ino`) |
|---|---|---|
| reads EMG (4 analog pins) | no | yes |
| gravity-EMG filter chain  | yes | no |
| 12-dim feature extraction | yes | no |
| measures ankle angle      | yes (camera + clicked markers) | no |
| trains models             | yes (5 in parallel + ensemble) | no |
| runs prediction           | yes | no |
| drives servo              | sends angle commands | applies them |

## Repo layout

```
emgProsthetic/
├── live_common.py       # shared: filter, features, models, save/load
├── live_train.py        # online training + camera GUI + dashboard
├── live_deploy.py       # load a saved bundle, drive servo (no camera)
├── train_offline.py     # train from scratch on logged CSVs
└── esp32_online/
    └── esp32_online.ino # ESP32 firmware (dumb node)
```

Created at runtime (gitignored): `logs/` for session CSVs, `models/` for
saved model bundles.

## Hardware

- ESP32 with four analog EMG inputs and one servo output:
  - `GPIO32` — Gravity Analog EMG (SEN0240), processed to envelope on the laptop
  - `GPIO34` — m1 (MyoWare, raw EMG)
  - `GPIO35` — m2 (MyoWare, raw EMG)
  - `GPIO33` — m3 (MyoWare, raw EMG)
  - `GPIO18` — servo signal

All four ADCs are on ADC1, so they keep working with WiFi on (ADC2 conflicts
with WiFi).
- Camera for the laptop (3 green markers on shin / ankle / foot for the
  ground-truth angle).

Pin assignments and servo limits live at the top of
`esp32_online/esp32_online.ino`.

## Software setup

```bash
# Python deps (LSTM/GRU are optional — torch is detected and skipped if absent)
pip install pyserial opencv-python scikit-learn scipy numpy
pip install torch          # optional, enables LSTM + GRU

# Arduino IDE + ESP32 board package + ESP32Servo library, then flash:
#   esp32_online/esp32_online.ino
```

Edit `SERIAL_PORT` at the top of `live_train.py` / `live_deploy.py` to your
ESP32's port (e.g. `/dev/cu.usbserial-XXXX`).  Default baud is 921600.

## Serial protocol  (line-delimited ASCII, 921600 baud)

ESP32 → laptop
- `R,t_us,grav,m1,m2,m3\n`  raw ADC sample at ~63.46 Hz
- `S,<status>\n`            boot / info

Laptop → ESP32
- `A,<angle_deg>\n`       drive servo to this angle (clamped on ESP32 too)

## Models

All defined in `live_common.py` and trained in parallel.  Inputs:

- **feature models** (16-dim engineered vector — RMS/MAV/VAR/WL per channel,
  with `LEVEL` instead of `RMS` for the gravity envelope):
  - `SGD`  — `SGDRegressor`, constant LR, `partial_fit` every 10 labels
  - `RLS`  — recursive least squares (online ridge), forgetting factor 0.999
  - `RF`   — `RandomForestRegressor` (60 trees × depth 12), refit in a
             background thread every 25 new labels
- **sequence models** (full `EMG_WINDOW=75` × 4 raw window, normalized):
  - `LSTM` — 2-layer, hidden 48, MLP head
  - `GRU`  — same shape as LSTM
- **ensemble**:
  - `ENS`  — error-weighted blend, weight ∝ 1 / (rolling_MAE + 1).  At deploy
             time, weights are frozen from each member's training-time MAE.

Models predict the angle `PRED_HORIZON_S = 100 ms` in the **future** — the
input window ends at `t`, target is `angle(t + 0.10s)`.  Absorbs the
EMG → motion electromechanical delay and makes the output useful for control.

## Usage

### 1. Live training  (laptop owns everything)

```bash
python live_train.py
```

- Three OpenCV windows open: camera, HSV mask, dashboard.
- Click **shin → ankle → foot** on the *Mask* window to start angle
  measurement.
- Move your foot. The trainer pairs features with the measured angle in a
  sliding buffer.  Each label triggers a prequential update of every model;
  the sequence nets also keep taking gradient steps from the shared replay
  buffer continuously.
- Pick which model drives the servo with the **"Servo src"** trackbar, or
  press keys `1..6`.  The selected model's needle on the camera frame is
  fattened and tagged `[SRV]`.
- Press **`S`** to save the current models to `models/bundle_<TS>.pkl`
  (includes EMA normalizer stats and ensemble weights).
- `Q` to quit.

CSV logs land in `logs/liveTrain/`:
- `live_train_<TS>.csv` — per-frame (angle + all model predictions + EMG snapshot)
- `live_train_<TS>_emg_raw.csv` — every raw EMG sample (used by offline training)

### 2. Deploy a saved model  (no camera, no training)

```bash
python live_deploy.py --model ENS
# or
python live_deploy.py --bundle models/bundle_20260622_140530.pkl --model RF
```

Loads the bundle, opens serial to the ESP32, runs the filter + feature
pipeline, predicts with the selected model, and sends `A,<deg>\n`.  Defaults
to the most recent bundle in `models/` and `--model ENS`.

Other flags: `--port`, `--baud`, `--no-invert` (disable the default servo
direction flip), `--print-every` (seconds between status prints, 0 = silent),
`--log PATH` (record predictions to a CSV).

### 3. Offline training  (retrain from scratch on logged data)

```bash
python train_offline.py logs/liveTrain/live_train_20260622_140530.csv \
                        [more_sessions.csv ...] \
                        --epochs 30
```

Pairs every EMG window with the angle label closest to `t + PRED_HORIZON_S`
(within `--tol-s`, default 50 ms), then trains:
- SGD over `--epochs` passes of `partial_fit`
- RLS in one streaming pass
- RF as a single `fit` on the last 1200 pairs
- LSTM + GRU in mini-batch epochs (`--no-seq` to skip)

Each session CSV's matching `_emg_raw.csv` is found automatically.  Output
bundle: `models/bundle_offline_<TS>.pkl` (loadable by `live_deploy.py`).

## Notes / knobs

- `EMG_WINDOW = 75` (~1.2 s of context at 63.46 Hz).  Change in
  `live_common.py` — make sure ESP32 and laptop agree.
- `PRED_HORIZON_S = 0.10` — how far in the future to predict.  Changing it
  invalidates saved bundles (they were trained for a specific horizon).
- `SERVO_INVERT = True` in `live_train.py` / `--no-invert` in `live_deploy.py`
  — flip if your servo horn is mounted such that increasing predicted angle
  drives it the wrong way.
- EMA normalizer stats (`feat_mean/var`, `raw_mean/var`, `tgt_mean/var`) are
  saved with the bundle, so deploy reproduces training-time normalization
  even though it never updates the EMAs itself.
- No GPU. The networks are tiny; `torch.set_num_threads(1)` is set and
  everything stays on CPU.
- Gravity-EMG filter chain: DC removal → band-pass 20–31 Hz (clamped at
  Nyquist for 63.46 Hz fs) → mains notch (skipped above Nyquist) → rectify →
  6 Hz envelope LP.  Lives in `GravityEMGProcessor` in `live_common.py`.
