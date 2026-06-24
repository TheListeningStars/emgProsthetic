# emgProsthetic — project report

A closed-loop EMG → servo system that predicts ankle angle in real time from
four surface-EMG channels and drives a servo with the prediction.
The laptop owns all signal processing, training, and inference; the ESP-32
is a dumb sensor / actuator node.

---

## A. Wiring diagram + parts

### A.1 Bill of materials

| qty | part | role |
|---:|---|---|
| 1 | [Gravity Analog EMG Sensor (DFRobot SEN0240)](https://wiki.dfrobot.com/Gravity__Analog_EMG_Sensor_by_OYMotion_SKU_SEN0240) | raw-EMG channel, **filter chain done on laptop** |
| 2 | [Techtonics Muscle Sensor Module](https://www.amazon.in/Techtonics-Muscle-Sensor-Module-Electrodes/dp/B07QD7DRGL) | envelope-output EMG channels (m1, m2) |
| 1 | [Advancer Tech EMG Muscle Sensor V3.0](https://robu.in/product/advancer-technologies-emg-muscle-sensor-v3-0-with-cable-and-electrodes/) | envelope-output EMG channel (m3) |
| 1 | ESP-32 WROOM dev board | ADC × 4 + PWM out + USB-serial to laptop |
| 1 | [Cytron MD-102 Breadboard Power Module](https://www.cytron.io/p-breadboard-power-module-md-102) | regulates 9 V → 5 V / 3.3 V rails for sensors |
| 1 | SG90 micro servo | end effector |
| 4 | 9 V batteries | power for the MD-102, the servo regulator, and spares |
| — | gel electrodes (3 per EMG sensor) | skin contact |
| — | breadboard + jumper wires | |

### A.2 Pin assignments (ESP-32 → world)

| ESP-32 pin | direction | connected to | code constant |
|---|---|---|---|
| `GPIO32` | input (ADC1) | Gravity EMG `Sig` | `ANALOG_PIN_GRAV` |
| `GPIO34` | input (ADC1) | Techtonics #1 `Sig` | `ANALOG_PIN_M1` |
| `GPIO35` | input (ADC1) | Techtonics #2 `Sig` | `ANALOG_PIN_M2` |
| `GPIO33` | input (ADC1) | Advancer V3 `Sig`   | `ANALOG_PIN_M3` |
| `GPIO18` | output (PWM) | SG90 signal wire | `SERVO_PIN` |
| `3V3`    | power | Gravity EMG `VCC` | — |
| `5V` (`VIN`) | power | Techtonics × 2 `VCC` + Advancer `VCC` | — |
| `GND`    | ground | sensor commons, servo GND, battery GND | — |

> All four ADC pins are on **ADC1** on purpose — ADC2 is used by the Wi-Fi
> radio, so reading ADC2 with Wi-Fi enabled returns garbage.  ADC1 reads
> stay valid regardless.

### A.3 Power topology

| rail | source | feeds | why |
|---|---|---|---|
| 5 V *(sensors)* | 9 V battery → MD-102 jack → 5 V rail | Techtonics × 2, Advancer V3 | clean regulated rail, no servo noise |
| 3.3 V *(sensor)* | MD-102 3.3 V rail | Gravity EMG | the OYMotion module is 3.3 V native |
| 5 V *(servo)* | dedicated 9 V → linear/buck regulator | SG90 + GND | servo stalls draw up to ~600 mA spikes; keeping it off the sensor rail prevents EMG-baseline shifts |
| 5 V *(MCU)* | USB from laptop | ESP-32 VIN | also the data path |
| GND | **single common ground** tied at the breadboard rail | every device | required for the EMG single-ended signals to make sense |

The remaining two of the four 9 V batteries are either used in parallel with
the active pair (for runtime) or kept as spares.  See the wiring diagram
([docs/wiring.svg](docs/wiring.svg)) for the full layout.

### A.4 Serial protocol (laptop ↔ ESP-32, 921600 baud)

ESP-32 → laptop
```
R,t_us,grav,m1,m2,m3\n      raw ADC sample at ~63.46 Hz
S,<status>\n                boot / info
```

Laptop → ESP-32
```
A,<angle_deg>\n             drive servo to this angle (clamped on ESP-32)
```

---

## B. Code explanation

### B.1 Architecture in one sentence

The laptop reads raw ADC samples over USB serial, runs a streaming gravity-EMG
filter, builds 16-dim features over a 75-sample (~1.2 s) window, trains five
models in parallel against camera-measured ankle angles, and ships the
selected model's prediction back to the ESP-32 as a servo angle.

```
   ┌── ESP-32 ──┐                ┌─────────────── Laptop ───────────────┐
   │ 4× ADC     │  R,t,g,m1..3   │  serial_reader → emg_buf  → trainer  │
   │ @ 63.5Hz   │ ─────────────► │       (gravity LP filter chain)      │
   │            │                │            │                          │
   │            │ ◄──────────────│  servo_command ◄── selected model     │
   │ PWM servo  │  A,<angle>     │                                       │
   └────────────┘                │  camera → markers → ankle angle →     │
                                 │  label buffer  → trainer pairs +H s   │
                                 └───────────────────────────────────────┘
```

### B.2 File map

| file | role |
|---|---|
| `esp32_online/esp32_online.ino` | ESP-32 firmware: read 4 ADCs, emit `R` lines, accept `A` lines, drive servo |
| `live_common.py` | single source of truth — gravity filter, features, normalizers, all model classes, `save_bundle` / `load_bundle` |
| `live_train.py` | live training: camera GUI + dashboard + multi-model trainer thread + servo dispatch |
| `live_deploy.py` | load a saved bundle and drive the servo. No camera, no training |
| `train_offline.py` | train from logged session CSVs; train/val split + horizon sweep + GPU support |
| `train_offline.ipynb` | notebook version (Colab-friendly, with ipywidgets file uploader) |
| `colab_train.sh` | one-shot helper: provision Colab VM → upload code+sessions → run training → download bundle → tear VM down |

### B.3 Signal chain (`live_common.GravityEMGProcessor`)

Causal, sample-at-a-time port of the offline gravity-EMG pipeline:

1. **DC removal** — slow EMA (`α = 1/500`) tracks baseline drift, subtracted out.
2. **Band-pass** — 4th-order Butterworth, 20–450 Hz, clamped at 0.99 × Nyquist for the actual fs.  At our 63.46 Hz the effective band is 20–31 Hz.
3. **Mains notch** — 60 Hz + harmonics via IIR notch filters; skipped automatically when above Nyquist (so at 63.46 Hz no notch is inserted — would alias).
4. **Full-wave rectify** — `abs(v)`.
5. **Envelope LP** — 4th-order low-pass at 6 Hz.

State (`zi_bp`, `zi_env`, notch `zi`) is persistent across samples, so the
filter is mathematically identical to the offline batch version once warmed up.

The two MyoWare-style channels (Techtonics, Advancer) already output an
envelope, so they bypass this filter and are read raw.

### B.4 Features (`features_from_arr`, 16-dim)

Per channel, 4 features over a 75-sample window:

| channel | features | rationale |
|---|---|---|
| m1, m2, m3 | RMS, MAV, VAR, WL of mean-removed signal | AC activity features (envelope-already, so DC is noise) |
| gravity | LEVEL (mean), MAV, VAR, WL | **LEVEL preserves** the absolute envelope amplitude — that's the activation strength |

Total feature vector = 4 × 4 = **16 dims**, fed to every "feature model".

### B.5 Sequence models

LSTM and GRU consume the **raw normalized window** of shape `(EMG_WINDOW=75, N_CH=4)` directly — no engineered features — so they can learn their own temporal patterns.  Both are:
`SEQ_HIDDEN=80, SEQ_LAYERS=3, dropout=0.1` recurrent body + 2-layer MLP head, trained with Adam (`lr=3e-3, wd=1e-4`) and Smooth-L1 loss.

### B.6 Normalizers (`Normalizers`)

EMA-tracked per-feature / per-channel / per-target mean & variance, with
half-lives of ~200 samples (features), ~2000 (raw), ~300 (target).  Stored
inside the bundle so deploy reproduces training-time normalization exactly.

### B.7 Models trained in parallel

| model | input | training | strengths |
|---|---|---|---|
| **RLS** | 16-dim feat | recursive least squares, λ=0.999 forgetting factor | fast, adaptive, no batch needed |
| **RF**  | 16-dim feat | `RandomForestRegressor(n=120, max_depth=20)`, refit every 25 labels in a background thread | non-linear, no normalization sensitivity |
| **LSTM** | (75, 4) raw | minibatch SGD on shared replay buffer (size 2000) | learns temporal dynamics |
| **GRU**  | (75, 4) raw | same as LSTM | cheaper twin of LSTM |
| **ENS**  | (predictions) | error-weighted blend, weight ∝ 1 / (rolling MAE + 1) | robust to any single model going off |

> The original SGD model was removed (always lost to RLS / RF; dead weight on the prediction loop).

### B.8 Prediction horizon

Targets are **angle(t + `PRED_HORIZON_S`)** rather than angle(t) — currently
0.10 s.  This trains the model to *lead* the signal, absorbing the EMG → motion
electromechanical delay and the implicit lag of the feature window.

### B.9 Live training loop (`live_train.py`)

- `serial_reader` thread reads `R,…` lines → pushes raw + gravity-envelope sample to `emg_buf` (a deque of `EMG_WINDOW` samples).
- `trainer` thread (~250 Hz):
  - snapshots the current window
  - for every new camera angle label, finds the window whose end-time matches `label_time − horizon`, normalizes, runs prequential eval (predict-then-train) for every model
  - takes a few gradient steps for the sequence nets from the shared replay buffer
  - ~50 Hz: computes a fresh prediction from the *current* window, ships the **selected** model's value to the ESP-32 via `A,<deg>\n`
- camera loop: HSV mask → 3 markers (shin/ankle/foot) → ankle angle, posted to a thread-safe queue + drawn on screen
- dashboard window shows per-model rolling + cumulative MAE/RMSE/R², predictions vs truth sparkline, error sparklines, and the 4 EMG channels.

#### B.9.1 Servo source dropdown + anti-jitter knobs

Three trackbars on the camera window:

- **Servo src** — which model drives the servo (`RLS / RF / LSTM / GRU / ENS`).  Also bindable to number keys 1..5.
- **Smooth** — EMA window applied to the servo command (1 = none, larger = smoother).
- **Slew** — max degree change per command (0 = disabled).

These two knobs let you trade jitter for lag at runtime without retraining.

#### B.9.2 Save bundle

Press **`S`** in the camera window → writes `models/bundle_<TS>.pkl` containing every trained model, the EMA normalizer state, the ensemble weights (computed from current rolling MAE), and metadata.  Save runs in a background thread so the GUI stays responsive.

### B.10 Deploy (`live_deploy.py`)

No camera, no training.  Loads a bundle, opens serial, runs the same filter +
feature pipeline, predicts with one model (chosen via `--model NAME`, default
`ENS`), and writes the angle to the ESP-32.  Defaults to the most recent
`models/bundle_*.pkl`.

### B.11 Offline training (`train_offline.py` + `train_offline.ipynb`)

- Reads one or more session main CSVs + their matching `_emg_raw.csv` files.
- `build_pairs` slides a 75-sample window through the raw EMG, pairs each window end with the closest angle label at `t + horizon` (within `--tol-s`).
- **Temporal train/val split**: last `--val-frac` (default 20%) of *each session* is held out for validation, so the metric reflects actual generalization on later data.
- Trains all four base models from scratch.
- Reports per-model **train MAE** and **val MAE**.
- `--horizons "0.05,0.10,0.15,0.20"` mode: re-pair and retrain at each horizon, print a sweep summary — no bundle saved.
- `--device auto|cpu|cuda|mps` plumbs through to LSTM/GRU training.  Net is moved back to CPU before save so bundles stay portable.

The notebook ([`train_offline.ipynb`](train_offline.ipynb)) is the same pipeline as cells, plus an `ipywidgets.FileUpload` block at the top that auto-saves dropped files to `/content/` so the workflow is one-click in Colab / VSCode-Colab.

### B.12 Colab automation (`colab_train.sh`)

End-to-end shell helper:

1. `colab new -s emg --gpu T4` (configurable)
2. upload `live_common.py`, `train_offline.py`, plus each session's CSV pair
3. `colab install …` deps
4. run `train_offline.py` with whatever args follow `--`
5. find the newest `models/bundle_*.pkl`, `colab download` it locally
6. `colab stop -s emg`  (skipped with `--no-stop`)

A bash `trap` ensures step 6 always runs.

---

## C. EMG sensor placement

### C.1 Anatomy: which muscles drive ankle angle

For pred-and-control of ankle dorsiflexion ↔ plantarflexion you want the four
muscles that **do** the motion, and one of them as your reference:

| function | muscle | where it sits |
|---|---|---|
| dorsiflexion (foot up) | **Tibialis Anterior (TA)** | front of shin, just lateral to the tibia, upper third |
| plantarflexion (foot down) | **Gastrocnemius — medial head** | inner upper calf, prominent belly |
| plantarflexion (foot down) | **Gastrocnemius — lateral head** | outer upper calf |
| plantarflexion / posture | **Soleus** | deeper, lower calf — palpate below the gastroc bellies |
| eversion / lateral stabilizer | **Peroneus Longus** | lateral lower leg, runs along the fibula |

### C.2 Recommended mapping for this code's 4 channels

| code channel | sensor | muscle | reasoning |
|---|---|---|---|
| `gravity` *(GPIO32)* | Gravity Analog EMG (SEN0240) | **Tibialis Anterior** | gravity sensor gives the cleanest raw signal post-filter; TA is the most-isolated dorsiflexor — pairs well with a high-quality channel |
| `m1` *(GPIO34)* | Techtonics #1 | **Gastrocnemius — medial head** | dominant plantarflexor; large, easy belly to palpate |
| `m2` *(GPIO35)* | Techtonics #2 | **Gastrocnemius — lateral head** | second head of the antagonist; redundancy helps the ensemble |
| `m3` *(GPIO33)* | Advancer Tech V3 | **Peroneus Longus** | adds an off-axis stabilizer signal that disambiguates pure plantarflexion from inversion / eversion |

Alternative: swap `m3` for **Soleus** if you don't care about eversion and
want pure plantarflexion redundancy.

### C.3 Electrode placement (per sensor)

Each MyoWare-style sensor uses three electrodes:

```
  ╔══════════ muscle belly ══════════╗
  ║                                  ║
  ║   ● + (mid-belly)                ║
  ║   ●  (2 cm proximal, in-line)    ║
  ║                                  ║
  ╚══════════════════════════════════╝

  ● REF on bony landmark — patella, medial malleolus (ankle bone), or kneecap
```

- **`+` and `–` electrodes**: along the muscle fibers, **2 cm centre-to-centre**, both on the muscle *belly* (not over tendon or bone).
- **`REF`**: on a nearby bony, electrically-quiet spot — patella for TA, medial malleolus for the calf muscles works well.
- Two electrodes per muscle is *bipolar differential*; the reference cancels common-mode noise (mains, motion).

### C.4 Skin prep (matters a lot for signal quality)

1. Shave hair from the patch.
2. Abrade lightly with prep gel or an alcohol wipe — drop skin impedance.
3. Let alcohol fully evaporate before sticking electrodes.
4. Press the electrode for ~5 seconds to set the gel.
5. Tape the sensor cable down so it doesn't tug the leads.

### C.5 Specific landmarks

**Tibialis Anterior (gravity / GPIO32)**
- Have the user dorsiflex against resistance — feel the belly bulge ~4 cm below and 2 cm lateral of the tibial tuberosity.
- `+` and `–` along that bulge, fibers run nearly vertical.
- `REF` on the patella.

**Gastrocnemius — medial head (m1 / GPIO34)**
- User goes onto toes — the inner half of the upper calf hardens.
- `+` and `–` along the long axis of that bulge.
- `REF` on the medial malleolus.

**Gastrocnemius — lateral head (m2 / GPIO35)**
- Outer half of the upper calf, also fires on toe-rises.
- Mirror the medial-head setup.
- `REF` on the lateral malleolus (or share medial — common ground anyway).

**Peroneus Longus (m3 / GPIO33)**
- Outside of the lower leg, running along the fibula head down to the foot.
- User everts the foot (turns sole outward) — the belly pops just below the fibular head.
- `REF` on the medial malleolus.

### C.6 Sanity checks before training

1. With the firmware running, open `live_train.py` and watch the dashboard EMG sparklines while the subject contracts each muscle in isolation.  Each channel should jump on its corresponding contraction and stay quiet otherwise.
2. If two channels move together, electrodes are picking up cross-talk — move them further apart or onto a more isolated muscle belly.
3. If a channel sits at ~0 or at the ADC ceiling, the electrode contact is bad — re-prep skin or replace the pad.
4. Warm up for ~30 s before starting a session — the gravity filter chain has a short transient, and EMG amplitudes settle once the user finds a comfortable posture.

### C.7 Reproducibility

- Mark electrode positions with a skin-safe pen so future sessions land on the same spot.
- Photograph the placement after the first session for the lab notebook.
- Note the subject's posture during recording — same chair, same ankle angle range — because feature distributions shift if the resting joint angle differs.

---

## Appendix · key constants

Lives in `live_common.py`.  If you change any of these, **bundles trained
before the change are invalid** (the input contract or model shape changes).

| constant | value | meaning |
|---|---|---|
| `N_CH` | 4 | EMG channels |
| `EMG_WINDOW` | 75 | samples per feature window (~1.2 s at 63.5 Hz) |
| `ESP32_FS_HZ` | 63.4635 | ESP-32 sampling rate; must match firmware |
| `PRED_HORIZON_S` | 0.10 | how far in the future to predict |
| `RF_TREES` / `RF_MAX_DEPTH` | 120 / 20 | RandomForest size |
| `SEQ_HIDDEN` / `SEQ_LAYERS` | 80 / 3 | LSTM / GRU shape |
| `REPLAY_MAX` | 2000 | sequence-net replay buffer cap |
| `ANGLE_MIN_DEG` / `ANGLE_MAX_DEG` | 90 / 180 | servo clamp |
