# live_train — closed-loop online training, ESP32 servo target

Computer runs the camera GUI + trainer. ESP32 runs the realtime EMG -> features -> linear pred -> servo loop. They talk over a single USB serial line.

## What runs where

| | Computer (`live_train.py`) | ESP32 (`esp32_online.ino`) |
|---|---|---|
| reads EMG | no | yes (3 analog pins) |
| runs filter chain + features | no | yes |
| measures angle | yes (camera + clicked markers) | no |
| runs trained model | no | yes (linear regression on features) |
| drives the servo | no | yes |
| retrains the model | yes (Ridge, every 20s) | no |
| ships weights | yes (over serial) | no |

## Serial protocol (line-delimited ASCII, 921600 baud)

ESP32 -> computer:
- `F,t_us,f0,f1,...,f146\n` — 147-feature vector at 10 Hz.
- `P,t_us,angle_deg\n` — ESP32's current servo target.
- `S,<status>\n` — boot / model-loaded / errors.

Computer -> ESP32:
- `M,w0,w1,...,w146,bias\n` — Ridge weights; ESP32 applies on receipt.

## Setup

1. Wire EMG sensors to ESP32 (defaults: gravity=GPIO32, m1=GPIO34, m2=GPIO35) and the servo signal pin to GPIO18. Edit the top of `esp32_online.ino` if you use different pins.
2. Drop `emg_filters.h` into the same Arduino folder as `esp32_online.ino`.
3. Install Arduino IDE + ESP32 board package + `ESP32Servo` library.
4. Flash `esp32_online.ino`.
5. On the laptop, edit `SERIAL_PORT` at the top of `live_train.py` to your ESP32's port (e.g. `/dev/cu.usbserial-XXXX`).
6. `pip install pyserial opencv-python scikit-learn` if you haven't already.
7. Run: `python live_train.py`.

## Live workflow

1. ESP32 boots, prints `S,boot ...`, holds the servo at `SERVO_REST_DEG`.
2. Filters warm up silently for `SETTLE_S = 4 s`, then ESP32 starts streaming features.
3. Click shin -> ankle -> foot on the Mask window to start angle measurement.
4. Move your foot. The trainer pairs features with measured angles in a sliding buffer.
5. Every `TRAIN_PERIOD_S = 20 s`, the computer fits a Ridge regression, sends weights to the ESP32; ESP32 prints `S,model_loaded_n=K`; servo now tracks predictions.
6. Each subsequent training cycle pairs the *cumulative* buffer (last 2000 features, 1000 angles), so the model keeps improving as more data arrives.

## Notes / knobs

- `MIN_PAIRS = 80` — won't ship a model until that many (feature, angle) pairs are matched. Tune down for faster first upload.
- `MATCH_TOL_S = 0.10` — max time gap between a feature timestamp and angle timestamp to count as a pair.
- The ESP32 expects `EMG_FS_HZ` from the baked filter coefficients in `emg_filters.h` (currently 63.5 Hz, the sample rate of your training data). If you wire a different ADC schedule, regenerate `emg_filters.h` from the notebook's export cell with the new `fs_emg`.
- Ridge weights are sent as ASCII for debuggability (~1.5 KB / upload, < 100 ms over 921600 baud). Switch to packed binary if you ever push to a much lower baud.
- No smoothing on the ESP32 side. If the servo is twitchy in deployment, add an EMA on `yhat` before `servo.write()`.

## Files

- `live_train.py` — computer-side GUI + trainer.
- `esp32_online/esp32_online.ino` — ESP32 firmware.
- `esp32_online/emg_filters.h` — baked filter coefficients (copy of the notebook's export).
