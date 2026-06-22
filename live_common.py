"""live_common.py — shared model definitions, EMG filter, features, and
save/load used by live_train.py, live_deploy.py, and train_offline.py.

The single source of truth for:
  - the gravity-EMG filter chain (DC -> BP -> notch -> rectify -> envelope)
  - the 16-dim feature vector layout (4 features x 4 channels: m1, m2, m3, grav)
  - the EMA normalizer state (features / raw / target)
  - all model classes (SGD, RLS, RF, LSTM, GRU, ENS)
  - bundle serialization (save_bundle / load_bundle)

Anything that affects the model contract lives here so the deploy script
gets exactly the same inputs the training script produced.
"""

import math
import pickle
import threading
from collections import deque

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

try:
    from scipy.signal import butter, iirnotch, sosfilt, lfilter
    HAS_SCIPY = True
except Exception as _e:
    HAS_SCIPY = False
    print(f"[live_common] WARNING: scipy unavailable — gravity uses fallback. ({_e})")

try:
    import torch
    import torch.nn as nn
    _probe = torch.from_numpy(np.zeros((1, 1), dtype=np.float32))  # noqa: F841
    HAS_TORCH = True
except Exception as _e:
    HAS_TORCH = False
    print(f"[live_common] WARNING: torch unavailable — LSTM/GRU disabled. ({_e})")


# ====== I/O CONTRACT (must match esp32_online.ino + recordings) =============
N_CH         = 4
CH_NAMES     = ["m1", "m2", "m3", "grav"]
GRAV_IDX     = 3
EMG_WINDOW   = 75
ESP32_FS_HZ  = 63.4635

# ====== GRAVITY FILTER ======================================================
GRAV_MAINS     = 60.0
GRAV_HARMONICS = 2
GRAV_BP_LO     = 20.0
GRAV_BP_HI     = 450.0
GRAV_ENV_CUT   = 6.0
GRAV_DC_ALPHA  = 1.0 / 500

# ====== TRAINING HYPERPARAMETERS ============================================
PRED_HORIZON_S = 0.10
ROLL_WINDOW    = 300

FEAT_EMA_ALPHA = 1.0 / 200
RAW_EMA_ALPHA  = 1.0 / 2000
TGT_EMA_ALPHA  = 1.0 / 300

SGD_ETA0       = 0.005
RETRAIN_EVERY  = 10

RLS_LAMBDA = 0.999
RLS_P0     = 1e3

RF_TREES       = 60
RF_MAX_DEPTH   = 12
RF_REFIT_EVERY = 25
RF_REPLAY_MAX  = 1200
RF_MIN_SAMPLES = 40

SEQ_HIDDEN          = 48
SEQ_LAYERS          = 2
SEQ_LR              = 3e-3
SEQ_WD              = 1e-4
SEQ_BATCH           = 32
SEQ_STEPS_PER_TICK  = 3
REPLAY_MAX          = 1500

# Display colors (BGR)
COL_TRUE = (120, 220, 255)
COL_SGD  = (60, 160, 255)
COL_RLS  = (200, 120, 255)
COL_RF   = (70, 220, 220)
COL_LSTM = (90, 230, 110)
COL_GRU  = (255, 170, 90)
COL_ENS  = (245, 245, 245)
COL_LABEL = (240, 200, 80)


# ====== GRAVITY EMG STREAM FILTER ==========================================
class GravityEMGProcessor:
    """Online sEMG chain for the Gravity Analog EMG sensor.

      1. DC-offset removal (slow EMA)
      2. band-pass 20-450 Hz   (causal SOS IIR, persistent state)
      3. mains notch 60/50 Hz  (+ harmonics, if below Nyquist)
      4. full-wave rectify
      5. linear envelope LP
    """

    def __init__(self, mains=GRAV_MAINS, harmonics=GRAV_HARMONICS,
                 bp_lo=GRAV_BP_LO, bp_hi=GRAV_BP_HI, env_cut=GRAV_ENV_CUT):
        self.mains = mains
        self.harmonics = harmonics
        self.bp_lo = bp_lo
        self.bp_hi = bp_hi
        self.env_cut = env_cut
        self.fs = None
        self.ready = False
        self.dc = None
        self._fallback_env = 0.0
        self.sos_bp = self.zi_bp = None
        self.notch_ba = []
        self.notch_zi = []
        self.sos_env = self.zi_env = None

    def set_fs(self, fs):
        if fs is None or fs <= 0:
            return
        if self.ready and self.fs is not None and abs(fs - self.fs) / self.fs < 0.05:
            return
        self.fs = float(fs)
        if HAS_SCIPY:
            self._build()

    def reset(self):
        self.dc = None
        self._fallback_env = 0.0
        if HAS_SCIPY and self.fs is not None:
            self._build()

    def _build(self):
        nyq = self.fs / 2.0
        hi = min(self.bp_hi, nyq * 0.99)
        lo = min(self.bp_lo, hi * 0.5)
        self.sos_bp = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        self.zi_bp = np.zeros((self.sos_bp.shape[0], 2))
        self.notch_ba = []
        self.notch_zi = []
        f = self.mains
        h = self.harmonics
        while f < nyq and h > 0:
            b, a = iirnotch(f / nyq, 30.0)
            self.notch_ba.append((b, a))
            self.notch_zi.append(np.zeros(max(len(a), len(b)) - 1))
            f += self.mains
            h -= 1
        self.sos_env = butter(4, self.env_cut / nyq, btype="low", output="sos")
        self.zi_env = np.zeros((self.sos_env.shape[0], 2))
        self.ready = True

    def push(self, x):
        x = float(x)
        if self.dc is None:
            self.dc = x
        else:
            self.dc = (1 - GRAV_DC_ALPHA) * self.dc + GRAV_DC_ALPHA * x
        centered = x - self.dc
        if not (HAS_SCIPY and self.ready):
            self._fallback_env = 0.9 * self._fallback_env + 0.1 * abs(centered)
            return self._fallback_env
        y, self.zi_bp = sosfilt(self.sos_bp, [centered], zi=self.zi_bp)
        v = y[0]
        for i, (b, a) in enumerate(self.notch_ba):
            yv, self.notch_zi[i] = lfilter(b, a, [v], zi=self.notch_zi[i])
            v = yv[0]
        r = abs(v)
        e, self.zi_env = sosfilt(self.sos_env, [r], zi=self.zi_env)
        return float(e[0])


# ====== FEATURES ===========================================================
def features_from_arr(arr):
    """arr: [T, >=1+N_CH] — col 0 = timestamp, cols 1..N_CH = m1, m2, m3, grav_env.

    Returns a 16-dim feature vector: per channel,
      m1/m2/m3: RMS, MAV, VAR, WL of the mean-removed raw signal.
      grav    : LEVEL, MAV, VAR, WL of the envelope (level preserved).
    """
    out = []
    for ci in range(1, 1 + N_CH):
        x = arr[:, ci]
        if (ci - 1) == GRAV_IDX:
            level = float(np.mean(x))
            mav = float(np.mean(np.abs(x)))
            var = float(np.var(x))
            wl = float(np.sum(np.abs(np.diff(x))))
            out.extend([level, mav, var, wl])
        else:
            xc = x - x.mean()
            rms = float(np.sqrt(np.mean(xc * xc)))
            mav = float(np.mean(np.abs(xc)))
            var = float(np.var(xc))
            wl = float(np.sum(np.abs(np.diff(xc))))
            out.extend([rms, mav, var, wl])
    return np.array(out, dtype=float)


# ====== EMA NORMALIZERS (single object passed around) ======================
class Normalizers:
    def __init__(self, n_ch=N_CH):
        self.feat_mean = None
        self.feat_var = None
        self.feat_n = 0
        self.raw_mean = np.zeros(n_ch, dtype=float)
        self.raw_var = np.ones(n_ch, dtype=float) * 10000.0
        self.tgt_mean = None
        self.tgt_var = None

    def update_feat(self, x, alpha=FEAT_EMA_ALPHA):
        self.feat_n += 1
        if self.feat_mean is None:
            self.feat_mean = x.copy()
            self.feat_var = np.ones_like(x)
            return
        self.feat_mean = (1 - alpha) * self.feat_mean + alpha * x
        self.feat_var = (1 - alpha) * self.feat_var + \
            alpha * (x - self.feat_mean) ** 2

    def norm_feat(self, x):
        if self.feat_mean is None:
            return x
        std = np.sqrt(self.feat_var) + 1e-6
        return (x - self.feat_mean) / std

    def update_raw(self, window_2d, alpha=RAW_EMA_ALPHA):
        m = window_2d.mean(axis=0)
        v = window_2d.var(axis=0)
        self.raw_mean = (1 - alpha) * self.raw_mean + alpha * m
        self.raw_var = (1 - alpha) * self.raw_var + alpha * v

    def norm_raw(self, window_2d):
        std = np.sqrt(self.raw_var) + 1e-6
        return (window_2d - self.raw_mean) / std

    def update_tgt(self, y, alpha=TGT_EMA_ALPHA):
        y = float(y)
        if self.tgt_mean is None:
            self.tgt_mean = y
            self.tgt_var = 1.0
            return
        self.tgt_mean = (1 - alpha) * self.tgt_mean + alpha * y
        self.tgt_var = (1 - alpha) * self.tgt_var + \
            alpha * (y - self.tgt_mean) ** 2

    @property
    def t_mean(self):
        return 0.0 if self.tgt_mean is None else self.tgt_mean

    @property
    def t_std(self):
        if self.tgt_var is None:
            return 1.0
        return math.sqrt(self.tgt_var) + 1e-6

    def to_dict(self):
        return {
            "feat_mean": self.feat_mean, "feat_var": self.feat_var,
            "feat_n": self.feat_n,
            "raw_mean": self.raw_mean, "raw_var": self.raw_var,
            "tgt_mean": self.tgt_mean, "tgt_var": self.tgt_var,
        }

    @classmethod
    def from_dict(cls, d):
        n = cls()
        n.feat_mean = d.get("feat_mean")
        n.feat_var = d.get("feat_var")
        n.feat_n = int(d.get("feat_n", 0))
        if d.get("raw_mean") is not None:
            n.raw_mean = np.asarray(d["raw_mean"], dtype=float)
        if d.get("raw_var") is not None:
            n.raw_var = np.asarray(d["raw_var"], dtype=float)
        n.tgt_mean = d.get("tgt_mean")
        n.tgt_var = d.get("tgt_var")
        return n


# ====== METRICS ============================================================
class RunningRegMetrics:
    def __init__(self):
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.y_sum = 0.0
        self.y_sum_sq = 0.0

    def update(self, y_true, y_pred):
        self.n += 1
        e = y_pred - y_true
        self.sum_abs += abs(e)
        self.sum_sq += e * e
        self.y_sum += y_true
        self.y_sum_sq += y_true * y_true

    def get(self):
        if self.n < 2:
            return 0.0, 0.0, 0.0
        mae = self.sum_abs / self.n
        rmse = math.sqrt(self.sum_sq / self.n)
        y_mean = self.y_sum / self.n
        ss_tot = self.y_sum_sq - self.n * y_mean * y_mean
        r2 = 1.0 - (self.sum_sq / ss_tot) if ss_tot > 1e-9 else 0.0
        return mae, rmse, r2


class RollingRegMetrics:
    def __init__(self, n=ROLL_WINDOW):
        self.yt = deque(maxlen=n)
        self.yp = deque(maxlen=n)

    def update(self, y_true, y_pred):
        self.yt.append(float(y_true))
        self.yp.append(float(y_pred))

    @property
    def n(self):
        return len(self.yt)

    def get(self):
        if len(self.yt) < 2:
            return 0.0, 0.0, 0.0
        yt = np.array(self.yt); yp = np.array(self.yp)
        e = yp - yt
        mae = float(np.mean(np.abs(e)))
        rmse = float(np.sqrt(np.mean(e * e)))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - float(np.sum(e * e)) / ss_tot if ss_tot > 1e-9 else 0.0
        return mae, rmse, r2

    def mae(self):
        return self.get()[0]


# ====== MODEL CLASSES ======================================================
HIST_LEN = 300


class BaseModel:
    kind = "feat"

    def __init__(self, name, color=(255, 255, 255), thick=2):
        self.name = name
        self.color = color
        self.thick = thick
        self.metrics = RunningRegMetrics()
        self.roll = RollingRegMetrics()
        self.hist = deque(maxlen=HIST_LEN)
        self.err_hist = deque(maxlen=HIST_LEN)
        self.smoothed = None
        self.last_pred = None
        self.trained = False
        self.extra = ""

    def update(self, x, y): pass
    def predict(self, x): return None

    def serialize(self):
        return {"trained": self.trained}

    def load_state(self, data):
        self.trained = bool(data.get("trained", False))


class SGDModel(BaseModel):
    kind = "feat"

    def __init__(self, name=("SGD"), color=COL_SGD, thick=2):
        super().__init__(name, color, thick)
        self.reg = SGDRegressor(alpha=1e-4, learning_rate="constant",
                                eta0=SGD_ETA0)
        self.pX = []
        self.py = []

    def update(self, x, y):
        self.pX.append(x)
        self.py.append(float(y))
        if len(self.pX) >= RETRAIN_EVERY:
            try:
                self.reg.partial_fit(np.array(self.pX), np.array(self.py))
                self.trained = True
            except Exception as e:
                print(f"sgd partial_fit: {e}")
            self.pX.clear()
            self.py.clear()

    def predict(self, x):
        if not self.trained:
            return None
        try:
            return float(self.reg.predict(x.reshape(1, -1))[0])
        except Exception:
            return None

    def serialize(self):
        return {"trained": self.trained,
                "reg": self.reg if self.trained else None}

    def load_state(self, data):
        super().load_state(data)
        if data.get("reg") is not None:
            self.reg = data["reg"]


class RLSModel(BaseModel):
    kind = "feat"

    def __init__(self, name="RLS", color=COL_RLS, thick=2):
        super().__init__(name, color, thick)
        self.w = None
        self.P = None
        self.lam = RLS_LAMBDA

    def _ensure(self, d):
        if self.w is None:
            self.w = np.zeros(d + 1)
            self.P = np.eye(d + 1) * RLS_P0

    def update(self, x, y):
        self._ensure(len(x))
        xb = np.append(x, 1.0)
        Px = self.P @ xb
        denom = self.lam + float(xb @ Px)
        if denom < 1e-9:
            return
        k = Px / denom
        e = float(y) - float(self.w @ xb)
        self.w = self.w + k * e
        self.P = (self.P - np.outer(k, Px)) / self.lam
        self.trained = True

    def predict(self, x):
        if not self.trained or self.w is None:
            return None
        return float(self.w @ np.append(x, 1.0))

    def serialize(self):
        return {"trained": self.trained,
                "w": self.w.copy() if self.w is not None else None}

    def load_state(self, data):
        super().load_state(data)
        if data.get("w") is not None:
            self.w = np.asarray(data["w"], dtype=float)


class RFModel(BaseModel):
    kind = "feat"

    def __init__(self, name="RF", color=COL_RF, thick=2):
        super().__init__(name, color, thick)
        self.buf = deque(maxlen=RF_REPLAY_MAX)
        self.model = None
        self.lock = threading.Lock()
        self.fitting = False
        self.count = 0

    def update(self, x, y):
        self.buf.append((x.copy(), float(y)))
        self.count += 1
        if (self.count % RF_REFIT_EVERY == 0 and not self.fitting
                and len(self.buf) >= RF_MIN_SAMPLES):
            self._launch_fit()

    def _launch_fit(self):
        self.fitting = True
        data = list(self.buf)

        def work():
            try:
                X = np.array([d[0] for d in data])
                Y = np.array([d[1] for d in data])
                m = RandomForestRegressor(
                    n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH, n_jobs=1)
                m.fit(X, Y)
                with self.lock:
                    self.model = m
                self.trained = True
                self.extra = f"trees={RF_TREES} n={len(data)}"
            except Exception as e:
                print(f"rf fit: {e}")
            finally:
                self.fitting = False

        threading.Thread(target=work, daemon=True).start()

    def fit_now(self, X, Y):
        """Synchronous fit (used by train_offline)."""
        m = RandomForestRegressor(
            n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH, n_jobs=1)
        m.fit(X, Y)
        with self.lock:
            self.model = m
        self.trained = True
        self.extra = f"trees={RF_TREES} n={len(X)}"

    def predict(self, x):
        with self.lock:
            m = self.model
        if m is None:
            return None
        try:
            return float(m.predict(x.reshape(1, -1))[0])
        except Exception:
            return None

    def serialize(self):
        with self.lock:
            return {"trained": self.trained, "model": self.model}

    def load_state(self, data):
        super().load_state(data)
        if data.get("model") is not None:
            self.model = data["model"]


if HAS_TORCH:
    torch.set_num_threads(1)

    class SeqNet(nn.Module):
        def __init__(self, net_kind, input_size=N_CH,
                     hidden=SEQ_HIDDEN, layers=SEQ_LAYERS):
            super().__init__()
            rnn_cls = nn.LSTM if net_kind == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=input_size, hidden_size=hidden,
                num_layers=layers, batch_first=True,
                dropout=(0.1 if layers > 1 else 0.0),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    class TorchSeqModel(BaseModel):
        kind = "seq"

        def __init__(self, name, color, thick, net_kind):
            super().__init__(name, color, thick)
            self.net_kind = net_kind
            self.net = SeqNet(net_kind)
            self.opt = torch.optim.Adam(self.net.parameters(),
                                        lr=SEQ_LR, weight_decay=SEQ_WD)
            self.loss_fn = nn.SmoothL1Loss()
            self.last_loss = 0.0

        def train_steps(self, replay, norms):
            if len(replay) < SEQ_BATCH:
                return
            mean = norms.t_mean
            std = norms.t_std
            for _ in range(SEQ_STEPS_PER_TICK):
                idx = np.random.randint(0, len(replay), size=SEQ_BATCH)
                xs = np.stack([replay[i][0] for i in idx]).astype(np.float32)
                ys_raw = np.array([replay[i][1] for i in idx], dtype=np.float32)
                ys = ((ys_raw - mean) / std).astype(np.float32)
                xt = torch.from_numpy(xs)
                yt = torch.from_numpy(ys)
                self.net.train()
                self.opt.zero_grad()
                pred = self.net(xt)
                loss = self.loss_fn(pred, yt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                self.opt.step()
                self.last_loss = float(loss.detach().item())
            self.trained = True
            self.extra = f"loss={self.last_loss:.3f}"

        def predict_norm(self, x_norm, norms):
            """Predict on a normalized raw window; un-standardize target."""
            if not self.trained:
                return None
            with torch.no_grad():
                self.net.eval()
                xt = torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(0)
                out = float(self.net(xt).item())
            return out * norms.t_std + norms.t_mean

        # Backwards-compat for live_train (passes norms via a module-level set).
        def predict(self, x_norm):
            return self.predict_norm(x_norm, _ACTIVE_NORMS)

        def serialize(self):
            sd = {k: v.detach().cpu().clone()
                  for k, v in self.net.state_dict().items()}
            return {"trained": self.trained, "state_dict": sd,
                    "net_kind": self.net_kind,
                    "hidden": SEQ_HIDDEN, "layers": SEQ_LAYERS}

        def load_state(self, data):
            super().load_state(data)
            sd = data.get("state_dict")
            if sd is not None:
                self.net.load_state_dict(sd)


# Module-level "active normalizers" so TorchSeqModel.predict(x) can find them
# without rewriting every caller. live_train/live_deploy set this at startup.
_ACTIVE_NORMS = Normalizers()


def set_active_norms(norms):
    global _ACTIVE_NORMS
    _ACTIVE_NORMS = norms


class EnsembleModel(BaseModel):
    kind = "ens"

    def __init__(self, name="ENS", color=COL_ENS, thick=4, members=None):
        super().__init__(name, color, thick)
        self.members = members or []
        self.trained = True
        self.static_mae = None   # populated from bundle at deploy time

    def blend(self, preds):
        items = [(m, preds.get(m.name)) for m in self.members]
        items = [(m, p) for (m, p) in items if p is not None and np.isfinite(p)]
        if not items:
            return None
        if self.static_mae:
            ws = np.array([1.0 / (self.static_mae.get(m.name, 1.0) + 1.0)
                           for m, _ in items])
        else:
            ws = np.array([1.0 / (m.roll.mae() + 1.0) for m, _ in items])
        ws = ws / ws.sum()
        return float(sum(w * p for w, (_, p) in zip(ws, items)))

    def serialize(self):
        member_mae = {}
        for m in self.members:
            if m.roll.n >= 2:
                member_mae[m.name] = float(m.roll.mae())
            else:
                member_mae[m.name] = 1.0
        return {"trained": True, "member_mae": member_mae}

    def load_state(self, data):
        super().load_state(data)
        self.static_mae = dict(data.get("member_mae", {}))


# ====== FACTORY ============================================================
def make_models(include_seq=None):
    """Build a fresh (untrained) model set. include_seq=None auto-detects torch."""
    if include_seq is None:
        include_seq = HAS_TORCH
    models = [SGDModel(), RLSModel(), RFModel()]
    if include_seq and HAS_TORCH:
        models.append(TorchSeqModel("LSTM", COL_LSTM, 3, "lstm"))
        models.append(TorchSeqModel("GRU",  COL_GRU,  3, "gru"))
    ens = EnsembleModel(members=models)
    display = models + [ens]
    return models, ens, display


# ====== SAVE / LOAD ========================================================
BUNDLE_VERSION = 1


def save_bundle(path, models, ensemble, norms, extra_meta=None):
    bundle = {
        "version": BUNDLE_VERSION,
        "meta": {
            "n_ch": N_CH, "grav_idx": GRAV_IDX, "emg_window": EMG_WINDOW,
            "esp32_fs_hz": ESP32_FS_HZ, "pred_horizon_s": PRED_HORIZON_S,
            **(extra_meta or {}),
        },
        "norms": norms.to_dict(),
        "models": {m.name: m.serialize() for m in models},
        "ensemble": ensemble.serialize(),
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if bundle.get("version") != BUNDLE_VERSION:
        print(f"[live_common] bundle version mismatch: "
              f"{bundle.get('version')} vs {BUNDLE_VERSION}")
    norms = Normalizers.from_dict(bundle["norms"])
    models, ensemble, display = make_models(include_seq=HAS_TORCH)
    for m in models:
        data = bundle["models"].get(m.name)
        if data:
            try:
                m.load_state(data)
            except Exception as e:
                print(f"[live_common] load {m.name}: {e}")
    ens_data = bundle.get("ensemble")
    if ens_data:
        ensemble.load_state(ens_data)
    set_active_norms(norms)
    return {
        "meta": bundle["meta"],
        "norms": norms,
        "models": models,
        "ensemble": ensemble,
        "display": display,
    }


def get_pred(model, feat_norm, raw_norm, all_models=None):
    """Dispatch a prediction for one of feat/seq/ens models."""
    if model.kind == "feat":
        return model.predict(feat_norm)
    if model.kind == "seq":
        return model.predict(raw_norm)
    if model.kind == "ens":
        preds = {}
        for m in (all_models or model.members):
            inp = feat_norm if m.kind == "feat" else raw_norm
            preds[m.name] = m.predict(inp)
        return model.blend(preds)
    return None
