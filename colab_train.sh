#!/usr/bin/env bash
# colab_train.sh — run train_offline.py on a Google Colab GPU VM.
#
# What it does:
#   1) provisions a fresh Colab session (default GPU=A100)
#   2) uploads live_common.py + train_offline.py + each given session CSV
#      (and its matching _emg_raw.csv)
#   3) installs deps on the VM (torch, sklearn, scipy, numpy)
#   4) runs train_offline.py with the args you pass after `--`
#   5) downloads the newest models/bundle_*.pkl back to ./models/
#   6) stops the VM (use --no-stop to keep it running)
#
# Usage:
#   ./colab_train.sh [OPTIONS] SESSION_CSV [SESSION_CSV ...] -- [train_offline args]
#
# Options:
#   --gpu GPU         T4, L4, A100, H100. Default: T4
#   --session NAME    Colab session name. Default: emg
#   --no-stop         Leave the VM running after training
#   -h, --help        Show this help
#
# Examples:
#   ./colab_train.sh logs/liveTrain/sess.csv -- --epochs 50 --horizon 0.15
#   ./colab_train.sh --gpu H100 sess1.csv sess2.csv -- --horizons "0.05,0.10,0.15"

set -euo pipefail

SESSION="emg"
GPU="T4"
KEEP=0
POSITIONAL=()
EXTRA_ARGS=()

usage() {
  awk '/^# /{print substr($0,3)} /^set -euo/{exit}' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)     GPU="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --no-stop) KEEP=1; shift ;;
    -h|--help) usage 0 ;;
    --)        shift; EXTRA_ARGS=("$@"); break ;;
    -*)        echo "unknown flag: $1" >&2; usage 1 ;;
    *)         POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ ${#POSITIONAL[@]} -eq 0 ]]; then
  echo "error: need at least one SESSION_CSV path" >&2
  usage 1
fi

if ! command -v colab >/dev/null 2>&1; then
  echo "colab CLI not found. Install with: uv tool install google-colab-cli" >&2
  exit 1
fi

for f in live_common.py train_offline.py; do
  [[ -f "$f" ]] || { echo "missing local file: $f" >&2; exit 1; }
done

REMOTE_DIR="/content/emg"

echo ">> provisioning Colab VM: $SESSION (GPU=$GPU)"
colab new -s "$SESSION" --gpu "$GPU"

cleanup() {
  if [[ "$KEEP" -eq 0 ]]; then
    echo ">> stopping Colab VM: $SESSION"
    colab stop -s "$SESSION" || true
  else
    echo ">> leaving Colab VM running (--no-stop). Stop with: colab stop -s $SESSION"
  fi
}
trap cleanup EXIT

echo ">> creating remote dirs"
cat <<PY | colab exec -s "$SESSION"
import os
os.makedirs('$REMOTE_DIR/logs/liveTrain', exist_ok=True)
os.makedirs('$REMOTE_DIR/models', exist_ok=True)
print('ok')
PY

echo ">> uploading code"
colab upload -s "$SESSION" live_common.py "$REMOTE_DIR/live_common.py"
colab upload -s "$SESSION" train_offline.py "$REMOTE_DIR/train_offline.py"

echo ">> installing deps"
colab install -s "$SESSION" scikit-learn scipy numpy torch

echo ">> uploading sessions"
REMOTE_SESSIONS=()
for csv in "${POSITIONAL[@]}"; do
  if [[ ! -f "$csv" ]]; then
    echo "not found: $csv" >&2; exit 1
  fi
  raw="${csv%.csv}_emg_raw.csv"
  if [[ ! -f "$raw" ]]; then
    echo "matching raw not found: $raw" >&2; exit 1
  fi
  base="$(basename "$csv")"
  base_raw="$(basename "$raw")"
  colab upload -s "$SESSION" "$csv"  "$REMOTE_DIR/logs/liveTrain/$base"
  colab upload -s "$SESSION" "$raw"  "$REMOTE_DIR/logs/liveTrain/$base_raw"
  REMOTE_SESSIONS+=("logs/liveTrain/$base")
done

# Build a JSON list of args (sessions + extras) for the remote python.
ARGS_JSON=$(python3 -c '
import json, sys
print(json.dumps(sys.argv[1:]))
' "${REMOTE_SESSIONS[@]}" "${EXTRA_ARGS[@]}")

echo ">> running train_offline.py with args: $ARGS_JSON"
cat <<PY | colab exec -s "$SESSION"
import os, subprocess, sys, json
os.chdir('$REMOTE_DIR')
args = json.loads(r'''$ARGS_JSON''')
cmd = [sys.executable, 'train_offline.py'] + args
print('>>>', ' '.join(cmd))
sys.stdout.flush()
sys.exit(subprocess.run(cmd).returncode)
PY

echo ">> locating produced bundle"
BUNDLE_LINE=$(cat <<PY | colab exec -s "$SESSION" 2>/dev/null | grep '^BUNDLE_PATH=' | tail -1
import glob
paths = sorted(glob.glob('$REMOTE_DIR/models/bundle_*.pkl'))
print('BUNDLE_PATH=' + (paths[-1] if paths else ''))
PY
)
BUNDLE_REMOTE="${BUNDLE_LINE#BUNDLE_PATH=}"

if [[ -z "$BUNDLE_REMOTE" ]]; then
  echo ">> no bundle produced (sweep mode? or training failed)"
else
  mkdir -p models
  LOCAL_BUNDLE="models/$(basename "$BUNDLE_REMOTE")"
  echo ">> downloading $BUNDLE_REMOTE -> $LOCAL_BUNDLE"
  colab download -s "$SESSION" "$BUNDLE_REMOTE" "$LOCAL_BUNDLE"
  echo "done: $LOCAL_BUNDLE"
fi
