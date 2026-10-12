#!/usr/bin/env bash
# Start a MiniCPM-o full-duplex server, warm it with one session, run one open-loop load cell, then stop it.
# Run from the repository root with the serving environment active. Arguments after -- go to the server:
#   benchmarks/duplex/load/run_cell.sh --model-path MODEL --sessions-dir SESSIONS --warmup-wav QUESTION.wav \
#     --output-dir runs/main-s16 --sessions 16 --profile-at 35 --profile-window 40 -- --mps off --max_sessions 32
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
model_path=""; sessions_dir=""; warmup_wav=""; output_dir=""
sessions=16; mean_gap_s=3; seed=11; port=18975
profile_at=""; profile_window=40; torch_trace=false
ready_timeout_s=1500
while [ $# -gt 0 ]; do
  case $1 in
    --model-path) model_path=$2; shift 2 ;;
    --sessions-dir) sessions_dir=$2; shift 2 ;;
    --warmup-wav) warmup_wav=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --sessions) sessions=$2; shift 2 ;;
    --mean-gap-s) mean_gap_s=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    --profile-at) profile_at=$2; shift 2 ;;
    --profile-window) profile_window=$2; shift 2 ;;
    --torch-trace) torch_trace=true; shift ;;
    --) shift; break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
if [ -z "$model_path" ] || [ -z "$sessions_dir" ] || [ -z "$output_dir" ]; then
  echo "need --model-path, --sessions-dir and --output-dir" >&2; exit 2
else
  :
fi
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
wavs=("$sessions_dir"/s*.wav)
warmup_wav=${warmup_wav:-${wavs[0]}}

if $torch_trace; then export SGLANG_TORCH_PROFILER_WITH_STACK=1; else :; fi
python -u -m sglang_omni.cli serve --model-path "$model_path" --variant session --enable-realtime \
  --host 127.0.0.1 --port "$port" "$@" > "$output_dir/server.log" 2>&1 &
server=$!
smi=""
trap 'kill $server $smi 2>/dev/null || true' EXIT

# The launcher can move the port, so read the one uvicorn reports.
start=$(date +%s)
while :; do
  bound=$(grep -o "Uvicorn running on http://127.0.0.1:[0-9]*" "$output_dir/server.log" | tail -n 1 | grep -o "[0-9]*$" || true)
  if [ -n "$bound" ] && curl -sf "http://127.0.0.1:$bound/v1/realtime/capabilities" | grep -q '"native_full_duplex": *true'; then
    break
  else
    :
  fi
  kill -0 $server 2>/dev/null || { echo "server exited, see $output_dir/server.log" >&2; exit 1; }
  [ $(( $(date +%s) - start )) -lt $ready_timeout_s ] || { echo "server not ready after $ready_timeout_s s" >&2; exit 1; }
  sleep 5
done
echo "ready on port $bound after $(( $(date +%s) - start )) s"

python -u "$here/client.py" --port "$bound" --output-dir "$output_dir/warmup" --wav "$warmup_wav" > "$output_dir/warmup.log"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 2000 > "$output_dir/smi.csv" 2>&1 &
smi=$!
python -u "$here/client.py" --port "$bound" --output-dir "$output_dir/cell" --wav "${wavs[@]}" \
  --sessions "$sessions" --mean-gap-s "$mean_gap_s" --seed "$seed" > "$output_dir/cell.log" &
client=$!
if [ -n "$profile_at" ]; then
  sleep "$profile_at"
  curl -sf -X POST "http://127.0.0.1:$bound/start_profile" -H 'Content-Type: application/json' -d \
    "{\"run_id\": \"window\", \"event_dir\": \"$output_dir/events\", \"enable_torch\": $torch_trace, \"trace_path_template\": \"$output_dir/trace-{run_id}/{stage}/trace\"}" > /dev/null
  sleep "$profile_window"
  curl -sf -X POST "http://127.0.0.1:$bound/stop_profile" -H 'Content-Type: application/json' -d '{"run_id": "window"}' > /dev/null
else
  :
fi
wait $client
cat "$output_dir/cell.log"
echo "server tracebacks: $(grep -ci traceback "$output_dir/server.log" || true)"
