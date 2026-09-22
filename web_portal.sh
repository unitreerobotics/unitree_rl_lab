#!/usr/bin/env bash
set -euo pipefail

# Starts both halves of the Go2 control panel together:
#   1. websocketd, exposing go2_ctrl as a websocket on port 8888
#   2. a plain HTTP server for index.html (+ tb.png) on port 8000
#
# Ctrl+C stops both.
#
# IMPORTANT (same caveat as before): websocketd forks a *new* go2_ctrl
# process per websocket connection and kills it when that connection
# closes. --maxforks=1 caps it at one instance at a time, but reloading
# the browser tab still kills the running process -- taking DDS/FSM state
# with it -- and starts a fresh one. Keep the physical controller within
# reach as your real emergency stop.

BIN="./deploy/robots/go2/build/go2_ctrl"
NETWORK_IFACE="eth0"   # match what you'd normally pass as --network
WS_PORT=8888
UI_PORT=8889
UI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # folder this script lives in; adjust if index.html is elsewhere

pids=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting websocketd on port ${WS_PORT} -> ${BIN} --network ${NETWORK_IFACE}"
websocketd --port="${WS_PORT}" --maxforks=1 -- "${BIN}" --network "${NETWORK_IFACE}" &
pids+=($!)

echo "Serving UI from ${UI_DIR} on port ${UI_PORT}"
python3 -m http.server "${UI_PORT}" --directory "${UI_DIR}" &
pids+=($!)

echo ""
echo "Open: http://localhost:${UI_PORT}/index.html"
echo "Press Ctrl+C to stop both."

wait
