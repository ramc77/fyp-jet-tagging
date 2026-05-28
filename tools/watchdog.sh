#!/bin/bash
#
# Watchdog for run_full_pipeline.py / run_study.py on macOS.
#
# Stops the pipeline (with SIGINT, so per-epoch checkpoints flush cleanly)
# on EITHER of:
#   - battery percentage < BATTERY_MIN          (default 10)
#   - CPU_Speed_Limit < SPEED_MIN for N_THROTTLE consecutive ticks
#                                               (default <90 for 3 ticks)
#
# Notes:
#   * `sysctl machdep.xcpm.cpu_thermal_level` is NOT degrees Celsius —
#     it is an opaque kernel pressure index (0..100). We log it for
#     diagnostics but DO NOT use it as a stop condition.
#   * `CPU_Speed_Limit < 100` means the kernel itself has decided to
#     throttle the CPU. That is the real "running too hot" signal.
#     We require it to stay throttled for several ticks to avoid
#     stopping on a momentary blip.
#
# Usage:
#   ./tools/watchdog.sh <PID>
#   BATTERY_MIN=15 SPEED_MIN=85 N_THROTTLE=5 ./tools/watchdog.sh $PID
#
set -u
PID="${1:?usage: watchdog.sh <pid>}"
BATTERY_MIN="${BATTERY_MIN:-10}"
SPEED_MIN="${SPEED_MIN:-90}"
N_THROTTLE="${N_THROTTLE:-3}"
INTERVAL="${INTERVAL:-60}"

if ! ps -p "$PID" >/dev/null 2>&1; then
    echo "[watchdog] no process with PID $PID — exiting." >&2
    exit 1
fi

echo "[watchdog] watching PID=$PID  batt<${BATTERY_MIN}%  "\
"speed<${SPEED_MIN} for ${N_THROTTLE} consecutive ticks  every ${INTERVAL}s"
echo "[watchdog] (thermal pressure index is reported for info only,"\
"not used as a stop trigger — it is NOT degrees Celsius.)"

throttle_streak=0

while ps -p "$PID" >/dev/null 2>&1; do
    BATT=$(pmset -g batt | awk '/InternalBattery/ {
        match($0, /[0-9]+%/); if (RSTART) print substr($0, RSTART, RLENGTH-1)}' | head -1)
    THERM_IDX=$(sysctl -n machdep.xcpm.cpu_thermal_level 2>/dev/null || echo 0)
    SPEED=$(pmset -g therm | awk '/CPU_Speed_Limit/ {print $NF}' | tail -1)
    SPEED="${SPEED:-100}"
    BATT="${BATT:-100}"

    TS=$(date "+%F %T")

    if [ "$SPEED" -lt "$SPEED_MIN" ]; then
        throttle_streak=$((throttle_streak + 1))
    else
        throttle_streak=0
    fi

    echo "[watchdog $TS] batt=${BATT}%  speed=${SPEED}  "\
"throttle_streak=${throttle_streak}  (thermal_idx=${THERM_IDX} info-only)"

    if [ "$BATT" -lt "$BATTERY_MIN" ]; then
        echo "[watchdog $TS] battery $BATT% < $BATTERY_MIN% — SIGINT to PID $PID"
        kill -INT "$PID"
        exit 0
    fi
    if [ "$throttle_streak" -ge "$N_THROTTLE" ]; then
        echo "[watchdog $TS] kernel throttling for ${throttle_streak} ticks "\
"(speed=$SPEED < $SPEED_MIN) — SIGINT to PID $PID"
        kill -INT "$PID"
        exit 0
    fi

    sleep "$INTERVAL"
done

echo "[watchdog] PID $PID has exited; watchdog stops."
