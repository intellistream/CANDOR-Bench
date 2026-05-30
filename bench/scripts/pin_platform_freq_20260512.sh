#!/usr/bin/env bash
# Pin platform frequencies to remove hardware-confound side effects from
# search-latency measurements. Run as root once before any latency-sensitive
# experiment. Idempotent.
set -euo pipefail
if [[ "$EUID" -ne 0 ]]; then
  echo "must be run as root: sudo $0" >&2; exit 1
fi

echo "[1] Pin CPU governor to performance on all CPUs"
for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
  echo performance > "$c" || true
done

echo "[2] Pin uncore freq to its max on all packages"
for pkg in /sys/devices/system/cpu/intel_uncore_frequency/package_*/; do
  max=$(cat "$pkg/max_freq_khz")
  echo "$max" > "$pkg/min_freq_khz"
  echo "$pkg pinned to $(cat $pkg/min_freq_khz) / $(cat $pkg/max_freq_khz) kHz"
done

echo "[3] Disable deep C-states (keep only POLL + C1) via cpu_dma_latency"
# Holding /dev/cpu_dma_latency open with value 0 forces all CPUs to stay in C0/C1.
# We use a background process that opens it and stays alive.
PIDFILE=/var/run/anns_cstate_pin.pid
if [[ -f $PIDFILE ]] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
  echo "cstate pin already active PID=$(cat $PIDFILE)"
else
  (python3 -c '
import os, struct, time, signal
fd = os.open("/dev/cpu_dma_latency", os.O_WRONLY)
os.write(fd, struct.pack("i", 0))
signal.pause()
' &
   echo $! > $PIDFILE)
  sleep 0.5
  echo "cstate pin PID=$(cat $PIDFILE)"
fi

echo "[4] Verify"
echo "  governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "  uncore pkg0 cur min/max: $(cat /sys/devices/system/cpu/intel_uncore_frequency/package_00_die_00/min_freq_khz) / $(cat /sys/devices/system/cpu/intel_uncore_frequency/package_00_die_00/max_freq_khz)"
echo "  uncore pkg1 cur min/max: $(cat /sys/devices/system/cpu/intel_uncore_frequency/package_01_die_00/min_freq_khz) / $(cat /sys/devices/system/cpu/intel_uncore_frequency/package_01_die_00/max_freq_khz)"
echo "[done]"
