#!/bin/sh
# Waits until this Mac is quiet: CPU use (all cores) and GPU use both under their limits in
# 6 samples in a row, 10 s apart. Prints each sample. Usage: bench/wait-quiet.sh [CPU% [GPU%]]
cpu_max=${1:-15} gpu_max=${2:-10} ok=0
while [ $ok -lt 6 ]; do
  idle=$(top -l 2 -n 0 -s 1 | grep "CPU usage" | tail -1 | sed -E 's/.* ([0-9.]+)% idle.*/\1/')
  cpu=$(echo "100 - $idle" | bc)
  gpu=$(ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)
  if [ "$(echo "$cpu < $cpu_max" | bc)" = 1 ] && [ "${gpu:-0}" -lt "$gpu_max" ]; then ok=$((ok + 1)); else ok=0; fi
  echo "$(date +%T) cpu ${cpu}% gpu ${gpu}% quiet ${ok}/6"
  [ $ok -lt 6 ] && sleep 8
done
exit 0
