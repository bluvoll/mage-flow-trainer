#!/bin/bash
# One clean look at the pipeline: its last events, the current run's last steps, the last uploads.
echo "== pipeline =="; tail -6 /root/work/pipeline/status.txt 2>/dev/null || echo "(not started)"
log=$(ls -t /root/work/logs/*.log 2>/dev/null | head -1)
if [ -n "$log" ]; then
  echo "== $(basename "$log") =="
  tr '\r' '\n' < "$log" | grep -E "step [0-9]+/[0-9]+|samples |saved |save_quit|Error|Traceback|out of memory" | tail -4
fi
echo "== uploaded =="; tail -3 /root/work/output/.uploaded 2>/dev/null || echo "(none yet)"
echo "== disk =="; df -h / | tail -1
