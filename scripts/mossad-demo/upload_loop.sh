#!/bin/bash
# Uploads each finished checkpoint (.safetensors + its config .json, never optimizer state), the run's samples and
# the configs to the HF repo, one folder per run: <resolution>-<lr>/.
REPO=${REPO:-RicemanT/MossAD-Demo}
OUT=/root/work/output
DONE=$OUT/.uploaded
mkdir -p "$OUT" && touch "$DONE"
source /root/work/venv/bin/activate
echo "watching $OUT/<run>/ for new checkpoints -> $REPO"
while true; do
  for f in "$OUT"/*/*.safetensors; do
    [ -e "$f" ] || continue
    name=$(basename "$f"); run=$(basename "$(dirname "$f")")
    case "$run" in MossAD-Demo-*) ;; *) continue ;; esac
    grep -qxF "$name" "$DONE" && continue
    [ $(( $(date +%s) - $(stat -c %Y "$f") )) -lt 120 ] && continue      # still being written
    stage=${run#MossAD-Demo-}
    if hf upload "$REPO" "$f" "$stage/$name" > /dev/null; then
      echo "$name" >> "$DONE"; echo "$(date +%H:%M) uploaded $stage/$name"
      [ -f "${f%.safetensors}.json" ] && hf upload "$REPO" "${f%.safetensors}.json" "$stage/${name%.safetensors}.json" > /dev/null
      [ -d "$OUT/$run/samples" ] && hf upload "$REPO" "$OUT/$run/samples" "$stage/samples" > /dev/null && echo "$(date +%H:%M) samples synced"
      hf upload "$REPO" /root/work/configs configs > /dev/null && echo "$(date +%H:%M) configs synced"
    fi
  done
  sleep 60
done
