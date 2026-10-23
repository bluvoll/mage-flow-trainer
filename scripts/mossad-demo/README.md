# MossAD-Demo pilot scripts (MI300X droplet)

Not part of the trainer: the unattended pipeline for the MossAD-Demo MOSS-Image pilot.

- `pipeline.py`: waits for the speed test, picks batch sizes, runs the LR test (A 4e-5, B 2e-5, 10 epochs each at
  512 px), then continues the winner (A unless `/root/work/pipeline/winner` says B) through 512 -> 768 -> 1024.
  Deadline watchdog: `PIPELINE_DEADLINE` (default 2026-10-23T23:00Z), save-and-quit 45 min before.
- `upload_loop.sh`: uploads checkpoints (no optimizer state), samples and configs to `RicemanT/MossAD-Demo`.
- `status.sh`: one-glance progress.

## What the pilot found (Oct 20-22, 2026)

3,685 images, 57 artists (35-75 images each), one MI300X. The run that worked, and the recipe the Studio's training
export now writes (`moss-512/768/1024.toml`):

| Stage | Peak LR | Epochs | Effective batch | Views, median artist | Result |
|---|---|---|---|---|---|
| 512 | 4e-5 | 70 | 512 (128 x 4) | ~4,550 | ~70-75% style, hands shaky |
| 768 | 2e-5 | 40 | 256 (64 x 4) | ~2,600 | ~85-90% style, carries to other subjects |
| 1024 | 1e-5 | 14 | 128 (32 x 4) | ~910 | ~90% style, solid hands |

- Halve the peak LR at each higher resolution: 768 at 4e-5 (the effective batch had halved too) was as unstable
  as the end of 512 by epoch 5 and was restarted at 2e-5.
- About 7k views per artist give a style that transfers across subjects; about 4k (an artist that missed its
  repeat) only showed in the artist's own subject domain. Train from the Studio's `folders.csv`, whose repeats bring
  small artists up to the others.
- ROCm: eager mode with `torch_varlen` (compile with varlen crashes, compile with SDPA stalls), SDNQ patched for
  the gfx arch string, `MIOPEN_FIND_MODE=FAST` before latent caching. Text caching re-encodes every caption per
  resolution (~95 min each here): a known design flaw of the per-crop caption pools.
- The pipeline's run names follow the LR test ("lr4e5"), so the restarted stages ran directly from configs
  (`MossAD-Demo-768-lr2e5`, `MossAD-Demo-1024-lr1e5`); `plan.json` can now also set a stage's peak LR.
