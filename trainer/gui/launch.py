"""What the GUI launches, as plain argv/env -- no Qt here.

Split out of `process.py` so the remote job server (`trainer.remote`) builds exactly the same
commands on a GPU box that has no PySide6 installed. `process.py` re-exports everything, so
existing imports keep working.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The repository root, as `widgets.PROJECT_ROOT` computes it (both live in trainer/gui/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Launch:
    """One thing the GUI can run. `argv` is passed to Popen verbatim -- no shell."""

    argv: list[str]
    label: str
    # Marks a run that must NOT be treated as training (no graph reset, no "training finished").
    is_training: bool = True
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Job:
    """A queued `Launch` plus what a non-zero exit should do to the rest of the queue.

    The two callers want opposite things and both are right. Auditing three dataset folders is
    three independent questions -- one failing says nothing about the others, so the queue carries
    on. A training pipeline is one job in several steps: if caching fails the training trains on
    missing latents, and if an arm fails the merge silently produces a LoRA with one parent in it.
    Both of those are worse than stopping, and both are silent, so the default is `stop`.
    """

    launch: Launch
    training: bool = False
    on_failure: str = "stop"      # "stop" cancels the remaining queue; "continue" does not
    config_path: Path | None = None
    config_digest: str | None = None
    queue_index: int | None = None


def _python() -> str:
    return sys.executable


def train_launch(config_path: str | Path, num_processes: int = 1, gpus: str = "") -> Launch:
    """Always through Accelerate, single GPU included -- the same choice sd-scripts makes.

    One launch path means one set of behaviours to reason about: the same `BatchSamplerShard`,
    the same `AcceleratedScheduler` horizon scaling, the same trailing-partial-group sync. A bug
    that only appears under the launcher cannot then hide from single-GPU testing, and a config
    that runs on one card runs on two without a second code path having to agree with the first.

    `python -m accelerate.commands.launch` rather than the `accelerate` console script: the script
    lives in the venv's bin/ and may not be on PATH when the GUI is started from a desktop entry,
    while the module is importable by construction if accelerate is installed at all. Verified to
    give identical results to the console script (MULTI_GPU, correct ranks and devices).
    """
    # `--multi_gpu` explicitly: with a saved Accelerate config that says distributed_type NO (left by
    # `accelerate config` or another tool on a shared box), `--num_processes N` alone is ignored and ONE
    # process trains while the other GPUs idle. Seen on a JupyterHub server.
    argv = [_python(), "-u", "-m", "accelerate.commands.launch",
            *(["--multi_gpu"] if num_processes > 1 else []),
            "--num_processes", str(num_processes),
            "-m", "trainer.training.train", str(config_path)]
    label = f"train ({num_processes} GPU{'s, DDP' if num_processes > 1 else ''})"
    return Launch(argv, label)


def cache_launch(
    dataset_path: str,
    model_path: str,
    resolutions: list[int],
    *,
    min_bucket_reso: int,
    max_bucket_reso: int,
    bucket_reso_steps: int,
    upscale: bool,
    multires_training: bool,
    overwrite: bool = False,
    dry_run: bool = False,
    gpus: str = "",
    vae_path: str | None = None,
    flux2_vae: bool = False,
    model_family: str = "auto",
    cache_validation: str = "warn",
) -> Launch:
    argv = [_python(), "-u", "-m", "trainer.tools.cache_latents", "cache", dataset_path,
            "--model-path", model_path,
            "--resolution", *[str(r) for r in resolutions],
            "--min-bucket-reso", str(min_bucket_reso),
            "--max-bucket-reso", str(max_bucket_reso),
            "--bucket-reso-steps", str(bucket_reso_steps),
            "--cache-validation", cache_validation]
    if vae_path:
        argv.extend(["--vae-path", vae_path])
    if model_family != "auto":
        argv.extend(["--model-family", model_family])
    if flux2_vae:
        argv.append("--flux2-vae")
    if gpus and "," in gpus:
        argv.extend(["--devices", gpus])
    if upscale:
        argv.append("--upscale")
    if multires_training:
        argv.append("--multires")
    if overwrite:
        argv.append("--overwrite")
    if dry_run:
        argv.append("--dry-run")
    tiers = "/".join(str(r) for r in resolutions)
    return Launch(argv, f"cache latents ({tiers})", is_training=False,
                  env={"CUDA_VISIBLE_DEVICES": gpus} if gpus else {})


def cache_config_launch(config_path: str | Path, *, gpus: str = "", dry_run: bool = False,
                        overwrite: bool = False) -> Launch:
    """Cache every folder a config uses -- inline subsets, `subsets_file`, `[eval]` -- in one
    process, with the config's own bucket, VAE and storage-precision settings.

    Replaces one `cache_launch` per folder: each of those loaded the VAE again, which is seconds
    for a handful of folders and hours for a per-artist dataset of thousands.
    """
    argv = [_python(), "-u", "-m", "trainer.tools.cache_latents", "cache-config", str(config_path)]
    if gpus and "," in gpus:
        argv.extend(["--devices", gpus])
    if overwrite:
        argv.append("--overwrite")
    if dry_run:
        argv.append("--dry-run")
    return Launch(argv, "cache latents" + (" (dry run)" if dry_run else ""), is_training=False,
                  env={"CUDA_VISIBLE_DEVICES": gpus} if gpus else {})


def concat_launch(output: str | Path, parents: list[tuple[str, float]]) -> Launch:
    """Combine finished LoRAs into their exact weighted sum (see `trainer.tools.concat_lora`).

    Not a merge algorithm with knobs -- stacking the factors reproduces `sum_i w_i * dW_i` exactly,
    so there is nothing to tune and nothing to get subtly wrong. That is why the pipeline can end
    with it unattended.
    """
    argv = [_python(), "-u", "-m", "trainer.tools.concat_lora", str(output)]
    argv += [f"{path}:{weight}" for path, weight in parents]
    return Launch(argv, f"concat {len(parents)} LoRA(s)", is_training=False)


def audit_launch(dataset_path: str, bucket_reso_steps: int) -> Launch:
    """Reports the source-size distribution and proposes a resolution ladder. Reads nothing but
    image headers, so it needs no GPU and no model."""
    return Launch(
        [_python(), "-u", "-m", "trainer.tools.cache_latents", "audit", dataset_path,
         "--bucket-reso-steps", str(bucket_reso_steps)],
        "audit dataset", is_training=False,
    )


_DEVICE_LIST_RE = re.compile(r"^\d+(,\d+)*$")


def training_env(gpus: str = "") -> dict[str, str]:
    """Environment for a spawned run: the repo on PYTHONPATH so `-m trainer.…` resolves however the
    GUI itself was started.

    `gpus` must be a bare device list. Torch treats an unparseable `CUDA_VISIBLE_DEVICES` as "no
    devices" rather than as an error, so passing something like `all` through here silently moves
    a 2B model onto the CPU -- which is how this check came to exist. Anything that is not
    digits-and-commas is rejected loudly instead.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (f"{PROJECT_ROOT}{os.pathsep}{existing}" if existing else str(PROJECT_ROOT))
    gpus = (gpus or "").strip()
    if gpus:
        if not _DEVICE_LIST_RE.match(gpus):
            raise ValueError(
                f"GPU selection must be device indices like '0' or '0,1', got {gpus!r}. "
                f"Leave it empty to use every GPU."
            )
        env["CUDA_VISIBLE_DEVICES"] = gpus
    return env
