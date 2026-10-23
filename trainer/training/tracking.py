"""Experiment tracking: TensorBoard, Weights & Biases and Trackio, any combination.

Ported from the diffusion-pipe-mageflow-ft fork's tracker abstraction (`utils/tracking.py`), with
the two rules that fork learned the hard way kept as the design:

* **Tracking never takes a run down.** Every call into a backend is guarded. A backend that fails to
  start is skipped; one that fails mid-run prints one warning and disables itself, and training
  and the other backends carry on. A 403 from a hosted service used to kill jobs before step 1.
* **Every log call carries an explicit step.** wandb and Trackio otherwise advance their own counter
  once per call, which desynchronises the x-axis from the training step as soon as one step logs
  more than once (training metrics, eval, samples).

Only the main process tracks; other ranks get an inert `Tracker` and can call it unconditionally.

Credentials never go through the config. `training_config` is embedded in every exported
checkpoint's safetensors metadata, so a `wandb_api_key` key would publish the key with every
shared LoRA. wandb reads `WANDB_API_KEY` / `wandb login`; Trackio Spaces read the Hugging Face
login.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BACKENDS = ("tensorboard", "wandb", "trackio")
RUN_FILE = "tracking_run.json"


class Backend:
    name = "none"

    def start(self, ctx: dict) -> bool:
        raise NotImplementedError

    def log(self, metrics: dict, step: int) -> None:
        raise NotImplementedError

    def log_images(self, key: str, entries: list, step: int) -> None:
        """`entries`: (name, PIL image, caption). Each backend picks its own gallery shape."""

    def finish(self) -> None:
        pass


class TensorBoardBackend(Backend):
    name = "tensorboard"

    def start(self, ctx):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(Path(ctx["log_dir"]) / "tensorboard"))
        print(f"tracking tensorboard -> {Path(ctx['log_dir']) / 'tensorboard'} "
              f"(view with: tensorboard --logdir \"{ctx['log_dir']}\")", flush=True)
        return True

    def log(self, metrics, step):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def log_images(self, key, entries, step):
        import numpy as np

        for name, image, caption in entries:
            self.writer.add_image(f"{key}/{name}", np.asarray(image.convert("RGB")), step,
                                  dataformats="HWC")
            # TensorBoard's image pane has no captions; the prompt goes beside it as text.
            self.writer.add_text(f"{key}/{name}", caption.replace("\n", "  \n"), step)
        self.writer.flush()

    def finish(self):
        self.writer.close()


class WandbBackend(Backend):
    name = "wandb"

    def start(self, ctx):
        import wandb

        self.wandb = wandb
        cfg = ctx["cfg"]
        if cfg.wandb_base_url:
            os.environ["WANDB_BASE_URL"] = cfg.wandb_base_url
        kwargs = dict(project=ctx["project"], name=ctx["run_name"], config=ctx["config"],
                      dir=ctx["log_dir"], entity=cfg.wandb_entity or None,
                      tags=list(cfg.wandb_tags) or None)
        if ctx.get("wandb_id"):
            kwargs.update(id=ctx["wandb_id"], resume="allow")
        try:
            self.run = wandb.init(mode=cfg.wandb_mode, **kwargs)
        except Exception as exc:
            if not (cfg.wandb_offline_on_failure and cfg.wandb_mode == "online"):
                raise
            print(f"WARNING: wandb online start failed ({type(exc).__name__}: {exc}); retrying "
                  f"offline. Upload later with `wandb sync \"{ctx['log_dir']}\"`.", flush=True)
            self.run = wandb.init(mode="offline", **kwargs)
        ctx["wandb_id"] = self.run.id
        print(f"tracking wandb run {self.run.id} ({getattr(self.run, 'url', None) or 'offline'})",
              flush=True)
        return True

    def log(self, metrics, step):
        self.wandb.log(metrics, step=step)

    def log_images(self, key, entries, step):
        # A list under one key renders as one gallery.
        self.wandb.log({key: [self.wandb.Image(img, caption=cap) for _, img, cap in entries]},
                       step=step)

    def finish(self):
        self.wandb.finish()


class TrackioBackend(Backend):
    """Local SQLite + dashboard by default (no network at all), or a Space / self-hosted server."""

    name = "trackio"

    def start(self, ctx):
        cfg = ctx["cfg"]
        # Must precede the import: trackio reads TRACKIO_DIR once, at import time, so setting it
        # afterwards is silently ignored and the run lands in ~/.cache instead of beside the run.
        if not (cfg.trackio_space_id or cfg.trackio_server_url):
            os.environ.setdefault("TRACKIO_DIR", str(Path(ctx["log_dir"]) / "trackio"))
        import trackio

        self.trackio = trackio
        kwargs = dict(project=ctx["project"], name=ctx["run_name"], config=ctx["config"],
                      resume="allow" if ctx.get("resumed") else "never")
        if cfg.trackio_space_id:
            kwargs["space_id"] = cfg.trackio_space_id
        if cfg.trackio_server_url:
            kwargs["server_url"] = cfg.trackio_server_url
        trackio.init(**kwargs)
        where = (cfg.trackio_space_id or cfg.trackio_server_url
                 or f"{os.environ.get('TRACKIO_DIR')} (view with: trackio show --project "
                    f"\"{ctx['project']}\")")
        print(f"tracking trackio -> {where}", flush=True)
        return True

    def log(self, metrics, step):
        self.trackio.log(metrics, step=step)

    def log_images(self, key, entries, step):
        # Trackio has no list-of-media branch: a list falls through to JSON serialisation and the
        # whole batch is dropped. One key per image instead.
        self.trackio.log({f"{key}/{name}": self.trackio.Image(img, caption=cap)
                          for name, img, cap in entries}, step=step)

    def finish(self):
        self.trackio.finish()


_FACTORIES = {"tensorboard": TensorBoardBackend, "wandb": WandbBackend, "trackio": TrackioBackend}


class Tracker:
    """Fan-out over the started backends. Inert when none are configured or off the main rank."""

    def __init__(self, backends: list[Backend] | None = None):
        self.backends = list(backends or [])

    @property
    def enabled(self) -> bool:
        return bool(self.backends)

    def _guard(self, backend: Backend, what: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            self.backends.remove(backend)
            print(f"WARNING: {backend.name} {what} failed ({type(exc).__name__}: {exc}); "
                  f"{backend.name} is disabled for the rest of this run. Training continues.",
                  flush=True)

    def log(self, metrics: dict, step: int) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if not clean:
            return
        for backend in list(self.backends):
            self._guard(backend, "logging", lambda b=backend: b.log(clean, step))

    def log_images(self, key: str, entries: list, step: int) -> None:
        if not entries:
            return
        for backend in list(self.backends):
            self._guard(backend, "image logging", lambda b=backend: b.log_images(key, entries, step))

    def finish(self) -> None:
        for backend in list(self.backends):
            try:
                backend.finish()
            except Exception:
                pass
        self.backends = []


def build_tracker(cfg, out_dir: Path, is_main: bool, resumed: bool = False) -> Tracker:
    """Start every configured backend. `cfg` is the whole training Config.

    On a resumed run, wandb continues the same run id (recorded in `<out_dir>/tracking_run.json`)
    and Trackio continues the run of the same name, so the curves stay one line instead of
    restarting at the resume step.
    """
    tcfg = cfg.tracking
    if not is_main or not tcfg.backends:
        return Tracker()
    from ..modeling.checkpoint_metadata import encode_settings

    out_dir = Path(out_dir)
    log_dir = Path(tcfg.log_dir) if tcfg.log_dir else out_dir / "tracking"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_file = out_dir / RUN_FILE
    previous = {}
    if resumed and run_file.is_file():
        try:
            previous = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
    ctx = dict(cfg=tcfg, project=tcfg.project, run_name=tcfg.run_name or cfg.train.run_name,
               config=json.loads(encode_settings(cfg)), log_dir=str(log_dir),
               resumed=resumed, wandb_id=previous.get("wandb_id") if tcfg.resume_run else None)
    started = []
    for name in tcfg.backends:
        backend = _FACTORIES[name]()
        try:
            if backend.start(ctx):
                started.append(backend)
        except ImportError as exc:
            print(f"WARNING: tracking backend {name!r} is not installed ({exc}); continuing "
                  f"without it. Install it with `pip install {name}`.", flush=True)
        except Exception as exc:
            print(f"WARNING: tracking backend {name!r} failed to start ({type(exc).__name__}: "
                  f"{exc}); continuing without it.", flush=True)
    if ctx.get("wandb_id"):
        try:
            run_file.write_text(json.dumps({"wandb_id": ctx["wandb_id"]}, indent=2),
                                encoding="utf-8")
        except OSError:
            pass
    return Tracker(started)
