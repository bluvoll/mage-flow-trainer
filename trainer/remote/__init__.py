"""Run the GUI on your laptop and train on a remote GPU box (Colab, JupyterHub, a rented pod).

On the GPU machine, in a notebook cell or a terminal:

    from trainer.remote import serve
    serve()                          # jobs mode: the GUI's Start Training runs here

or, to keep training in the notebook (LoRA_Easy_Training_scripts' Colab workflow):

    from trainer.remote import receive_config, run
    config = receive_config()        # waits until the GUI sends its config, returns its path
    run(config)                      # next cell: cache latents, then train, output in the cell

Each prints a connect link (a Cloudflare quick tunnel by default, no account needed). Paste it into
the GUI's Remote bar and press Connect. Terminal equivalents: `python -m trainer.remote serve`,
`... receive`, `... run <config.toml>`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .server import PROJECT_ROOT, RemoteServer

__all__ = ["serve", "receive_config", "run", "RemoteServer"]


def _start(mode, port, tunnel, host, token, url, **kw) -> RemoteServer:
    host = host or ("127.0.0.1" if tunnel == "cloudflared" else "0.0.0.0")
    server = RemoteServer(host=host, port=port, token=token, mode=mode, **kw)
    try:
        server.start(tunnel=tunnel)
    except Exception:
        server.stop()
        raise
    print(server.banner(url), flush=True)
    return server


def serve(port: int = 8765, tunnel: str = "cloudflared", host: str | None = None,
          token: str | None = None, url: str | None = None, block: bool = True):
    """Start the job server. `tunnel="none"` with `url=` your pod's exposed address (e.g. a RunPod
    proxy URL) skips Cloudflare. `block=False` returns the server and frees the notebook cell;
    the server then lives as long as the kernel, or until `.stop()`."""
    server = _start("jobs", port, tunnel, host, token, url)
    if not block:
        return server
    try:
        while server.alive:      # ends on Ctrl+C / interrupt, or a GUI "shutdown"
            server.join(1.0)
    except KeyboardInterrupt:
        print("stopping the remote server", flush=True)
    finally:
        server.jobs.stop()
        server.stop()
    return None


def receive_config(port: int = 8765, tunnel: str = "cloudflared", host: str | None = None,
                   token: str | None = None, url: str | None = None,
                   save_dir: str | Path | None = None, timeout: float | None = None) -> Path:
    """Wait for the GUI to send a config, save it on this machine and return its path.

    The GUI validates against this machine first (paths, folders), so what arrives loads."""
    server = _start("receive", port, tunnel, host, token, url,
                    receive_dir=Path(save_dir) if save_dir else None)
    try:
        path = server.wait_for_config(timeout)
    finally:
        server.stop()
    print(f"received config: {path}", flush=True)
    return path


def run(config: str | Path, steps=("cache", "train"), gpus: str = "",
        num_processes: int | None = None) -> int:
    """Cache latents for every folder the config uses, then train -- in the foreground, with the
    output in this cell. The same commands the GUI runs locally. Returns the exit code."""
    from ..gui.launch import cache_config_launch, train_launch, training_env
    from .server import detect_gpus

    config = Path(config)
    visible = [g for g in gpus.split(",") if g.strip()] if gpus else detect_gpus()
    n = num_processes or max(1, len(visible))
    for step in steps:
        launch = (train_launch(config, n) if step == "train" else
                  cache_config_launch(config, gpus=gpus, dry_run=step == "cache_dry"))
        env = {**training_env(gpus), **launch.env}
        print(f"\n=== {launch.label}: {' '.join(launch.argv)}", flush=True)
        proc = subprocess.Popen(launch.argv, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            print("\ninterrupted", flush=True)
            return 130
        if code != 0:
            print(f"\n{launch.label} failed with exit code {code}", flush=True)
            return code
    return 0
