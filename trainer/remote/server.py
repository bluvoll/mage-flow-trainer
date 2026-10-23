"""The remote job server: the GUI on your laptop, the training on a GPU box.

Runs on the GPU machine (a notebook cell, a terminal, a pod) and speaks small JSON over HTTP. The
GUI pastes one connect link, validates its config against the remote filesystem, then either:

* **jobs mode** (`serve`): asks the server to run cache-config and training there. The GUI tails
  the log into its console and live graphs, and Stop / Save now / Save & stop act remotely; or
* **receive mode** (`receive_config`): only uploads the config. The notebook cell that was waiting
  returns its path, and the next cell runs training -- the workflow LoRA_Easy_Training_scripts'
  Colab backend uses.

Stdlib only (`http.server`), so a GPU box needs nothing beyond the trainer. Every request except
the bare landing page needs the session token, and the connect link carries it after `#`, which
browsers and proxies never send to the server. A public tunnel URL without the token can do
nothing; the commands the server runs are fixed -- the GUI sends a config, never a command line.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..gui.launch import PROJECT_ROOT, cache_config_launch, train_launch, training_env

API_VERSION = 1
MAX_BODY = 8 * 1024 * 1024
LOG_CHUNK = 256 * 1024
STEPS = ("cache", "cache_dry", "train")
JOBS_DIR = PROJECT_ROOT / "remote_jobs"
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def detect_gpus() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            gpus.append({"index": int(parts[0]), "name": parts[1],
                         "memory_gb": round(float(parts[2]) / 1024, 1) if parts[2] else None})
    return gpus


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "")).strip("-.")
    return cleaned[:80] or "run"


def validate_toml(text: str) -> dict:
    """Load the config exactly as the trainer would, on this machine's filesystem."""
    from ..training.config import load_config

    fd, tmp = tempfile.mkstemp(suffix=".toml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        try:
            cfg = load_config(tmp)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        from ..training.config import config_warnings
        warnings = config_warnings(cfg)
        folders = []
        try:
            folders = [s.path for s in cfg.dataset.effective_subsets()]
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        missing = [p for p in folders if not Path(p).is_dir()]
        if missing:
            warnings.append(f"{len(missing)} of {len(folders)} dataset folder(s) do not exist here "
                            f"(e.g. {missing[0]})")
        model = Path(cfg.train.transformer_path or Path(cfg.train.model_path) / "transformer")
        if not (model if model.is_absolute() else PROJECT_ROOT / model).exists():
            warnings.append(f"model not found here: {model}")
        return {"ok": True, "run_name": cfg.train.run_name, "folders": len(folders),
                "out_dir": str(run_dir(cfg)), "warnings": warnings}
    finally:
        Path(tmp).unlink(missing_ok=True)


def run_dir(cfg) -> Path:
    out = Path(cfg.train.output_dir) / cfg.train.run_name
    return out if out.is_absolute() else PROJECT_ROOT / out


@dataclass
class Job:
    id: str
    folder: Path
    config: Path
    steps: list[str]
    gpus: str
    num_processes: int
    run_dir: Path
    log_path: Path
    step: str | None = None
    exit_code: int | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    process: subprocess.Popen | None = None
    cancelled: bool = False

    @property
    def running(self) -> bool:
        return self.finished is None

    def summary(self) -> dict:
        return {"id": self.id, "steps": self.steps, "step": self.step, "running": self.running,
                "exit_code": self.exit_code, "config": str(self.config),
                "run_dir": str(self.run_dir), "num_processes": self.num_processes,
                "gpus": self.gpus, "started": self.started, "finished": self.finished}


class JobRunner:
    """One job at a time: its steps run in order and a failed step cancels the rest -- the same
    rule as the GUI's local queue (training past a failed cache trains on missing latents)."""

    def __init__(self):
        self.job: Job | None = None
        self.lock = threading.Lock()
        # Spawning a step and cancelling the job are serialised: a stop that lands while a step
        # is being launched must not miss the process and let it run to completion.
        self._spawn_lock = threading.Lock()

    def start(self, toml_text: str, name: str, steps: list[str], gpus: str,
              num_processes: int | None) -> Job:
        with self.lock:
            if self.job is not None and self.job.running:
                raise RuntimeError(f"job {self.job.id} is still running")
            from ..training.config import load_config

            stamp = time.strftime("%Y%m%d-%H%M%S")
            folder = JOBS_DIR / f"{stamp}-{safe_name(name)}"
            folder.mkdir(parents=True, exist_ok=True)
            config = folder / f"{safe_name(name)}.toml"
            config.write_text(toml_text, encoding="utf-8")
            cfg = load_config(config)
            visible = [g for g in gpus.split(",") if g.strip()] if gpus else detect_gpus()
            n = num_processes or max(1, len(visible))
            job = Job(id=folder.name, folder=folder, config=config, steps=list(steps), gpus=gpus,
                      num_processes=n, run_dir=run_dir(cfg), log_path=folder / "log.txt")
            job.log_path.touch()
            self.job = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _launch(self, job: Job, step: str):
        if step == "train":
            return train_launch(job.config, job.num_processes)
        return cache_config_launch(job.config, gpus=job.gpus, dry_run=step == "cache_dry")

    def _run(self, job: Job):
        code = 0
        with open(job.log_path, "a", encoding="utf-8", errors="replace") as log:
            for step in job.steps:
                if job.cancelled:
                    code = code or -1
                    break
                job.step = step
                launch = self._launch(job, step)
                env = {**training_env(job.gpus), **launch.env}
                log.write(f"\n=== {launch.label}: {' '.join(launch.argv)}\n")
                log.flush()
                options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                           if os.name == "nt" else {"start_new_session": True})
                with self._spawn_lock:
                    if job.cancelled:
                        code = -1
                        break
                    try:
                        job.process = subprocess.Popen(
                            launch.argv, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1, **options)
                    except OSError as exc:
                        log.write(f"could not start {launch.label}: {exc}\n")
                        code = -1
                        break
                for line in job.process.stdout:
                    log.write(_ANSI.sub("", line))
                    log.flush()
                code = job.process.wait()
                log.write(f"=== {launch.label} exited with {code}\n")
                log.flush()
                if code != 0:
                    break
        job.exit_code = code if not job.cancelled else (code or -1)
        job.finished = time.time()

    def stop(self) -> bool:
        job = self.job
        if job is None or not job.running:
            return False
        with self._spawn_lock:
            job.cancelled = True
            proc = job.process
        if proc is not None and proc.poll() is None:
            # The whole tree: accelerate's launcher, every rank, their dataloader workers.
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, timeout=30)
            else:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(30)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        return True

    def read_log(self, offset: int) -> dict:
        job = self.job
        if job is None:
            return {"job": None, "text": "", "offset": 0}
        size = job.log_path.stat().st_size if job.log_path.exists() else 0
        offset = max(0, min(offset, size))
        with open(job.log_path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read(LOG_CHUNK)
        # Never split a UTF-8 sequence: back off to the last complete line in a full chunk.
        if len(chunk) == LOG_CHUNK and b"\n" in chunk:
            chunk = chunk[: chunk.rindex(b"\n") + 1]
        return {"job": job.summary(), "text": chunk.decode("utf-8", errors="replace"),
                "offset": offset + len(chunk)}


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # Tunnels close idle keep-alive connections; that is not worth a traceback in the cell.
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
                            TimeoutError)):
            return
        super().handle_error(request, client_address)


class RemoteServer:
    """The HTTP server plus its state. `mode` is "jobs" or "receive"."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, token: str | None = None,
                 mode: str = "jobs", receive_dir: Path | None = None):
        if mode not in ("jobs", "receive"):
            raise ValueError("mode must be 'jobs' or 'receive'")
        self.token = token or os.environ.get("MAGEFLOW_REMOTE_TOKEN") or secrets.token_urlsafe(24)
        self.mode = mode
        self.receive_dir = Path(receive_dir) if receive_dir else PROJECT_ROOT / "configs" / "remote"
        self.received: Path | None = None
        self.received_event = threading.Event()
        self.jobs = JobRunner()
        self.public_url: str | None = None
        self.tunnel = None
        handler = type("Handler", (_Handler,), {"server_state": self})
        self.httpd = _HTTPServer((host, port), handler)
        self.host, self.port = self.httpd.server_address[:2]
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------------------------------

    def start(self, tunnel: str = "none") -> "RemoteServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        if tunnel == "cloudflared":
            from .tunnel import CloudflaredTunnel

            self.tunnel = CloudflaredTunnel(self.port)
            self.public_url = self.tunnel.start()
        elif tunnel != "none":
            raise ValueError("tunnel must be 'cloudflared' or 'none'")
        return self

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def stop(self) -> None:
        if self.tunnel is not None:
            self.tunnel.stop()
            self.tunnel = None
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def local_url(self) -> str:
        host = self.host if self.host not in ("0.0.0.0", "::") else socket.gethostname()
        return f"http://{host}:{self.port}"

    def connect_link(self, base: str | None = None) -> str:
        return f"{(base or self.public_url or self.local_url).rstrip('/')}/#token={self.token}"

    def banner(self, base: str | None = None) -> str:
        link = self.connect_link(base)
        what = ("send your config from the GUI; this cell returns its path"
                if self.mode == "receive" else "Start Training in the GUI runs here")
        lines = ["", "=" * 72, "Mage-Flow remote server ready -- paste this link into the GUI's "
                 "Remote bar and press Connect:", "", f"    {link}", "",
                 f"({what}. The part after # is the access token: share the link only with "
                 f"people who may train on this machine.)"]
        if not self.public_url and not base:
            lines.append("No tunnel: use a URL your laptop can reach (an exposed pod port, an SSH "
                         "tunnel) in place of the host above, keeping the #token part.")
        lines.append("=" * 72)
        return "\n".join(lines)

    def wait_for_config(self, timeout: float | None = None) -> Path:
        if not self.received_event.wait(timeout):
            raise TimeoutError("no config received")
        return self.received

    # -- handlers --------------------------------------------------------------------------------

    def status(self) -> dict:
        job = self.jobs.job
        return {"server": "mageflow-remote", "api": API_VERSION, "mode": self.mode,
                "hostname": socket.gethostname(), "root": str(PROJECT_ROOT),
                "commit": _git_commit(), "python": sys.version.split()[0],
                "platform": sys.platform, "gpus": detect_gpus(),
                "job": job.summary() if job else None,
                "received": str(self.received) if self.received else None}

    def receive(self, text: str, name: str) -> Path:
        result = validate_toml(text)
        if not result["ok"]:
            raise ValueError(result["error"])
        self.receive_dir.mkdir(parents=True, exist_ok=True)
        path = self.receive_dir / f"{safe_name(name)}.toml"
        path.write_text(text, encoding="utf-8")
        self.received = path
        self.received_event.set()
        return path

    def signal(self, name: str) -> Path:
        if name not in ("save", "save_quit"):
            raise ValueError("signal must be 'save' or 'save_quit'")
        job = self.jobs.job
        if job is None or not job.running or job.step != "train":
            raise RuntimeError("no training is running")
        job.run_dir.mkdir(parents=True, exist_ok=True)
        target = job.run_dir / name
        target.touch()
        return target


class _Handler(BaseHTTPRequestHandler):
    server_state: RemoteServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep notebook output clean
        pass

    # -- plumbing --------------------------------------------------------------------------------

    def _send(self, code: int, payload, content_type="application/json"):
        body = (json.dumps(payload) if content_type == "application/json" else payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else self.headers.get("X-Mageflow-Token", "")
        return bool(token) and hmac.compare_digest(token, self.server_state.token)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError("request too large")
        data = self.rfile.read(length) if length else b"{}"
        return json.loads(data.decode("utf-8") or "{}")

    def _route(self, method: str):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html") and method == "GET":
            # Nothing about the machine without the token -- just how to use the link.
            return self._send(200, "Mage-Flow remote server. Paste the full connect link (with its "
                                   "#token=...) into the trainer GUI's Remote bar.\n", "text/plain")
        if not url.path.startswith("/api/"):
            return self._send(404, {"error": "not found"})
        if not self._authorized():
            return self._send(401, {"error": "missing or wrong token -- copy the whole connect "
                                             "link, including #token=..."})
        state = self.server_state
        name = url.path[len("/api/"):]
        try:
            if method == "GET" and name == "status":
                return self._send(200, state.status())
            if method == "GET" and name == "log":
                offset = int(parse_qs(url.query).get("offset", ["0"])[0])
                return self._send(200, state.jobs.read_log(offset))
            if method != "POST":
                return self._send(405, {"error": "method not allowed"})
            body = self._body()
            if name == "validate":
                return self._send(200, validate_toml(str(body.get("toml", ""))))
            if name == "config":
                path = state.receive(str(body.get("toml", "")), str(body.get("name", "run")))
                return self._send(200, {"path": str(path), "mode": state.mode})
            if name == "run":
                if state.mode != "jobs":
                    return self._send(409, {"error": "this server only receives configs; run "
                                                     "training from the notebook"})
                steps = [s for s in body.get("steps", ["cache", "train"])]
                if not steps or any(s not in STEPS for s in steps):
                    return self._send(400, {"error": f"steps must be from {list(STEPS)}"})
                gpus = str(body.get("gpus") or "").strip()
                training_env(gpus)    # rejects anything but a device list, as locally
                check = validate_toml(str(body.get("toml", "")))
                if not check["ok"]:
                    return self._send(400, {"error": check["error"]})
                job = state.jobs.start(str(body["toml"]), str(body.get("name", "run")), steps,
                                       gpus, body.get("num_processes"))
                return self._send(200, {"job": job.summary(), "warnings": check["warnings"]})
            if name == "stop":
                return self._send(200, {"stopped": state.jobs.stop()})
            if name == "signal":
                return self._send(200, {"path": str(state.signal(str(body.get("name"))))})
            if name == "shutdown":
                job = state.jobs.job
                if job is not None and job.running and not body.get("force"):
                    return self._send(409, {"error": "a job is running; stop it first"})
                threading.Thread(target=state.stop, daemon=True).start()
                return self._send(200, {"shutdown": True})
            return self._send(404, {"error": f"unknown endpoint {name!r}"})
        except (ValueError, RuntimeError) as exc:
            return self._send(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - surfaced to the GUI rather than swallowed
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")
