"""Cloudflare quick tunnels: a public https URL for a server listening on localhost.

`cloudflared tunnel --url http://127.0.0.1:<port>` needs no account and no open ports -- the
connection is outbound from the GPU box, so it works on Colab, JupyterHub and pods without exposed
ports alike. The binary is taken from PATH, `MAGEFLOW_CLOUDFLARED`, or downloaded once from
Cloudflare's GitHub releases into `~/.cache/mageflow-trainer/`.

Modelled on LoRA_Easy_Training_scripts' backend (which uses pycloudflared), without the
dependency, and with one difference that matters for long runs: cloudflared's stderr is drained
for the life of the tunnel. Reading only the first lines leaves the pipe to fill, and a full pipe
blocks cloudflared mid-session.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_RELEASES = "https://github.com/cloudflare/cloudflared/releases/latest/download/"


def _asset() -> tuple[str, bool]:
    """(release asset name, is_tgz) for this machine."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    if system == "linux":
        return f"cloudflared-linux-{arch}", False
    if system == "windows":
        return "cloudflared-windows-amd64.exe", False
    if system == "darwin":
        return f"cloudflared-darwin-{arch}.tgz", True
    raise RuntimeError(f"no cloudflared build for {system}/{machine}; install it and put it on PATH")


def cloudflared_path(download: bool = True) -> Path:
    explicit = os.environ.get("MAGEFLOW_CLOUDFLARED")
    if explicit:
        return Path(explicit)
    found = shutil.which("cloudflared")
    if found:
        return Path(found)
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "mageflow-trainer"
    exe = cache / ("cloudflared.exe" if platform.system() == "Windows" else "cloudflared")
    if exe.is_file():
        return exe
    if not download:
        raise FileNotFoundError("cloudflared not found")
    asset, is_tgz = _asset()
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / (asset + ".part")
    print(f"downloading cloudflared ({asset}) ...", flush=True)
    urllib.request.urlretrieve(_RELEASES + asset, tmp)
    if is_tgz:
        with tarfile.open(tmp) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("cloudflared"))
            with tar.extractfile(member) as src, open(exe, "wb") as dst:
                shutil.copyfileobj(src, dst)
        tmp.unlink()
    else:
        os.replace(tmp, exe)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


class CloudflaredTunnel:
    def __init__(self, port: int, timeout: float = 60.0):
        self.port = port
        self.url: str | None = None
        self.process: subprocess.Popen | None = None
        self._log: list[str] = []
        self._found = threading.Event()
        self.timeout = timeout

    def start(self) -> str:
        exe = cloudflared_path()
        self.process = subprocess.Popen(
            [str(exe), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace")
        threading.Thread(target=self._drain, daemon=True).start()
        if not self._found.wait(self.timeout) or self.url is None:
            self.stop()
            tail = "\n".join(self._log[-15:])
            raise RuntimeError(f"cloudflared did not report a tunnel URL within "
                               f"{self.timeout:.0f}s. Its output:\n{tail}")
        return self.url

    def _drain(self):
        for line in self.process.stderr:
            if len(self._log) < 500:
                self._log.append(line.rstrip())
            if self.url is None:
                match = _URL_RE.search(line)
                if match:
                    self.url = match.group(0)
                    self._found.set()
        self._found.set()

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def wait_until_reachable(url: str, timeout: float = 45.0) -> bool:
    """A fresh quick-tunnel hostname takes a few seconds to resolve; poll until it answers."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/", timeout=5):
                return True
        except Exception:
            time.sleep(2)
    return False
