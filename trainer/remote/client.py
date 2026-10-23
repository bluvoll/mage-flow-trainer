"""Client side of the remote job server, for the GUI. Stdlib only (urllib)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


class RemoteError(RuntimeError):
    pass


def parse_link(link: str, token: str = "") -> tuple[str, str]:
    """`https://host/#token=XYZ` -> ("https://host", "XYZ"). A bare URL needs `token`."""
    link = (link or "").strip()
    if not link:
        raise RemoteError("paste the connect link the server printed")
    base, _, fragment = link.partition("#")
    if fragment.startswith("token="):
        token = fragment[len("token="):]
    elif fragment and not token:
        token = fragment
    url = urlparse(base if "://" in base else "http://" + base)
    if url.scheme not in ("http", "https") or not url.netloc:
        raise RemoteError(f"not a URL: {link!r}")
    if not token:
        raise RemoteError("the link has no #token=... part; copy the whole line the server printed")
    return f"{url.scheme}://{url.netloc}{url.path.rstrip('/')}", token


class RemoteClient:
    def __init__(self, link: str, token: str = "", timeout: float = 20.0):
        self.base, self.token = parse_link(link, token)
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict | None = None,
                 timeout: float | None = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/api/{path}", data=data, method=method,
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": "application/json",
                     "User-Agent": "mageflow-trainer-gui"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read().decode("utf-8")).get("error", exc.reason)
            except Exception:
                message = exc.reason
            raise RemoteError(f"{exc.code}: {message}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RemoteError(f"cannot reach {self.base}: {reason}") from None

    def status(self, retries: int = 0) -> dict:
        """With `retries`, keep trying for a fresh quick tunnel whose hostname is not resolvable
        yet -- that takes a few seconds after the server prints its link."""
        for attempt in range(retries + 1):
            try:
                status = self._request("GET", "status")
                if status.get("server") != "mageflow-remote":
                    raise RemoteError("that URL answered, but it is not a Mage-Flow remote server")
                return status
            except RemoteError as exc:
                if attempt == retries or str(exc).startswith(("401", "403")):
                    raise
                time.sleep(2)
        raise AssertionError("unreachable")

    def validate(self, toml: str) -> dict:
        return self._request("POST", "validate", {"toml": toml}, timeout=120)

    def send_config(self, toml: str, name: str) -> dict:
        return self._request("POST", "config", {"toml": toml, "name": name}, timeout=120)

    def run(self, toml: str, name: str, steps: list[str], gpus: str = "",
            num_processes: int | None = None) -> dict:
        return self._request("POST", "run", {"toml": toml, "name": name, "steps": steps,
                                             "gpus": gpus, "num_processes": num_processes},
                             timeout=120)

    def log(self, offset: int) -> dict:
        return self._request("GET", f"log?offset={int(offset)}")

    def stop(self) -> dict:
        return self._request("POST", "stop", {})

    def signal(self, name: str) -> dict:
        return self._request("POST", "signal", {"name": name})
