"""A remote job, seen by the GUI exactly like a local subprocess.

`RemoteRunner` has `ProcessRunner`'s signals and `stop()`, so the console, the live graphs, the job
queue and the Stop button need no remote-specific code: the log is tailed from the job server
(`trainer.remote`) and replayed line by line through the same classification.
"""

from __future__ import annotations

import re
import time

from PySide6.QtCore import QThread, Signal

from ..remote.client import RemoteClient, RemoteError
from .process import ProcessRunner
from .widgets import ANSI_ESCAPE_RE


class RemoteCall(QThread):
    """Run one blocking client call off the UI thread."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as exc:
            self.failed.emit(str(exc))


class RemoteRunner(QThread):
    logSignal = Signal(str)
    progressSignal = Signal(str, bool)
    finishedSignal = Signal(int)
    errorSignal = Signal(str)
    metricsSignal = Signal(str)

    # How long a lost connection is tolerated before giving up on following the job. The job
    # itself keeps running on the server either way; reconnecting attaches to it again.
    PATIENCE_S = 180

    def __init__(self, client: RemoteClient, label: str, start=None, attach: dict | None = None):
        """`start()` submits the job (it runs in this thread, off the UI); `attach` follows a job
        that is already running instead."""
        super().__init__()
        self.client = client
        self.label = label
        self.start_fn = start
        self.attach = attach
        self.stop_requested = False
        self.detached = False

    def _emit_line(self, line: str) -> None:
        line = ANSI_ESCAPE_RE.sub("", line.rstrip("\r"))
        if not line.strip():
            return
        low = line.lower()
        if any(hint in low for hint in ProcessRunner.FATAL_HINTS):
            self.logSignal.emit(f"*** {line} ***")
        else:
            is_progress = "\r" in line or bool(re.match(r"^\s*\d+%\|", line))
            self.progressSignal.emit(line.split("\r")[-1], is_progress)
        self.metricsSignal.emit(line.split("\r")[-1])

    def run(self):
        try:
            job = self.attach or self.start_fn()["job"]
        except Exception as exc:
            self.errorSignal.emit(f"Remote start failed: {exc}")
            self.finishedSignal.emit(-1)
            return
        where = self.client.base
        verb = "attached to" if self.attach else "started"
        self.logSignal.emit(f"INFO: {self.label} {verb} on {where} (job {job['id']}, "
                            f"{job.get('num_processes')} process(es))")
        offset, partial, lost_since = 0, "", None
        while not self.detached:
            try:
                chunk = self.client.log(offset)
                if lost_since is not None:
                    self.logSignal.emit("INFO: reconnected to the remote server")
                lost_since = None
            except RemoteError as exc:
                if lost_since is None:
                    lost_since = time.time()
                    self.logSignal.emit(f"WARNING: lost the remote server ({exc}); retrying. "
                                        f"The job keeps running there.")
                if time.time() - lost_since > self.PATIENCE_S:
                    self.errorSignal.emit("Gave up following the remote job. It is still running "
                                          "on the server: Connect again to reattach.")
                    self.finishedSignal.emit(-1)
                    return
                time.sleep(3)
                continue
            text = chunk.get("text", "")
            offset = chunk.get("offset", offset)
            if text:
                partial += text
                *lines, partial = partial.split("\n")
                for line in lines:
                    self._emit_line(line)
            state = chunk.get("job") or {}
            if not text and state and not state.get("running", True):
                if partial:
                    self._emit_line(partial)
                code = state.get("exit_code")
                self.finishedSignal.emit(int(code) if code is not None else -1)
                return
            if not text:
                time.sleep(1.0)

    def detach(self) -> None:
        """Stop following without stopping the job (closing the GUI must not kill a remote run)."""
        self.detached = True

    def stop(self):
        self.stop_requested = True
        try:
            self.client.stop()
            self.logSignal.emit("Stop requested on the remote server.")
        except RemoteError as exc:
            self.errorSignal.emit(f"Remote stop failed: {exc}")
