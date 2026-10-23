"""Subprocess launching for the GUI.

`ProcessRunner` is vendored from Aozora Trainer (Apache-2.0) with the Windows-only branches kept
intact; the launch specs below are ours.

The GUI never imports the trainer -- it spawns it and reads stdout. That is what keeps a crashing
run from taking the window down with it, and it is why a torch/CUDA import never happens in the Qt
process.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .launch import (  # noqa: F401  (re-exported: the launch specs moved to launch.py)
    Job, Launch, _DEVICE_LIST_RE, _python, audit_launch, cache_config_launch, cache_launch,
    concat_launch, train_launch, training_env,
)
from .widgets import ANSI_ESCAPE_RE, IS_WINDOWS, PROJECT_ROOT


class ProcessRunner(QThread):
    """Vendored from Aozora (`gui/gui.py::ProcessRunner`), retargeted at our log lines.

    stdout and stderr are merged into one pipe so ordering is preserved; the trainer runs under
    `-u` so a crash traceback is not lost in a half-flushed buffer.
    """

    logSignal = Signal(str)
    progressSignal = Signal(str, bool)
    finishedSignal = Signal(int)
    errorSignal = Signal(str)
    metricsSignal = Signal(str)

    # Substrings that mean the run is already doomed -- surfaced in the log with emphasis rather
    # than scrolling past at the same weight as a step line.
    FATAL_HINTS = (
        "out of memory", "cuda error", "nccl error", "device-side assert",
        "nan/inf", "memory inaccessible",
    )

    def __init__(self, launch: Launch, working_dir: str, env=None):
        super().__init__()
        self.launch = launch
        self.working_dir = working_dir
        self.env = env
        self.process = None
        self.stop_requested = False

    @staticmethod
    def _clean(line: str) -> str:
        return ANSI_ESCAPE_RE.sub("", line)

    def run(self):
        try:
            popen_options = {}
            if IS_WINDOWS:
                popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                # One process group for the trainer, its dataloader workers, and -- under DDP --
                # every rank Accelerate spawns. Without this, stopping kills the launcher and
                # leaves N ranks holding the GPUs.
                popen_options["start_new_session"] = True
            self.process = subprocess.Popen(
                self.launch.argv, cwd=self.working_dir, env=self.env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1, **popen_options)
            self.logSignal.emit(f"INFO: {self.launch.label} started (PID {self.process.pid})")
            for line in iter(self.process.stdout.readline, ''):
                line = self._clean(line.rstrip("\n"))
                if not line.strip():
                    continue
                low = line.lower()
                if any(hint in low for hint in self.FATAL_HINTS):
                    self.logSignal.emit(f"*** {line} ***")
                else:
                    # tqdm redraws with \r; keep only the newest frame.
                    is_progress = '\r' in line or bool(re.match(r'^\s*\d+%\|', line))
                    self.progressSignal.emit(line.split('\r')[-1], is_progress)
                self.metricsSignal.emit(line)
            self.finishedSignal.emit(self.process.wait())
        except Exception as e:
            self.errorSignal.emit(f"Subprocess error: {e}")
            self.finishedSignal.emit(-1)

    def _kill_tree_windows(self) -> bool:
        """Kill the launcher *and* everything under it. Returns False if taskkill was unusable.

        `Popen.terminate()` is TerminateProcess on one PID, and Windows has no killpg, so it takes
        down `accelerate.commands.launch` and leaves the process it spawned alive -- simple_launcher
        is an unconditional `subprocess.Popen` (accelerate/commands/launch.py:989), so there is
        always at least one grandchild, single GPU included. Two things then go wrong at once: the
        real trainer keeps running on the GPU, and because it inherited the stdout handle the pipe
        never reaches EOF, so `iter(readline, '')` blocks forever and `finishedSignal` is never
        emitted. The GUI keeps its Stop button up and keeps printing step lines -- which is exactly
        what "stop does nothing on Windows" looks like from the outside. `/T` walks the tree, `/F`
        is required because the ranks have no window to accept a polite close.
        """
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                capture_output=True,
                # No console flash: the GUI has no terminal attached to borrow.
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=15,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    def stop(self):
        if self.process and self.process.poll() is None:
            self.stop_requested = True
            killed_tree = False
            if IS_WINDOWS:
                killed_tree = self._kill_tree_windows()
                if not killed_tree:
                    # Last resort. Only reaches the launcher, so a grandchild may survive and hold
                    # the GPU -- said out loud rather than left for the user to find in nvidia-smi.
                    self.process.terminate()
            else:
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if IS_WINDOWS:
                    self.process.kill()
                else:
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            if IS_WINDOWS and not killed_tree:
                self.logSignal.emit(
                    "WARNING: taskkill unavailable -- only the launcher was stopped. Check Task "
                    "Manager for a surviving python.exe still holding the GPU."
                )
            self.logSignal.emit("Process stopped.")
