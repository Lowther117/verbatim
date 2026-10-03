"""Starting, quietening and stopping the engine process.

Every transcription runs in a second copy of the program ("Verbatim engine
--job ..."), so that Stop is immediate, the window never freezes, and a fault
in one file cannot take the window down with it.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import List, Optional

from . import paths


def no_window() -> dict:
    """Keep a child process from flashing a console window on Windows."""
    if sys.platform.startswith("win"):
        return {"creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {}


def engine_command(job_path: str) -> List[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "engine", "--job", job_path]
    return [sys.executable, os.path.join(paths.app_dir(), "verbatim.py"), "engine", "--job", job_path]


def start_engine(job_path: str, log) -> subprocess.Popen:
    """Launch the engine in its own process group, so that it and the ffmpeg
    it starts can be stopped together."""
    kwargs: dict = {}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(engine_command(job_path), stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, cwd=paths.app_dir(), **kwargs)


def kill_tree(proc: Optional[subprocess.Popen]) -> None:
    """Stop a process and everything it started. proc.kill() alone would leave
    ffmpeg running, still writing to (and on Windows, locking) the temp folder."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, **no_window())
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    try:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def be_a_good_neighbour() -> None:
    """Called by the engine on itself: run below normal priority so the
    computer stays usable, and keep the machine awake until the job is done
    (a laptop going to sleep is the likeliest way an hour-long job fails)."""
    try:
        if sys.platform.startswith("win"):
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)       # BELOW_NORMAL_PRIORITY_CLASS
            k32.SetThreadExecutionState(0x80000000 | 0x00000001)            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        else:
            os.nice(5)
            if sys.platform == "darwin":
                subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001 - a nicety, never a reason to fail
        pass
