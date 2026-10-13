#!/usr/bin/env python3
"""Frozen entry point for Verbatim (PyInstaller builds).

Handles what a double-clicked, windowed executable has to handle: Finder's
-psn argument, a missing stdout, and start-up failures that would otherwise
be silent.

  Verbatim                       launch the window
  Verbatim FILE ...              launch with files queued
  Verbatim selftest              run the self-test, write verbatim-selftest.txt, exit
  Verbatim engine --job FILE     one transcription job (the window starts these itself)
  Verbatim transcribe ...        command-line transcription (see --help)
  Verbatim speedtest             time this computer and choose the fastest way to run
"""
import multiprocessing
import os
import sys
import traceback


_HEADLESS = ("engine", "selftest", "transcribe", "fetch", "models", "speedtest", "-h", "--help", "help", "-V", "--version")


def _app_dir():
    if getattr(sys, "frozen", False):
        exe = os.path.abspath(sys.executable)
        parts = exe.split(os.sep)
        for i, p in enumerate(parts):
            if p.endswith(".app"):
                return os.sep.join(parts[:i]) or os.sep
        return os.path.dirname(exe)
    return os.path.dirname(os.path.abspath(__file__))


def _attach_console():
    """Windows: a windowed build has no console, so "Verbatim.exe transcribe ..."
    typed at a prompt would print nothing. Borrow the console it was typed in."""
    if not sys.platform.startswith("win") or sys.stdout is not None:
        return
    try:
        import ctypes
        if ctypes.windll.kernel32.AttachConsole(-1):   # ATTACH_PARENT_PROCESS
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")
            sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace")
    except Exception:
        pass


def _fix_streams():
    """A windowed build has no console: sys.stdout and sys.stderr are None."""
    class _Null:
        def write(self, *_a):
            return 0

        def flush(self):
            pass

        def isatty(self):
            return False

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            setattr(sys, name, _Null())
        else:
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


def _log_dirs():
    """Where the crash log can go: the app's data folder (beside the app, or the
    per-user folder when the app sits in /Applications or Program Files), then
    beside the app, then the home folder."""
    dirs = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from verbatimlib import paths
        dirs.append(paths.data_dir())
    except Exception:
        pass
    for d in (_app_dir(), os.path.expanduser("~")):
        if d not in dirs:
            dirs.append(d)
    return dirs


def _crash(exc, dialog=True):
    body = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    path = None
    for d in _log_dirs():
        candidate = os.path.join(d, "verbatim-crash.log")
        try:
            with open(candidate, "a", encoding="utf-8") as fh:
                fh.write(body + "\n" + "-" * 60 + "\n")
            path = candidate
            break
        except Exception:
            continue
    if path is None:
        path = "(could not be written anywhere)"
    if dialog:  # never from the engine or the self-test: those run with no one watching
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Verbatim could not start",
                                 f"{type(exc).__name__}: {exc}\n\nDetails written to:\n{path}")
            root.destroy()
        except Exception:
            pass
    print(body, file=sys.stderr)


def main():
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] in _HEADLESS and sys.argv[1] not in ("engine", "selftest"):
        _attach_console()
    _fix_streams()
    # Finder passes -psn_0_12345 to a bundle; a strict parser would exit(2)
    # here and the window would never appear.
    args = [a for a in sys.argv[1:] if not a.startswith("-psn_")]
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        from verbatimlib.cli import main as cli_main
        return cli_main(args)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        _crash(exc, dialog=not (args and args[0] in _HEADLESS))
        return 1


if __name__ == "__main__":
    sys.exit(main())
