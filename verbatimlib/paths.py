"""Where things live: the app folder, settings, models and the default save folder."""
from __future__ import annotations

import json
import os
import sys

APP = "Verbatim"
VERSION = "1.1.0"
SETTINGS_NAME = "verbatim-settings.json"
MODELS_NAME = "verbatim-models"


def app_dir() -> str:
    """Folder the app lives in: beside the exe, or beside the .app bundle on macOS."""
    if getattr(sys, "frozen", False):
        exe = os.path.abspath(sys.executable)
        parts = exe.split(os.sep)
        for i, p in enumerate(parts):
            if p.endswith(".app"):
                return os.sep.join(parts[:i]) or os.sep
        return os.path.dirname(exe)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_dir() -> str:
    """Bundled resources (PyInstaller unpacks them under sys._MEIPASS)."""
    return getattr(sys, "_MEIPASS", app_dir())


_DATA_DIR = None


def data_dir() -> str:
    """Where settings, downloaded models and logs are kept.

    Beside the app when that folder can be written to - the house rule, and
    it keeps everything in one place. An app sitting in /Applications or
    Program Files usually cannot write there, so the per-user application
    data folder is used instead.
    """
    global _DATA_DIR
    if _DATA_DIR:
        return _DATA_DIR
    here = app_dir()
    probe = os.path.join(here, f".verbatim-write-test-{os.getpid()}")
    try:
        with open(probe, "w") as fh:
            fh.write("x")
        os.remove(probe)
        _DATA_DIR = here
        return here
    except OSError:
        pass
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    fallback = os.path.join(base, APP)
    try:
        os.makedirs(fallback, exist_ok=True)
        _DATA_DIR = fallback
    except OSError:
        _DATA_DIR = here
    return _DATA_DIR


def models_dirs() -> list:
    """Every folder a model may be in. The first one is where downloads go."""
    out = [os.path.join(data_dir(), MODELS_NAME)]
    beside = os.path.join(app_dir(), MODELS_NAME)
    if beside not in out:
        out.append(beside)
    env = os.environ.get("VERBATIM_MODELS")
    if env:
        out.insert(0, env)
    bundled = os.path.join(resource_dir(), "models")
    if bundled not in out:
        out.append(bundled)
    return out


def settings_path() -> str:
    return os.path.join(data_dir(), SETTINGS_NAME)


def load_settings() -> dict:
    try:
        with open(settings_path(), encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def save_settings(settings: dict) -> None:
    try:
        with open(settings_path(), "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=1)
    except Exception:  # noqa: BLE001
        pass


def downloads_dir() -> str:
    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class GUID(ctypes.Structure):
                _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                            ("Data4", wintypes.BYTE * 8)]

            folderid = GUID(0x374DE290, 0x123F, 0x4565,
                            (wintypes.BYTE * 8)(0x91, 0x64, 0x39, 0xC4, 0x92, 0x5E, 0x46, 0x7B))
            path_ptr = ctypes.c_wchar_p()
            shell32 = ctypes.windll.shell32
            if shell32.SHGetKnownFolderPath(ctypes.byref(folderid), 0, None, ctypes.byref(path_ptr)) == 0:
                path = path_ptr.value
                ctypes.windll.ole32.CoTaskMemFree(path_ptr)
                if path and os.path.isdir(path):
                    return path
        except Exception:  # noqa: BLE001
            pass
    cand = os.path.join(home, "Downloads")
    return cand if os.path.isdir(cand) else home


def default_save_dir(settings: dict | None = None) -> str:
    """The saved "save_dir" if it still exists, otherwise Downloads/Verbatim."""
    settings = settings if settings is not None else load_settings()
    d = settings.get("save_dir")
    if d and os.path.isdir(d):
        return d
    return os.path.join(downloads_dir(), APP)
