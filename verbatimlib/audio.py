"""Turning any audio or video file into 16 kHz mono samples, through ffmpeg.

ffmpeg does the format work, so "any format" really is whatever ffmpeg can
open: mp3, m4a, wav, flac, ogg, opus, wma, aac, amr, aiff, and the audio track
of mp4, mkv, mov, webm, avi and the rest. The file extension is never trusted -
anything dropped on the window is simply tried.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from typing import Callable, Optional

import numpy as np

RATE = 16000

# Used only to decide what "Add folder" picks up. Any file can still be added by hand.
EXTENSIONS = {
    ".mp3", ".m4a", ".m4b", ".aac", ".wav", ".wave", ".flac", ".ogg", ".oga", ".opus", ".wma",
    ".aif", ".aiff", ".aifc", ".amr", ".awb", ".3gp", ".3g2", ".3ga", ".caf", ".ac3", ".eac3",
    ".dts", ".mka", ".mp2", ".mpa", ".ape", ".wv", ".tta", ".au", ".snd", ".gsm", ".spx", ".ra",
    ".rm", ".dss", ".ds2", ".voc", ".w64", ".weba", ".m4r", ".qcp", ".vox",
    ".mp4", ".m4v", ".mkv", ".mov", ".webm", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".ts",
    ".mts", ".m2ts", ".ogv", ".vob", ".asf", ".f4v", ".mxf",
}


class AudioError(Exception):
    """A file could not be decoded; the message is fit to show the user."""


def _no_window() -> dict:
    """Keep ffmpeg from flashing a console window on Windows."""
    if sys.platform.startswith("win"):
        return {"creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {}


_FFMPEG: Optional[str] = None


def find_ffmpeg() -> str:
    """The bundled static ffmpeg first (a known quantity), then whatever is installed."""
    global _FFMPEG
    if _FFMPEG and os.path.isfile(_FFMPEG):
        return _FFMPEG
    env = os.environ.get("VERBATIM_FFMPEG")
    if env and os.path.isfile(env):
        _FFMPEG = env
        return env
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            _FFMPEG = exe
            return exe
    except Exception:  # noqa: BLE001
        pass
    exe = shutil.which("ffmpeg")
    if not exe:
        for cand in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
            if os.path.isfile(cand):
                exe = cand
                break
    if not exe:
        raise AudioError("ffmpeg is missing. Rebuild the app with its build file, or install ffmpeg.")
    _FFMPEG = exe
    return exe


_DUR = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_AUDIO = re.compile(r"Stream #\d+:\d+.*?: Audio: (.+)")


def probe(path: str) -> dict:
    """Duration in seconds (None if the container does not say) and the audio stream description."""
    try:
        proc = subprocess.run([find_ffmpeg(), "-hide_banner", "-nostdin", "-i", path],
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=60, **_no_window())
    except subprocess.TimeoutExpired:
        return {"duration": None, "audio": None, "streams": 0}
    text = proc.stderr.decode("utf-8", "replace")
    dur = None
    m = _DUR.search(text)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    streams = _AUDIO.findall(text)
    return {"duration": dur, "audio": streams[0].strip() if streams else None, "streams": len(streams)}


def _friendly(stderr: str, path: str) -> str:
    low = stderr.lower()
    name = os.path.basename(path)
    if "does not contain any stream" in low or "matches no streams" in low or "output file does not contain" in low:
        return f"{name} has no audio track."
    if "invalid data found" in low or "could not find codec parameters" in low or "moov atom not found" in low:
        return f"{name} is not a readable audio or video file (or it is damaged)."
    if "no such file" in low:
        return f"{name} could not be found - was it moved or renamed?"
    if "permission denied" in low:
        return f"{name} could not be opened (permission denied)."
    last = [ln.strip() for ln in stderr.strip().splitlines() if ln.strip()]
    return f"{name} could not be decoded: {last[-1] if last else 'ffmpeg gave no reason'}"


def decode(path: str, raw_path: str, duration: Optional[float] = None,
           progress: Optional[Callable[[float], None]] = None,
           cancelled: Optional[Callable[[], bool]] = None,
           highpass: bool = True, notes: Optional[list] = None) -> np.ndarray:
    """Decode the first audio stream of `path` to 16 kHz mono float32 in `raw_path`.

    Returns a read-only memory map, so a three-hour recording does not need
    three hours of samples in RAM just to be opened. `highpass` removes
    sub-speech rumble (mains hum, handling noise, wind) below 70 Hz, which
    carries no speech at all and only confuses the speech detector. If ffmpeg
    reports a fault part-way through (a damaged or cut-off file) the audio it
    did recover is still returned, and a line saying so is added to `notes`.
    """
    if not os.path.isfile(path):
        raise AudioError(f"{os.path.basename(path)} could not be found - was it moved or renamed?")
    cmd = [find_ffmpeg(), "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", path,
           "-vn", "-sn", "-dn", "-map", "0:a:0", "-ac", "1", "-ar", str(RATE)]
    if highpass:
        cmd += ["-af", "highpass=f=70"]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", raw_path]
    # A file left from an earlier decode must not be mistaken for this one's
    # output when ffmpeg fails before writing anything.
    try:
        os.remove(raw_path)
    except OSError:
        pass
    err_path = raw_path + ".err"
    with open(err_path, "wb") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err, **_no_window())
        expected = duration * RATE * 4 if duration else None
        while proc.poll() is None:
            if cancelled and cancelled():
                proc.kill()
                proc.wait()
                raise AudioError("Cancelled.")
            if progress and expected:
                try:
                    progress(min(0.99, os.path.getsize(raw_path) / expected))
                except OSError:
                    pass
            time.sleep(0.1)
    try:
        with open(err_path, encoding="utf-8", errors="replace") as fh:
            stderr = fh.read()
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass
    size = os.path.getsize(raw_path) if os.path.exists(raw_path) else 0
    if size < 4 * 160:  # under 10 ms of audio: nothing usable came out
        raise AudioError(_friendly(stderr, path))
    if proc.returncode != 0 and notes is not None:
        got = size / 4 / RATE
        notes.append(f"This file is damaged or incomplete: only the first {int(got // 60)} min {int(got % 60)} s "
                     f"could be read, and that is what was transcribed.")
    if progress:
        progress(1.0)
    return np.memmap(raw_path, dtype=np.float32, mode="r", shape=(size // 4,))


def write_wav(path: str, samples: np.ndarray, rate: int = RATE) -> None:
    """16-bit mono WAV, for the self-test and for 'save cleaned audio'."""
    import wave
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((pcm * 32767.0).astype("<i2").tobytes())


def frame_power(samples: np.ndarray, frame: int = 320) -> np.ndarray:
    """Mean power of consecutive 20 ms frames."""
    n = (len(samples) // frame) * frame
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.empty(n // frame, dtype=np.float64)
    step = frame * 4096
    for i in range(0, n, step):  # in blocks, so a memory-mapped file is never loaded whole
        block = np.asarray(samples[i:min(n, i + step)], dtype=np.float32).reshape(-1, frame)
        out[i // frame:i // frame + len(block)] = np.mean(block.astype(np.float64) ** 2, axis=1)
    return out


def snr_db(samples: np.ndarray, speech: list) -> Optional[float]:
    """Rough signal-to-noise ratio: speech level against the background between speech.

    `speech` is a list of (start_sample, end_sample). The background is the
    median power of the frames that are well clear of any speech (a 0.3 s
    guard either side, so the tails of words do not count as noise), capped by
    the quietest moments of the whole file so that applause in the gaps is not
    taken for a steady background; when a recording has no gaps at all, the
    quietest frames stand in. Returns None
    when there is no speech and 99 when the background is digitally silent.
    """
    power = frame_power(samples)
    if len(power) == 0 or not speech:
        return None
    mask = np.zeros(len(power), dtype=bool)
    near = np.zeros(len(power), dtype=bool)
    guard = 15  # frames = 0.3 s
    for s, e in speech:
        a, b = s // 320, max(s // 320 + 1, e // 320)
        mask[a:b] = True
        near[max(0, a - guard):b + guard] = True
    sp = power[mask]
    if len(sp) == 0:
        return None
    rest = power[~near]
    floor = float(np.percentile(power, 8))   # the quietest moments anywhere, pauses between words included
    if len(rest) >= 50:  # at least a second of clear background to measure
        # Whichever is lower: applause or laughter in the gaps is loud but is
        # not a steady background, and must not read as a noisy recording.
        noise = min(float(np.median(rest)), max(floor, 1e-12) * 4.0) if floor > 1e-11 else float(np.median(rest))
    else:
        noise = floor
    signal = float(np.mean(sp))
    if signal <= 0:
        return None
    if noise <= 1e-11:
        return 99.0
    if signal <= noise * 1.05:  # the "speech" is no louder than the background: call it 0 dB, not minus infinity
        return float(max(-20.0, 10.0 * np.log10(signal / noise)))
    return float(min(99.0, 10.0 * np.log10((signal - noise) / noise)))
