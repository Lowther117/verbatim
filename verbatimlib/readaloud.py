"""readaloud - the shared read-aloud / word-highlight / RSVP module.

Identical in every one of Dan's reading apps (Hayloft, Lectern). One text in,
one word stream out: the same stream drives the spoken audio, the highlight
on the page and the RSVP flash, so they can never drift apart.

    doc = tokenize(text)                      # words, sentences, char offsets
    reader = Reader(settings, log=print)      # synthesis + playback engine
    bar = ReadBar(parent, app_like, ...)      # the control strip (Tk)

Engines (any can be missing - the bar only lists what works):
    kokoro   offline neural voices (kokoro-onnx + ~140 MB of model files,
             downloaded once into the per-user ReadAloud folder)
    edge     Microsoft's online neural voices (edge-tts; exact word timings)
    system   the OS voices: SAPI on Windows, `say` on macOS, espeak-ng on Linux

Timing: edge reports a timestamp for every word. The others give one audio
clip per sentence; the sentence boundaries are exact and the words inside a
sentence are placed by weight (phoneme count for kokoro, letter count
otherwise), which lands within about a tenth of a second.

Audio out: miniaudio when installed (one streaming device, also decodes
edge's MP3); otherwise one process per sentence (winsound / afplay / aplay)
timed by the wall clock.

Two clocks: with audio on, the audio position is the clock and the WPM
slider is just a readout. With audio off, RSVP runs on its own timer at the
chosen WPM and the page highlight follows it.
"""
from __future__ import annotations

import array
import base64
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave

SAMPLE_RATE = 24000
VERSION = "1.0.0"

# Two sets of model files exist and they are NOT interchangeable: the
# "model-files-v1.0" release matches kokoro-onnx 0.4.x (the last version that
# installs on Python 3.14), "model-files-v1.1" matches kokoro-onnx 0.5+. The
# wrong pair fails every sentence with "expected float, got int32".
KOKORO_FILE_NAMES = ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")
KOKORO_MB = 140


def kokoro_release():
    """Which GitHub release the installed kokoro-onnx wants."""
    try:
        from importlib.metadata import version
        v = tuple(int(x) for x in version("kokoro-onnx").split(".")[:2])
    except Exception:
        v = (0, 4)
    return "model-files-v1.0" if v < (0, 5) else "model-files-v1.1"


def kokoro_files():
    rel = kokoro_release()
    return {n: f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/{rel}/{n}" for n in KOKORO_FILE_NAMES}

# Kokoro v1.0 voice ids: first letter = language, second = f/m
KOKORO_LANG = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it",
               "j": "ja", "p": "pt-br", "z": "cmn"}
KOKORO_LANG_LABEL = {"a": "American", "b": "British", "e": "Spanish", "f": "French", "h": "Hindi",
                     "i": "Italian", "j": "Japanese", "p": "Portuguese", "z": "Mandarin"}
# Voice policy: only female voices speaking UK or US English are offered, by every engine.
VOICE_LOCALES = ("en-GB", "en-US")
KOKORO_VOICE_PREFIXES = ("bf", "af")      # b = British, a = American; f = female
KOKORO_DEFAULT_VOICES = ["bf_emma", "bf_isabella", "bf_alice", "bf_lily",
                         "af_heart", "af_bella", "af_sarah", "af_nicole", "af_sky"]
EDGE_DEFAULT_VOICES = [
    ("en-GB-SoniaNeural", "Sonia", "en-GB", "Female"), ("en-GB-LibbyNeural", "Libby", "en-GB", "Female"),
    ("en-US-AriaNeural", "Aria", "en-US", "Female"),
    ("en-US-JennyNeural", "Jenny", "en-US", "Female"), ("en-US-EmmaNeural", "Emma", "en-US", "Female"),
    ("en-US-AvaNeural", "Ava", "en-US", "Female"), ("en-US-MichelleNeural", "Michelle", "en-US", "Female"),
]
# Microsoft's child voices - the voice list reports no age, so they are named here and never offered.
EDGE_CHILD_VOICES = {"en-GB-MaisieNeural", "en-US-AnaNeural"}
# macOS `say` does not report a voice's sex, so it is asked of the system (_MAC_VOICES_JXA); these
# names are the fallback if that fails. Novelty voices (sound effects, singing) are never offered.
MAC_FEMALE_VOICES = {"Samantha", "Allison", "Ava", "Susan", "Vicki", "Victoria", "Kathy", "Zoe", "Nicky",
                     "Joelle", "Noelle", "Kate", "Serena", "Stephanie", "Martha", "Fiona", "Flo", "Grandma",
                     "Sandy", "Shelley"}
MAC_NOVELTY_VOICES = {"Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Good News", "Jester",
                      "Organ", "Pipe Organ", "Superstar", "Trinoids", "Whisper", "Wobble", "Zarvox", "Deranged",
                      "Hysterical"}

SETTINGS_DEFAULTS = {
    "tts_voice": "",           # "engine:voice id"; blank = best available
    "tts_speed": 1.0,          # 0.5 .. 3.0 (x natural)
    "tts_audio": True,         # False = RSVP/highlight on the timer, no sound
    "rsvp_wpm": 350,
    "rsvp_show": False,
    "read_autoadvance": True,  # carry on into the next chapter / page
}


# ------------------------------------------------------------------ folders
def models_dir():
    """Per-user folder shared by every app that uses this module."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "ReadAloud", "models")


def kokoro_paths():
    d = models_dir()
    return [os.path.join(d, n) for n in KOKORO_FILE_NAMES]


def _release_marker():
    return os.path.join(models_dir(), "kokoro-release.txt")


def kokoro_files_on_disk():
    return all(os.path.isfile(p) and os.path.getsize(p) > 1_000_000 for p in kokoro_paths())


def kokoro_models_present():
    """The files are there AND they are the set the installed kokoro-onnx wants."""
    if not kokoro_files_on_disk():
        return False
    try:
        with open(_release_marker(), encoding="utf-8") as fh:
            return fh.read().strip() == kokoro_release()
    except OSError:
        return False


def download_kokoro_models(progress=None, cancel=None):
    """Fetch the model files (≈140 MB) into models_dir(). progress(done, total,
    name) is called as bytes arrive; cancel() returning True aborts."""
    d = models_dir()
    os.makedirs(d, exist_ok=True)
    files = kokoro_files()
    if not kokoro_models_present():
        for p in kokoro_paths():            # a mismatched set from another kokoro-onnx version: start again
            try:
                os.remove(p)
            except OSError:
                pass
    total = 0
    sizes = {}
    for name, url in files.items():
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "ReadAloud/" + VERSION})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                sizes[name] = int(r.headers.get("Content-Length") or 0)
        except Exception:
            sizes[name] = 0
        total += sizes[name]
    done = 0
    for name, url in files.items():
        dest = os.path.join(d, name)
        if os.path.isfile(dest) and sizes.get(name) and os.path.getsize(dest) == sizes[name]:
            done += sizes[name]
            continue
        part = dest + ".part"
        req = urllib.request.Request(url, headers={"User-Agent": "ReadAloud/" + VERSION})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as fh:
            while True:
                if cancel and cancel():
                    fh.close()
                    os.remove(part)
                    raise RuntimeError("download cancelled")
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total, name)
        os.replace(part, dest)
    with open(_release_marker(), "w", encoding="utf-8") as fh:
        fh.write(kokoro_release())
    return kokoro_models_present()


# ---------------------------------------------------------------- tokenizer
class Word:
    __slots__ = ("i", "start", "end", "text", "clean", "sent", "pause", "para")

    def __init__(self, i, start, end, text, clean, pause, para):
        self.i, self.start, self.end, self.text, self.clean = i, start, end, text, clean
        self.sent = 0
        self.pause = pause          # "", "comma", "sentence", "paragraph"
        self.para = para            # first word of a paragraph

    def __repr__(self):
        return f"Word({self.i}, {self.text!r}, s{self.sent}, {self.pause})"


class Sentence:
    __slots__ = ("i", "first", "last", "start", "end", "text")

    def __init__(self, i, first, last, start, end, text):
        self.i, self.first, self.last, self.start, self.end, self.text = i, first, last, start, end, text

    @property
    def nwords(self):
        return self.last - self.first + 1


class Doc:
    """Words and sentences of one text, with character offsets into it."""

    def __init__(self, text, words, sentences):
        self.text = text
        self.words = words
        self.sentences = sentences

    def __len__(self):
        return len(self.words)

    def word_at_char(self, pos):
        """Index of the word at (or the first word after) a character offset."""
        lo, hi = 0, len(self.words)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.words[mid].end <= pos:
                lo = mid + 1
            else:
                hi = mid
        return min(lo, len(self.words) - 1) if self.words else 0

    def sentence_of(self, i):
        return self.sentences[self.words[i].sent] if self.words else None

    def speech_text(self, s):
        """What to hand the synthesiser for sentence s."""
        return _speakable(s.text)


_TOKEN = re.compile(r"\S+")
_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "no", "col", "gen",
           "lt", "sgt", "capt", "cf", "approx", "dept", "est", "fig", "inc", "ltd", "co", "mt", "pp", "vol",
           "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
_CLOSERS = "\"'’”)]}»"
_OPENERS = "\"'‘“([{«"
MAX_SENT_WORDS = 32                 # longer sentences are cut at a comma (keeps the audio start snappy)


def _clean(tok):
    t = tok.strip("\"'‘’“”()[]{}«»<>*_~`#|.,;:!?…—–-")
    return t


def tokenize(text):
    """Split text into words and sentences. Paragraphs are '\\n'-separated."""
    words = []
    pos = 0
    para_first = True
    n = len(text)
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        gap = text[pos:m.start()]
        if "\n" in gap and words:
            words[-1].pause = "paragraph"
            para_first = True
        pos = m.end()
        clean = _clean(tok)
        if not clean:
            # pure punctuation (an em dash, "***"): fold its pause into the previous word
            if words:
                if any(ch in tok for ch in ".!?…") and words[-1].pause in ("", "comma"):
                    words[-1].pause = "sentence"
                elif words[-1].pause == "":
                    words[-1].pause = "comma"
            continue
        w = Word(len(words), m.start(), m.end(), tok, clean, "", para_first)
        para_first = False
        tail = tok.rstrip(_CLOSERS)
        if tail.endswith(("…", "...")):
            w.pause = "sentence"
        elif tail and tail[-1] in ".!?":
            core = tail[:-1].lower().strip(_OPENERS)
            if tail[-1] == "." and (core in _ABBREV or (len(core) == 1 and core.isalpha())
                                    or re.fullmatch(r"[a-z]\.[a-z](\.[a-z])*", core)
                                    or re.fullmatch(r"\d+", core)):
                w.pause = "comma"       # "Mr." / "J." / "e.g." / "3." are not the end
            else:
                w.pause = "sentence"
        elif tail and tail[-1] in ",;:—–" or tok.endswith(("—", "–")):
            w.pause = "comma"
        words.append(w)
    if words and words[-1].pause in ("", "comma"):
        words[-1].pause = "sentence"

    # Sentence boundaries: paragraph > sentence punctuation, with a cap so a
    # run-on paragraph still starts speaking quickly.
    sentences = []
    first = 0
    for i, w in enumerate(words):
        end_here = w.pause in ("sentence", "paragraph") or i == len(words) - 1
        if w.pause == "sentence" and i + 1 < len(words):
            nxt = words[i + 1].text.lstrip(_OPENERS)
            # "he said. and then" - lowercase continuation after a period is usually an abbreviation we missed
            if nxt and nxt[0].islower() and w.text.rstrip(_CLOSERS).endswith(".") and not w.text.rstrip(_CLOSERS).endswith(("…", "...")):
                end_here = False
        if not end_here and i - first + 1 >= MAX_SENT_WORDS:
            # cut at the last comma in the window, else right here
            cut = None
            for j in range(i, first + MAX_SENT_WORDS // 2, -1):
                if words[j].pause == "comma":
                    cut = j
                    break
            if cut is not None and cut != i:
                _add_sentence(sentences, words, text, first, cut)
                first = cut + 1
                continue
            end_here = True
        if end_here:
            _add_sentence(sentences, words, text, first, i)
            first = i + 1
    return Doc(text, words, sentences)


def _add_sentence(sentences, words, text, first, last):
    s = Sentence(len(sentences), first, last, words[first].start, words[last].end,
                 text[words[first].start:words[last].end])
    for j in range(first, last + 1):
        words[j].sent = s.i
    sentences.append(s)


_SPEAK_MAP = {"“": '"', "”": '"', "‘": "'", "’": "'", "—": ", ", "–": ", ", "…": "...", " ": " ",
              "*": "", "_": "", "`": "", "#": "", "|": " ", "~": ""}


def _speakable(s):
    out = "".join(_SPEAK_MAP.get(ch, ch) for ch in s)
    out = re.sub(r"\s+", " ", out).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out


# --------------------------------------------------------------- timing
def word_weights(words, phoneme_lens=None):
    """Relative speaking time of each word in a sentence."""
    ws = []
    for k, w in enumerate(words):
        if phoneme_lens is not None:
            base = phoneme_lens[k] + 0.6
        else:
            letters = sum(1 for ch in w.clean if ch.isalnum())
            base = max(1, letters) + 0.8
            if any(ch.isdigit() for ch in w.clean):
                base *= 1.7           # "1,250" is read as five words
        if w.pause == "comma":
            base += 2.2
        elif w.pause in ("sentence", "paragraph"):
            base += 1.0               # trailing silence is mostly trimmed by the engines
        ws.append(base)
    return ws


def spread_times(words, duration, phoneme_lens=None, span=None):
    """[(start, end)] for the words of one clip of `duration` seconds. `span` is the
    (start, end) of the audible part (_voiced_span): the words share that, not the
    silence an engine pads each sentence with - otherwise the highlight runs ahead."""
    a, b = span if span else (0.0, duration)
    if not (0.0 <= a < b <= duration) or b - a < 0.05:
        a, b = 0.0, duration
    ws = word_weights(words, phoneme_lens)
    total = sum(ws) or 1.0
    out, t = [], a
    for wgt in ws:
        d = (b - a) * wgt / total
        out.append((t, t + d))
        t += d
    return out


def _voiced_span(pcm, rate=SAMPLE_RATE):
    """(start, end) seconds of the audible part of a clip of 16-bit mono PCM."""
    s = array.array("h")
    s.frombytes(pcm[:len(pcm) // 2 * 2])
    n = len(s)
    if not n:
        return 0.0, 0.0
    peak = max(max(s), -min(s))
    if peak < 300:                      # silence or near enough: nothing to trim against
        return 0.0, n / rate
    thr = max(200, peak // 20)
    win = max(1, rate // 100)           # 10 ms
    a = 0
    while a < n and max(map(abs, s[a:a + win])) < thr:
        a += win
    b = n
    while b > a and max(map(abs, s[max(a, b - win):b])) < thr:
        b -= win
    pad = rate // 50                    # keep 20 ms either side: soft consonants sit below the threshold
    return max(0, a - pad) / rate, min(n, b + pad) / rate


def rsvp_dwell(word, wpm):
    """Seconds to show one word in timer mode."""
    base = 60.0 / max(60, wpm)
    m = 1.0
    n = len(word.clean)
    if n > 6:
        m += min(0.9, 0.09 * (n - 6))
    if any(ch.isdigit() for ch in word.clean):
        m += 0.35
    if word.pause == "comma":
        m += 0.55
    elif word.pause == "sentence":
        m += 1.3
    elif word.pause == "paragraph":
        m += 1.9
    return base * m


def orp_index(word):
    """Optimal recognition point: the letter to pin at the pivot."""
    n = len(word)
    if n <= 1:
        return 0
    if n <= 5:
        return 1
    if n <= 9:
        return 2
    if n <= 13:
        return 3
    return 4


# --------------------------------------------------------------- audio utils
def _wav_to_pcm(path):
    """Any PCM WAV -> (mono 16-bit bytes at SAMPLE_RATE, seconds)."""
    with wave.open(path, "rb") as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        samples = array.array("h")
        samples.frombytes(raw)
        if sys.byteorder == "big":
            samples.byteswap()
    elif width == 1:
        samples = array.array("h", ((b - 128) << 8 for b in raw))
    elif width == 4:
        samples = array.array("h", (v >> 16 for v in array.array("i", raw)))
    else:
        raise ValueError(f"unsupported wav sample width {width}")
    if ch > 1:
        samples = array.array("h", (samples[i] for i in range(0, len(samples), ch)))
    if rate != SAMPLE_RATE:
        samples = _resample(samples, rate, SAMPLE_RATE)
    return samples.tobytes(), len(samples) / SAMPLE_RATE


def _resample(samples, src, dst):
    try:
        import numpy as np
        a = np.frombuffer(samples.tobytes(), dtype="<i2").astype("float32")
        n = int(len(a) * dst / src)
        x = np.linspace(0, len(a) - 1, n)
        out = np.interp(x, np.arange(len(a)), a).astype("<i2")
        r = array.array("h")
        r.frombytes(out.tobytes())
        return r
    except Exception:
        n = int(len(samples) * dst / src)
        out = array.array("h", bytes(2 * n))
        step = src / dst
        last = len(samples) - 1
        for i in range(n):
            x = i * step
            j = int(x)
            if j >= last:
                out[i] = samples[last]
            else:
                f = x - j
                out[i] = int(samples[j] * (1 - f) + samples[j + 1] * f)
        return out


def pcm_to_wav(pcm, path, rate=SAMPLE_RATE):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def _float_to_pcm(samples):
    try:
        import numpy as np
        a = np.asarray(samples, dtype="float32")
        return (np.clip(a, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    except Exception:
        out = array.array("h", (int(max(-1.0, min(1.0, float(v))) * 32767) for v in samples))
        return out.tobytes()


# ---------------------------------------------------------------- engines
class Clip:
    """One synthesised sentence: mono 16-bit PCM at SAMPLE_RATE plus per-word
    (start, end) seconds relative to the clip."""
    __slots__ = ("pcm", "seconds", "times", "first")

    def __init__(self, pcm, times):
        self.pcm = pcm
        self.seconds = len(pcm) / 2 / SAMPLE_RATE
        self.times = times
        self.first = 0              # doc index of the first word in `times`


class Engine:
    id = "base"
    label = "Base"
    online = False

    def __init__(self, log=None):
        self.log = log or (lambda *_: None)

    @classmethod
    def available(cls):
        """(ok, why-not)"""
        return False, "not implemented"

    def voices(self):
        """[(voice id, label)] in display order."""
        return []

    def default_voice(self):
        v = self.voices()
        return v[0][0] if v else ""

    def allowed(self, vid):
        """Is this voice one the list offers (a saved choice may predate the voice policy)?"""
        return any(v == vid for v, _ in self.voices())

    def synth(self, words, text, voice, speed):
        raise NotImplementedError

    def close(self):
        pass


_KOKORO = {"model": None, "lock": threading.Lock()}     # one model per process, however many readers


class KokoroEngine(Engine):
    id = "kokoro"
    label = "Kokoro (offline)"

    def __init__(self, log=None):
        super().__init__(log)
        self._lock = _KOKORO["lock"]

    @property
    def _k(self):
        return _KOKORO["model"]

    @classmethod
    def available(cls):
        try:
            import kokoro_onnx  # noqa: F401
        except Exception as e:
            return False, f"kokoro-onnx is not installed ({e.__class__.__name__})"
        if not kokoro_models_present():
            if kokoro_files_on_disk():
                return False, "model files on disk are for a different kokoro-onnx version - re-download from the Voice list"
            return False, "model files not downloaded"
        return True, ""

    @staticmethod
    def voice_label(vid):
        lang = KOKORO_LANG_LABEL.get(vid[:1], vid[:1])
        sex = {"f": "female", "m": "male"}.get(vid[1:2], "")
        name = vid.split("_", 1)[-1].capitalize()
        return f"{name} ({lang} {sex})".replace("  ", " ")

    def voices(self):
        ids = list(KOKORO_DEFAULT_VOICES)
        if self._k is not None:
            try:
                extra = [v for v in self._k.get_voices() if v not in ids and v[:2] in KOKORO_VOICE_PREFIXES]
                ids += sorted(extra)
            except Exception:
                pass
        return [(v, self.voice_label(v)) for v in ids]

    def default_voice(self):
        return "bf_emma"

    def allowed(self, vid):
        return vid[:2] in KOKORO_VOICE_PREFIXES

    def _load(self):
        if _KOKORO["model"] is None:
            from kokoro_onnx import Kokoro
            m, v = kokoro_paths()
            k = Kokoro(m, v)
            try:                                  # a one-word warm-up catches a model/package mismatch at once
                k.create("Hello.", voice="bf_emma", speed=1.0, lang="en-gb")
            except Exception as e:
                if "data type" in str(e) or "INVALID_ARGUMENT" in str(e):
                    raise RuntimeError("Kokoro model files do not match the installed kokoro-onnx - choose "
                                       "'Download Kokoro offline voices' in the Voice list to fetch the right set") from e
                raise
            _KOKORO["model"] = k
        return _KOKORO["model"]

    def synth(self, words, text, voice, speed):
        with self._lock:
            k = self._load()
            lang = KOKORO_LANG.get(voice[:1], "en-us")
            spd = max(0.5, min(2.0, float(speed)))       # the package asserts 0.5..2.0
            try:
                samples, rate = k.create(text, voice=voice, speed=spd, lang=lang)
            except TypeError:
                samples, rate = k.create(text, voice, spd, lang)
            pcm = _float_to_pcm(samples)
            if rate != SAMPLE_RATE:
                s = array.array("h")
                s.frombytes(pcm)
                pcm = _resample(s, rate, SAMPLE_RATE).tobytes()
            phon = None
            try:
                ph = k.tokenizer.phonemize(text, lang)
                chunks = [c for c in re.split(r"\s+", ph.strip()) if c]
                if len(chunks) == len(words):
                    phon = [len(c) for c in chunks]
                else:
                    # punctuation and numbers split or merge chunks: phonemise word by word instead
                    lens = [len(re.sub(r"\s+", "", k.tokenizer.phonemize(w.clean or w.text, lang)))
                            if (w.clean or w.text).strip() else 0 for w in words]
                    if any(lens):
                        phon = lens
            except Exception:
                phon = None
            seconds = len(pcm) / 2 / SAMPLE_RATE
            return Clip(pcm, spread_times(words, seconds, phon, _voiced_span(pcm)))


class EdgeEngine(Engine):
    id = "edge"
    label = "Online (Microsoft)"
    online = True

    @classmethod
    def available(cls):
        try:
            import edge_tts  # noqa: F401
        except Exception as e:
            return False, f"edge-tts is not installed ({e.__class__.__name__})"
        try:
            import miniaudio  # noqa: F401
        except Exception as e:
            return False, f"miniaudio could not be imported ({e.__class__.__name__}: {e}) - needed to decode the audio"
        return True, ""

    def _cache_path(self):
        return os.path.join(models_dir(), "edge-voices.json")

    def voices(self):
        try:
            with open(self._cache_path(), encoding="utf-8") as fh:
                data = json.load(fh)
            out = []
            for v in data:
                sn = v.get("ShortName", "")
                loc = v.get("Locale", "")
                if not sn or loc not in VOICE_LOCALES or v.get("Gender") != "Female" or sn in EDGE_CHILD_VOICES:
                    continue
                name = sn.split("-")[-1].replace("Neural", "")
                out.append((sn, f"{name} ({loc} female)"))
            out.sort(key=lambda x: (VOICE_LOCALES.index(x[0][:5]) if x[0][:5] in VOICE_LOCALES else 9, x[1]))
            if out:
                return out
        except Exception:
            pass
        return [(sn, f"{name} ({loc} {g.lower()})") for sn, name, loc, g in EDGE_DEFAULT_VOICES]

    def default_voice(self):
        return "en-GB-SoniaNeural"

    def refresh_voices(self):
        """Fetch the full voice list from Microsoft (needs the network)."""
        import asyncio
        import edge_tts
        data = asyncio.run(edge_tts.list_voices())
        os.makedirs(models_dir(), exist_ok=True)
        with open(self._cache_path(), "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return len(data)

    def synth(self, words, text, voice, speed):
        import asyncio
        import edge_tts
        import miniaudio
        # the service sends no audio past double speed
        pct = int(round((max(0.5, min(2.0, float(speed))) - 1.0) * 100))
        rate = f"{pct:+d}%"

        async def run():
            try:
                # edge-tts 7+ sends sentence boundaries unless asked for words - without them the
                # highlight could only be spread evenly across each sentence
                comm = edge_tts.Communicate(text, voice=voice, rate=rate, boundary="WordBoundary")
            except TypeError:           # older edge-tts: no option, word boundaries always on
                comm = edge_tts.Communicate(text, voice=voice, rate=rate)
            audio = bytearray()
            marks = []
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    audio.extend(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    marks.append((chunk["offset"] / 1e7, chunk["duration"] / 1e7, chunk.get("text", "")))
            return bytes(audio), marks

        err = None
        for attempt in range(2):            # the service drops the odd request: one quiet retry
            try:
                audio, marks = asyncio.run(run())
                if audio:
                    break
                err = RuntimeError("no audio returned")
            except Exception as e:
                err = e
            if attempt == 0:
                time.sleep(0.4)
        else:
            raise err
        dec = miniaudio.decode(audio, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1,
                               sample_rate=SAMPLE_RATE)
        pcm = dec.samples.tobytes()
        seconds = len(pcm) / 2 / SAMPLE_RATE
        return Clip(pcm, _align_marks(words, marks, seconds, _voiced_span(pcm)))


def _align_marks(words, marks, seconds, span=None):
    """Map edge's word boundaries onto our tokens; interpolate anything unmatched."""
    times = [None] * len(words)
    if not marks:
        return spread_times(words, seconds, span=span)
    k = 0
    for off, dur, txt in marks:
        key = re.sub(r"[^\w]", "", txt).lower()
        hit = None
        for j in range(k, min(len(words), k + 4)):
            wk = re.sub(r"[^\w]", "", words[j].clean).lower()
            if key and (wk == key or wk.startswith(key) or key.startswith(wk) and wk):
                hit = j
                break
        if hit is None:
            if k < len(words):
                hit = k
            else:
                break
        if times[hit] is None:
            times[hit] = [off, off + dur]
        else:
            times[hit][1] = max(times[hit][1], off + dur)
        k = hit + 1
    # fill gaps by sharing the span between known neighbours
    i = 0
    n = len(words)
    while i < n:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < n and times[j] is None:
            j += 1
        start = times[i - 1][1] if i > 0 else 0.0
        end = times[j][0] if j < n else seconds
        if end <= start:
            end = start + 0.05 * (j - i)
        step = (end - start) / (j - i)
        for q in range(i, j):
            times[q] = [start + step * (q - i), start + step * (q - i + 1)]
        i = j
    # ends must not overlap the next start
    for q in range(n - 1):
        if times[q][1] > times[q + 1][0]:
            times[q][1] = times[q + 1][0]
    times[-1][1] = max(times[-1][1], min(seconds, times[-1][0] + 0.05))
    return [(a, b) for a, b in times]


_SAPI_SERVER = r"""
$ErrorActionPreference = 'Continue'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(24000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$in = [Console]::In
while ($true) {
    $line = $in.ReadLine()
    if ($line -eq $null) { break }
    if ($line.Length -eq 0) { continue }
    try {
        $parts = $line.Split('|')
        $text = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($parts[0]))
        $voice = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($parts[1]))
        $rate = [int]$parts[2]
        $out = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($parts[3]))
        if ($voice.Length -gt 0) { try { $s.SelectVoice($voice) } catch { } }
        $s.Rate = $rate
        $s.SetOutputToWaveFile($out, $fmt)
        $s.Speak($text)
        $s.SetOutputToNull()
        [Console]::Out.WriteLine('ok')
    } catch {
        try { $s.SetOutputToNull() } catch { }
        [Console]::Out.WriteLine('err ' + $_.Exception.Message)
    }
    [Console]::Out.Flush()
}
"""
_MAC_VOICES_JXA = r"""
ObjC.import('AppKit');
var vs = $.NSSpeechSynthesizer.availableVoices, out = [];
for (var i = 0; i < vs.count; i++) {
  var a = $.NSSpeechSynthesizer.attributesForVoice(vs.objectAtIndex(i));
  out.push([ObjC.unwrap(a.objectForKey('VoiceName')), ObjC.unwrap(a.objectForKey('VoiceLocaleIdentifier')),
            ObjC.unwrap(a.objectForKey('VoiceGender'))].join('|'));
}
out.join('\n');
"""
_SAPI_VOICES = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "foreach ($v in $s.GetInstalledVoices()) { if ($v.Enabled) { [Console]::Out.WriteLine($v.VoiceInfo.Name + '|' "
                "+ $v.VoiceInfo.Gender + '|' + $v.VoiceInfo.Culture.Name) } }")


def _hidden_popen_kwargs():
    kw = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        kw["startupinfo"] = si
        kw["creationflags"] = 0x08000000      # CREATE_NO_WINDOW
    return kw


def _b64(s):
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


class SystemEngine(Engine):
    id = "system"
    label = "System voices"

    _cache = None               # voice list, shared by every instance in the process

    def __init__(self, log=None):
        super().__init__(log)
        self._proc = None
        self._lock = threading.Lock()
        self._tmp = tempfile.mkdtemp(prefix="readaloud-")

    @classmethod
    def available(cls):
        if os.name == "nt":
            return True, ""
        if sys.platform == "darwin":
            return (True, "") if shutil.which("say") else (False, "`say` not found")
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            return True, ""
        return False, "espeak-ng is not installed"

    @staticmethod
    def is_natural(vid):
        """macOS Enhanced / Premium voices are neural; SAPI desktop, classic macOS and espeak are not."""
        return sys.platform == "darwin" and bool(re.search(r"\((Enhanced|Premium)\)", vid or ""))

    def voices(self, fetch=True):
        """The female UK/US voices this system offers - and while a neural engine (Kokoro or the
        online voices) is available, only the natural-sounding ones among them: the robotic
        voices are kept for when there is nothing better."""
        full = self._os_voices(fetch)
        if neural_engine_available():
            return [x for x in full if self.is_natural(x[0])]
        return full

    def _os_voices(self, fetch=True):
        """The OS voice list. Asking the OS takes up to a second or two
        (PowerShell), so fetch=False returns the cached list or a placeholder."""
        if SystemEngine._cache is not None:
            return SystemEngine._cache
        if not fetch:
            return [("", "Default voice")]
        out = []
        try:
            if os.name == "nt":
                kw = _hidden_popen_kwargs()
                kw["stdin"] = subprocess.DEVNULL
                p = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", _SAPI_VOICES],
                                   capture_output=True, text=True, timeout=30,
                                   startupinfo=kw.get("startupinfo"), creationflags=kw.get("creationflags", 0))
                for line in p.stdout.splitlines():
                    parts = line.strip().split("|")
                    if len(parts) >= 3 and parts[0] and parts[1] == "Female" and parts[2] in VOICE_LOCALES:
                        out.append((parts[0], f"{parts[0].replace('Microsoft ', '').replace(' Desktop', '')} "
                                              f"({parts[2]} female)"))
                out.sort(key=lambda x: (VOICE_LOCALES.index(x[1].split("(")[-1][:5]), x[0]))
            elif sys.platform == "darwin":
                sexes = self._mac_voice_sexes()
                p = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=30)
                for line in p.stdout.splitlines():
                    # anchored on the locale: a name that fills its column ("Samantha (Enhanced)",
                    # "Eddy (English (UK))") is followed by one space, not two
                    m = re.match(r"^(.+?)\s+([a-z]{2,3}_[A-Za-z0-9_\-]+)\s+#", line)
                    if not m:
                        continue
                    name, loc = m.group(1).strip(), m.group(2)
                    base = re.sub(r"\s*\(.*$", "", name)
                    if not loc.startswith(("en_GB", "en_US")) or base in MAC_NOVELTY_VOICES:
                        continue
                    sex = sexes.get(name) or sexes.get(base)      # the system may call "Flo (English (UK))" just "Flo"
                    if (sex.endswith("Female") if sex else base in MAC_FEMALE_VOICES):
                        out.append((name, f"{name} ({loc[:5].replace('_', '-')} female)"))
                out.sort(key=lambda x: (not x[1].endswith("(en-GB female)"), x[0]))
            else:
                exe = shutil.which("espeak-ng") or shutil.which("espeak")
                p = subprocess.run([exe, "--voices=en"], capture_output=True, text=True, timeout=30)
                for line in p.stdout.splitlines()[1:]:
                    parts = line.split()
                    if len(parts) >= 4 and parts[1] in ("en-gb", "en-us"):
                        out.append((parts[1] + "+f3", f"{parts[3]} ({parts[1]} female)"))   # +f3: female variant
        except Exception as e:
            self.log(f"system voices: {e!r}")
        if not out:
            out = [("", "Default voice")]
        SystemEngine._cache = out
        return out

    @staticmethod
    def _mac_voice_sexes():
        """{voice name: 'VoiceGenderFemale' / 'VoiceGenderMale' / ...} from the system, or {}."""
        try:
            p = subprocess.run(["osascript", "-l", "JavaScript", "-e", _MAC_VOICES_JXA],
                               capture_output=True, text=True, timeout=20)
            out = {}
            for line in p.stdout.splitlines():
                parts = line.split("|")
                if len(parts) == 3 and parts[0] and parts[2]:
                    out[parts[0]] = parts[2]
            return out
        except Exception:
            return {}

    def default_voice(self, fetch=True):
        v = self.voices(fetch=fetch)          # sorted UK first; fetched once, then cached
        return v[0][0] if v else ""

    def allowed(self, vid):
        return any(v == vid for v, _ in self.voices())

    def _rate_param(self, speed):
        speed = max(0.5, min(3.0, float(speed)))
        if os.name == "nt":
            return int(round(math.log(speed) / math.log(3) * 10))      # -10 .. 10
        return int(round(175 * speed))                                 # words per minute

    def synth(self, words, text, voice, speed):
        with self._lock:
            out = os.path.join(self._tmp, f"s{int(time.time() * 1000) % 100000000}.wav")
            if os.name == "nt":
                self._sapi(text, voice, self._rate_param(speed), out)
            elif sys.platform == "darwin":
                src = out[:-4] + ".txt"
                with open(src, "w", encoding="utf-8") as fh:
                    fh.write(text)
                cmd = ["say", "-r", str(self._rate_param(speed)), "-o", out, "--file-format=WAVE",
                       "--data-format=LEI16@24000", "-f", src]
                if voice:
                    cmd[1:1] = ["-v", voice]
                subprocess.run(cmd, check=True, timeout=120, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(src)
            else:
                exe = shutil.which("espeak-ng") or shutil.which("espeak")
                cmd = [exe, "-s", str(self._rate_param(speed)), "-w", out, "--", text]
                if voice:
                    cmd[1:1] = ["-v", voice]
                subprocess.run(cmd, check=True, timeout=120, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pcm, seconds = _wav_to_pcm(out)
            try:
                os.remove(out)
            except OSError:
                pass
            return Clip(pcm, spread_times(words, seconds, span=_voiced_span(pcm)))

    def _sapi(self, text, voice, rate, out):
        if self._proc is None or self._proc.poll() is not None:
            script = os.path.join(self._tmp, "sapi_server.ps1")
            with open(script, "w", encoding="utf-8-sig") as fh:
                fh.write(_SAPI_SERVER)
            self._proc = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script],
                                          text=True, encoding="utf-8", errors="replace", bufsize=1,
                                          **_hidden_popen_kwargs())
        p = self._proc
        p.stdin.write(f"{_b64(text)}|{_b64(voice or '')}|{rate}|{_b64(out)}\n")
        p.stdin.flush()
        line = p.stdout.readline().strip()
        if line != "ok":
            raise RuntimeError(line[4:] if line.startswith("err ") else f"voice server stopped ({line!r})")

    def close(self):
        p, self._proc = self._proc, None
        if p is not None:
            try:
                p.stdin.close()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        shutil.rmtree(self._tmp, ignore_errors=True)


ENGINES = {"kokoro": KokoroEngine, "edge": EdgeEngine, "system": SystemEngine}
ENGINE_ORDER = ("kokoro", "edge", "system")


def neural_engine_available():
    """Kokoro or the online voices can speak - so the robotic system voices are not offered."""
    return KokoroEngine.available()[0] or EdgeEngine.available()[0]


def engine_status():
    """{engine id: (ok, reason)} - cheap, never loads a model."""
    return {k: cls.available() for k, cls in ENGINES.items()}


# ---------------------------------------------------------------- playback
MA_PERIODS = 3                  # periods miniaudio keeps queued when callback_periods is left at 0


class _Timeline:
    """Which sentence/word the audio output is on. Shared by both players."""

    def __init__(self):
        self.lock = threading.Lock()
        self.segs = []          # [(sentence index, nframes)], in play order
        self.cur = 0            # index into segs
        self.frame = 0          # frames consumed inside segs[cur]
        self.generation = 0

    def reset(self, generation):
        with self.lock:
            self.segs = []
            self.cur = 0
            self.frame = 0
            self.generation = generation

    def position(self):
        with self.lock:
            if self.cur < len(self.segs):
                return self.segs[self.cur][0], self.frame / SAMPLE_RATE, False
            if self.segs:
                return self.segs[-1][0], self.segs[-1][1] / SAMPLE_RATE, True
            return None, 0.0, True


class _MiniaudioOut:
    """Streams clips through one miniaudio device; silence while waiting."""

    def __init__(self):
        import miniaudio
        self._ma = miniaudio
        self.tl = _Timeline()
        self.paused = False
        self._clips = {}            # position in segs -> pcm bytes
        self._offset = 0            # bytes consumed in the current clip
        # What is audible lags what the callback hands over by the periods queued ahead of it
        # (buffersize_msec is miniaudio's period; it keeps MA_PERIODS of them) plus the OS mixer.
        self.latency = 0.02
        self._dev_frames = 0        # frames handed to the device, silence included
        self._chunk = 0.08          # seconds asked for by the last callback
        self._stamp = None          # monotonic time of the last callback
        self._hist = []             # [(device frame, segs position, clip frame, nframes)] handed lately
        self.dev = miniaudio.PlaybackDevice(output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1,
                                            sample_rate=SAMPLE_RATE, buffersize_msec=80)
        g = self._gen()
        next(g)
        self.dev.start(g)

    def _gen(self):
        need = yield b""
        while True:
            nbytes = need * 2
            out = bytearray()
            with self.tl.lock:
                base = self._dev_frames
                if not self.paused:
                    while len(out) < nbytes and self.tl.cur < len(self.tl.segs):
                        pcm = self._clips.get(self.tl.cur)
                        if pcm is None:
                            break
                        take = min(nbytes - len(out), len(pcm) - self._offset)
                        if take > 0:
                            self._hist.append((base + len(out) // 2, self.tl.cur, self._offset // 2, take // 2))
                        out += pcm[self._offset:self._offset + take]
                        self._offset += take
                        self.tl.frame = self._offset // 2
                        if self._offset >= len(pcm):
                            self._clips.pop(self.tl.cur, None)
                            self.tl.cur += 1
                            self._offset = 0
                            self.tl.frame = 0
                del self._hist[:-64]
                self._dev_frames = base + need
                self._chunk = need / SAMPLE_RATE
                self._stamp = time.monotonic()
            if len(out) < nbytes:
                out += bytes(nbytes - len(out))
            need = yield bytes(out)

    def reset(self, generation):
        with self.tl.lock:
            self._clips = {}
            self._offset = 0
            self._hist = []
            self.tl.segs, self.tl.cur, self.tl.frame, self.tl.generation = [], 0, 0, generation

    def enqueue(self, sent_idx, pcm, generation):
        with self.tl.lock:
            if self.tl.generation != generation:
                return
            self._clips[len(self.tl.segs)] = pcm
            self.tl.segs.append((sent_idx, len(pcm) // 2))

    def position(self):
        """(sentence, seconds into it, done) of what is audible now - not of what the callback
        last handed over, which runs ahead by the device buffer, steps a whole period at a
        time and crosses into the next sentence while the last one is still sounding."""
        with self.tl.lock:
            segs, cur = self.tl.segs, self.tl.cur
            if self._hist and self._stamp is not None:
                since = min(max(0.0, time.monotonic() - self._stamp), self._chunk)
                aud = self._dev_frames - (MA_PERIODS * self._chunk + self.latency - since) * SAMPLE_RATE
                pick = None
                for span in reversed(self._hist):
                    if span[0] <= aud:
                        pick = span
                        break
                if pick is None:                    # this run's audio is not out of the buffer yet
                    pick, frame = self._hist[0], self._hist[0][2]
                else:                               # past the span's end: silence (paused / waiting)
                    frame = pick[2] + min(pick[3], aud - pick[0])
                last = self._hist[-1]
                if not (cur >= len(segs) and aud >= last[0] + last[3]):
                    return segs[pick[1]][0], frame / SAMPLE_RATE, False
            elif cur < len(segs):
                return segs[cur][0], self.tl.frame / SAMPLE_RATE, False
            if segs:
                return segs[-1][0], segs[-1][1] / SAMPLE_RATE, True
            return None, 0.0, True

    def waiting(self):
        with self.tl.lock:
            return self.tl.cur >= len(self.tl.segs)

    def set_paused(self, p):
        with self.tl.lock:
            self.paused = p

    def close(self):
        try:
            self.dev.stop()
            self.dev.close()
        except Exception:
            pass


class _ProcessOut:
    """No miniaudio: one OS player process per sentence, wall-clock timed."""

    def __init__(self):
        self.tl = _Timeline()
        self.paused = False
        self._clips = {}
        self._proc = None
        self._started = None          # wall clock when the current clip started
        self._pause_at = 0.0
        self._tmp = tempfile.mkdtemp(prefix="readaloud-play-")
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        self.latency = 0.05

    def _play_file(self, path):
        if os.name == "nt":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            return None
        exe = "afplay" if sys.platform == "darwin" else (shutil.which("aplay") or shutil.which("paplay") or "aplay")
        return subprocess.Popen([exe, path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

    def _kill(self):
        if os.name == "nt":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        elif self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(0.02)
            with self.tl.lock:
                if self.paused or self.tl.cur >= len(self.tl.segs):
                    continue
                pcm = self._clips.get(self.tl.cur)
                if pcm is None:
                    continue
                nframes = self.tl.segs[self.tl.cur][1]
                if self._started is None:
                    # (re)start this clip from _pause_at
                    skip = int(self._pause_at * SAMPLE_RATE) * 2
                    path = os.path.join(self._tmp, f"p{self.tl.cur}.wav")
                    try:
                        pcm_to_wav(pcm[skip:], path)
                        self._proc = self._play_file(path)
                    except Exception:
                        # no player / locked file / winsound refused: carry on silently on the clock rather
                        # than let this thread die and the highlight freeze
                        self._proc = None
                    self._started = time.monotonic() - self._pause_at
                elapsed = time.monotonic() - self._started
                self.tl.frame = min(nframes, int(elapsed * SAMPLE_RATE))
                if elapsed >= nframes / SAMPLE_RATE + 0.02:
                    if self._proc is not None and self._proc.poll() is None and elapsed < nframes / SAMPLE_RATE + 2.0:
                        continue    # the player started late and is still sounding: the next one would talk over it
                    try:
                        os.remove(os.path.join(self._tmp, f"p{self.tl.cur}.wav"))   # one WAV per sentence otherwise piles up until close
                    except OSError:
                        pass
                    self._clips.pop(self.tl.cur, None)
                    self.tl.cur += 1
                    self.tl.frame = 0
                    self._started = None
                    self._pause_at = 0.0
                    self._proc = None

    def reset(self, generation):
        with self.tl.lock:
            self._kill()
            self._clips = {}
            self._started = None
            self._pause_at = 0.0
            self.tl.segs, self.tl.cur, self.tl.frame, self.tl.generation = [], 0, 0, generation

    def enqueue(self, sent_idx, pcm, generation):
        with self.tl.lock:
            if self.tl.generation != generation:
                return
            self._clips[len(self.tl.segs)] = pcm
            self.tl.segs.append((sent_idx, len(pcm) // 2))

    def position(self):
        s, t, done = self.tl.position()
        return s, max(0.0, t - self.latency), done      # the player process takes a moment to start sounding

    def waiting(self):
        with self.tl.lock:
            return self.tl.cur >= len(self.tl.segs)

    def set_paused(self, p):
        with self.tl.lock:
            if p and not self.paused:
                if self._started is not None:
                    self._pause_at = time.monotonic() - self._started
                self._kill()
                self._started = None
            self.paused = p

    def close(self):
        self._stop.set()
        with self.tl.lock:
            self._kill()
        shutil.rmtree(self._tmp, ignore_errors=True)


def _make_output():
    try:
        return _MiniaudioOut()
    except Exception:
        return _ProcessOut()


# ------------------------------------------------------------------ reader
class Reader:
    """Turns a Doc into speech with a word clock. All methods are safe to call
    from the Tk thread; synthesis runs on its own thread and never touches
    widgets. Poll `current()` from a Tk after-loop."""

    LOOKAHEAD = 3

    def __init__(self, settings, log=None):
        self.s = settings
        self.log = log or (lambda *_: None)
        self.doc = None
        self.state = "idle"             # idle / loading / buffering / playing / paused / done / error
        self.error = ""
        self.progress = ""              # a one-line note for the status label
        self._fallback = None           # (generation, engine, voice) used after the chosen engine failed
        self.engines = ENGINES          # a host can narrow these (Verbatim reads its recording, no voices)
        self.engine_order = ENGINE_ORDER
        self._engines = {}
        self._out = None
        self._gen = 0
        self._clips = {}                # sentence index -> Clip (this generation)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._from = 0                  # first sentence of the current run
        self._next = 0                  # next sentence to synthesise
        self._thread = None
        self._start_word = 0
        self._cursor = 0                # last word reported

    # -- configuration
    def voice(self, fetch=True):
        """(engine id, voice id) to speak with. The Tk thread passes fetch=False: checking a
        system voice means asking the OS for its list (seconds, the first time), which only
        the synthesis thread may wait on - until then a saved system voice is taken as is."""
        v = self.s.get("tts_voice") or ""
        if ":" in v:
            eng, vid = v.split(":", 1)
            if eng in self.engines and self.engines[eng].available()[0]:
                e = self.engine(eng)
                if eng == "system" and neural_engine_available() and not SystemEngine.is_natural(vid):
                    ok = False                       # robotic, and something natural is there: decided by name, no fetch
                elif eng == "system" and not fetch and SystemEngine._cache is None:
                    return eng, vid
                else:
                    try:
                        ok = e.allowed(vid)
                    except Exception:
                        ok = False
                if ok:
                    return eng, vid
                # a saved voice the list no longer offers (male, other accents, robotic) gives way to that
                # engine's default - or, if it has none left to offer, to the first engine that has
                d = e.default_voice(fetch=False) if eng == "system" and not fetch else e.default_voice()
                if d:
                    return eng, d
        for eng in self.engine_order:
            if self.engines[eng].available()[0]:
                e = self.engine(eng)
                return eng, (e.default_voice(fetch=False) if eng == "system" and not fetch else e.default_voice())
        return "system", ""

    def engine(self, eid):
        e = self._engines.get(eid)
        if e is None:
            e = self._engines[eid] = self.engines[eid](self.log)
        return e

    def speed(self):
        try:
            return max(0.5, min(3.0, float(self.s.get("tts_speed") or 1.0)))
        except (TypeError, ValueError):
            return 1.0

    # -- control
    def load(self, doc):
        self.stop()
        self.doc = doc

    def play(self, word_idx=0):
        """Start (or restart) speaking from a word."""
        if self.doc is None or not self.doc.words:
            return
        word_idx = max(0, min(len(self.doc.words) - 1, word_idx))
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._clips = {}
            s = self.doc.words[word_idx].sent
            self._from = s
            self._next = s
            self._start_word = word_idx
            self._cursor = word_idx
            self.error = ""
            self.progress = ""
            self.state = "loading"
            if self._out is None:
                self._out = _make_output()
            self._out.reset(gen)
            self._out.set_paused(False)
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._synth_loop, daemon=True)
            self._thread.start()
        self._wake.set()

    def pause(self):
        with self._lock:            # the synth thread moves loading -> playing under this lock
            if self.state in ("playing", "buffering", "loading"):
                self.state = "paused"
                if self._out:
                    self._out.set_paused(True)

    def resume(self):
        with self._lock:
            if self.state == "paused":
                self.state = "playing"
                if self._out:
                    self._out.set_paused(False)

    def toggle(self):
        if self.state == "paused":
            self.resume()
        elif self.state in ("playing", "buffering", "loading"):
            self.pause()
        else:
            self.play(self._cursor)

    def stop(self):
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._clips = {}
            self.state = "idle"
        if self._out:
            self._out.reset(gen)
            self._out.set_paused(False)

    def seek_sentence(self, delta):
        if self.doc is None or not self.doc.words:
            return
        s = self.doc.words[self._cursor].sent + delta
        s = max(0, min(len(self.doc.sentences) - 1, s))
        was_paused = self.state == "paused"
        self.play(self.doc.sentences[s].first)
        if was_paused:
            self.pause()

    def set_speed(self, speed):
        self.s["tts_speed"] = float(speed)
        if self.state in ("playing", "buffering", "loading", "paused"):
            was_paused = self.state == "paused"
            self.play(self._cursor)
            if was_paused:
                self.pause()

    def set_voice(self, eng, vid):
        self.s["tts_voice"] = f"{eng}:{vid}"
        if self.state in ("playing", "buffering", "loading", "paused"):
            was_paused = self.state == "paused"
            self.play(self._cursor)
            if was_paused:
                self.pause()

    def current(self):
        """(word index or None, state) - call from the UI clock."""
        if self.doc is None or self._out is None or self.state in ("idle", "error"):
            return None, self.state
        if self.state == "done":
            return self._cursor, "done"
        sent, t, done = self._out.position()
        if sent is None:
            return self._cursor, self.state
        clip = self._clips.get(sent)
        if clip is None:
            return self._cursor, self.state
        s = self.doc.sentences[sent]
        idx = clip.first
        for k, (a, b) in enumerate(clip.times):
            if t >= a:
                idx = clip.first + k
            else:
                break
        with self._lock:
            if self.state in ("idle", "error"):     # the synth thread failed since the check above: keep its error
                return None, self.state
            if done and sent >= len(self.doc.sentences) - 1 and self._next >= len(self.doc.sentences):
                self.state = "done"
                self._cursor = s.last
                return self._cursor, "done"
            if self.state == "paused":
                pass
            elif done:
                self.state = "buffering"
            else:
                self.state = "playing"
            self._cursor = idx
            return idx, self.state

    def close(self):
        self.stop()
        self._stop.set()
        self._wake.set()
        if self._out:
            self._out.close()
            self._out = None
        for e in self._engines.values():
            try:
                e.close()
            except Exception:
                pass

    # -- synthesis thread
    def _synth_loop(self):
        while not self._stop.is_set():
            self._wake.wait(0.1)
            self._wake.clear()
            while not self._stop.is_set():
                with self._lock:
                    gen = self._gen
                    doc = self.doc
                    nxt = self._next
                    if doc is None or self.state in ("idle", "error", "done") or nxt >= len(doc.sentences):
                        break
                    playing_sent = self._out.position()[0] if self._out else None
                    if playing_sent is None:
                        playing_sent = self._from
                    if nxt > playing_sent + self.LOOKAHEAD:
                        break
                    sent = doc.sentences[nxt]
                    speed = self.speed()
                    frm, start_word = self._from, self._start_word     # same snapshot as doc/nxt: a play() now cannot mix runs
                # outside the lock: the first look at the system voice list can take a second or two, and the
                # UI clock must not wait on it (a voice changed meanwhile is caught by the generation check below)
                eng_id, vid = self.voice()
                fb = self._fallback
                if fb is not None and fb[0] == gen:      # the chosen engine already failed this run: don't wait on it again
                    eng_id, vid = fb[1], fb[2]
                first = sent.first
                if nxt == frm and start_word > sent.first:
                    first = start_word          # clicked mid-sentence: speak from that word
                words = doc.words[first:sent.last + 1]
                text = _speakable(doc.text[words[0].start:sent.end])
                try:
                    clip = self.engine(eng_id).synth(words, text, vid, speed)
                    clip.first = first
                except Exception as e:
                    self.log(f"read aloud: {eng_id} failed on sentence {nxt}: {e!r}")
                    clip = None
                    err = f"{self.engines[eng_id].label}: {e}"
                    # natural voices first (Kokoro is offline, so it survives a dropped connection), the
                    # system voice last; whatever works carries on for the rest of this run
                    for alt in self.engine_order:
                        if alt == eng_id or not self.engines[alt].available()[0]:
                            continue
                        try:
                            ae = self.engine(alt)
                            av = ae.default_voice()
                            clip = ae.synth(words, text, av, speed)
                            clip.first = first
                            self._fallback = (gen, alt, av)
                            self.progress = f"{self.engines[eng_id].label} failed - using {self.engines[alt].label}"
                            break
                        except Exception as e2:
                            clip = None
                            err += f"\n{self.engines[alt].label}: {e2}"
                    if clip is None:
                        with self._lock:
                            if gen == self._gen:          # an abandoned run must not kill the new one
                                self.error = err
                                self.state = "error"
                        break
                with self._lock:
                    if gen != self._gen:
                        continue
                    self._clips[nxt] = clip
                    self._next = nxt + 1
                    if self.state == "loading":
                        self.state = "playing"
                    out = self._out
                if out is not None:
                    out.enqueue(nxt, clip.pcm, gen)
                # a short sleep lets the UI thread in between sentences
                time.sleep(0.005)


# ------------------------------------------------------------------ Tk UI
def _tk():
    import tkinter as tk
    from tkinter import ttk, font as tkfont
    return tk, ttk, tkfont


class TextHighlighter:
    """Paints the current word / sentence in a tk.Text that holds doc.text."""

    def __init__(self, text, doc, palette):
        self.text = text
        self.doc = doc
        self.cur = None
        # Tk 8.6 counts a character outside the BMP (an emoji) as two; Python (and Tk 9) as one.
        # Measure rather than assume: if Tk's count matches Python's, no correction is needed.
        # line starts, so an offset becomes "line.col" by bisection - Tk's own "1.0+Nc" walks every
        # line from the top, which made each highlighted word slower the further into a book it was
        self._lines = [0] + [k + 1 for k, ch in enumerate(doc.text) if ch == "\n"]
        self._astral = [k for k, ch in enumerate(doc.text) if ord(ch) > 0xFFFF]
        if self._astral:
            try:
                tk_chars = int(self.text.tk.call(self.text._w, "count", "-chars", "1.0", "end-1c"))
                if tk_chars == len(doc.text):
                    self._astral = []
            except Exception:
                pass
        self.text.tag_configure("ra_sent")
        self.text.tag_configure("ra_word")
        self.text.tag_raise("ra_sent")
        self.text.tag_raise("ra_word")
        self.apply_palette(palette)

    def apply_palette(self, c):
        self.text.tag_configure("ra_sent", background=c["sel"])
        self.text.tag_configure("ra_word", background=c["accent"], foreground=c["accent_text"])

    def _idx(self, off):
        import bisect
        off = max(0, min(off, len(self.doc.text)))
        line = bisect.bisect_right(self._lines, off) - 1
        col = off - self._lines[line]
        if self._astral:        # Tk 8.6 counts each character outside the BMP as two columns
            col += bisect.bisect_left(self._astral, off) - bisect.bisect_left(self._astral, self._lines[line])
        return f"{line + 1}.{col}"

    def show(self, i):
        if i == self.cur:
            return
        t = self.text
        t.tag_remove("ra_word", "1.0", "end")
        if i is None:
            t.tag_remove("ra_sent", "1.0", "end")
            self.cur = None
            return
        w = self.doc.words[i]
        if self.cur is None or self.doc.words[self.cur].sent != w.sent:
            t.tag_remove("ra_sent", "1.0", "end")
            s = self.doc.sentences[w.sent]
            t.tag_add("ra_sent", self._idx(s.start), self._idx(s.end))
        t.tag_add("ra_word", self._idx(w.start), self._idx(w.end))
        self.cur = i
        try:
            info = t.dlineinfo(self._idx(w.start))
            h = t.winfo_height()
            if info is None or info[1] < h * 0.15 or info[1] + info[3] > h * 0.8:
                t.see(self._idx(w.start))
                # keep the line about a third of the way down
                info = t.dlineinfo(self._idx(w.start))
                if info is not None and h > 0:
                    frac = t.yview()[0] + (info[1] - h * 0.33) / max(1, self._pixel_height())
                    t.yview_moveto(max(0.0, frac))
        except Exception:
            pass

    def _pixel_height(self):
        try:
            first, last = self.text.yview()
            vis = max(1e-6, last - first)
            return self.text.winfo_height() / vis
        except Exception:
            return 1

    def clear(self):
        self.show(None)

    def word_at(self, tk_index):
        """Word index under a Text index like '@x,y'."""
        try:
            line, col = map(int, self.text.index(tk_index).split("."))
        except Exception:
            return None
        line = max(1, min(line, len(self._lines)))      # the line starts, not a split of the whole book per click
        start = self._lines[line - 1]
        cur = self.doc.text[start:self._lines[line] - 1 if line < len(self._lines) else len(self.doc.text)]
        if not self._astral:
            # Tk's count matches Python's (no astral chars, or Tk 9): the column is the offset
            return self.doc.word_at_char(start + min(col, len(cur)))
        # Tk 8.6's column counts a character outside the BMP as two; walk back to Python's count
        py = seen = 0
        while seen < col and py < len(cur):
            seen += 2 if ord(cur[py]) > 0xFFFF else 1
            py += 1
        return self.doc.word_at_char(start + py)


class RsvpView:
    """The flash panel: one word, pivot letter in the accent colour."""

    def __init__(self, parent, palette, ui_family="Segoe UI", height=120):
        tk, ttk, tkfont = _tk()
        self.tk = tk
        self.frame = ttk.Frame(parent)
        self.canvas = tk.Canvas(self.frame, height=height, highlightthickness=1, bd=0)
        self.canvas.pack(fill="x", expand=True)
        self.font = tkfont.Font(family=ui_family, size=30, weight="bold")
        self.small = tkfont.Font(family=ui_family, size=10)
        self.c = palette
        self.word = None
        self.note = ""
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.apply_palette(palette)

    def apply_palette(self, c):
        self.c = c
        self.canvas.configure(bg=c["sidebar"], highlightbackground=c["border"], highlightcolor=c["border"])
        self.redraw()

    def set_size(self, size):
        self.font.configure(size=size)
        self.redraw()

    def show(self, word, note=""):
        self.word = word
        self.note = note
        self.redraw()

    def redraw(self):
        cv = self.canvas
        cv.delete("all")
        W = cv.winfo_width() or 600
        H = int(cv["height"]) if str(cv["height"]).isdigit() else cv.winfo_height() or 120
        c = self.c
        px = int(W * 0.42)
        mid = H // 2
        cv.create_line(px, 6, px, mid - 28, fill=c["border"], width=2)
        cv.create_line(px, mid + 28, px, H - 18, fill=c["border"], width=2)
        cv.create_line(int(W * 0.12), 6, int(W * 0.88), 6, fill=c["border"])
        cv.create_line(int(W * 0.12), H - 18, int(W * 0.88), H - 18, fill=c["border"])
        if self.note:
            cv.create_text(W - 10, H - 4, text=self.note, anchor="se", fill=c["dim"], font=self.small)
        w = self.word
        if not w:
            cv.create_text(px, mid, text="RSVP", fill=c["dim"], font=self.font, anchor="center")
            cv.create_text(px, mid + 34, text="words flash here once you press Read - each pinned on its recognition letter",
                           fill=c["dim"], font=self.small, anchor="center")
            return
        k = orp_index(w)
        left, piv, right = w[:k], w[k], w[k + 1:]
        f = self.font
        wl, wp = f.measure(left), f.measure(piv)
        x0 = px - wp // 2 - wl
        cv.create_text(x0, mid, text=left, anchor="w", fill=c["text"], font=f)
        cv.create_text(x0 + wl, mid, text=piv, anchor="w", fill=c["accent"], font=f)
        cv.create_text(x0 + wl + wp, mid, text=right, anchor="w", fill=c["text"], font=f)


class ReadBar:
    """The control strip + the clock. Needs an `app` with: .palette, .s
    (settings dict), .root, .ui (font family), .save_settings(), .log().

    get_doc()   -> Doc currently on screen (or None)
    on_word(i)  -> paint word i (None = clear)
    on_end()    -> reached the end; return True if you moved on (next chapter
                   or page) and want playback to carry on from word 0
    """

    def __init__(self, parent, app, get_doc, on_word, on_end=None, rsvp_parent=None, rsvp_before=None):
        tk, ttk, tkfont = _tk()
        self.tk, self.ttk = tk, ttk
        self.app = app
        self.s = app.s
        for k, v in SETTINGS_DEFAULTS.items():
            self.s.setdefault(k, v)
        self.get_doc = get_doc
        self.on_word = on_word
        self.on_end = on_end or (lambda: False)
        self.reader = Reader(self.s, log=app.log)
        self.doc = None
        self.mode = "idle"              # idle / audio / timer
        self.cursor = 0
        self._timer_job = None
        self._clock_job = None
        self._timer_due = 0.0
        self._paused = False
        self._voice_items = []          # [(engine, voice id)] parallel to the combo
        self._last_word = None
        self.rsvp_before = rsvp_before

        self.frame = outer = ttk.Frame(parent, padding=(10, 2, 10, 4))
        f = ttk.Frame(outer)                 # row 1: transport, voice, speed
        f.pack(fill="x")
        f2 = ttk.Frame(outer)                # row 2: audio / RSVP / WPM / status
        f2.pack(fill="x", pady=(4, 0))
        self.btn_play = ttk.Button(f, text="▶ Read", width=9, command=self.toggle)
        self.btn_play.pack(side="left")
        ttk.Button(f, text="■", width=3, command=self.stop).pack(side="left", padx=(4, 0))
        ttk.Button(f, text="⏮", width=3, command=lambda: self.skip(-1)).pack(side="left", padx=(8, 0))
        ttk.Button(f, text="⏭", width=3, command=lambda: self.skip(1)).pack(side="left", padx=(2, 0))

        ttk.Label(f, text="Voice", style="Dim.TLabel").pack(side="left", padx=(12, 4))
        self.var_voice = tk.StringVar()
        self.cmb_voice = ttk.Combobox(f, textvariable=self.var_voice, state="readonly", width=30)
        self.cmb_voice.pack(side="left")
        self.cmb_voice.bind("<<ComboboxSelected>>", self._voice_picked)

        ttk.Label(f, text="Speed", style="Dim.TLabel").pack(side="left", padx=(12, 4))
        self.var_speed = tk.DoubleVar(value=self.reader.speed())
        self.scl_speed = ttk.Scale(f, from_=0.5, to=3.0, variable=self.var_speed, length=110, command=self._speed_moved)
        self.scl_speed.pack(side="left")
        self.lbl_speed = ttk.Label(f, text="", width=5)
        self.lbl_speed.pack(side="left")
        self.scl_speed.bind("<ButtonRelease-1>", lambda e: self._speed_set())

        self.var_audio = tk.BooleanVar(value=bool(self.s.get("tts_audio", True)))
        ttk.Checkbutton(f2, text="Audio (off = silent speed-reading on the WPM clock)", variable=self.var_audio,
                        command=self._audio_toggled).pack(side="left")
        self.var_rsvp = tk.BooleanVar(value=bool(self.s.get("rsvp_show", False)))
        ttk.Checkbutton(f2, text="RSVP panel", variable=self.var_rsvp, command=self._rsvp_toggled).pack(side="left", padx=(14, 0))

        ttk.Label(f2, text="WPM", style="Dim.TLabel").pack(side="left", padx=(14, 4))
        self.var_wpm = tk.IntVar(value=int(self.s.get("rsvp_wpm") or 350))
        self.scl_wpm = ttk.Scale(f2, from_=100, to=1000, variable=self.var_wpm, length=140, command=self._wpm_moved)
        self.scl_wpm.pack(side="left")
        self.lbl_wpm = ttk.Label(f2, text="", width=5)
        self.lbl_wpm.pack(side="left")

        self.lbl_status = ttk.Label(f2, text="", style="Dim.TLabel")
        self.lbl_status.pack(side="left", padx=(12, 0))

        self._rsvp_packed = False
        self.last_ui_error = ""
        self.rsvp = RsvpView(rsvp_parent or parent, app.palette, app.ui)
        self._fill_voices()
        self._speed_moved(None)
        self._wpm_moved(None)
        self._update_mode_widgets()

    # -- wiring
    def pack(self, **kw):
        self.frame.pack(**kw)

    def apply_palette(self, c):
        self.rsvp.apply_palette(c)

    def bind_keys(self, widget):
        widget.bind("<Control-space>", lambda e: (self.toggle(), "break")[1])
        widget.bind("<Control-Right>", lambda e: (self.skip(1), "break")[1])
        widget.bind("<Control-Left>", lambda e: (self.skip(-1), "break")[1])
        widget.bind("<Control-Up>", lambda e: (self._nudge_rate(1), "break")[1])
        widget.bind("<Control-Down>", lambda e: (self._nudge_rate(-1), "break")[1])
        widget.bind("<Control-r>", lambda e: (self.var_rsvp.set(not self.var_rsvp.get()), self._rsvp_toggled(), "break")[2])

    def _fill_voices(self):
        try:
            if not self.frame.winfo_exists():
                return
        except Exception:
            return
        items, labels = [], []
        status = engine_status()
        for eid in ENGINE_ORDER:
            ok, why = status[eid]
            if not ok:
                continue
            eng = self.reader.engine(eid)
            if eid == "system" and SystemEngine._cache is None:
                # ask the OS on a thread; the strip refills when the list arrives
                def fetch(eng=eng):
                    eng.voices(fetch=True)

                def refill():
                    if SystemEngine._cache is not None:
                        self._fill_voices()
                t = threading.Thread(target=fetch, daemon=True)
                t.start()
                self._after_thread(t, refill)
            for vid, lab in (eng.voices(fetch=False) if eid == "system" else eng.voices()):
                items.append((eid, vid))
                labels.append(f"{ENGINES[eid].label.split(' (')[0]}: {lab}")
        if not status["kokoro"][0]:
            try:
                import kokoro_onnx  # noqa: F401
                items.append(("_kokoro_download", ""))
                labels.append(f"Download Kokoro offline voices ({KOKORO_MB} MB)…" if not kokoro_files_on_disk()
                              else f"Re-download Kokoro voices for this version ({KOKORO_MB} MB)…")
            except Exception:
                pass
        if status["edge"][0]:
            items.append(("_edge_refresh", ""))
            labels.append("Fetch the full online voice list…")
        self._voice_items = items
        self.cmb_voice["values"] = labels
        eid, vid = self.reader.voice(fetch=False)
        for k, (e, v) in enumerate(items):
            if e == eid and v == vid:
                self.cmb_voice.current(k)
                break
        else:
            for k, (e, _) in enumerate(items):
                if e == eid:
                    self.cmb_voice.current(k)
                    break
            else:
                if items:
                    self.cmb_voice.current(0)

    def _voice_picked(self, event=None):
        k = self.cmb_voice.current()
        if k < 0 or k >= len(self._voice_items):
            return
        eid, vid = self._voice_items[k]
        if eid == "_kokoro_download":
            self._download_kokoro()
            return
        if eid == "_edge_refresh":
            self._refresh_edge()
            return
        self.reader.set_voice(eid, vid)
        self.app.save_settings()

    def _download_kokoro(self):
        from tkinter import messagebox
        if not messagebox.askyesno("Kokoro voices", f"Download the Kokoro offline voices now?\n\nAbout {KOKORO_MB} MB, "
                                   f"once, into\n{models_dir()}", parent=self.frame):
            self._fill_voices()
            return
        self.set_status("Downloading Kokoro voices…")
        prog = {"done": 0, "total": 0, "err": None}

        def cb(done, total, name):
            prog["done"], prog["total"] = done, total

        def work():                 # never touches Tk: tick() picks up the result
            try:
                download_kokoro_models(cb)
                prog["err"] = ""
            except Exception as e:
                prog["err"] = str(e)

        def finish(err):
            if err:
                self.set_status("Download failed: " + err)
            else:
                self.set_status("Kokoro voices ready")
                self.reader.set_voice("kokoro", "bf_emma")
                self.app.save_settings()
            self._fill_voices()

        def tick():
            if prog["total"]:
                self.set_status(f"Downloading Kokoro voices… {prog['done'] // 1048576} / {prog['total'] // 1048576} MB")
            if t.is_alive():
                self.app.root.after(300, tick)
            elif prog["err"] is not None:
                finish(prog["err"])

        t = threading.Thread(target=work, daemon=True)
        t.start()
        tick()

    def _refresh_edge(self):
        self.set_status("Fetching voice list…")
        res = {}

        def work():                 # never touches Tk: done() runs on the Tk thread
            try:
                n = self.reader.engine("edge").refresh_voices()
                res["msg"] = f"{n} online voices"
            except Exception as e:
                res["msg"] = "Voice list: " + str(e)

        def done():
            if "msg" in res:
                self.set_status(res["msg"])
                self._fill_voices()

        t = threading.Thread(target=work, daemon=True)
        t.start()
        self._after_thread(t, done)

    def _after_thread(self, t, fn):
        """Run fn() on the Tk thread once worker thread t has finished
        (worker threads never call into Tk themselves)."""
        def poll():
            if t.is_alive():
                self.app.root.after(100, poll)
            else:
                fn()
        self.app.root.after(100, poll)

    def _speed_moved(self, _):
        self.lbl_speed.configure(text=f"{self.var_speed.get():.2g}×")

    def _speed_set(self):
        v = round(self.var_speed.get(), 2)
        self.reader.set_speed(v)
        self.app.save_settings()
        eid = self.reader.voice(fetch=False)[0]
        if v > 2.0 and eid in ("kokoro", "edge"):
            self.set_status(f"{ENGINES[eid].label.split(' (')[0]} voices top out at 2× (the system voices go to 3×)")

    def _wpm_moved(self, _):
        v = int(self.var_wpm.get())
        self.lbl_wpm.configure(text=str(v))
        self.s["rsvp_wpm"] = v

    def _nudge_rate(self, d):
        if self.var_audio.get():
            self.var_speed.set(max(0.5, min(3.0, round(self.var_speed.get() + 0.1 * d, 2))))
            self._speed_moved(None)
            self._speed_set()
        else:
            self.var_wpm.set(max(100, min(1000, int(self.var_wpm.get()) + 25 * d)))
            self._wpm_moved(None)

    def _audio_toggled(self):
        self.s["tts_audio"] = bool(self.var_audio.get())
        self.app.save_settings()
        self._update_mode_widgets()
        if self.mode != "idle":
            pos = self.cursor
            paused = self._paused
            self.stop(keep_cursor=True)
            self.play(pos)
            if paused:
                self.pause()

    def _rsvp_toggled(self):
        self.s["rsvp_show"] = bool(self.var_rsvp.get())
        self.app.save_settings()
        self._update_mode_widgets()

    def _update_mode_widgets(self):
        audio = self.var_audio.get()
        self.scl_speed.state(["!disabled"] if audio else ["disabled"])
        self.scl_wpm.state(["disabled"] if audio else ["!disabled"])
        if self.var_rsvp.get():
            if not self._rsvp_packed:
                kw = {"before": self.rsvp_before} if self.rsvp_before is not None else {}
                self.rsvp.frame.pack(side="top", fill="x", padx=10, pady=(0, 6), **kw)
                self._rsvp_packed = True
                self.rsvp.redraw()
        elif self._rsvp_packed:
            self.rsvp.frame.pack_forget()
            self._rsvp_packed = False

    def diagnostics(self):
        """One text block for a Help menu: what is installed and what is showing."""
        lines = [f"readaloud {VERSION}", f"models folder: {models_dir()}", f"kokoro release wanted: {kokoro_release()}",
                 f"kokoro files on disk: {kokoro_files_on_disk()}  matching: {kokoro_models_present()}"]
        for k, (ok, why) in engine_status().items():
            lines.append(f"engine {k}: {'available' if ok else 'not available - ' + why}")
        lines.append(f"voice setting: {self.s.get('tts_voice')!r} -> using {self.reader.voice(fetch=False)}")
        lines.append(f"audio: {self.var_audio.get()}  rsvp checkbox: {self.var_rsvp.get()}  panel packed: {self._rsvp_packed}")
        try:
            fr = self.rsvp.frame
            lines.append(f"panel mapped: {bool(fr.winfo_ismapped())}  size: {fr.winfo_width()}x{fr.winfo_height()}  "
                         f"parent: {fr.winfo_parent()}  canvas: {self.rsvp.canvas.winfo_width()}x{self.rsvp.canvas.winfo_height()}")
        except Exception as e:
            lines.append(f"panel: {e!r}")
        try:
            cv = self.rsvp.canvas
            texts = [cv.itemcget(i, "text") for i in cv.find_all() if cv.type(i) == "text"]
            lines.append(f"panel word: {self.rsvp.word!r}  items drawn: {len(cv.find_all())}  texts: {texts[:4]}  "
                         f"font: {self.rsvp.font.actual('family')} {self.rsvp.font.actual('size')}  colours: text {self.rsvp.c['text']} on {cv['bg']}")
        except Exception as e:
            lines.append(f"panel drawing: {e!r}")
        lines.append(f"mode: {self.mode}  cursor: {self.cursor}  last word: {self._last_word}  reader state: {self.reader.state}")
        if self.reader.error:
            lines.append("last error: " + self.reader.error)
        if self.last_ui_error:
            lines.append("last ui error: " + self.last_ui_error[-600:])
        return "\n".join(lines)

    def set_status(self, text):
        if text != getattr(self, "_status_text", None):    # the clock calls this every 30 ms
            self._status_text = text
            self.lbl_status.configure(text=text)

    # -- transport
    def toggle(self):
        if self.mode == "idle":
            self.play(self.cursor)
        elif self._paused:
            self.resume()
        else:
            self.pause()

    def play(self, word_idx=0):
        doc = self.get_doc()
        if doc is None or not doc.words:
            if self.mode != "idle":         # auto-advance into an empty page: stop, don't sit there showing "Pause"
                self.stop(keep_cursor=True)
            self.set_status("Nothing to read")
            return
        self.doc = doc
        self.cursor = max(0, min(len(doc.words) - 1, word_idx))
        self._paused = False
        self._last_word = None
        for attr in ("_timer_job", "_clock_job"):          # a restart must not leave the old clock ticking
            job = getattr(self, attr)
            if job:
                self.frame.after_cancel(job)
                setattr(self, attr, None)
        self.btn_play.configure(text="⏸ Pause")
        if self.var_audio.get():
            self.mode = "audio"
            eid, _ = self.reader.voice(fetch=False)
            self.reader.load(doc)
            self.reader.play(self.cursor)
            self.set_status("Loading voice…" if eid == "kokoro" else "Starting…")
            self._start_clock()
        else:
            self.mode = "timer"
            self.reader.stop()
            self.set_status("")
            self._timer_due = time.monotonic()
            self._paint(self.cursor)
            self._timer_tick()

    def pause(self):
        if self.mode == "idle" or self._paused:
            return
        self._paused = True
        self.btn_play.configure(text="▶ Resume")
        if self.mode == "audio":
            self.reader.pause()
        elif self._timer_job:
            self.frame.after_cancel(self._timer_job)
            self._timer_job = None

    def resume(self):
        if self.mode == "idle" or not self._paused:
            return
        self._paused = False
        self.btn_play.configure(text="⏸ Pause")
        if self.mode == "audio":
            self.reader.resume()
        else:
            self._timer_due = time.monotonic()
            self._timer_tick()

    def stop(self, keep_cursor=False):
        if self._timer_job:
            self.frame.after_cancel(self._timer_job)
            self._timer_job = None
        if self._clock_job:
            self.frame.after_cancel(self._clock_job)
            self._clock_job = None
        self.reader.stop()
        self.mode = "idle"
        self._paused = False
        self.btn_play.configure(text="▶ Read")
        if not keep_cursor:
            self._last_word = None
            self.on_word(None)
            self.rsvp.show(None)
        self.set_status("")

    def skip(self, delta):
        doc = self.doc or self.get_doc()
        if doc is None or not doc.words:
            return
        s = doc.words[self.cursor].sent + delta
        if s < 0:
            s = 0
        if s >= len(doc.sentences):
            if self.mode != "idle" and self.on_end():
                self.play(0)
            return
        target = doc.sentences[s].first
        if self.mode == "idle":
            self.cursor = target
            self._paint(target)
        else:
            paused = self._paused
            self.play(target)
            if paused:
                self.pause()

    def set_cursor(self, word_idx, restart=True):
        """Click on a word: start there (or just move the marker when idle)."""
        if self.mode != "idle" and restart:
            paused = self._paused
            self.play(word_idx)
            if paused:
                self.pause()
        else:
            self.cursor = word_idx
            self.doc = self.get_doc()
            self._paint(word_idx)

    def document_changed(self):
        """The page/chapter on screen was replaced by the owner (manual navigation)."""
        self.stop()
        self.cursor = 0
        self.doc = None
        self._last_word = None

    def close(self):
        self.stop()
        self.reader.close()

    # -- the two clocks
    def _paint(self, i):
        if i == self._last_word:
            return
        self._last_word = i
        self.cursor = i
        # neither painter may kill the clock: log and carry on
        try:
            self.on_word(i)
        except Exception:
            self._ui_error("highlight")
        if self.var_rsvp.get() and self.doc is not None and i is not None:
            try:
                w = self.doc.words[i]
                self.rsvp.show(w.clean or w.text, f"{i + 1} / {len(self.doc.words)}")
            except Exception:
                self._ui_error("rsvp")

    def _ui_error(self, where):
        import traceback
        self.last_ui_error = f"{where}: " + traceback.format_exc()
        try:
            self.app.log("read aloud ui error in " + self.last_ui_error)
        except Exception:
            pass

    def _start_clock(self):
        if self._clock_job:
            self.frame.after_cancel(self._clock_job)
        self._clock_job = self.frame.after(30, self._clock_tick)

    def _clock_tick(self):
        self._clock_job = None
        if self.mode != "audio":
            return
        r = self.reader
        idx, state = r.current()
        if state == "error":
            self.app.log("read aloud: " + r.error)
            self.stop(keep_cursor=True)         # before the message: stop() blanks the status line
            self.set_status(r.error.splitlines()[0][:120] if r.error else "Voice failed")
            return
        if state == "done":
            if idx is not None:
                self._paint(idx)
            if bool(self.s.get("read_autoadvance", True)) and self.on_end():
                self.play(0)
            else:
                self.stop(keep_cursor=True)
                self.cursor = 0              # the highlight stays on the last word; Read starts over
                self.set_status("End")
            return
        if idx is not None:
            self._paint(idx)
        note = {"loading": "Loading voice…", "buffering": "Buffering…", "paused": "Paused", "playing": ""}.get(state, "")
        if r.progress:
            note = r.progress
        self.set_status(note)
        self._clock_job = self.frame.after(30, self._clock_tick)

    def _timer_tick(self):
        self._timer_job = None
        if self.mode != "timer" or self._paused or self.doc is None:
            return
        i = self.cursor
        self._paint(i)
        w = self.doc.words[i]
        dwell = rsvp_dwell(w, int(self.var_wpm.get()))
        self._timer_due += dwell
        nxt = i + 1
        if nxt >= len(self.doc.words):
            def finish():
                if bool(self.s.get("read_autoadvance", True)) and self.on_end():
                    self.play(0)
                else:
                    self.stop(keep_cursor=True)
                    self.cursor = 0
                    self.set_status("End")
            self._timer_job = self.frame.after(max(1, int(dwell * 1000)), finish)
            return
        delay = max(1, int((self._timer_due - time.monotonic()) * 1000))

        def step():
            self.cursor = nxt
            self._timer_tick()
        self._timer_job = self.frame.after(delay, step)


# ---------------------------------------------------------------- selftest
def selftest_checks(check):
    """Offline checks for an app's self-test; `check(name, cond, detail)`."""
    d = tokenize('Mr. Smith went home. "It was late," he said - very late!\n\nA new paragraph (e.g. this one)... Ends here')
    check("tokenize: word count", len(d.words) == 19, str(len(d.words)))
    check("tokenize: 'Mr.' is not a sentence end", d.words[0].pause == "comma" and d.words[0].sent == d.words[1].sent)
    check("tokenize: sentences", [s.text for s in d.sentences] == [
        "Mr. Smith went home.", '"It was late," he said - very late!', "A new paragraph (e.g. this one)...", "Ends here"],
        str([s.text for s in d.sentences]))
    check("tokenize: paragraph pause", d.words[10].pause == "paragraph" and d.words[11].para, repr(d.words[10]))
    check("tokenize: em dash folded into pause", d.words[8].pause == "comma", repr(d.words[8]))
    check("tokenize: offsets", d.text[d.words[1].start:d.words[1].end] == "Smith" and d.word_at_char(d.words[5].start + 1) == 5)
    long = tokenize(" ".join(f"w{i}," if i % 7 == 6 else f"w{i}" for i in range(80)))
    check("tokenize: run-on cut at commas", all(s.nwords <= MAX_SENT_WORDS for s in long.sentences) and len(long.sentences) >= 3,
          str([s.nwords for s in long.sentences]))
    check("speech text", _speakable("“Quoted” — done") == '"Quoted" , done.')
    t = spread_times(d.words[0:4], 2.0)
    check("spread times: monotone, ends at duration", all(t[i][1] <= t[i + 1][0] + 1e-9 for i in range(3)) and abs(t[-1][1] - 2.0) < 1e-6, str(t))
    marks = [(0.0, 0.3, "Mr"), (0.35, 0.4, "Smith"), (0.8, 0.2, "went"), (1.05, 0.4, "home")]
    a = _align_marks(d.words[0:4], marks, 1.6)
    check("edge marks aligned", [round(x[0], 2) for x in a] == [0.0, 0.35, 0.8, 1.05], str(a))
    a = _align_marks(d.words[0:4], [(0.0, 0.3, "Mr"), (1.05, 0.4, "home")], 1.6)
    check("edge marks: gaps interpolated", 0.3 <= a[1][0] < a[2][0] < 1.05, str(a))
    check("rsvp dwell: long word and sentence end take longer",
          rsvp_dwell(d.words[1], 300) < rsvp_dwell(d.words[3], 300) and rsvp_dwell(Word(0, 0, 0, "a", "a", "", False), 300) < rsvp_dwell(Word(0, 0, 0, "extraordinary", "extraordinary", "", False), 300))
    check("orp index", [orp_index("a"), orp_index("word"), orp_index("reading"), orp_index("extraordinary")] == [0, 1, 2, 3])
    pcm = bytes(4800)
    tmp = tempfile.mkdtemp(prefix="ra-")
    try:
        p = os.path.join(tmp, "t.wav")
        with wave.open(p, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(bytes(48000 * 2 * 2))
        got, secs = _wav_to_pcm(p)
        check("wav: stereo 48k -> mono 24k, 1 s", abs(secs - 1.0) < 0.01 and len(got) == 48000, f"{secs} {len(got)}")
        pcm_to_wav(pcm, p)
        got, secs = _wav_to_pcm(p)
        check("wav round trip", got == pcm and abs(secs - 0.1) < 1e-6)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    st = engine_status()
    check("engine status reported", set(st) == set(ENGINES), str(st))
    return st


if __name__ == "__main__":
    problems = []

    def _check(name, cond, detail=""):
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail and not cond else ""))
        if not cond:
            problems.append(name)

    st = selftest_checks(_check)
    for k, (ok, why) in st.items():
        print(f"  engine {k}: {'available' if ok else 'unavailable - ' + why}")
    print("RESULT:", "OK" if not problems else "PROBLEMS " + ", ".join(problems))
    sys.exit(1 if problems else 0)
