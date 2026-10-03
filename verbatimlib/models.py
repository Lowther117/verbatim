"""The model catalogue, and fetching models that are not on disk yet.

Every model is an ONNX file (or a few) published on the sherpa-onnx project's
GitHub releases. Nothing is fetched from anywhere else, no account or token is
needed, and once a model is on disk the app never touches the network for it
again - audio is always processed on this machine.

A model lives in  <models folder>/<id>/  with exactly the file names listed
in its catalogue entry. The models folder is "verbatim-models" beside the app;
models baked into a build sit inside the bundle and are found there too.
"""
from __future__ import annotations

import os
import shutil
import ssl
import tarfile
import time
import urllib.request
from typing import Callable, Dict, List, Optional

from . import paths

_REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


class ModelError(Exception):
    """A model is missing or could not be fetched; the message is fit to show the user."""


def _whisper(name: str, label: str, mb: int, blurb: str, speed: str) -> dict:
    return {
        "kind": "whisper", "label": label, "mb": mb, "blurb": blurb, "speed": speed, "languages": "99 languages",
        "url": f"{_REL}/asr-models/sherpa-onnx-whisper-{name}.tar.bz2",
        "files": {"encoder": f"{name}-encoder.int8.onnx", "decoder": f"{name}-decoder.int8.onnx",
                  "tokens": f"{name}-tokens.txt"},
    }


# Speech-to-text models, in the order the picker shows them.
ASR: Dict[str, dict] = {
    "parakeet-v2": {
        "kind": "transducer", "label": "Parakeet v2 - English (recommended)", "mb": 630, "speed": "fast",
        "languages": "English only",
        "blurb": "The most accurate English model here and the least prone to inventing words. "
                 "Punctuation and capitals included. English only.",
        "url": f"{_REL}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
        "files": {"encoder": "encoder.int8.onnx", "decoder": "decoder.int8.onnx", "joiner": "joiner.int8.onnx",
                  "tokens": "tokens.txt"},
    },
    "parakeet-v3": {
        "kind": "transducer", "label": "Parakeet v3 - 25 European languages", "mb": 640, "speed": "fast",
        "languages": "25 European languages, detected automatically",
        "blurb": "Same family as v2 but multilingual: English, French, German, Spanish, Italian, Dutch, Polish, "
                 "Portuguese, Russian, Ukrainian and 15 more. Not Welsh - use a Whisper model for that.",
        "url": f"{_REL}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        "files": {"encoder": "encoder.int8.onnx", "decoder": "decoder.int8.onnx", "joiner": "joiner.int8.onnx",
                  "tokens": "tokens.txt"},
    },
    "whisper-turbo": _whisper("turbo", "Whisper large-v3 turbo - any language, best quality", 1040,
                              "OpenAI's large model, 99 languages including Welsh. The most accurate "
                              "multilingual choice; noticeably slower without a fast processor.", "slow"),
    "whisper-small": _whisper("small", "Whisper small - any language, balanced", 360,
                              "99 languages including Welsh. A reasonable middle ground for other languages "
                              "on an ordinary laptop.", "medium"),
    "whisper-base": _whisper("base", "Whisper base - any language, quick draft", 155,
                             "Quick and light. Fine for clear recordings, shaky on accents and crosstalk.", "fast"),
    "whisper-tiny": _whisper("tiny", "Whisper tiny - any language, fastest", 100,
                             "The smallest model. For a rough gist only.", "very fast"),
}

# Supporting models. Small, and always baked into a build.
SUPPORT: Dict[str, dict] = {
    "vad": {
        "label": "Speech detector (Silero VAD)", "mb": 1,
        "url": f"{_REL}/asr-models/silero_vad.onnx", "files": {"model": "silero_vad.onnx"},
    },
    "denoise": {
        "label": "Noise reduction (GTCRN)", "mb": 1,
        "url": f"{_REL}/speech-enhancement-models/gtcrn_simple.onnx", "files": {"model": "gtcrn_simple.onnx"},
    },
    "segmentation": {
        "label": "Speaker change detector (pyannote segmentation 3.0)", "mb": 6,
        "url": f"{_REL}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "files": {"model": "model.onnx"},
    },
    "embedding": {
        "label": "Voice fingerprint (NVIDIA TitaNet small)", "mb": 39,
        # "recongition" is how the release tag is spelt upstream
        "url": f"{_REL}/speaker-recongition-models/nemo_en_titanet_small.onnx",
        "files": {"model": "nemo_en_titanet_small.onnx"},
    },
}

CATALOGUE: Dict[str, dict] = {**ASR, **SUPPORT}
DEFAULT_ASR = "parakeet-v2"

# Whisper language codes offered in the picker ("" = detect). Anything Whisper
# knows can still be typed into the settings file.
LANGUAGES = [
    ("", "Detect automatically"), ("en", "English"), ("cy", "Welsh"), ("fr", "French"), ("de", "German"),
    ("es", "Spanish"), ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"), ("pl", "Polish"),
    ("ro", "Romanian"), ("sv", "Swedish"), ("da", "Danish"), ("no", "Norwegian"),
    ("fi", "Finnish"), ("cs", "Czech"), ("el", "Greek"), ("hu", "Hungarian"), ("tr", "Turkish"),
    ("uk", "Ukrainian"), ("ru", "Russian"), ("ar", "Arabic"), ("he", "Hebrew"), ("hi", "Hindi"),
    ("ur", "Urdu"), ("bn", "Bengali"), ("pa", "Punjabi"), ("zh", "Chinese"), ("ja", "Japanese"),
    ("ko", "Korean"), ("vi", "Vietnamese"), ("th", "Thai"), ("id", "Indonesian"),
]


def find(model_id: str) -> Optional[Dict[str, str]]:
    """{role: absolute path} if every file of the model is on disk, else None."""
    spec = CATALOGUE.get(model_id)
    if not spec:
        return None
    for root in paths.models_dirs():
        folder = os.path.join(root, model_id)
        found = {role: os.path.join(folder, name) for role, name in spec["files"].items()}
        if all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in found.values()):
            return found
    return None


def installed(model_id: str) -> bool:
    return find(model_id) is not None


def require(model_id: str) -> Dict[str, str]:
    got = find(model_id)
    if got is None:
        label = CATALOGUE.get(model_id, {}).get("label", model_id)
        raise ModelError(f"The model \"{label}\" is not installed. Download it from the Models menu.")
    return got


def missing(model_ids: List[str]) -> List[str]:
    return [m for m in model_ids if not installed(m)]


def _context() -> ssl.SSLContext:
    """The system's certificates plus certifi's.

    A frozen app on a Mac has no system bundle to lean on, which is what
    certifi is for. The system store is loaded as well because a network that
    inspects TLS (some offices do) signs with its own root, which only the
    system store knows about.
    """
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        pass
    return ctx


def _download(url: str, dest: str, progress: Optional[Callable[[int, int], None]],
              cancelled: Optional[Callable[[], bool]]) -> None:
    """Fetch url to dest, resuming a partial file left by an interrupted attempt."""
    part = dest + ".part"
    last_error: Optional[Exception] = None
    for attempt in range(4):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"User-Agent": f"Verbatim/{paths.VERSION}"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60, context=_context()) as resp:
                if have and resp.status != 206:  # server ignored the range: start again
                    have = 0
                length = resp.headers.get("Content-Length")
                total = have + int(length) if length else 0
                with open(part, "ab" if have else "wb") as fh:
                    while True:
                        if cancelled and cancelled():
                            raise ModelError("Download cancelled.")
                        block = resp.read(1 << 18)
                        if not block:
                            break
                        fh.write(block)
                        have += len(block)
                        if progress:
                            progress(have, total)
                if total and have < total:
                    raise OSError("the connection closed early")
            os.replace(part, dest)
            return
        except ModelError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:  # the partial file is already the whole file, or stale
                os.remove(part)
                continue
            last_error = exc
        except Exception as exc:  # noqa: BLE001 - timeouts, resets, DNS
            last_error = exc
        time.sleep(1.5 * (attempt + 1))
    raise ModelError(f"Could not download {os.path.basename(dest)}: {last_error}. Check the internet connection "
                     f"and try again - the download carries on from where it stopped.")


def install(model_id: str, progress: Optional[Callable[[str, int, int], None]] = None,
            cancelled: Optional[Callable[[], bool]] = None, into: Optional[str] = None) -> Dict[str, str]:
    """Download one model into the models folder (or `into`) and return its file paths.

    progress(stage, done, total) is called with stage "download" (bytes) and
    then "unpack" (files).
    """
    spec = CATALOGUE.get(model_id)
    if not spec:
        raise ModelError(f"Unknown model: {model_id}")
    if into is None:
        got = find(model_id)
        if got:
            return got
    root = into or paths.models_dirs()[0]
    folder = os.path.join(root, model_id)
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as exc:
        raise ModelError(f"Could not create {folder}: {exc}") from exc
    url = spec["url"]
    wanted = dict(spec["files"])
    archive = url.endswith(".tar.bz2")
    target = os.path.join(folder, "download.tar.bz2" if archive else next(iter(wanted.values())))
    free = shutil.disk_usage(folder).free
    need = spec["mb"] * (3 if archive else 1) * 1024 * 1024
    if free < need:
        raise ModelError(f"Not enough free disk space for {spec['label']}: about {need // (1 << 20)} MB is needed "
                         f"in {root}.")
    _download(url, target, (lambda d, t: progress("download", d, t)) if progress else None, cancelled)
    if archive:
        names = set(wanted.values())
        done = 0
        try:
            with tarfile.open(target, "r:bz2") as tar:
                for member in tar:
                    if cancelled and cancelled():
                        raise ModelError("Download cancelled.")
                    base = os.path.basename(member.name)
                    if not member.isfile() or base not in names:
                        continue
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    tmp = os.path.join(folder, base + ".part")
                    with open(tmp, "wb") as out:
                        shutil.copyfileobj(src, out, 1 << 20)
                    os.replace(tmp, os.path.join(folder, base))
                    done += 1
                    if progress:
                        progress("unpack", done, len(names))
                    if done == len(names):
                        break
        except (tarfile.TarError, EOFError, OSError) as exc:
            try:
                os.remove(target)
            except OSError:
                pass
            raise ModelError(f"The download of {spec['label']} was damaged ({exc}). Try again.") from exc
        try:
            os.remove(target)
        except OSError:
            pass
    found = {role: os.path.join(folder, name) for role, name in wanted.items()}
    absent = [n for n, p in found.items() if not os.path.isfile(p)]
    if absent:
        raise ModelError(f"{spec['label']} unpacked without its {', '.join(absent)} file.")
    return found


def remove(model_id: str) -> bool:
    """Delete a downloaded model. Models baked into the app cannot be removed."""
    folder = os.path.join(paths.models_dirs()[0], model_id)
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)
        return True
    return False


def needed_for(asr: str, speakers: bool, denoise: bool) -> List[str]:
    out = [asr, "vad"]
    if denoise:
        out.append("denoise")
    if speakers:
        out += ["segmentation", "embedding"]
    return out
