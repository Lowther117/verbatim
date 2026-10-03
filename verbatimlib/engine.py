"""The transcription pipeline: one audio file in, one speaker-labelled transcript out.

    decode  ->  find speech  ->  clean (if noisy)  ->  who spoke when  ->  transcribe  ->  filter

Everything runs on this machine through sherpa-onnx (ONNX Runtime underneath).

Why the stages are in this order, since it is what keeps invented text out:

1. A speech detector (Silero VAD) marks where somebody is actually talking.
   Only those stretches ever reach the speech-to-text model, so silence,
   music and hiss cannot be "transcribed" into phantom sentences.
2. The noise level is measured from the gaps between speech. If the recording
   is noisy the whole file goes through a noise-reduction model (GTCRN) and
   speech is detected again on the cleaned audio, which stops noise bursts
   being mistaken for speech and stops the speaker stage splitting one voice
   into two because the background changed.
3. Speaker turns come from a segmentation model plus a voice-fingerprint model;
   voices that barely appear are folded into their neighbour rather than
   becoming a "Speaker 7" who said one word.
4. Each stretch of speech is transcribed on its own, and what comes back is
   checked (filters.py) before it is accepted.
"""
from __future__ import annotations

import gc
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import audio, filters, models, paths

RATE = audio.RATE
MAX_UNIT = 24.0          # seconds of audio handed to the model in one go
HARD_MAX = 28.0          # Whisper's window is 30 s; never exceed it
PAD = 0.12               # context added either side of a unit (Whisper)
AUTO_DENOISE_BELOW_DB = 20.0


class Cancelled(Exception):
    pass


class EngineError(Exception):
    """A failure with a message fit to show the user."""


@dataclass
class Options:
    model: str = models.DEFAULT_ASR
    language: str = ""            # Whisper only; "" = detect
    translate: bool = False       # Whisper only: translate into English
    speakers: bool = True
    num_speakers: int = 0         # 0 = work it out
    denoise: str = "auto"         # auto | on | off
    filtering: str = "balanced"   # off | balanced | strict
    threads: int = 0              # 0 = choose from the processor
    asr_audio: str = "original"   # original | cleaned: what the text model listens to after noise reduction

    @classmethod
    def from_dict(cls, d: dict) -> "Options":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        o = cls(**known)
        if o.model not in models.ASR:
            o.model = models.DEFAULT_ASR
        if o.denoise not in ("auto", "on", "off"):
            o.denoise = "auto"
        if o.filtering not in filters.LEVELS:
            o.filtering = "balanced"
        try:
            o.num_speakers = max(0, min(20, int(o.num_speakers)))
        except (TypeError, ValueError):
            o.num_speakers = 0
        return o


def default_threads() -> int:
    """How many threads to give the models.

    More is not better: threads beyond the fast physical cores (hyper-threads,
    or the efficiency cores of an Apple or recent Intel chip) slow ONNX Runtime
    down rather than speed it up.
    """
    import sys
    n = os.cpu_count() or 2
    if sys.platform == "darwin":
        try:
            import subprocess
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5).stdout
            perf = int(out.decode().strip())
            if perf > 0:
                return max(1, min(8, perf))
        except Exception:  # noqa: BLE001 - Intel Macs have no perflevel0
            pass
    if n <= 4:
        return max(1, n)
    return max(4, min(8, n // 2))


def _sherpa():
    try:
        import sherpa_onnx
        return sherpa_onnx
    except Exception as exc:  # noqa: BLE001
        raise EngineError(f"The speech engine (sherpa-onnx) could not be loaded: {exc}") from exc


# --- stages -------------------------------------------------------------------

def detect_speech(samples: np.ndarray, vad_path: str, level: str, threads: int,
                  progress: Optional[Callable[[float], None]] = None,
                  cancelled: Optional[Callable[[], bool]] = None) -> List[Tuple[int, int]]:
    """Speech regions as (start_sample, end_sample), in order."""
    so = _sherpa()
    cfg = so.VadModelConfig()
    cfg.silero_vad.model = vad_path
    # A higher threshold means "be surer it is speech". Strict raises it;
    # filtering off lowers it so that nothing faint is skipped.
    cfg.silero_vad.threshold = {"strict": 0.6, "off": 0.35}.get(level, 0.5)
    cfg.silero_vad.min_silence_duration = 0.35
    cfg.silero_vad.min_speech_duration = 0.3 if level == "strict" else 0.2
    cfg.silero_vad.max_speech_duration = 18.0
    cfg.sample_rate = RATE
    cfg.num_threads = 1
    if not cfg.validate():
        raise EngineError("The speech detector model could not be loaded.")
    vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
    window = cfg.silero_vad.window_size
    block = window * 64
    out: List[Tuple[int, int]] = []

    def drain():
        while not vad.empty():
            seg = vad.front
            start = int(seg.start)
            out.append((start, start + len(seg.samples)))
            vad.pop()

    total = len(samples)
    for i in range(0, total, block):
        if cancelled and cancelled():
            raise Cancelled()
        chunk = np.asarray(samples[i:i + block], dtype=np.float32)
        for j in range(0, len(chunk), window):
            piece = chunk[j:j + window]
            if len(piece) < window:
                piece = np.concatenate([piece, np.zeros(window - len(piece), dtype=np.float32)])
            vad.accept_waveform(piece)
        drain()
        if progress and (i // block) % 8 == 0:
            progress(min(1.0, i / max(1, total)))
    vad.flush()
    drain()
    out = [(max(0, s), min(total, e)) for s, e in out if e > s]
    out.sort()
    if progress:
        progress(1.0)
    return out


def reduce_noise(samples: np.ndarray, model_path: str, out_path: str, threads: int,
                 progress: Optional[Callable[[float], None]] = None,
                 cancelled: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """Run the whole file through the denoiser, a minute at a time, into a new file."""
    so = _sherpa()
    cfg = so.OfflineSpeechDenoiserConfig(
        model=so.OfflineSpeechDenoiserModelConfig(
            gtcrn=so.OfflineSpeechDenoiserGtcrnModelConfig(model=model_path),
            num_threads=max(1, min(4, threads)), debug=False, provider="cpu"))
    if not cfg.validate():
        raise EngineError("The noise-reduction model could not be loaded.")
    den = so.OfflineSpeechDenoiser(cfg)
    total = len(samples)
    block = RATE * 60
    overlap = RATE // 2  # half a second of lead-in so each block starts with settled state
    with open(out_path, "wb") as fh:
        for i in range(0, total, block):
            if cancelled and cancelled():
                raise Cancelled()
            lead = min(overlap, i)
            chunk = np.ascontiguousarray(samples[i - lead:i + block], dtype=np.float32)
            res = den(chunk, RATE)
            got = np.asarray(res.samples, dtype=np.float32)
            if int(res.sample_rate) != RATE:
                raise EngineError("The noise-reduction model returned an unexpected sample rate.")
            want = len(chunk) - lead
            got = got[lead:lead + want]
            if len(got) < want:  # the model drops a partial last frame: pad it back
                got = np.concatenate([got, np.zeros(want - len(got), dtype=np.float32)])
            got.tofile(fh)
            if progress:
                progress(min(1.0, (i + block) / max(1, total)))
    return np.memmap(out_path, dtype=np.float32, mode="r", shape=(total,))


def diarize(samples: np.ndarray, seg_path: str, emb_path: str, num_speakers: int, threads: int,
            progress: Optional[Callable[[float], None]] = None) -> Tuple[List[Tuple[float, float, int]], int]:
    """Speaker turns as (start_seconds, end_seconds, speaker_index), plus how many
    stray voices were folded into a real speaker.

    Two passes. The library's own pass finds the turns and groups them by
    voice, deliberately erring towards too many groups. The second pass
    (refine_speakers) averages each group's voice fingerprint over everything
    it said - far steadier than any single snippet - and uses those averages
    to merge groups that are the same person and to rehome scraps.
    """
    so = _sherpa()
    emb_cfg = so.SpeakerEmbeddingExtractorConfig(model=emb_path, num_threads=threads)
    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg_path),
            num_threads=threads),
        embedding=emb_cfg,
        clustering=so.FastClusteringConfig(num_clusters=-1, threshold=CLUSTER_THRESHOLD),
        min_duration_on=0.3, min_duration_off=0.5)
    if not cfg.validate():
        raise EngineError("The speaker models could not be loaded.")
    sd = so.OfflineSpeakerDiarization(cfg)
    data = np.ascontiguousarray(samples, dtype=np.float32)

    def cb(done: int, total: int) -> int:
        if progress and total:
            progress(0.9 * done / total)
        return 0

    result = sd.process(data, callback=cb).sort_by_start_time()
    turns = [(float(r.start), float(r.end), int(r.speaker)) for r in result]
    if len({k for _s, _e, k in turns}) <= 1 and num_speakers <= 1:
        return turns, 0
    if num_speakers > 0 and len({k for _s, _e, k in turns}) < num_speakers:
        # the first pass found fewer voices than the user says there are: ask for exactly that many
        cfg.clustering = so.FastClusteringConfig(num_clusters=num_speakers, threshold=CLUSTER_THRESHOLD)
        sd = so.OfflineSpeakerDiarization(cfg)
        result = sd.process(data).sort_by_start_time()
        turns = [(float(r.start), float(r.end), int(r.speaker)) for r in result]
    extractor = so.SpeakerEmbeddingExtractor(emb_cfg)

    def fingerprint(start: float, end: float) -> Optional[np.ndarray]:
        a, b = int(start * RATE), int(min(end, start + 15.0) * RATE)
        if b - a < int(0.4 * RATE):
            return None
        st = extractor.create_stream()
        st.accept_waveform(sample_rate=RATE, waveform=np.ascontiguousarray(data[a:b]))
        st.input_finished()
        if not extractor.is_ready(st):
            return None
        v = np.asarray(extractor.compute(st), dtype=np.float64)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    turns, folded = refine_speakers(turns, fingerprint, num_speakers)
    if progress:
        progress(1.0)
    return turns, folded


# First pass: distance threshold for grouping voice snippets. Low on purpose -
# it is far easier to merge two halves of one speaker afterwards than to pull
# apart two people who were lumped together.
CLUSTER_THRESHOLD = 0.5
# Second pass: two groups whose averaged fingerprints are at least this alike
# (cosine similarity) are the same person.
MERGE_SIMILARITY = 0.78
# A group with less audio than this in total is a scrap, not a participant.
MINOR_SECONDS = 4.0


def refine_speakers(turns: List[Tuple[float, float, int]], fingerprint: Callable[[float, float], Optional[np.ndarray]],
                    num_speakers: int = 0) -> Tuple[List[Tuple[float, float, int]], int]:
    """Merge speaker groups using each group's average voice fingerprint.

    Without a fixed count: scraps (under MINOR_SECONDS in total - a cough, a
    burst of noise, one word in an odd tone) join whichever real speaker they
    sound most like, then real speakers that sound alike enough are merged.
    With a fixed count: the smallest scraps are rehomed first so they cannot
    use up one of the requested places, then the most alike pair is merged
    until exactly that many remain.
    """
    if not turns:
        return turns, 0
    dur: Dict[int, float] = {}
    vec: Dict[int, np.ndarray] = {}
    for s, e, k in turns:
        dur[k] = dur.get(k, 0.0) + (e - s)
        v = fingerprint(s, e)
        if v is not None:
            vec[k] = vec.get(k, 0.0) + v * (e - s)
    if len(dur) <= 1:
        return turns, 0
    remap: Dict[int, int] = {k: k for k in dur}

    def unit(v) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def sim(a: int, b: int) -> float:
        if a not in vec or b not in vec:
            return -1.0
        return float(unit(vec[a]) @ unit(vec[b]))

    def merge(src: int, dst: int) -> None:
        dur[dst] += dur.pop(src)
        if src in vec:
            vec[dst] = vec.get(dst, 0.0) + vec.pop(src)
        for k, v in remap.items():
            if v == src:
                remap[k] = dst

    def nearest_in_time(k: int, pool: List[int]) -> Optional[int]:
        best, best_gap = None, 1e9
        mine = [(s, e) for s, e, kk in turns if remap[kk] == k]
        for s2, e2, k2 in turns:
            r = remap[k2]
            if r not in pool:
                continue
            for s, e in mine:
                gap = max(0.0, s - e2, s2 - e)
                if gap < best_gap:
                    best, best_gap = r, gap
        return best

    folded = 0
    if num_speakers > 0:
        while len(dur) > num_speakers:
            small = min(dur, key=dur.get)
            others = [k for k in dur if k != small]
            if dur[small] < MINOR_SECONDS:
                target = max(others, key=lambda k: sim(small, k))
                if sim(small, target) < 0:
                    target = nearest_in_time(small, others) or target
                merge(small, target)
                folded += 1
                continue
            pairs = [(sim(a, b), a, b) for i, a in enumerate(sorted(dur)) for b in sorted(dur)[i + 1:]]
            _best, a, b = max(pairs)
            if dur[a] < dur[b]:
                a, b = b, a
            merge(b, a)
    else:
        majors = [k for k in dur if dur[k] >= MINOR_SECONDS]
        if majors:
            for k in [k for k in dur if k not in majors]:
                target = max(majors, key=lambda m: sim(k, m))
                if sim(k, target) < 0:
                    target = nearest_in_time(k, majors) or target
                merge(k, target)
                folded += 1
        while len(dur) > 1:
            keys = sorted(dur)
            pairs = [(sim(a, b), a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
            best, a, b = max(pairs)
            if best < MERGE_SIMILARITY:
                break
            if dur[a] < dur[b]:
                a, b = b, a
            merge(b, a)
    return [(s, e, remap[k]) for s, e, k in turns], folded


def speaker_frames(turns: List[Tuple[float, float, int]], total: int) -> np.ndarray:
    """Who is speaking in each 10 ms frame of the file (-1 = nobody labelled).

    Longest turns are laid down first and never overwritten: a two-word
    interjection that sits entirely inside someone else's sentence must not
    chop that sentence in half.
    """
    nframes = total // (RATE // 100) + 1
    label = np.full(nframes, -1, dtype=np.int32)
    for s, e, k in sorted(turns, key=lambda t: t[0] - t[1]):
        a, b = max(0, int(s * 100)), min(nframes, int(e * 100) + 1)
        view = label[a:b]
        view[view == -1] = k
    return label


def words_to_segments(words: List[Tuple[float, float, str]], label: Optional[np.ndarray]) -> List[dict]:
    """Give every timed word a speaker, then group the words into sentences.

    Each word takes whichever speaker holds most of its own duration. A word
    or two flipped to someone else in the middle of one person's sentence
    (the speaker model wobbling for a fraction of a second) is flipped back.
    Sentences end at the model's own full stops and wherever the speaker
    changes. Returns [{"start", "end", "speaker", "text"}].
    """
    if not words:
        return []
    who: List[Optional[int]] = []
    for ws, we, _t in words:
        k = None
        if label is not None:
            a = max(0, min(len(label) - 1, int(ws * 100)))
            b = max(a + 1, min(len(label), int(we * 100) + 1))
            frames = label[a:b]
            frames = frames[frames >= 0]
            if len(frames):
                vals, counts = np.unique(frames, return_counts=True)
                k = int(vals[np.argmax(counts)])
        who.append(k)
    # words nobody was labelled for take the nearest labelled word's speaker
    known = [i for i, k in enumerate(who) if k is not None]
    if known:
        for i, k in enumerate(who):
            if k is None:
                j = min(known, key=lambda x: abs(x - i))
                who[i] = who[j]
        # flip back short wobbles: a run of under 0.5 s between two runs of one speaker
        runs: List[List[int]] = []
        for i, k in enumerate(who):
            if runs and who[runs[-1][0]] == k:
                runs[-1][1] = i
            else:
                runs.append([i, i])
        for n in range(1, len(runs) - 1):
            a, b = runs[n]
            left, right = who[runs[n - 1][0]], who[runs[n + 1][0]]
            if left == right and words[b][1] - words[a][0] < 0.5 and b - a < 2:
                for i in range(a, b + 1):
                    who[i] = left
        # A change of speaker two words into a sentence, or two words before
        # its end, is the speaker model being a fraction of a second early or
        # late - people do not swap mid-sentence like that. Snap those scraps
        # to whoever says the bulk of the sentence.
        start = 0
        for i, (_ws, _we, text) in enumerate(words):
            last = i == len(words) - 1
            if not (last or (text.endswith((".", "?", "!")) and not text.endswith("..."))):
                continue
            idx = list(range(start, i + 1))
            start = i + 1
            share: Dict[int, float] = {}
            for j in idx:
                share[who[j]] = share.get(who[j], 0.0) + max(0.02, words[j][1] - words[j][0])
            if len(share) < 2:
                continue
            major = max(share, key=share.get)
            if share[major] < 0.6 * sum(share.values()):
                continue
            j = 0
            while j < len(idx):
                k = j
                while k + 1 < len(idx) and who[idx[k + 1]] == who[idx[j]]:
                    k += 1
                if who[idx[j]] != major and k - j < 2 and words[idx[k]][1] - words[idx[j]][0] < 0.8:
                    for m in range(j, k + 1):
                        who[idx[m]] = major
                j = k + 1
    out: List[dict] = []
    cur: Optional[dict] = None
    for (ws, we, text), k in zip(words, who):
        if cur is None or cur["speaker"] != k:
            if cur is not None:
                out.append(cur)
            cur = {"start": ws, "end": we, "speaker": k, "text": text}
        else:
            cur["text"] += " " + text
            cur["end"] = we
        if text.endswith((".", "?", "!")) and not text.endswith("..."):
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def build_units(speech: List[Tuple[int, int]], turns: Optional[List[Tuple[float, float, int]]],
                total: int, join_gap: float = 1.0) -> List[dict]:
    """Cut the speech regions at speaker changes and join neighbours of one speaker.

    Neighbouring stretches by one speaker are joined across pauses of up to
    `join_gap` seconds, so the model hears whole sentences with their context
    rather than fragments. Returns [{"start": sample, "end": sample,
    "speaker": index or None}], each at most MAX_UNIT seconds, in time order.
    """
    step = RATE // 100  # 10 ms frames
    label = speaker_frames(turns, total) if turns else None

    pieces: List[List] = []
    for s, e in speech:
        if e - s < RATE // 10:
            continue
        if label is None:
            pieces.append([s, e, None])
            continue
        a, b = s // step, max(s // step + 1, e // step)
        lab = label[a:b].copy()
        known = np.nonzero(lab >= 0)[0]
        if len(known) == 0:
            pieces.append([s, e, None])
            continue
        # frames inside detected speech that no turn covers take the nearest speaker
        idx = np.arange(len(lab))
        pos = np.searchsorted(known, idx).clip(0, len(known) - 1)
        prev = known[(pos - 1).clip(0)]
        nxt = known[pos]
        lab = lab[np.where(np.abs(idx - prev) <= np.abs(nxt - idx), prev, nxt)]
        runs: List[List] = []
        start = 0
        for i in range(1, len(lab) + 1):
            if i == len(lab) or lab[i] != lab[start]:
                runs.append([start, i, int(lab[start])])
                start = i
        # a run under 0.4 s is not a turn: give it to the longer neighbour
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for i, r in enumerate(runs):
                if r[1] - r[0] < 40:
                    left = runs[i - 1] if i > 0 else None
                    right = runs[i + 1] if i + 1 < len(runs) else None
                    if left is not None and (right is None or left[1] - left[0] >= right[1] - right[0]):
                        left[1] = r[1]
                    else:
                        right[0] = r[0]
                    del runs[i]
                    changed = True
                    break
            merged: List[List] = []
            for r in runs:
                if merged and merged[-1][2] == r[2]:
                    merged[-1][1] = r[1]
                else:
                    merged.append(r)
            runs = merged
        for i, (ra, rb, k) in enumerate(runs):
            ps = s if i == 0 else (a + ra) * step
            pe = e if i == len(runs) - 1 else (a + rb) * step
            pieces.append([ps, pe, k])

    # hard split anything too long for the model
    limit = int(HARD_MAX * RATE)
    split: List[List] = []
    for s, e, k in pieces:
        while e - s > limit:
            split.append([s, s + limit, k])
            s += limit
        split.append([s, e, k])

    units: List[dict] = []
    for s, e, k in split:
        last = units[-1] if units else None
        if (last is not None and last["speaker"] == k and s - last["end"] <= int(join_gap * RATE)
                and e - last["start"] <= int(MAX_UNIT * RATE)):
            last["end"] = e
        else:
            units.append({"start": s, "end": e, "speaker": k})
    return units


class Recognizer:
    """One loaded speech-to-text model."""

    def __init__(self, model_id: str, language: str, translate: bool, threads: int):
        so = _sherpa()
        spec = models.ASR[model_id]
        files = models.require(model_id)
        self.kind = spec["kind"]
        self.model_id = model_id
        if self.kind == "whisper":
            self.rec = so.OfflineRecognizer.from_whisper(
                encoder=files["encoder"], decoder=files["decoder"], tokens=files["tokens"],
                language=language or "", task="translate" if translate else "transcribe",
                num_threads=threads, enable_segment_timestamps=True)
        else:
            self.rec = so.OfflineRecognizer.from_transducer(
                files["encoder"], files["decoder"], files["joiner"], files["tokens"],
                num_threads=threads, provider="cpu", debug=False, decoding_method="greedy_search",
                model_type="nemo_transducer")

    def run(self, samples: np.ndarray, offset: float) -> dict:
        """Recognise one stretch of audio; `offset` is where it starts in the file (seconds).

        Returns {"text", "parts": [(start, end, text)], "words": [(start, end,
        word)], "confidence", "language"}. "parts" are the model's own timed
        phrases (Whisper); "words" are individually timed words (Parakeet),
        empty for a model that does not time them.
        """
        stream = self.rec.create_stream()
        stream.accept_waveform(RATE, samples)
        self.rec.decode_stream(stream)
        res = stream.result
        text = (res.text or "").strip()
        length = len(samples) / RATE
        parts: List[Tuple[float, float, str]] = []
        words: List[Tuple[float, float, str]] = []
        conf = None
        if self.kind == "whisper":
            texts = list(res.segment_texts)
            starts = list(res.segment_timestamps)
            durs = list(res.segment_durations)
            if texts and len(texts) == len(starts) == len(durs):
                for t, s, d in zip(texts, starts, durs):
                    t = t.strip()
                    if t:
                        s = min(max(0.0, float(s)), length)
                        parts.append((offset + s, offset + min(length, s + max(0.2, float(d))), t))
        else:
            tokens = list(res.tokens)
            starts = [float(x) for x in res.timestamps]
            durs = [float(x) for x in res.durations]
            conf = filters.confidence(res.ys_log_probs)
            if tokens and len(tokens) == len(starts):
                if len(durs) != len(tokens):
                    durs = [0.08] * len(tokens)
                words = _words(tokens, starts, durs, offset, length)
        if not parts and text:
            parts = [(offset, offset + length, text)]
        return {"text": text, "parts": parts, "words": words, "confidence": conf,
                "language": (res.lang or "").strip("<|>")}


def _words(tokens: List[str], starts: List[float], durs: List[float], offset: float,
           length: float) -> List[Tuple[float, float, str]]:
    """Join word-piece tokens into whole words, each with its start and end time in the file."""
    out: List[Tuple[float, float, str]] = []
    cur, cur_start, cur_end = "", 0.0, 0.0
    for tok, s, d in zip(tokens, starts, durs):
        if tok.startswith((" ", "\u2581")) and cur.strip():
            out.append((offset + min(cur_start, length), offset + min(max(cur_end, cur_start + 0.02), length),
                        cur.replace("\u2581", " ").strip()))
            cur = ""
        if not cur.strip():
            cur_start = s
        cur += tok
        cur_end = s + d
    if cur.strip():
        out.append((offset + min(cur_start, length), offset + min(max(cur_end, cur_start + 0.02), length),
                    cur.replace("\u2581", " ").strip()))
    return out


def _db(x: float) -> float:
    return 10.0 * float(np.log10(max(x, 1e-12)))


# --- the whole job --------------------------------------------------------------

def transcribe(path: str, options: Options, emit: Optional[Callable[[dict], None]] = None,
               cancelled: Optional[Callable[[], bool]] = None, workdir: Optional[str] = None) -> dict:
    """Run the full pipeline on one file and return the transcript dict (see export.py)."""
    emit = emit or (lambda _e: None)
    started = time.time()
    threads = options.threads or default_threads()
    spec = models.ASR[options.model]
    want_denoise = options.denoise != "off"
    need = models.needed_for(options.model, options.speakers, want_denoise)
    absent = models.missing(need)
    if absent:
        names = ", ".join(models.CATALOGUE[m]["label"] for m in absent)
        raise EngineError(f"Missing model{'s' if len(absent) > 1 else ''}: {names}. Use Models > Download.")

    # stage weights for the overall progress bar
    w = {"decode": 0.04, "speech": 0.05, "clean": 0.08 if want_denoise else 0.0,
         "speakers": 0.30 if options.speakers else 0.0, "text": 0.0}
    w["text"] = 1.0 - sum(w.values())
    order = ["decode", "speech", "clean", "speakers", "text"]
    labels = {"decode": "Reading the audio", "speech": "Finding speech", "clean": "Reducing noise",
              "speakers": "Working out who spoke when", "text": "Transcribing"}

    def stage(name: str):
        base = sum(w[s] for s in order[:order.index(name)])

        def report(frac: float, detail: str = ""):
            if cancelled and cancelled():
                raise Cancelled()
            emit({"type": "progress", "stage": name, "label": labels[name],
                  "fraction": min(1.0, base + w[name] * max(0.0, min(1.0, frac))), "detail": detail})
        return report

    notes: List[str] = []
    samples = work = listen = chunk = seg = None
    own_tmp = None
    if workdir is None:
        own_tmp = tempfile.TemporaryDirectory(prefix="verbatim-", ignore_cleanup_errors=True)
        workdir = own_tmp.name
    raw_path = os.path.join(workdir, "audio.f32")
    clean_path = os.path.join(workdir, "clean.f32")
    try:
        # 1. decode
        rep = stage("decode")
        rep(0.0)
        info = audio.probe(path)
        samples = audio.decode(path, raw_path, info.get("duration"), lambda f: rep(f), cancelled,
                               highpass=want_denoise)
        total = len(samples)
        duration = total / RATE
        if duration < 0.2:
            raise EngineError(f"{os.path.basename(path)} holds no audio to speak of.")

        # 2. speech detection
        rep = stage("speech")
        rep(0.0)
        vad_path = models.require("vad")["model"]
        speech = detect_speech(samples, vad_path, options.filtering, threads, lambda f: rep(f), cancelled)
        snr = audio.snr_db(samples, speech)

        # 3. noise reduction
        work = samples
        cleaned = False
        if speech and want_denoise and (options.denoise == "on" or (snr is not None and snr < AUTO_DENOISE_BELOW_DB)):
            rep = stage("clean")
            rep(0.0)
            work = reduce_noise(samples, models.require("denoise")["model"], clean_path, threads,
                                lambda f: rep(f), cancelled)
            cleaned = True
            speech = detect_speech(work, vad_path, options.filtering, threads, None, cancelled)
            if snr is not None:
                notes.append(f"Noisy recording (speech about {snr:.0f} dB above the background): noise reduction applied.")
            else:
                notes.append("Noise reduction applied.")
        speech_seconds = sum(e - s for s, e in speech) / RATE

        result = {
            "app": f"{paths.APP} {paths.VERSION}", "file": path, "duration": round(duration, 2),
            "model": options.model, "language": "", "options": asdict(options), "speakers": {},
            "segments": [], "removed": [], "notes": notes,
            "stats": {"speech_seconds": round(speech_seconds, 1), "snr_db": None if snr is None else round(snr, 1),
                      "noise_reduction": cleaned},
        }
        if not speech:
            notes.append("No speech was found in this file.")
            emit({"type": "progress", "stage": "text", "label": labels["text"], "fraction": 1.0, "detail": ""})
            result["stats"]["seconds_taken"] = round(time.time() - started, 1)
            return result

        # 4. speakers
        turns = None
        if options.speakers:
            rep = stage("speakers")
            rep(0.0)
            turns, folded = diarize(work, models.require("segmentation")["model"],
                                    models.require("embedding")["model"], options.num_speakers, threads,
                                    lambda f: rep(f))
            if folded:
                notes.append(f"{folded} stray voice{'s' if folded != 1 else ''} with only a moment of audio "
                             f"{'were' if folded != 1 else 'was'} merged into the speaker "
                             f"{'they' if folded != 1 else 'it'} sounded most like.")
            rep(1.0)

        # 5. transcribe
        rep = stage("text")
        rep(0.0, "loading the model")
        rec = Recognizer(options.model, options.language, options.translate, threads)
        # The model is given whole stretches of speech, cut only at pauses, and
        # the text is shared out between the speakers afterwards from its
        # timestamps - word by word with Parakeet, phrase by phrase with
        # Whisper, which only times phrases. Cutting the audio itself at each
        # change of speaker would clip the words either side of the cut.
        timed = spec["kind"] != "whisper"
        units = build_units(speech, None, total, 1.0 if timed else 1.5)
        label = speaker_frames(turns, total) if turns else None

        def majority(start: float, end: float) -> Optional[int]:
            if label is None:
                return None
            a_ = max(0, min(len(label) - 1, int(start * 100)))
            frames = label[a_:max(a_ + 1, min(len(label), int(end * 100) + 1))]
            frames = frames[frames >= 0]
            if not len(frames):
                return None
            vals, counts = np.unique(frames, return_counts=True)
            return int(vals[np.argmax(counts)])

        listen = work if options.asr_audio == "cleaned" else samples
        pad = int(PAD * RATE)
        locked = options.language
        if spec["kind"] == "whisper" and not options.language and not options.translate and len(units) > 1:
            # Whisper guesses the language afresh for every piece it is given,
            # and on a short or noisy piece the guess can be anything - which
            # is how a stray line of Welsh or Spanish ends up in an English
            # meeting. Decide once, from the three longest stretches of
            # speech, and hold the whole file to it.
            votes: Dict[str, float] = {}
            for u in sorted(units, key=lambda u: u["start"] - u["end"])[:3]:
                a, b = max(0, u["start"] - pad), min(total, u["end"] + pad)
                lang = rec.run(np.ascontiguousarray(listen[a:b], dtype=np.float32), a / RATE)["language"]
                if lang:
                    votes[lang] = votes.get(lang, 0.0) + (b - a)
            if votes:
                locked = max(votes, key=votes.get)
                rec = Recognizer(options.model, locked, False, threads)
        # loudness of each unit against the typical unit, for the phantom-phrase check
        levels = []
        for u in units:
            seg = np.asarray(work[u["start"]:u["end"]], dtype=np.float32)
            levels.append(_db(float(np.mean(seg.astype(np.float64) ** 2)) if len(seg) else 0.0))
        typical = float(np.median(levels)) if levels else 0.0
        done_audio = 0
        all_audio = sum(u["end"] - u["start"] for u in units) or 1
        langs: Dict[str, float] = {}
        segments: List[dict] = []
        for i, (u, lvl) in enumerate(zip(units, levels)):
            if cancelled and cancelled():
                raise Cancelled()
            # a little context either side, but never past the middle of the
            # pause that separates this unit from its neighbour
            before = u["start"] - (units[i - 1]["end"] if i else 0)
            after = (units[i + 1]["start"] if i + 1 < len(units) else total) - u["end"]
            a = max(0, u["start"] - max(min(pad, before // 2), 0))
            b = min(total, u["end"] + max(min(pad, after // 2), 0))
            chunk = np.ascontiguousarray(listen[a:b], dtype=np.float32)
            out = rec.run(chunk, a / RATE)
            length = (u["end"] - u["start"]) / RATE
            if out["language"]:
                langs[out["language"]] = langs.get(out["language"], 0.0) + length
            kept, reason = filters.judge(out["text"], length, out["confidence"], options.filtering, lvl - typical)
            if not kept:
                if out["text"]:
                    result["removed"].append({"start": round(u["start"] / RATE, 2), "end": round(u["end"] / RATE, 2),
                                              "text": out["text"], "reason": reason or "empty"})
            else:
                # stripping a [Music] tag leaves the model's own timing usable; a
                # collapsed loop does not, and becomes one piece
                intact = kept == out["text"] or reason == "non-speech markers removed"
                if reason and not intact:
                    result["removed"].append({"start": round(u["start"] / RATE, 2), "end": round(u["end"] / RATE, 2),
                                              "text": out["text"], "reason": reason, "kept_as": kept})
                conf = None if out["confidence"] is None else round(out["confidence"], 3)
                if timed and out["words"] and intact:
                    for seg_ in words_to_segments(out["words"], label):
                        segments.append({"start": round(seg_["start"], 2), "end": round(seg_["end"], 2),
                                         "speaker": seg_["speaker"], "text": seg_["text"], "confidence": conf})
                else:
                    parts = out["parts"] if intact else [(u["start"] / RATE, u["end"] / RATE, kept)]
                    for ps, pe, pt in parts:
                        if options.filtering != "off":
                            why = filters.phantom(pt, options.filtering) if len(parts) > 1 else None
                            if why:
                                result["removed"].append({"start": round(ps, 2), "end": round(pe, 2), "text": pt,
                                                          "reason": why})
                                continue
                            pt = filters.strip_tags(pt)
                        if not pt:
                            continue
                        ps = max(u["start"] / RATE - PAD, min(ps, u["end"] / RATE))
                        pe = max(ps + 0.05, min(pe, u["end"] / RATE + PAD))
                        who = majority(ps, pe)
                        if who is None:
                            who = majority(u["start"] / RATE, u["end"] / RATE)
                        segments.append({"start": round(ps, 2), "end": round(pe, 2), "speaker": who, "text": pt,
                                         "confidence": conf})
            done_audio += u["end"] - u["start"]
            rep(done_audio / all_audio, f"{export_clock(u['end'] / RATE)} of {export_clock(duration)}")

        # 6. name the speakers in order of first appearance
        if options.speakers:
            order_seen: List[int] = []
            for s in segments:
                if s["speaker"] is not None and s["speaker"] not in order_seen:
                    order_seen.append(s["speaker"])
            ids = {k: f"S{i + 1}" for i, k in enumerate(order_seen)}
            last = None
            for s in segments:
                if s["speaker"] is None:
                    s["speaker"] = last
                else:
                    s["speaker"] = ids[s["speaker"]]
                    last = s["speaker"]
            first = next((s["speaker"] for s in segments if s["speaker"]), None)
            for s in segments:
                if s["speaker"] is None:
                    s["speaker"] = first
            result["speakers"] = {v: f"Speaker {i + 1}" for i, v in enumerate(ids.values())}
        else:
            for s in segments:
                s["speaker"] = None
        result["segments"] = segments
        if spec["kind"] == "whisper":
            result["language"] = locked or (max(langs, key=langs.get) if langs else "")
            if options.translate:
                result["language"] = (result["language"] + " > en").strip()
        elif options.model == "parakeet-v2":
            result["language"] = "en"
        dropped = [r for r in result["removed"] if "kept_as" not in r]
        if dropped:
            notes.append(f"{len(dropped)} piece{'s' if len(dropped) != 1 else ''} of doubtful text "
                         f"{'were' if len(dropped) != 1 else 'was'} left out (listed under Removed).")
        result["stats"]["seconds_taken"] = round(time.time() - started, 1)
        result["stats"]["words"] = sum(len(s["text"].split()) for s in segments)
        return result
    finally:
        # Windows will not delete a file that is still memory-mapped, and a
        # slice of a memory map keeps the whole map alive: drop every reference.
        samples = work = listen = chunk = seg = None  # noqa: F841
        gc.collect()
        for p in (raw_path, clean_path):
            try:
                os.remove(p)
            except OSError:
                pass
        if own_tmp is not None:
            try:
                own_tmp.cleanup()
            except Exception:  # noqa: BLE001
                pass


def export_clock(seconds: float) -> str:
    from .export import clock
    return clock(seconds)
