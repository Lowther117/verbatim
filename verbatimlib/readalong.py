"""Read along with the original recording.

The finished transcript plays back in the recording's own voice while each word
lights up as it is said, and the RSVP panel can flash the words one at a time in
step with it. Untick Audio and the same panel speed-reads the transcript on a
words-per-minute clock instead.

The playback, highlight and RSVP machinery is readaloud.py, the module Hayloft
and Lectern read stories aloud with (keep all three copies byte-identical). This
file adds the one thing those apps do not have: an "engine" that, instead of
synthesising a sentence, cuts that sentence out of the recording.

Timing comes from the transcript. Parakeet gives every word its own time, and
those are saved with each segment ("words"); for Whisper transcripts (and ones
saved before word times were kept) the words of each segment are spread over
the segment by their length, which is close but not exact.
"""
from __future__ import annotations

import bisect
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from typing import List, Optional, Tuple

from . import audio, export
from . import readaloud as ra

RATE = ra.SAMPLE_RATE
LEAD = 0.12         # seconds of lead-in before a sentence that does not follow straight on from the last
LEAD_MID = 0.08     # ... and before a word clicked in the middle of a sentence
KEEP_GAP = 1.5      # pauses up to this long between sentences are played as they are
TAIL = 0.45         # a longer pause is cut down to this much after the last word
_WORD = re.compile(r"\S+")


class Track:
    """A transcript laid out as one read-along text, with a time for every word.

    doc        the readaloud Doc of the text (paragraph bodies, blank line between)
    para_off   doc offset where each paragraph's text starts (export.paragraphs order)
    times      [(start, end)] seconds in the recording, per doc word
    spans      [(from, to)] seconds of recording that play for each sentence
    """

    def __init__(self, t: dict):
        segs = t.get("segments") or []
        paras = export.paragraphs(t)
        pieces: List[str] = []
        self.para_off: List[int] = []
        ranges: List[Tuple[int, int, int]] = []       # (doc start, doc end, segment index)
        off = 0
        for k, p in enumerate(paras):
            if k:
                pieces.append("\n\n")
                off += 2
            self.para_off.append(off)
            built = []
            pos = off
            for n in p["segments"]:
                s = str(segs[n].get("text", "")).strip()
                if not s:
                    continue
                if built:
                    pos += 1
                ranges.append((pos, pos + len(s), n))
                built.append(s)
                pos += len(s)
            if " ".join(built) != p["text"]:          # never expected; fall back to the paragraph as one piece
                ranges = [r for r in ranges if r[0] < off]
                ranges.append((off, off + len(p["text"]), p["segments"][0]))
            pieces.append(p["text"])
            off += len(p["text"])
        self.text = "".join(pieces)
        self.doc = ra.tokenize(self.text)
        words = self.doc.words
        self.duration = float(t.get("duration") or (segs[-1]["end"] + 1.0 if segs else 0.0))
        self.times: List[Tuple[float, float]] = [(0.0, 0.0)] * len(words)
        starts = [w.start for w in words]
        for a, b, n in ranges:
            lo, hi = bisect.bisect_left(starts, a), bisect.bisect_left(starts, b)
            if lo < hi:
                self._time_segment(words[lo:hi], segs[n], a)
        filled = 0.0
        for i, (a, b) in enumerate(self.times):        # a word outside every segment: hold the last time
            if b <= 0.0 and a <= 0.0:
                self.times[i] = (filled, filled)
            filled = self.times[i][1]
        self.spans = self._sentence_spans()

    def _time_segment(self, words, seg: dict, base: int):
        s0, s1 = float(seg["start"]), float(seg["end"])
        text = str(seg.get("text", "")).strip()
        wt = seg.get("words")
        toks = [m.start() for m in _WORD.finditer(text)]
        if isinstance(wt, list) and len(wt) == len(toks) and toks:
            # Parakeet's own word times: each doc word takes the time of the word it sits in
            # (a token the tokenizer splits in two shares its time out between the halves)
            groups = {}
            for w in words:
                k = max(0, bisect.bisect_right(toks, w.start - base) - 1)
                groups.setdefault(k, []).append(w)
            for k, ws in groups.items():
                try:
                    a, b = float(wt[k][0]), float(wt[k][1])
                except (TypeError, ValueError, IndexError):
                    a, b = s0, s1
                if len(ws) == 1:
                    self.times[ws[0].i] = (a, max(a, b))
                else:
                    for w, (x, y) in zip(ws, ra.spread_times(ws, max(0.05, b - a))):
                        self.times[w.i] = (a + x, a + y)
            return
        for w, (x, y) in zip(words, ra.spread_times(words, max(0.05, s1 - s0))):
            self.times[w.i] = (s0 + x, s0 + y)

    def _sentence_spans(self) -> List[Tuple[float, float]]:
        sents = self.doc.sentences
        out: List[Tuple[float, float]] = []
        prev_to = 0.0
        for k, s in enumerate(sents):
            first = self.times[s.first][0]
            last_end = self.times[s.last][1]
            if k + 1 < len(sents):
                nxt = self.times[sents[k + 1].first][0]
                to = nxt if nxt - last_end <= KEEP_GAP else last_end + TAIL
            else:
                to = last_end + TAIL
            if self.duration:
                to = min(to, self.duration)
            frm = min(first, max(prev_to, first - LEAD))
            to = max(to, frm + 0.05)
            out.append((frm, to))
            prev_to = to
        return out


def _atempo(speed: float) -> str:
    """ffmpeg's atempo, chained so that every link stays within 0.5-2x (older ffmpegs refuse more)."""
    parts = []
    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    parts.append(f"atempo={speed:.4f}")
    return ",".join(parts)


class RecordingEngine(ra.Engine):
    """Speaks a sentence by cutting it out of the recording.

    The recording is decoded once, on first use, to a plain WAV in the temp folder,
    so every cut lands on the exact sample whatever the source format (a seek into
    a variable-bitrate MP3 can be seconds out). Other speeds go through ffmpeg's
    atempo, which keeps the pitch.
    """

    id = "recording"
    label = "Original recording"

    def __init__(self, log=None):
        super().__init__(log)
        self.src = ""
        self.track: Optional[Track] = None
        self.note = ""                  # "Preparing the recording… 40%" while decoding
        self._lock = threading.Lock()   # one decode at a time
        self._wav = ""
        self._wav_src = None            # (path, size, mtime) the WAV was made from
        self._tmp = ""
        self._proc: Optional[subprocess.Popen] = None
        self._closed = False

    @classmethod
    def available(cls):
        return True, ""

    def voices(self, fetch=True):
        return [("original", "Original recording")]

    def default_voice(self, fetch=True):
        return "original"

    def allowed(self, vid):
        return True

    def set_source(self, path: str, track: Track):
        self.src = path
        self.track = track

    def _stamp(self, path: str):
        st = os.stat(path)
        return (path, st.st_size, int(st.st_mtime))

    def _prepare(self) -> str:
        """The WAV of the current recording, decoding it first if need be (synthesis thread only)."""
        path = self.src
        name = os.path.basename(path) or "the recording"
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"{name} is not where it was (moved or renamed?) - untick Audio to speed-read without it")
        with self._lock:
            stamp = self._stamp(path)
            if self._wav and self._wav_src == stamp and os.path.isfile(self._wav):
                return self._wav
            self._drop_wav()
            if not self._tmp or not os.path.isdir(self._tmp):
                self._tmp = tempfile.mkdtemp(prefix="verbatim-listen-")
            out = os.path.join(self._tmp, "recording.wav")
            cmd = [audio.find_ffmpeg(), "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", path,
                   "-vn", "-sn", "-dn", "-map", "0:a:0", "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", out]
            err_path = out + ".err"
            expected = (self.track.duration if self.track and self.track.duration else 0) * RATE * 2
            self.note = "Preparing the recording…"
            try:
                with open(err_path, "wb") as err:
                    self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                                  stderr=err, **audio._no_window())
                    try:
                        while self._proc.poll() is None:
                            if self._closed or self.src != path:    # quitting, or another recording picked
                                raise RuntimeError("stopped")
                            if expected:
                                try:
                                    pct = int(100 * min(0.99, os.path.getsize(out) / expected))
                                    self.note = f"Preparing the recording… {pct}%"
                                except OSError:
                                    pass
                            time.sleep(0.1)
                    finally:
                        if self._proc.poll() is None:
                            self._proc.kill()
                        self._proc.wait()
                code = self._proc.returncode
                with open(err_path, encoding="utf-8", errors="replace") as fh:
                    stderr = fh.read()
            finally:
                self._proc = None
                self.note = ""
                try:
                    os.remove(err_path)
                except OSError:
                    pass
            ok = os.path.isfile(out) and os.path.getsize(out) > 44 + RATE // 50
            if not ok:
                raise RuntimeError(audio._friendly(stderr, path) if stderr.strip() else f"{name} gave no audio (ffmpeg exit {code})")
            self._wav, self._wav_src = out, stamp
            return out

    def _drop_wav(self):
        if self._wav:
            try:
                os.remove(self._wav)
            except OSError:
                pass
        self._wav, self._wav_src = "", None

    def _cut(self, frm: float, to: float, speed: float) -> bytes:
        path = self._prepare()
        with wave.open(path, "rb") as w:
            total = w.getnframes()
            a = max(0, min(total, int(round(frm * RATE))))
            b = max(a, min(total, int(round(to * RATE))))
            w.setpos(a)
            pcm = w.readframes(b - a)
        if abs(speed - 1.0) < 0.01 or not pcm:
            return pcm
        cmd = [audio.find_ffmpeg(), "-hide_banner", "-nostdin", "-v", "error", "-f", "s16le", "-ar", str(RATE),
               "-ac", "1", "-i", "pipe:0", "-af", _atempo(speed), "-f", "s16le", "-ar", str(RATE), "-ac", "1", "pipe:1"]
        r = subprocess.run(cmd, input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
                           **audio._no_window())
        if r.returncode != 0 or not r.stdout:
            raise RuntimeError("ffmpeg could not change the speed: "
                               + (r.stderr.decode("utf-8", "replace").strip().splitlines() or ["no reason given"])[-1])
        return r.stdout

    def synth(self, words, text, voice, speed):
        tr = self.track
        if tr is None or not words or words[-1].i >= len(tr.times):
            raise RuntimeError("the transcript changed - press Read again")
        first = words[0]
        frm, to = tr.spans[first.sent]
        if first.i != tr.doc.sentences[first.sent].first:     # started on a clicked word mid-sentence
            frm = max(frm, tr.times[first.i][0] - LEAD_MID)
        pcm = self._cut(frm, to, speed)
        times = [((tr.times[w.i][0] - frm) / speed, (tr.times[w.i][1] - frm) / speed) for w in words]
        return ra.Clip(pcm, times)

    def close(self):
        self._closed = True
        p = self._proc
        if p is not None:
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            self._drop_wav()
            if self._tmp:
                shutil.rmtree(self._tmp, ignore_errors=True)
                self._tmp = ""


class RecordingReader(ra.Reader):
    """readaloud's Reader with the recording as its one and only voice."""

    def __init__(self, settings, log=None):
        super().__init__(settings, log)
        self.engines = {"recording": RecordingEngine}
        self.engine_order = ("recording",)

    @property
    def rec(self) -> RecordingEngine:
        return self.engine("recording")

    def voice(self, fetch=True):
        return "recording", "original"

    def current(self):
        idx, state = super().current()
        note = self.rec.note
        self.progress = note or ("Opening the recording…" if state == "loading" else "")
        return idx, state


class RecordingBar(ra.ReadBar):
    """The read-along strip: transport, speed, Audio / RSVP / WPM - no voice to choose."""

    def __init__(self, parent, app, get_doc, on_word, on_end=None, rsvp_parent=None, rsvp_before=None):
        super().__init__(parent, app, get_doc, on_word, on_end, rsvp_parent, rsvp_before)
        self.reader = RecordingReader(self.s, log=app.log)
        self.var_speed.set(self.reader.speed())
        self._speed_moved(None)

    def _fill_voices(self):
        self._voice_items = [("recording", "original")]
        self.cmb_voice["values"] = ["Original recording"]
        self.cmb_voice.current(0)
        self.cmb_voice.state(["disabled"])

    def _voice_picked(self, event=None):
        pass

    def _speed_set(self):
        self.reader.set_speed(round(self.var_speed.get(), 2))
        self.app.save_settings()

    def diagnostics(self):
        rec = self.reader.rec
        return "\n".join([f"readaloud {ra.VERSION} (read along)", f"recording: {rec.src!r}",
                          f"decoded copy: {rec._wav!r}", f"mode: {self.mode}  cursor: {self.cursor}  "
                          f"reader state: {self.reader.state}", "last error: " + (self.reader.error or "none")])


class TranscriptHighlighter(ra.TextHighlighter):
    """Paints the current word and sentence in Verbatim's transcript pane.

    The pane holds more than the read-along text (speaker names, times, notes),
    so a doc offset is found through the "body{n}" tag on each paragraph's text
    rather than by counting lines from the top.
    """

    def __init__(self, text, track: Track, palette):
        self.text = text
        self.doc = track.doc
        self.para_off = track.para_off
        self.cur = None
        self._base = {}
        self.text.tag_configure("ra_sent")
        self.text.tag_configure("ra_word")
        self.raise_tags()
        self.apply_palette(palette)

    def raise_tags(self):
        self.text.tag_raise("ra_sent")
        self.text.tag_raise("ra_word")

    def rebuilt(self):
        """The pane was redrawn (same text): forget the cached positions and repaint."""
        self._base = {}
        cur, self.cur = self.cur, None
        self.raise_tags()
        if cur is not None:
            self.show(cur)

    def _para(self, off):
        return max(0, bisect.bisect_right(self.para_off, off) - 1)

    def _idx(self, off):
        n = self._para(off)
        base = self._base.get(n)
        if base is None:
            try:
                line, col = map(int, self.text.index(f"body{n}.first").split("."))
            except Exception:  # noqa: BLE001 - the pane no longer shows this transcript
                return "end"
            base = self._base[n] = (line, col)
        return f"{base[0]}.{base[1] + max(0, off - self.para_off[n])}"

    def word_at(self, tk_index):
        """Word index under a Text index like '@x,y' (a click on a speaker line picks that paragraph's first word)."""
        try:
            index = self.text.index(tk_index)
            tags = self.text.tag_names(index)
        except Exception:  # noqa: BLE001
            return None
        para = body = None
        for tag in tags:
            if tag.startswith("body") and tag[4:].isdigit():
                body = int(tag[4:])
            elif tag.startswith("para") and tag[4:].isdigit():
                para = int(tag[4:])
        n = body if body is not None else para
        if n is None or n >= len(self.para_off):
            return None
        off = self.para_off[n]
        if body is not None:
            try:
                off += int(self.text.count(f"body{n}.first", index, "chars")[0] or 0)
            except Exception:  # noqa: BLE001
                pass
        return self.doc.word_at_char(off)


def selftest_checks(check):
    """Offline checks of the timing; `check(name, cond, detail)`."""
    t = {"duration": 20.0, "speakers": {"S1": "A", "S2": "B"}, "segments": [
        {"start": 1.0, "end": 3.0, "speaker": "S1", "text": "Hello there, friend.",
         "words": [[1.0, 1.4], [1.5, 2.0], [2.2, 2.9]]},
        {"start": 3.1, "end": 5.0, "speaker": "S1", "text": "How are you?"},
        {"start": 12.0, "end": 14.0, "speaker": "S2", "text": "Fine thanks."}]}
    tr = Track(t)
    check("read along: text is the paragraphs", tr.text == "Hello there, friend. How are you?\n\nFine thanks.", repr(tr.text))
    check("read along: Parakeet word times kept", tr.times[:3] == [(1.0, 1.4), (1.5, 2.0), (2.2, 2.9)], str(tr.times[:3]))
    a = [x[0] for x in tr.times[3:6]]
    check("read along: other words spread over their segment", 3.1 <= a[0] < a[1] < a[2] < 5.0 and abs(tr.times[5][1] - 5.0) < 1e-6,
          str(tr.times[3:6]))
    check("read along: three sentences", len(tr.spans) == 3, str(tr.spans))
    s0, s1, s2 = tr.spans
    check("read along: sentences play back to back", abs(s0[1] - s1[0]) < 1e-9 and s0[0] < 1.0, str(tr.spans))
    check("read along: a long pause is cut short", s1[1] < 6.0 and s2[0] >= s1[1] and 11.0 < s2[0] <= 12.0, str(tr.spans))
    check("read along: atempo chain", _atempo(3.0) == "atempo=2.0,atempo=1.5000" and _atempo(1.25) == "atempo=1.2500")
    clip = None
    eng = RecordingEngine()
    eng.set_source("", tr)
    try:
        eng.synth(tr.doc.words[0:3], "", "original", 1.0)
    except RuntimeError as e:
        clip = str(e)
    check("read along: a missing recording says so", bool(clip) and "not where it was" in clip, str(clip))
