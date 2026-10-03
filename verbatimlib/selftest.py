"""Self-test: exercises every stage for real and writes verbatim-selftest.txt beside the app.

The build files run this against the finished app, because "it built" and "it
works" are different claims. A windowed build has no console, so the report is
a file as well as printed output. The line "PROBLEMS FOUND" is what the build
files look for.

A check that cannot run (a model that is not downloaded, no display for the
window test) is reported as skipped, not as a pass and not as a failure.
"""
from __future__ import annotations

import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from typing import Callable, List

from . import paths

SAMPLE_WORDS = ("afternoon train cardiff leaves platform four twenty past three thank long journey take "
                "fifty minutes short stop").split()


class Skip(Exception):
    pass


class Report:
    def __init__(self):
        self.lines: List[str] = []
        self.passed = self.failed = self.skipped = 0

    def say(self, text: str = ""):
        self.lines.append(text)
        try:
            print(text, flush=True)
        except Exception:  # noqa: BLE001
            pass

    def check(self, name: str, fn: Callable[[], object]):
        start = time.time()
        try:
            detail = fn()
            self.passed += 1
            took = time.time() - start
            extra = f" - {detail}" if detail else ""
            self.say(f"  ok    {name}{extra}" + (f"  ({took:.1f}s)" if took >= 1 else ""))
        except Skip as exc:
            self.skipped += 1
            self.say(f"  skip  {name} - {exc}")
        except Exception as exc:  # noqa: BLE001
            self.failed += 1
            self.say(f"  FAIL  {name} - {type(exc).__name__}: {exc}")
            tail = traceback.format_exc().strip().splitlines()[-6:]
            for ln in tail:
                self.say(f"          {ln}")


def _sample() -> str:
    for base in (os.path.dirname(os.path.abspath(__file__)), os.path.join(paths.resource_dir(), "verbatimlib")):
        p = os.path.join(base, "data", "selftest.ogg")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("the bundled sample recording (data/selftest.ogg) is missing from this build")


def run(argv: List[str]) -> int:
    quick = "--quick" in argv
    rep = Report()
    rep.say(f"{paths.APP} {paths.VERSION} self-test - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rep.say(f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()} {platform.machine()}"
            f"{' (frozen build)' if getattr(sys, 'frozen', False) else ' (from source)'}")
    rep.say(f"App folder: {paths.app_dir()}")
    if paths.data_dir() != paths.app_dir():
        rep.say(f"Settings and models: {paths.data_dir()} (the app folder cannot be written to)")
    rep.say()
    tmp = tempfile.mkdtemp(prefix="verbatim-selftest-")

    # --- the pieces ---------------------------------------------------------------
    rep.say("Pieces")

    def numpy_ok():
        import numpy
        return f"numpy {numpy.__version__}"
    rep.check("numpy loads", numpy_ok)

    def sherpa_ok():
        import sherpa_onnx
        for name in ("OfflineRecognizer", "VoiceActivityDetector", "OfflineSpeakerDiarization",
                     "SpeakerEmbeddingExtractor", "OfflineSpeechDenoiser"):
            if not hasattr(sherpa_onnx, name):
                raise RuntimeError(f"sherpa_onnx has no {name}")
        return f"sherpa-onnx {getattr(sherpa_onnx, '__version__', getattr(sherpa_onnx, 'version', '?'))}"
    rep.check("speech engine loads", sherpa_ok)

    def ffmpeg_ok():
        from . import audio
        exe = audio.find_ffmpeg()
        out = subprocess.run([exe, "-hide_banner", "-version"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=60, **audio._no_window()).stdout.decode("utf-8", "replace")
        if "ffmpeg version" not in out:
            raise RuntimeError("ffmpeg did not run")
        return out.split("\n")[0].strip()[:60]
    rep.check("ffmpeg runs", ffmpeg_ok)

    def certs_ok():
        from . import models
        ctx = models._context()
        n = len(ctx.get_ca_certs())
        try:
            import certifi  # noqa: F401
            return "certifi bundle"
        except Exception:  # noqa: BLE001
            if n == 0:
                raise Skip("no certificate bundle found - model downloads may fail; everything else is unaffected")
            return f"{n} system certificates"
    rep.check("certificates for model downloads", certs_ok)

    def save_ok():
        d = paths.downloads_dir()
        if not os.path.isdir(d):
            raise RuntimeError(f"{d} is not a folder")
        return d
    rep.check("Downloads folder found", save_ok)

    # --- logic ---------------------------------------------------------------------
    rep.say()
    rep.say("Logic")

    def filters_ok():
        from . import filters as f
        # (text, seconds of audio, model confidence or None, isolated by silence either side)
        drop = [("Thanks for watching!", 2.0, None, False), ("Subtitles by the Amara.org community", 4.0, None, False),
                ("[Music]", 6.0, None, False), ("(crying)", 5.0, None, False), ("[VIDEO PLAYBACK]", 1.0, None, False),
                ("Gracias por ver el video.", 5.0, None, False), ("H-h-h-h-h-h-h-h-h-h-h-h-h-h", 5.0, None, False),
                ("1/2 tbps", 12.0, None, False), ("so", 12.0, None, False), ("I think so. " * 8, 6.0, None, False),
                ("Okay.", 0.6, 0.2, False), ("Thank you.", 3.0, 0.3, False), ("the quick brown fox " * 30, 2.0, 0.9, False),
                ("you", 2.0, None, True), ("Thank you.", 1.5, None, True), ("Bye-bye.", 0.8, None, True),
                ("Thanks for watching, and I'll see you in the next video.", 3.0, None, False),
                ("Like, share, subscribe.", 2.0, None, False), ("3.5mm", 3.0, None, True),
                ("1.0.1.1.1.1.1.1.1", 3.0, None, False), ("google.com", 1.5, None, True)]
        keep = [("Good morning everyone, thanks for joining.", 3.0, 0.94), ("Yes.", 0.5, 0.9), ("No.", 0.7, 0.95),
                ("OK.", 0.8, 0.95), ("I do.", 1.4, 0.95), ("25.", 1.0, 0.95), ("1, 2, 3", 1.5, None),
                ("No, no, no, no, no, no.", 2.5, 0.95), ("Thank you.", 1.0, 0.95), ("Thank you.", 1.0, None),
                ("Bye-bye.", 0.8, None), ("See you next time.", 1.2, None), ("case-uk.co.uk", 1.5, 0.95),
                ("Thanks for watching the demo, Sarah will follow up.", 3.0, 0.9),
                ("That is a very, very, very good point.", 2.0, None),
                ("Translated by a colleague of mine, the document was then sent to the board for approval last week.",
                 7.0, 0.95),
                ("Subscribe to the newsletter if you want the monthly budget updates, and I will send the first one "
                 "on Friday.", 7.0, None),
                ("He said (sound good to you?) and I agreed", 3.0, 0.9),
                ("(Hello everyone and welcome to the meeting", 3.0, 0.9)]
        for text, dur, conf, isolated in drop:
            kept, _why = f.judge(text, dur, conf, "balanced", 0.0, isolated=isolated)
            if kept:
                raise AssertionError(f"should have been left out: {text[:40]!r}")
        for text, dur, conf in keep:
            kept, _why = f.judge(text, dur, conf, "balanced", 0.0)
            if kept != text:
                raise AssertionError(f"should have been kept as said: {text!r} -> {kept!r}")
        if f.judge("Thanks for watching!", 2.0, None, "off", 0.0)[0] != "Thanks for watching!":
            raise AssertionError("filtering 'off' must pass everything through")
        if f.phantom("See you next time.") or not f.phantom("Thanks for watching!"):
            raise AssertionError("a phrase tacked on after real speech was judged wrongly")
        start = time.time()
        f.judge("(a " * 40 + "x) hello", 3.0, None)
        if time.time() - start > 1.0:
            raise AssertionError("a pattern is backtracking")
        return f"{len(drop)} phantom pieces dropped, {len(keep)} real ones kept"
    rep.check("invented-text filter", filters_ok)

    def units_ok():
        from . import engine as e
        R = e.RATE
        speech = [(0, 5 * R), (6 * R, 16 * R), (20 * R, 21 * R)]
        turns = [(0.0, 5.0, 0), (6.0, 11.0, 1), (11.0, 16.0, 0), (8.0, 8.3, 0), (20.0, 21.0, 1)]
        units = e.build_units(speech, turns, 22 * R)
        got = [(round(u["start"] / R), round(u["end"] / R), u["speaker"]) for u in units]
        want = [(0, 5, 0), (6, 11, 1), (11, 16, 0), (20, 21, 1)]
        if got != want:
            raise AssertionError(f"{got} != {want}")
        plain = e.build_units([(0, 40 * R)], None, 40 * R)
        if any((u["end"] - u["start"]) / R > e.HARD_MAX + 0.01 for u in plain) or len(plain) < 2:
            raise AssertionError("a 40 s stretch was not split for the model")
        return "speech is cut at speaker changes; an interjection does not split a sentence"
    rep.check("speaker-change cutting", units_ok)

    def words_ok():
        from . import engine as e
        R = e.RATE
        # A says a sentence, B answers. The speaker model is 0.3 s late at the hand-over,
        # so B's first word "Oh," falls in A's turn; and it wobbles for one word mid-sentence.
        turns = [(0.0, 3.3, 0), (1.2, 1.4, 1), (3.3, 6.0, 1)]
        label = e.speaker_frames(turns, 6 * R)
        said = [(0.1, 0.5, "Shall"), (0.6, 0.9, "we"), (1.0, 1.4, "start"), (1.5, 1.9, "with"), (2.0, 2.3, "the"),
                (2.4, 2.9, "budget?"), (3.0, 3.25, "Oh,"), (3.4, 3.8, "yes,"), (3.9, 4.3, "of"), (4.4, 4.9, "course."),
                (5.0, 5.6, "Thanks.")]
        segs = e.words_to_segments(said, label)
        got = [(g["speaker"], g["text"]) for g in segs]
        want = [(0, "Shall we start with the budget?"), (1, "Oh, yes, of course."), (1, "Thanks.")]
        if got != want:
            raise AssertionError(f"{got}")
        if not (abs(segs[1]["start"] - 3.0) < 1e-6 and abs(segs[1]["end"] - 4.9) < 1e-6):
            raise AssertionError("sentence times are wrong")
        if [g["speaker"] for g in e.words_to_segments(said, None)] != [None, None, None]:
            raise AssertionError("without speaker labels every sentence must be unlabelled")
        doc = [(0.0, 0.2, "I"), (0.3, 0.5, "saw"), (0.6, 0.8, "Dr."), (0.9, 1.2, "Smith"), (1.3, 1.6, "today."),
               (1.8, 2.0, "Fine.")]
        if [g["text"] for g in e.words_to_segments(doc, None)] != ["I saw Dr. Smith today.", "Fine."]:
            raise AssertionError("an abbreviation ended a sentence")
        return "each word goes to one speaker; a late hand-over snaps to the sentence"
    rep.check("sharing words out between speakers", words_ok)

    def refine_ok():
        import numpy as np
        from . import engine as e
        a, b, c, d = np.eye(4)
        # two takes of voice A (one through a different microphone), voice B, voice C,
        # and a 1.5 s scrap that sounds a little like B and like nobody else
        voice = {0: a, 1: b, 2: 0.8 * a + 0.6 * d, 3: 0.35 * b + 0.94 * d - 0.1 * a, 4: c}
        turns = [(0, 10, 0), (10, 20, 1), (20, 30, 2), (30, 31.5, 3), (31.5, 41.5, 4)]

        def fp(s, _e):
            v = voice[[t[2] for t in turns if t[0] == s][0]]
            return v / np.linalg.norm(v)

        auto, folded = e.refine_speakers(turns, fp, 0)
        labels = [k for _s, _e, k in auto]
        if not (labels[0] == labels[2] and labels[1] == labels[3] and len(set(labels)) == 3 and folded == 1):
            raise AssertionError(f"auto: {labels} folded={folded}")
        fixed, _ = e.refine_speakers(turns, fp, 2)
        if len({k for _s, _e, k in fixed}) != 2:
            raise AssertionError("a fixed count of 2 was not honoured")
        # in a short clip a few seconds can be a whole person: 4 s against 5 s, quite unalike
        short = [(0, 5, 0), (5, 9, 1)]
        if len({k for _s, _e, k in e.refine_speakers(short, lambda s, _e: a if s == 0 else b, 0)[0]}) != 2:
            raise AssertionError("a second speaker in a short clip was folded away as a scrap")
        frames = e.speaker_frames([(0.0, 30.0, 0), (10.0, 15.0, 1), (20.0, 20.4, 1)], 30 * e.RATE)
        if not (frames[1200] == 1 and frames[2020] == 0 and frames[500] == 0):
            raise AssertionError("overlapping turns were laid down wrongly")
        return "same voice merged, a 1.5 s scrap rehomed, fixed count honoured, short clips keep both speakers"
    rep.check("speaker merging", refine_ok)

    transcript = {
        "file": "/x/Team meeting <1>.m4a", "duration": 3725.0, "language": "en", "model": "parakeet-v2",
        "speakers": {"S1": "Dan", "S2": "Speaker 2"},
        "segments": [
            {"start": 1.0, "end": 4.2, "speaker": "S1", "text": "Good morning & welcome.", "confidence": 0.93},
            {"start": 4.4, "end": 9.0, "speaker": "S1", "text": "Shall we start with the budget?", "confidence": 0.9},
            {"start": 9.6, "end": 31.0, "speaker": "S2", "text": "Yes. " + "We are under budget this quarter. " * 6,
             "confidence": None},
            {"start": 3700.0, "end": 3703.0, "speaker": "S1", "text": "Diolch yn fawr – £4,000 saved.",
             "confidence": 0.8},
        ],
        "removed": [], "notes": [],
    }

    def export_ok():
        from . import export as x
        folder = os.path.join(tmp, "out")
        written = x.write_all(transcript, folder, list(x.FORMATS))
        if set(written) != set(x.FORMATS):
            raise AssertionError("not every format was written")
        txt = open(written["txt"], encoding="utf-8").read()
        if "Dan:" not in txt or "[1:01:40]" not in txt or "£4,000" not in txt:
            raise AssertionError("text export lost a speaker, a timestamp or a character")
        srt = open(written["srt"], encoding="utf-8").read()
        if not re.search(r"^1\n00:00:01,000 --> 00:00:04,200\nDan: Good morning & welcome\.$", srt, re.M):
            raise AssertionError("subtitle timing or numbering is wrong")
        if max(len(ln) for ln in srt.splitlines()) > 60:
            raise AssertionError("a long segment was not split into subtitle-sized pieces")
        if not open(written["vtt"], encoding="utf-8").read().startswith("WEBVTT"):
            raise AssertionError("vtt header missing")
        with zipfile.ZipFile(written["docx"]) as z:
            if z.testzip() is not None:
                raise AssertionError("docx is a damaged zip")
            import xml.dom.minidom
            doc = xml.dom.minidom.parseString(z.read("word/document.xml"))
            text = "".join(n.firstChild.data for n in doc.getElementsByTagName("w:t") if n.firstChild)
            if "Good morning & welcome." not in text or "Dan" not in text:
                raise AssertionError("docx text is wrong")
        back = json.loads(open(written["json"], encoding="utf-8").read())
        if back["segments"][3]["text"] != transcript["segments"][3]["text"]:
            raise AssertionError("json did not round-trip")
        rows = open(written["csv"], encoding="utf-8-sig").read().splitlines()
        if len(rows) != 5 or not rows[0].startswith("start,end"):
            raise AssertionError("csv rows are wrong")
        again = x.write_all(transcript, folder, ["txt"])
        if not again["txt"].endswith("(2).txt"):
            raise AssertionError("an existing file would have been overwritten")
        if os.path.basename(written["txt"]) != "Team meeting _1_.txt":
            raise AssertionError(f"unsafe file name: {os.path.basename(written['txt'])}")
        return "txt, docx, md, srt, vtt, csv, json written and read back"
    rep.check("every export format", export_ok)

    # --- audio -------------------------------------------------------------------------
    rep.say()
    rep.say("Audio")

    def decode_ok():
        import numpy as np
        from . import audio
        t = np.arange(audio.RATE * 3) / audio.RATE
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        wav = os.path.join(tmp, "tone.wav")
        audio.write_wav(wav, tone)
        made = ["wav"]
        ff = audio.find_ffmpeg()
        for ext, args in (("mp3", ["-c:a", "libmp3lame"]), ("m4a", ["-c:a", "aac"]), ("flac", ["-c:a", "flac"]),
                          ("ogg", ["-c:a", "libvorbis"]), ("mkv", ["-c:a", "pcm_s16le"]),
                          ("mp4", ["-f", "lavfi", "-i", "color=c=black:s=64x64:d=3", "-shortest", "-c:a", "aac"])):
            out = os.path.join(tmp, f"tone.{ext}")
            r = subprocess.run([ff, "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", wav] + args + [out],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=120, **audio._no_window())
            if r.returncode == 0 and os.path.getsize(out) > 0:
                made.append(ext)
        for ext in made:
            src = os.path.join(tmp, f"tone.{ext}")
            raw = os.path.join(tmp, f"tone-{ext}.f32")
            info = audio.probe(src)
            data = audio.decode(src, raw, info["duration"])
            secs = len(data) / audio.RATE
            del data
            if not 2.8 <= secs <= 3.3:
                raise AssertionError(f"{ext} decoded to {secs:.2f}s, expected 3s")
        if len(made) < 4:
            raise AssertionError(f"ffmpeg could only write {made}")
        bad = os.path.join(tmp, "not-audio.mp3")
        with open(bad, "wb") as fh:
            fh.write(b"this is not audio" * 50)
        try:
            audio.decode(bad, os.path.join(tmp, "bad.f32"))
        except audio.AudioError:
            pass
        else:
            raise AssertionError("a file that is not audio was accepted")
        return "decoded " + ", ".join(made) + "; a non-audio file is refused cleanly"
    rep.check("any format in", decode_ok)

    # --- the real thing ---------------------------------------------------------------
    rep.say()
    rep.say("Models")
    from . import models
    for mid, spec in models.CATALOGUE.items():
        rep.say(f"  {'installed    ' if models.installed(mid) else 'not installed'} {mid:14} {spec['label']}")
    asr = next((m for m in [models.DEFAULT_ASR] + list(models.ASR) if models.installed(m)), None)
    support = all(models.installed(m) for m in models.SUPPORT)

    rep.say()
    rep.say("End to end")

    def full_ok():
        if asr is None:
            raise Skip("no speech model is downloaded yet")
        if not support:
            raise Skip("helper models are not downloaded yet")
        from . import engine
        stages = set()
        result = engine.transcribe(_sample(), engine.Options(model=asr, denoise="on"),
                                   lambda ev: stages.add(ev.get("stage")))
        text = " ".join(s["text"] for s in result["segments"]).lower()
        hit = [w for w in SAMPLE_WORDS if w in text]
        if len(hit) < 0.75 * len(SAMPLE_WORDS):
            raise AssertionError(f"only {len(hit)}/{len(SAMPLE_WORDS)} expected words found: {text[:140]!r}")
        n = len(result["speakers"])
        if n != 2:
            raise AssertionError(f"two people speak on the sample; {n} were found")
        order = [s["speaker"] for s in result["segments"]]
        if order[0] == order[-1]:
            raise AssertionError("the first and last lines are by different people but were given one label")
        if not {"decode", "speech", "clean", "speakers", "text"} <= stages:
            raise AssertionError(f"stages reported: {sorted(s for s in stages if s)}")
        return (f"{models.ASR[asr]['label']}: {len(hit)}/{len(SAMPLE_WORDS)} key words, 2 speakers, "
                f"{result['stats']['seconds_taken']}s")
    rep.check("sample recording -> speaker-labelled transcript", full_ok)

    def silence_ok():
        if asr is None or not models.installed("vad"):
            raise Skip("models are not downloaded yet")
        import numpy as np
        from . import audio, engine
        rng = np.random.default_rng(5)
        n = audio.RATE * 20
        noise = rng.standard_normal(n).astype(np.float32) * 0.03
        t = np.arange(n) / audio.RATE
        noise += (0.05 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
        for k in range(6):  # clicks and beeps
            s = int(rng.uniform(1, 18) * audio.RATE)
            noise[s:s + 3200] += (0.2 * np.sin(2 * np.pi * rng.uniform(400, 1200) * t[:3200])).astype(np.float32)
        wav = os.path.join(tmp, "noise.wav")
        audio.write_wav(wav, noise)
        result = engine.transcribe(wav, engine.Options(model=asr, speakers=False, denoise="off" if not
                                                       models.installed("denoise") else "auto"))
        if result["segments"]:
            raise AssertionError(f"text was invented from noise: {result['segments'][0]['text']!r}")
        return "20 s of hiss, hum and beeps gave an empty transcript"
    rep.check("noise with no speech -> no text", silence_ok)

    def engine_ok():
        if quick:
            raise Skip("--quick")
        if asr is None or not support:
            raise Skip("models are not downloaded yet")
        from . import proc
        jobdir = os.path.join(tmp, "job")
        os.makedirs(jobdir)
        job = {"file": _sample(), "options": {"model": asr, "speakers": True, "parallel": "on"},
               "events": os.path.join(jobdir, "events.jsonl"), "result": os.path.join(jobdir, "result.json"),
               "cancel": os.path.join(jobdir, "cancel"), "workdir": jobdir}
        with open(os.path.join(jobdir, "job.json"), "w", encoding="utf-8") as fh:
            json.dump(job, fh)
        with open(os.path.join(jobdir, "engine.log"), "wb") as log:
            engine_proc = proc.start_engine(os.path.join(jobdir, "job.json"), log)
            try:
                code = engine_proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                proc.kill_tree(engine_proc)
                raise AssertionError("the engine did not finish in 15 minutes")
        with open(job["events"], "rb") as fh:
            events = [json.loads(ln.decode("utf-8")) for ln in fh.read().splitlines() if ln.strip()]
        if code != 0 or not os.path.isfile(job["result"]):
            err = next((e.get("message") for e in events if e.get("type") == "error"), None)
            tail = open(os.path.join(jobdir, "engine.log"), encoding="utf-8", errors="replace").read()[-300:]
            raise AssertionError(f"engine exited {code}: {err or tail}")
        if not any(e.get("type") == "progress" for e in events) or events[-1].get("type") != "done":
            raise AssertionError("the engine did not report progress and completion")
        result = json.load(open(job["result"], encoding="utf-8"))
        if not result["segments"]:
            raise AssertionError("the engine returned an empty transcript")
        if not result["stats"].get("parallel"):
            raise AssertionError("the engine ignored the request to run both stages at once")
        return f"separate engine process ran both stages at once and reported {len(events)} events"
    rep.check("the window's engine process", engine_ok)

    def window_ok():
        try:
            import tkinter as tk
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"tkinter is missing: {exc}")
        from . import ui
        saved = paths.settings_path
        paths.settings_path = lambda: os.path.join(tmp, "settings.json")  # never touch the real settings
        try:
            try:
                root = ui.make_root()
            except tk.TclError as exc:
                raise Skip(f"no display ({str(exc)[:50]})")
            try:
                root.withdraw()
                app = ui.App(root, [_sample()])
                root.update()
                app.toggle_theme()
                root.update()
                app.toggle_theme()
                import copy
                item = next(iter(app.items.values()))
                item.result, item.state = copy.deepcopy(transcript), "done"
                app._show(item)
                root.update()
                shown = app.text.get("1.0", "end")
                if "Dan" not in shown or "budget" not in shown:
                    raise AssertionError("the transcript pane did not show the transcript")
                # hand the first paragraph to the other speaker, as the right-click menu does
                app._reassign(item, item.result, list(app._para_spans[0]["segments"]), "S2")
                root.update()
                if item.result["segments"][0]["speaker"] != "S2":
                    raise AssertionError("changing a paragraph's speaker did nothing")
                # a saved transcript opens again
                saved_json = os.path.join(tmp, "reopen.json")
                with open(saved_json, "w", encoding="utf-8") as fh:
                    json.dump(transcript, fh)
                before = len(app.items)
                app.add_paths([saved_json])
                root.update()
                if len(app.items) != before + 1 or "Dan" not in app.text.get("1.0", "end"):
                    raise AssertionError("a saved transcript did not reopen")
                tkv = root.tk.call("info", "patchlevel")
                dnd = "with drag and drop" if ui._HAVE_DND else "no drag and drop (tkinterdnd2 unavailable)"
                app._closing = True
            finally:
                root.destroy()
        finally:
            paths.settings_path = saved
        return f"Tk {tkv}, {dnd}"
    rep.check("the window builds, in both themes", window_ok)

    shutil.rmtree(tmp, ignore_errors=True)
    rep.say()
    rep.say(f"{rep.passed} passed, {rep.failed} failed, {rep.skipped} skipped")
    rep.say("PROBLEMS FOUND" if rep.failed else "ALL GOOD")
    for folder in (paths.app_dir(), paths.data_dir()):
        try:
            with io.open(os.path.join(folder, "verbatim-selftest.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(rep.lines) + "\n")
            break
        except OSError:
            continue
    return 1 if rep.failed else 0
