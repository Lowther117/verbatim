"""Measuring this computer, instead of guessing, to choose how to run.

Two things can make transcription faster and neither can be decided from a
specification sheet:

- running "who spoke when" and the transcription side by side, which pays
  only when there are processor cores to spare;
- handing the models to the graphics processor (Core ML on a Mac, CUDA or
  DirectML on Windows), which pays only if this build includes that back end,
  the hardware is there, and it really is quicker for these models - it is
  often not.

So each candidate is run for real on two minutes of the bundled sample and timed.
A candidate that fails, falls back to the processor, or is not clearly faster
is not chosen. Each run is a separate engine process, so a back end that
crashes outright only loses its own run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Callable, List, Optional

from . import audio, engine, models, paths, proc


def sample_path() -> str:
    for base in (os.path.dirname(os.path.abspath(__file__)), os.path.join(paths.resource_dir(), "verbatimlib")):
        p = os.path.join(base, "data", "selftest.ogg")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("the bundled sample recording (data/selftest.ogg) is missing from this build")


def _make_audio(folder: str, repeats: int = 8) -> str:
    import numpy as np
    raw = os.path.join(folder, "sample.f32")
    one = np.array(audio.decode(sample_path(), raw, None, highpass=False))
    wav = os.path.join(folder, "speedtest.wav")
    audio.write_wav(wav, np.tile(one, repeats))
    return wav


def candidates() -> List[dict]:
    """The ways of running worth timing here, the plain one first."""
    out = [{"name": "Processor, one stage at a time", "provider": "cpu", "parallel": "off"}]
    if (os.cpu_count() or 2) >= 4:
        out.append({"name": "Processor, both stages at once", "provider": "cpu", "parallel": "on"})
    for gpu in engine.gpu_candidates():
        out.append({"name": {"coreml": "Graphics processor (Core ML)", "cuda": "Graphics processor (NVIDIA CUDA)",
                             "directml": "Graphics processor (DirectML)"}[gpu], "provider": gpu, "parallel": None})
    return out


def _run_one(wav: str, folder: str, options: dict, timeout: float) -> dict:
    jobdir = tempfile.mkdtemp(prefix="run-", dir=folder)
    job = {"file": wav, "options": options, "events": os.path.join(jobdir, "events.jsonl"),
           "result": os.path.join(jobdir, "result.json"), "workdir": jobdir}
    job_path = os.path.join(jobdir, "job.json")
    with open(job_path, "w", encoding="utf-8") as fh:
        json.dump(job, fh)
    started = time.time()
    with open(os.path.join(jobdir, "engine.log"), "wb") as log:
        p = proc.start_engine(job_path, log)
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill_tree(p)
            return {"ok": False, "why": "took too long"}
    if code != 0 or not os.path.isfile(job["result"]):
        why = "failed to run"
        try:
            with open(job["events"], "rb") as fh:
                for line in fh.read().splitlines():
                    ev = json.loads(line.decode("utf-8"))
                    if ev.get("type") == "error":
                        why = ev.get("message") or why
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "why": why}
    with open(job["result"], encoding="utf-8") as fh:
        result = json.load(fh)
    stats = result.get("stats", {})
    timings = stats.get("timings", {})
    return {"ok": True, "seconds": round(float(timings.get("core", time.time() - started)), 2),
            "provider_used": stats.get("provider", "cpu"), "words": stats.get("words", 0),
            "speakers": len(result.get("speakers") or {})}


def run(model: Optional[str] = None, report: Optional[Callable[[str], None]] = None) -> dict:
    """Time every candidate and return {"results": [...], "choice": {"provider", "parallel"}, "summary"}."""
    report = report or (lambda _text: None)
    model = model if model in models.ASR and models.installed(model) else next(
        (m for m in models.ASR if models.installed(m)), None)
    lacking = models.missing(["vad", "segmentation", "embedding"])
    if model is None or lacking:
        raise models.ModelError("The speed test needs the speech model and the speaker models downloaded first.")
    folder = tempfile.mkdtemp(prefix="verbatim-speedtest-")
    try:
        wav = _make_audio(folder)
        base = {"model": model, "speakers": True, "denoise": "off", "filtering": "balanced"}
        results: List[dict] = []
        best_parallel = "off"
        for cand in candidates():
            parallel = cand["parallel"] or best_parallel
            report(f"Timing: {cand['name']}…")
            r = _run_one(wav, folder, dict(base, provider=cand["provider"], parallel=parallel), timeout=900)
            r.update(name=cand["name"], provider=cand["provider"], parallel=parallel)
            results.append(r)
            cpu = [x for x in results if x["ok"] and x["provider"] == "cpu"]
            if cpu:
                seq = next((x for x in cpu if x["parallel"] == "off"), None)
                par = next((x for x in cpu if x["parallel"] == "on"), None)
                # side by side has to be clearly quicker to be worth the extra memory
                best_parallel = "on" if (seq and par and par["seconds"] < 0.88 * seq["seconds"]) else "off"
        cpu_ok = [x for x in results if x["ok"] and x["provider"] == "cpu"]
        if not cpu_ok:
            raise engine.EngineError("The speed test could not complete even the plain run: "
                                     + (results[0].get("why") or "unknown reason"))
        cpu_best = next(x for x in cpu_ok if x["parallel"] == best_parallel)
        choice = {"provider": "cpu", "parallel": best_parallel}
        winner = cpu_best
        for x in results:
            # a back end that silently fell back to the processor shows no gain and is never chosen
            if x["ok"] and x["provider"] != "cpu" and x["seconds"] < 0.85 * winner["seconds"]:
                winner = x
                choice = {"provider": x["provider"], "parallel": x["parallel"]}
        lines = []
        for x in results:
            if x["ok"]:
                mark = "  <- chosen" if (x["provider"], x["parallel"]) == (choice["provider"], choice["parallel"]) else ""
                lines.append(f"{x['name']}: {x['seconds']:.1f} s{mark}")
            else:
                lines.append(f"{x['name']}: not usable here ({x.get('why', 'failed')})")
        gpu_tried = [x for x in results if x["provider"] != "cpu"]
        if choice["provider"] == "cpu" and gpu_tried:
            lines.append("The graphics processor was not quicker than the processor for these models on this "
                         "computer (or this build has no graphics back end), so the processor is used.")
        return {"results": results, "choice": choice, "summary": "\n".join(lines), "model": model,
                "when": time.strftime("%Y-%m-%d %H:%M")}
    finally:
        shutil.rmtree(folder, ignore_errors=True)
