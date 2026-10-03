"""Command line: open the window, transcribe from a terminal, fetch models, self-test.

  verbatim                          open the window
  verbatim FILE ...                 open the window with files queued
  verbatim transcribe FILE ...      transcribe without the window
  verbatim models                   list the models and whether each is installed
  verbatim fetch ID ...             download models (the build files use this)
  verbatim speedtest                time this computer and choose the fastest way to run
  verbatim selftest                 check the whole pipeline, write verbatim-selftest.txt
  verbatim engine --job FILE        (internal) one transcription job for the window
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import List

from . import paths


def _engine(argv: List[str]) -> int:
    """One job for the window. Progress goes to an events file, one JSON object
    per line, because a windowed build has no usable standard output. The
    window asks for a stop by creating the job's "cancel" file."""
    ap = argparse.ArgumentParser(prog="verbatim engine")
    ap.add_argument("--job", required=True)
    args, _ = ap.parse_known_args(argv)
    with open(args.job, encoding="utf-8") as fh:
        job = json.load(fh)
    events_path = job["events"]
    cancel_path = job.get("cancel") or os.path.join(os.path.dirname(os.path.abspath(args.job)), "cancel")

    def emit(event: dict) -> None:
        try:
            with open(events_path, "ab") as fh:
                fh.write((json.dumps(event) + "\n").encode("utf-8"))
        except OSError:
            pass

    try:
        from . import engine, proc
        proc.be_a_good_neighbour()
        options = engine.Options.from_dict(job.get("options") or {})
        result = engine.transcribe(job["file"], options, emit, cancelled=lambda: os.path.exists(cancel_path),
                                   workdir=job.get("workdir"))
        tmp = job["result"] + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        os.replace(tmp, job["result"])
        emit({"type": "done"})
        return 0
    except BaseException as exc:  # noqa: BLE001 - everything must reach the window
        friendly = type(exc).__name__ in ("AudioError", "EngineError", "ModelError", "Cancelled")
        emit({"type": "error", "message": str(exc) if friendly else f"{type(exc).__name__}: {exc}",
              "trace": "" if friendly else traceback.format_exc()})
        return 1


def _transcribe(argv: List[str]) -> int:
    from . import engine, export, filters, models
    ap = argparse.ArgumentParser(prog="verbatim transcribe", description="Transcribe audio or video files.")
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--output", help="folder to save into (default: the saved default, else Downloads/Verbatim)")
    ap.add_argument("--model", default=None, choices=list(models.ASR), help="speech model")
    ap.add_argument("--no-speakers", action="store_true", help="do not label speakers")
    ap.add_argument("--speakers", type=int, default=0, metavar="N", help="number of speakers if known (default: work it out)")
    ap.add_argument("--denoise", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--filter", default="balanced", choices=list(filters.LEVELS), dest="filtering")
    ap.add_argument("--formats", default="txt", help="comma-separated: " + ",".join(export.FORMATS))
    ap.add_argument("--no-timestamps", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="replace existing output files instead of numbering")
    ap.add_argument("--download", action="store_true", help="download any missing model without asking")
    ap.add_argument("--provider", default=None, choices=list(engine.PROVIDERS),
                    help="where the models run (default: what the speed test chose, else cpu)")
    args = ap.parse_args(argv)

    settings = paths.load_settings()
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in formats if f not in export.FORMATS]
    if bad:
        print(f"Unknown format: {', '.join(bad)}", file=sys.stderr)
        return 2
    opts = engine.Options.from_dict({
        "model": args.model or settings.get("model") or models.DEFAULT_ASR, "speakers": not args.no_speakers,
        "num_speakers": args.speakers, "denoise": args.denoise, "filtering": args.filtering,
        "provider": args.provider or settings.get("provider") or "cpu",
        "parallel": settings.get("parallel") or "auto"})
    need = models.missing(models.needed_for(opts.model, opts.speakers, opts.denoise != "off"))
    if need:
        names = ", ".join(models.CATALOGUE[m]["label"] for m in need)
        if not args.download:
            print(f"Missing models: {names}\nRun again with --download, or: verbatim fetch {' '.join(need)}",
                  file=sys.stderr)
            return 1
        for m in need:
            _fetch_one(m, None)
    folder = args.output or paths.default_save_dir(settings)
    failures = 0
    for path in args.files:
        name = os.path.basename(path)

        def emit(e, name=name):
            if e.get("type") == "progress":
                print(f"\r{name}: {e['label']} {e['fraction'] * 100:3.0f}%   ", end="", file=sys.stderr, flush=True)

        try:
            result = engine.transcribe(path, opts, emit)
        except Exception as exc:  # noqa: BLE001
            print(f"\n{name}: FAILED - {exc}", file=sys.stderr)
            failures += 1
            continue
        written = export.write_all(result, folder, formats, not args.no_timestamps, args.overwrite)
        st = result["stats"]
        print(f"\r{name}: done in {st.get('seconds_taken', 0):.0f}s - {st.get('words', 0)} words, "
              f"{len(result['speakers'])} speaker(s)                              ", file=sys.stderr)
        for note in result["notes"]:
            print(f"  note: {note}", file=sys.stderr)
        for p in written.values():
            print(p)
    return 1 if failures else 0


def _speedtest(_argv: List[str]) -> int:
    from . import speedtest
    settings = paths.load_settings()
    print("Timing this computer on two minutes of the sample recording. This takes a few minutes.", flush=True)
    try:
        out = speedtest.run(settings.get("model"), lambda text: print("  " + text, flush=True))
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(out["summary"])
    settings.update(out["choice"])
    settings["speedtest"] = {"when": out["when"], "summary": out["summary"]}
    paths.save_settings(settings)
    print("Saved. The window and the command line will use this from now on.")
    return 0


def _fetch_one(model_id: str, into) -> None:
    from . import models
    spec = models.CATALOGUE[model_id]
    state = {"pct": -1}

    def progress(stage, done, total):
        if stage == "download" and total:
            pct = int(done * 100 / total)
            if pct != state["pct"] and pct % 5 == 0:
                state["pct"] = pct
                print(f"  {spec['label']}: {pct}% of {total // (1 << 20)} MB", flush=True)
        elif stage == "unpack":
            print(f"  {spec['label']}: unpacked {done}/{total}", flush=True)

    print(f"Fetching {spec['label']} (~{spec['mb']} MB)...", flush=True)
    models.install(model_id, progress, into=into)


def _fetch(argv: List[str]) -> int:
    from . import models
    ap = argparse.ArgumentParser(prog="verbatim fetch", description="Download models.")
    ap.add_argument("ids", nargs="*", help="model ids (see: verbatim models). 'support' = the four small helper models")
    ap.add_argument("--into", help="folder to download into instead of the models folder")
    ap.add_argument("--stage", help="with --into: copy exactly the named models into this fresh folder afterwards "
                                    "(what a build bakes in, so that nothing else in --into comes along)")
    args = ap.parse_args(argv)
    ids: List[str] = []
    for i in args.ids:
        if i == "support":
            ids += list(models.SUPPORT)
        elif i in models.CATALOGUE:
            ids.append(i)
        else:
            print(f"Unknown model id: {i}", file=sys.stderr)
            return 2
    if not ids:
        ap.print_help()
        return 2
    try:
        for i in ids:
            if args.into:
                target = os.path.join(args.into, i)
                files = models.CATALOGUE[i]["files"].values()
                if all(os.path.isfile(os.path.join(target, f)) for f in files):
                    print(f"{models.CATALOGUE[i]['label']}: already there.")
                    continue
            elif models.installed(i):
                print(f"{models.CATALOGUE[i]['label']}: already installed.")
                continue
            _fetch_one(i, args.into)
    except models.ModelError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    if args.stage and args.into:
        import shutil
        shutil.rmtree(args.stage, ignore_errors=True)
        for i in ids:
            dest = os.path.join(args.stage, i)
            os.makedirs(dest, exist_ok=True)
            for name in models.CATALOGUE[i]["files"].values():
                shutil.copy2(os.path.join(args.into, i, name), os.path.join(dest, name))
        print(f"Staged {len(ids)} models for the build.")
    print("Models ready.")
    return 0


def _models(_argv: List[str]) -> int:
    from . import models
    print(f"Models folder: {paths.models_dirs()[0]}\n")
    for group, table in (("Speech-to-text", models.ASR), ("Helpers", models.SUPPORT)):
        print(group)
        for mid, spec in table.items():
            mark = "installed" if models.installed(mid) else "-"
            print(f"  {mid:15} {spec['mb']:5d} MB  {mark:10} {spec['label']}")
        print()
    return 0


def main(argv: List[str]) -> int:
    if argv and argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if argv and argv[0] in ("-V", "--version"):
        print(f"{paths.APP} {paths.VERSION}")
        return 0
    if argv and argv[0] == "engine":
        return _engine(argv[1:])
    if argv and argv[0] == "transcribe":
        return _transcribe(argv[1:])
    if argv and argv[0] == "fetch":
        return _fetch(argv[1:])
    if argv and argv[0] == "models":
        return _models(argv[1:])
    if argv and argv[0] == "speedtest":
        return _speedtest(argv[1:])
    if argv and argv[0] == "selftest":
        from .selftest import run
        return run(argv[1:])
    from .ui import launch
    return launch([a for a in argv if not a.startswith("-")])
