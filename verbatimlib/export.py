"""Writing a finished transcript out: text, Word, Markdown, subtitles, CSV and JSON.

A transcript is a plain dict (it is also exactly what the JSON export holds):

    {"file": ..., "duration": seconds, "language": "en", "model": "...",
     "speakers": {"S1": "Speaker 1", ...},          # id -> display name
     "segments": [{"start", "end", "speaker", "text", "confidence"}, ...],
     "removed":  [{"start", "end", "text", "reason"}, ...],
     "notes": [...]}

Nothing here needs a third-party package: the .docx is written directly as the
small zip of XML that it is.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import zipfile
from typing import Dict, List, Optional
from xml.sax.saxutils import escape

FORMATS = {
    "txt": "Plain text (.txt)",
    "docx": "Word document (.docx)",
    "md": "Markdown (.md)",
    "srt": "Subtitles (.srt)",
    "vtt": "Web subtitles (.vtt)",
    "csv": "Spreadsheet (.csv)",
    "json": "Everything, machine-readable (.json)",
}


def clock(seconds: float, always_hours: bool = False) -> str:
    seconds = max(0, int(seconds + 0.5))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h or always_hours:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _stamp(seconds: float, sep: str) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def speaker_name(t: dict, sid: Optional[str]) -> str:
    if not sid:
        return ""
    return (t.get("speakers") or {}).get(sid) or sid


def paragraphs(t: dict, gap: float = 2.5, max_chars: int = 1200) -> List[dict]:
    """Consecutive segments from one speaker, joined into readable paragraphs."""
    out: List[dict] = []
    for seg in t.get("segments", []):
        text = seg["text"].strip()
        if not text:
            continue
        last = out[-1] if out else None
        if (last is not None and last["speaker"] == seg.get("speaker")
                and seg["start"] - last["end"] <= gap and len(last["text"]) + len(text) < max_chars):
            last["text"] += " " + text
            last["end"] = seg["end"]
        else:
            out.append({"start": seg["start"], "end": seg["end"], "speaker": seg.get("speaker"), "text": text})
    return out


def _header_lines(t: dict) -> List[str]:
    lines = [os.path.basename(t.get("file") or "Transcript")]
    bits = []
    if t.get("duration"):
        bits.append(f"Length {clock(t['duration'])}")
    names = [speaker_name(t, s) for s in (t.get("speakers") or {})]
    if names:
        bits.append(f"{len(names)} speaker{'s' if len(names) != 1 else ''}: " + ", ".join(names))
    if t.get("language"):
        bits.append(f"Language {t['language']}")
    if bits:
        lines.append(" | ".join(bits))
    return lines


def to_txt(t: dict, timestamps: bool = True) -> str:
    hours = (t.get("duration") or 0) >= 3600
    out = _header_lines(t) + [""]
    for p in paragraphs(t):
        who = speaker_name(t, p["speaker"])
        stamp = f"[{clock(p['start'], hours)}] " if timestamps else ""
        if who:
            out.append(f"{stamp}{who}:")
            out.append(p["text"])
        else:
            out.append(f"{stamp}{p['text']}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def to_md(t: dict, timestamps: bool = True) -> str:
    hours = (t.get("duration") or 0) >= 3600
    head = _header_lines(t)
    out = [f"# {head[0]}", ""]
    if len(head) > 1:
        out += [f"*{head[1]}*", ""]
    for p in paragraphs(t):
        who = speaker_name(t, p["speaker"])
        stamp = f" `{clock(p['start'], hours)}`" if timestamps else ""
        if who:
            out.append(f"**{who}**{stamp}  ")
        elif timestamps:
            out.append(f"`{clock(p['start'], hours)}`  ")
        out.append(p["text"])
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def cues(t: dict, max_chars: int = 84, max_seconds: float = 7.0) -> List[dict]:
    """Subtitle-sized pieces: long segments are split at word boundaries, with
    the time shared out in proportion to the text."""
    out: List[dict] = []
    for seg in t.get("segments", []):
        text = " ".join(seg["text"].split())
        if not text:
            continue
        start, end = seg["start"], max(seg["end"], seg["start"] + 0.3)
        pieces = max(1, -(-len(text) // max_chars), int(-(-(end - start) // max_seconds)))
        if pieces == 1:
            out.append({"start": start, "end": end, "speaker": seg.get("speaker"), "text": text})
            continue
        words = text.split(" ")
        target = len(text) / pieces
        chunks, cur = [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > target * 1.15 and len(chunks) < pieces - 1:
                chunks.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}" if cur else w
        if cur:
            chunks.append(cur)
        total = sum(len(c) for c in chunks) or 1
        pos = start
        for c in chunks:
            span = (end - start) * len(c) / total
            out.append({"start": pos, "end": pos + span, "speaker": seg.get("speaker"), "text": c})
            pos += span
    return out


def _wrap(text: str, width: int = 42) -> str:
    if len(text) <= width:
        return text
    mid = len(text) // 2
    left = text.rfind(" ", 0, mid + 1)
    right = text.find(" ", mid)
    cut = left if (left != -1 and (right == -1 or mid - left <= right - mid)) else right
    if cut <= 0:
        return text
    return text[:cut] + "\n" + text[cut + 1:]


def to_srt(t: dict) -> str:
    out = []
    for i, c in enumerate(cues(t), 1):
        who = speaker_name(t, c["speaker"])
        text = _wrap(c["text"])
        if who:
            text = f"{who}: {text}"
        out.append(f"{i}\n{_stamp(c['start'], ',')} --> {_stamp(c['end'], ',')}\n{text}\n")
    return "\n".join(out)


def to_vtt(t: dict) -> str:
    out = ["WEBVTT", ""]
    for c in cues(t):
        who = speaker_name(t, c["speaker"])
        text = escape(_wrap(c["text"]))
        if who:
            text = f"<v {escape(who)}>{text}"
        out.append(f"{_stamp(c['start'], '.')} --> {_stamp(c['end'], '.')}\n{text}\n")
    return "\n".join(out)


def to_csv(t: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(["start", "end", "start_seconds", "end_seconds", "speaker", "text", "confidence"])
    for s in t.get("segments", []):
        conf = s.get("confidence")
        w.writerow([clock(s["start"], True), clock(s["end"], True), f"{s['start']:.2f}", f"{s['end']:.2f}",
                    speaker_name(t, s.get("speaker")), s["text"], "" if conf is None else f"{conf:.2f}"])
    return buf.getvalue()


def to_json(t: dict) -> str:
    return json.dumps(t, indent=1, ensure_ascii=False)


# --- Word ---------------------------------------------------------------------

_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_BAD_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def _run(text: str, bold: bool = False, size: int = 22, colour: Optional[str] = None, italic: bool = False) -> str:
    props = f'<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>'
    if bold:
        props += "<w:b/>"
    if italic:
        props += "<w:i/>"
    if colour:
        props += f'<w:color w:val="{colour}"/>'
    props += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    text = escape(_BAD_XML.sub("", text))
    return f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{text}</w:t></w:r>'


def _para(runs: str, after: int = 120, keep_next: bool = False) -> str:
    keep = "<w:keepNext/>" if keep_next else ""
    return f'<w:p><w:pPr>{keep}<w:spacing w:after="{after}"/></w:pPr>{runs}</w:p>'


def to_docx_bytes(t: dict, timestamps: bool = True) -> bytes:
    hours = (t.get("duration") or 0) >= 3600
    head = _header_lines(t)
    body = [_para(_run(head[0], bold=True, size=32), after=60)]
    if len(head) > 1:
        body.append(_para(_run(head[1], size=18, colour="666666"), after=280))
    for p in paragraphs(t):
        who = speaker_name(t, p["speaker"])
        stamp = clock(p["start"], hours) if timestamps else ""
        if who:
            runs = _run(who, bold=True)
            if stamp:
                runs += _run("   " + stamp, size=18, colour="808080")
            body.append(_para(runs, after=0, keep_next=True))
            body.append(_para(_run(p["text"]), after=200))
        else:
            runs = (_run(stamp + "   ", size=18, colour="808080") if stamp else "") + _run(p["text"])
            body.append(_para(runs, after=160))
    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {_W}><w:body>'
                + "".join(body)
                + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" '
                  'w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
                  '</w:body></w:document>')
    types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="xml" ContentType="application/xml"/>'
             '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.'
             'wordprocessingml.document.main+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'officeDocument" Target="word/document.xml"/></Relationships>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
    return buf.getvalue()


# --- files --------------------------------------------------------------------

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = _UNSAFE.sub("_", stem).strip(" .")
    return stem[:120] or "transcript"


def render(t: dict, fmt: str, timestamps: bool = True) -> bytes:
    if fmt == "txt":
        return to_txt(t, timestamps).encode("utf-8")
    if fmt == "md":
        return to_md(t, timestamps).encode("utf-8")
    if fmt == "srt":
        return to_srt(t).encode("utf-8")
    if fmt == "vtt":
        return to_vtt(t).encode("utf-8")
    if fmt == "csv":
        return ("﻿" + to_csv(t)).encode("utf-8")  # BOM so Excel reads it as UTF-8
    if fmt == "json":
        return to_json(t).encode("utf-8")
    if fmt == "docx":
        return to_docx_bytes(t, timestamps)
    raise ValueError(f"unknown format {fmt}")


def write_all(t: dict, folder: str, formats: List[str], timestamps: bool = True,
              overwrite: bool = False, stem: Optional[str] = None) -> Dict[str, str]:
    """Write every requested format into `folder`; returns {format: path}.

    Unless `overwrite` is set an existing file is never replaced: the new one
    becomes "name (2).txt", and every format of one transcript shares the
    same number so they stay together.
    """
    os.makedirs(folder, exist_ok=True)
    stem = stem or safe_stem(t.get("file") or "transcript")
    base = stem
    if not overwrite:
        n = 1
        while any(os.path.exists(os.path.join(folder, f"{base}.{f}")) for f in formats):
            n += 1
            base = f"{stem} ({n})"
    written = {}
    for fmt in formats:
        path = os.path.join(folder, f"{base}.{fmt}")
        with open(path, "wb") as fh:
            fh.write(render(t, fmt, timestamps))
        written[fmt] = path
    return written
