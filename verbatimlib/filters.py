"""Keeping invented text out of the transcript.

Speech models - Whisper above all - were trained on subtitled video, and when
they are handed silence, music, hiss or a cough they tend to answer with what
subtitles say at such moments: "Thanks for watching", a credit line, a lone
"you", or one phrase looped forty times. None of it was said.

The speech detector stops most of that by never showing non-speech to the
model at all. This module is the second line: it looks at each piece of text
that did come back and decides whether to keep it. Nothing is thrown away
silently - every rejected piece is returned with a reason, and lands in the
"removed" list of the transcript so it can be reviewed.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

LEVELS = ("off", "balanced", "strict")

# Subtitle furniture. Never real speech in a recording, at any confidence.
_CREDITS = [
    r"subtitles? (?:by|made by|provided by|created by)\b.*",
    r"(?:sub)?titles? by the amara\.?org community",
    r"amara\.?org",
    r"transcri(?:bed|ption) by\b.*",
    r"captions? (?:by|provided by)\b.*",
    r"translated by\b.*",
    r"(?:www\.)?[a-z0-9-]+\.(?:com|org|net|co\.uk)",
    r"copyright .*all rights reserved.*",
    r"\(c\) \d{4}.*",
    r"sous-titres? (?:r[eé]alis[eé]s?|par|fait)\b.*", r"untertitel\b.*", r"subt[ií]tulos? (?:por|realizados|de la comunidad)\b.*",
    r"sottotitoli (?:creati|a cura|e revisione)\b.*", r"legendas? (?:pela|por)\b.*", r"ondertitel(?:s|ing)?\b.*",
    r"napisy stworzone\b.*", r"субтитры .*", r"редактор субтитров.*", r"продолжение следует.*",
    r"字幕.*", r"ご視聴ありがとうございました.*", r"請不吝點贊.*", r"谢谢观看.*", r"謝謝觀看.*", r"시청해 ?주셔서 감사합니다.*",
]
# Things said at the end of videos. Dropped when they stand alone as a whole
# piece AND the evidence for them is weak (short, quiet or low confidence).
_OUTROS = [
    r"thanks? (?:you )?(?:so much |very much )?for watching(?: everyone| guys| and see you next time)?",
    r"thank you for watching(?: this video)?",
    r"(?:please )?(?:like(?:,| and) )?(?:share(?:,| and) )?subscribe(?: to (?:my|the|our) channel)?",
    r"(?:don'?t forget to |please )?(?:like and )?subscribe.*",
    r"see you (?:in the )?next (?:time|video|one)",
    r"i'?ll see you (?:guys )?(?:in the )?next (?:time|video|one)",
    r"bye[- ]?bye",
    r"(?:please )?leave a comment.*",
    r"so that'?s the end of (?:the|this) video",
    r"(?:muchas )?gracias por ver(?: el v[ií]deo)?", r"merci d'avoir regard[eé].*", r"vielen dank f[uü]r'?s zusehen.*",
    r"danke f[uü]rs zuschauen.*", r"grazie per (?:la visione|aver guardato).*", r"obrigad[oa] por assistir.*",
    r"bedankt voor het kijken.*", r"dzi[eę]kuj[eę] za (?:uwag[eę]|ogl[aą]danie).*", r"спасибо за просмотр.*",
]
# One- or two-word fillers that are plausible speech but are also what a model
# says to noise. Dropped only on weak evidence.
_FILLERS = {
    "you", "thank you", "thanks", "bye", "okay", "ok", "so", "oh", "uh", "um", "hmm", "mm", "mhm", "ah", "eh",
    "yeah", "yes", "no", "the", "and", "i", "a", "it", "well", "right", "hm", "huh", "the end",
}
# Non-speech event tags the models emit.
_TAGS = re.compile(r"[\[(（<♪*]+\s*(?:music|musik|applause|laughter|laughs?|silence|noise|inaudible|blank[ _]audio|"
                   r"sound|sighs?|coughs?|clears throat|background noise|no audio|no speech|beep|static|"
                   r"foreign|speaking foreign language)[^\])）>]*[\])）>♪*]+", re.I)
_NOTES = re.compile(r"[♪♫♬🎵🎶]+")

# A piece that is nothing but a stage direction: [Spring], (crying), *sighs*.
_ALL_BRACKET = re.compile(r"^\s*(?:[\[(（*<][^\])）*>]{0,60}[\])）*>]?\s*)+$")
_LETTERS = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_STUTTER = re.compile(r"(.{1,3}?)\1{5,}")

_CREDITS_RE = [re.compile(r"^(?:" + p + r")$", re.I) for p in _CREDITS]
_OUTROS_RE = [re.compile(r"^(?:" + p + r")$", re.I) for p in _OUTROS]
_PUNCT = re.compile(r"[^\w\s'.-]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def plain(text: str) -> str:
    """Lower-case, punctuation-free form used for matching."""
    t = text.lower().replace("’", "'")
    t = _PUNCT.sub(" ", t)
    t = _SPACE.sub(" ", t).strip(" .-'")
    return t


_SQUARE = re.compile(r"\[[^\]]{0,60}\]")


def strip_tags(text: str) -> str:
    """Remove [Music], (applause), musical notes and the like.

    Anything in square brackets goes: nobody speaks in square brackets, they
    are always the model describing a sound ("[VIDEO PLAYBACK]", "[BLANK_AUDIO]").
    """
    t = _TAGS.sub(" ", text)
    t = _SQUARE.sub(" ", t)
    t = _NOTES.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def phantom(text: str, level: str = "balanced") -> Optional[str]:
    """Pattern checks only, for one phrase inside a longer piece of text.

    Whisper hands back a stretch of speech as several timed phrases, and a
    phantom often arrives as the last of them, tacked on after real speech:
    "...see you on Friday." + "Thanks for watching!". Returns the reason the
    phrase should be dropped, or None to keep it.
    """
    if level not in _T:
        return None
    cleaned = strip_tags(text)
    if not cleaned or not plain(cleaned) or _ALL_BRACKET.match(cleaned):
        return "non-speech marker (music, noise)"
    key = plain(cleaned)
    for rx in _CREDITS_RE:
        if rx.match(key):
            return "subtitle credit - a known phantom phrase"
    for rx in _OUTROS_RE:
        if rx.match(key):
            return "video sign-off phrase standing alone - a known phantom phrase"
    if _STUTTER.search(key.replace(" ", "")) and len(set(key.replace(" ", "").replace("-", ""))) <= 4:
        return "a stuttered run of one sound - invented"
    return None


def collapse_repeats(text: str, limit: int = 3, word_limit: int = 6) -> Tuple[str, bool]:
    """Cut a phrase looped more than `limit` times in a row down to one.

    "I think so. I think so. I think so. I think so. I think so." -> "I think so."
    Three in a row is left alone, and a single word may repeat up to
    `word_limit` times - people do say "no, no, no, no".
    """
    words = text.split()
    if len(words) < limit + 1:
        return text, False
    limits = {1: word_limit}
    keys = [plain(w) for w in words]
    changed = False
    for size in range(1, min(12, len(words) // (limit + 1)) + 1):
        i = 0
        out_w: List[str] = []
        out_k: List[str] = []
        while i < len(words):
            unit = keys[i:i + size]
            n = 1
            while len(unit) == size and any(unit) and keys[i + n * size:i + (n + 1) * size] == unit:
                n += 1
            if n > limits.get(size, limit):
                out_w += words[i:i + size]
                out_k += unit
                i += n * size
                changed = True
            else:
                out_w.append(words[i])
                out_k.append(keys[i])
                i += 1
        words, keys = out_w, out_k
    return " ".join(words), changed


def confidence(log_probs) -> Optional[float]:
    """Mean token probability (0-1) from the model's token log-probabilities."""
    try:
        vals = [float(v) for v in log_probs]
    except Exception:  # noqa: BLE001
        return None
    vals = [v for v in vals if not math.isnan(v) and not math.isinf(v)]
    if not vals:
        return None
    # models report log-probabilities (<= 0); anything else is not one
    if max(vals) > 1e-6:
        return None
    return float(math.exp(sum(vals) / len(vals)))


# Thresholds per level: (minimum confidence, "weak evidence" confidence,
#                        maximum characters per second, "quiet" level in dB below the file's speech)
_T = {
    "balanced": (0.20, 0.55, 32.0, -28.0),
    "strict": (0.35, 0.70, 28.0, -22.0),
}


def judge(text: str, duration: float, conf: Optional[float], level: str = "balanced",
          rel_db: Optional[float] = None) -> Tuple[str, Optional[str]]:
    """Decide what to do with one piece of recognised text.

    duration  seconds of audio the text was recognised from
    conf      mean token probability, when the model reports one
    rel_db    loudness of this piece relative to the file's typical speech (0 = typical)

    Returns (text_to_keep, reason). Text "" means drop it; reason is None when
    the text was kept untouched.
    """
    original = text.strip()
    if not original:
        return "", None
    if level not in _T:  # "off": pass everything through untouched
        return original, None
    min_conf, weak_conf, max_cps, quiet_db = _T[level]

    cleaned = strip_tags(original)
    if not cleaned or not plain(cleaned) or _ALL_BRACKET.match(cleaned):
        return "", "non-speech marker (music, noise)"
    key = plain(cleaned)
    if _STUTTER.search(key.replace(" ", "")) and len(set(key.replace(" ", "").replace("-", ""))) <= 4:
        return "", "a stuttered run of one sound - invented"

    for rx in _CREDITS_RE:
        if rx.match(key):
            return "", "subtitle credit - a known phantom phrase"

    cleaned2, looped = collapse_repeats(cleaned)
    if looped:
        before = len(cleaned.split())
        if before >= 12 and len(cleaned2.split()) <= 0.4 * before:
            return "", "one phrase looped over and over - invented"
        cleaned, key = cleaned2, plain(cleaned2)

    weak = ((conf is not None and conf < weak_conf)
            or (rel_db is not None and rel_db < quiet_db)
            or duration < 0.35)
    # A model that reports no confidence (Whisper) gives nothing to weigh a
    # sign-off phrase against, so on its own it goes; with a confidence it
    # goes only when the evidence is weak.
    if weak or conf is None:
        for rx in _OUTROS_RE:
            if rx.match(key):
                return "", "video sign-off phrase standing alone - a known phantom phrase"
    if weak:
        if key in _FILLERS and (duration > 4.0 or (conf is not None and conf < weak_conf * 0.8)
                                or (rel_db is not None and rel_db < quiet_db)):
            return "", "lone filler word on weak audio"

    if conf is not None and conf < min_conf:
        return "", f"very low confidence ({conf:.0%})"

    if duration > 0:
        cps = len(key) / duration
        if cps > max_cps and len(key) > 20:
            return "", "far more text than the audio is long - invented"
        # Speech runs at 10-20 characters a second. Several seconds of "speech"
        # that came out as a word or two is noise the detector let through.
        slow = 2.0 if level == "strict" else 1.0
        if duration >= (3.0 if level == "strict" else 4.0) and cps < slow and (conf is None or conf < 0.9):
            return "", "seconds of audio produced almost no text - noise, not speech"
        # digits and symbols with no actual word, at no sensible pace ("1/2 tbps", "3.5mm")
        if not _LETTERS.search(cleaned) and (cps < 3.0 or looped):
            return "", "no real words"

    if looped:
        return cleaned, "a looped phrase was collapsed"
    if cleaned != original:
        return cleaned, "non-speech markers removed"
    return original, None
