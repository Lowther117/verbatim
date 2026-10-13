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
# Tails are bounded and long pieces are never matched (see _MAX_PHANTOM): a real
# sentence that happens to begin "Translated by a colleague of mine..." must
# survive.
_CREDITS = [
    r"subtitles? (?:by|made by|provided by|created by)\b.{0,40}",
    r"(?:sub)?titles? by the amara\.?org community",
    r"amara\.?org",
    r"transcri(?:bed|ption) by\b.{0,40}",
    r"captions? (?:by|provided by)\b.{0,40}",
    r"translated by\b.{0,40}",
    r"copyright .{0,30}all rights reserved.{0,20}",
    r"sous-titres? (?:r[eé]alis[eé]s?|par|fait)\b.{0,40}", r"sous-titrage\b.{0,40}", r"untertitel\b.{0,40}",
    r"subt[ií]tulos? (?:por|realizados|de la comunidad)\b.{0,40}",
    r"sottotitoli (?:creati|a cura|e revisione)\b.{0,40}", r"legendas? (?:pela|por)\b.{0,40}",
    r"ondertitel(?:s|ing)?\b.{0,40}", r"napisy stworzone\b.{0,40}", r"субтитры .{0,40}",
    r"редактор субтитров.{0,40}", r"продолжение следует.{0,20}",
    r"字幕.{0,30}", r"ご視聴ありがとうございました.{0,20}", r"請不吝點贊.{0,30}", r"谢谢观看.{0,20}", r"謝謝觀看.{0,20}",
    r"시청해 ?주셔서 감사합니다.{0,20}",
]
# A web address and nothing else. Real when someone reads one out, a phantom
# when it appears from nowhere - so it needs weak evidence to be dropped.
_DOMAIN = r"(?:www\.)?[a-z0-9-]+\.(?:com|org|net|co\.uk)"
# Things said at the end of videos and nowhere else. Dropped when they stand
# alone and the evidence is weak, or the model gives no confidence at all.
_VIDEO_OUTROS = [
    r"thanks?(?: you)?(?: so much| very much)? for watching(?: (?:everyone|guys|this video|the video))?",
    r"thanks?(?: you)?(?: so much| very much)? for watching\b.{0,40}\b(?:video|next time|subscribe|channel)\b.{0,20}",
    r"(?:please )?(?:don'?t forget to )?(?:like )?(?:and )?(?:share )?(?:and )?subscribe(?: to (?:my|the|our) channel)?",
    r"(?:please )?(?:don'?t forget to )?(?:like and )?subscribe\b.{0,30}\b(?:channel|video|videos|bell|notifications?)\b.{0,20}",
    r"(?:i'?ll )?see you (?:guys )?in the next (?:video|one)",
    r"(?:please )?leave a comment(?: (?:below|down below|in the comments))?",
    r"so that'?s the end of (?:the|this) video",
    r"(?:muchas )?gracias por ver(?: el v[ií]deo)?", r"merci d'avoir regard[eé].{0,30}",
    r"vielen dank f[uü]r'?s zusehen.{0,20}", r"danke f[uü]rs zuschauen.{0,20}",
    r"grazie per (?:la visione|aver guardato).{0,20}", r"obrigad[oa] por assistir.{0,20}",
    r"bedankt voor het kijken.{0,20}", r"dzi[eę]kuj[eę] za (?:uwag[eę]|ogl[aą]danie).{0,20}",
    r"спасибо за просмотр.{0,20}",
]
# Sign-offs people really do say at the end of a call. Dropped only on weak
# evidence, or when a model with no confidence produced one from an isolated
# blip with silence either side.
_SPOKEN_OUTROS = [
    r"bye[- ]?bye",
    r"(?:i'?ll )?see you next time",
]
# One- or two-word fillers that are plausible speech but are also what a model
# says to noise. Dropped only on weak evidence.
_FILLERS = {
    "you", "thank you", "thanks", "bye", "okay", "ok", "so", "oh", "uh", "um", "hmm", "mm", "mhm", "ah", "eh",
    "yeah", "yes", "no", "the", "and", "i", "a", "it", "well", "right", "hm", "huh", "the end",
}
_MAX_PHANTOM = 70   # a piece longer than this is a sentence, not a stock phrase
# Non-speech event tags the models emit.
_TAGS = re.compile(r"[\[(（<♪*]+\s*(?:music|musik|applause|laughter|laughs?|silence|noise|inaudible|blank[ _]audio|"
                   r"sound|sighs?|coughs?|clears throat|background noise|no audio|no speech|beep|static|"
                   r"foreign|speaking foreign language)\b(?:[ _-]+[\w']+){0,2}\s*[\])）>♪*]+", re.I)
_NOTES = re.compile(r"[♪♫♬🎵🎶]+")

# One closed stage direction: [Spring], (crying), *sighs*. Written so that it
# cannot backtrack: the body may not contain another opener.
_BRACKETED = re.compile(r"[\[(（*<][^\[\](（)）*<>]{0,60}[\])）*>]")
_WORD = re.compile(r"(?<![^\W_])[^\W\d_]+(?![^\W_])", re.UNICODE)   # a token made only of letters
_WORD2 = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
_STUTTER = re.compile(r"(.{1,3}?)\1{5,}")

_CREDITS_RE = [re.compile(r"^(?:" + p + r")$", re.I) for p in _CREDITS]
_DOMAIN_RE = re.compile(r"^(?:" + _DOMAIN + r")$", re.I)
_VIDEO_RE = [re.compile(r"^(?:" + p + r")$", re.I) for p in _VIDEO_OUTROS]
_SPOKEN_RE = [re.compile(r"^(?:" + p + r")$", re.I) for p in _SPOKEN_OUTROS]
_PUNCT = re.compile(r"[^\w\s'.-]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def _all_bracket(text: str) -> bool:
    """Is the piece nothing but stage directions?"""
    return not _BRACKETED.sub(" ", text).strip()


def _stuttered(key: str) -> bool:
    """One sound repeated over and over with no actual word in it: "h-h-h-h-h-h", "1.1.1.1.1.1"."""
    squashed = key.replace(" ", "")
    if not _STUTTER.search(squashed) or len(set(squashed.replace("-", ""))) > 4:
        return False
    return not _WORD2.search(key)


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


def phantom(text: str, level: str = "balanced", isolated: bool = False) -> Optional[str]:
    """Pattern checks only, for one phrase inside a longer piece of text.

    Whisper hands back a stretch of speech as several timed phrases, and a
    phantom often arrives as the last of them, tacked on after real speech:
    "...see you on Friday." + "Thanks for watching!". Returns the reason the
    phrase should be dropped, or None to keep it.
    """
    if level not in _T:
        return None
    cleaned = strip_tags(text)
    if not cleaned or not plain(cleaned) or _all_bracket(cleaned):
        return "non-speech marker (music, noise)"
    key = plain(cleaned)
    if len(key) <= _MAX_PHANTOM:
        for rx in _CREDITS_RE:
            if rx.match(key):
                return "subtitle credit - a known phantom phrase"
        for rx in _VIDEO_RE:
            if rx.match(key):
                return "video sign-off phrase - a known phantom phrase"
        if isolated or level == "strict":
            for rx in _SPOKEN_RE:
                if rx.match(key):
                    return "sign-off phrase with silence either side - a known phantom phrase"
    if _stuttered(key):
        return "a stuttered run of one sound - invented"
    return None


def collapse_repeats(text: str, limit: int = 3, word_limit: int = 6) -> Tuple[str, bool]:
    """Cut a phrase looped more than `limit` times in a row down to one.

    "I think so. I think so. I think so. I think so. I think so." -> "I think so."
    Three in a row is left alone, and a single word may repeat up to
    `word_limit` times - people do say "no, no, no, no".
    """
    words = text.split()
    keep, changed = collapse_repeat_words(words, limit, word_limit)
    if not changed:
        return text, False
    return " ".join(words[i] for i in keep), True


def collapse_repeat_words(words: List[str], limit: int = 3, word_limit: int = 6) -> Tuple[List[int], bool]:
    """collapse_repeats on a list of words: the indices of the words that
    survive, so individually timed words (Parakeet) keep their own timing."""
    idx = list(range(len(words)))
    if len(words) < limit + 1:
        return idx, False
    limits = {1: word_limit}
    keys = [plain(w) for w in words]
    changed = False
    for size in range(1, min(12, len(words) // (limit + 1)) + 1):
        i = 0
        out_i: List[int] = []
        out_k: List[str] = []
        while i < len(idx):
            unit = keys[i:i + size]
            n = 1
            while len(unit) == size and any(unit) and keys[i + n * size:i + (n + 1) * size] == unit:
                n += 1
            if n > limits.get(size, limit):
                out_i += idx[i:i + size]
                out_k += unit
                i += n * size
                changed = True
            else:
                out_i.append(idx[i])
                out_k.append(keys[i])
                i += 1
        idx, keys = out_i, out_k
    return idx, changed


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
          rel_db: Optional[float] = None, isolated: bool = False) -> Tuple[str, Optional[str]]:
    """Decide what to do with one piece of recognised text.

    duration  seconds of audio the text was recognised from
    conf      mean token probability, when the model reports one (Whisper does not)
    rel_db    loudness of this piece relative to the file's typical speech (0 = typical)
    isolated  the piece has two seconds or more of non-speech on both sides

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
    if not cleaned or not plain(cleaned) or _all_bracket(cleaned):
        return "", "non-speech marker (music, noise)"
    key = plain(cleaned)
    if _stuttered(key):
        return "", "a stuttered run of one sound - invented"

    short = len(key) <= _MAX_PHANTOM
    if short:
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
    # With no confidence from the model (Whisper), the only evidence that a
    # stock phrase is a phantom is where it sits: alone, with silence around it.
    blind = conf is None
    if short:
        # video sign-offs: nobody says these in a meeting
        if weak or blind:
            for rx in _VIDEO_RE:
                if rx.match(key):
                    return "", "video sign-off phrase standing alone - a known phantom phrase"
        if weak or (blind and (isolated or level == "strict")):
            for rx in _SPOKEN_RE:
                if rx.match(key):
                    return "", "sign-off phrase on weak audio - a known phantom phrase"
            if _DOMAIN_RE.match(key):
                return "", "a web address from nowhere - a known phantom phrase"
    if key in _FILLERS:
        if weak and (duration > 4.0 or (conf is not None and conf < weak_conf * 0.8)
                     or (rel_db is not None and rel_db < quiet_db)):
            return "", "lone filler word on weak audio"
        if blind and (isolated or level == "strict") and key in ("you", "thank you", "thanks", "bye", "so", "the end"):
            return "", "lone filler word with silence either side - a known phantom phrase"

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
        # Digits and symbols with not one actual word ("3.5mm", "1/2"), at no
        # sensible pace. A confident model, or a piece that sits inside a
        # conversation, gets the benefit of the doubt: "25." can be an answer.
        if not _WORD.search(cleaned) and (cps < 3.0 or looped):
            if looped or (conf is not None and conf < 0.8) or (blind and (isolated or level == "strict")):
                return "", "no real words"

    if looped:
        return cleaned, "a looped phrase was collapsed"
    if cleaned != original:
        return cleaned, "non-speech markers removed"
    return original, None
