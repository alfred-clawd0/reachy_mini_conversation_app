"""Normalize assistant text for natural speech synthesis."""

from __future__ import annotations
import re


_CODE_FENCE_RE = re.compile(r"```(?:[\w+-]+)?\s*(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MARKDOWN_LINK_RE = re.compile(r"\[([^]]+)]\((?:https?://|mailto:)[^)]+\)")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_APPLICATION_BUNDLE_RE = re.compile(
    r"(?<!\w)/Applications/(?:[^/\s]+/)*(?:[A-Z0-9][^/\s,;:!?]*\s){0,3}[^/\s,;:!?]*\.app(?=/|[\s.,;:!?)]|$)(?:/\S+)?"
)
_ABSOLUTE_PATH_RE = re.compile(r"(?<!\w)/(?:Applications|Users|private|var|tmp)/\S+")
_REPEAT_COUNTER_RE = re.compile(r"\(\s*[×x]\s*\d+\s*\)", re.IGNORECASE)
_NUMERIC_RANGE_RE = re.compile(r"(?<=\d)\s*–\s*(?=\d)")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")
_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s*")
_EMPHASIS_RE = re.compile(r"(?<![\w*])(\*{1,3}|__|_)(?![\s*_])([^\n*]+?)(?<![\s*])\1(?![\w*])")
_SPACE_RE = re.compile(r"\s+")


def _trailing_punctuation(value: str) -> str:
    return value[len(value.rstrip(".,;:!?)]")) :]


def _spoken_path(match: re.Match[str]) -> str:
    path = match.group().rstrip(".,;:!?)]")
    if re.search(r"\.app(?:/|$)", path):
        spoken = "the application"
    else:
        basename = path.rstrip("/").rsplit("/", 1)[-1]
        spoken = basename if re.fullmatch(r"[\w.-]{1,30}", basename) else "a file"
    return spoken + _trailing_punctuation(match.group())


def _spoken_emphasis(match: re.Match[str]) -> str:
    # Double-underscore identifiers are code names, not italic/bold prose.
    if match.group(1) == "__" and re.fullmatch(r"\w+", match.group(2)):
        return match.group()
    return match.group(2)


def normalize_for_speech(text: str) -> str:
    """Convert display-oriented assistant text into plain, pronounceable English."""
    clean = str(text or "").strip()
    if not clean:
        return ""
    clean = _CODE_FENCE_RE.sub(lambda match: match.group(1), clean)
    clean = _INLINE_CODE_RE.sub(lambda match: match.group(1), clean)
    clean = _MARKDOWN_LINK_RE.sub(lambda match: match.group(1), clean)
    clean = _URL_RE.sub(lambda match: "the link" + _trailing_punctuation(match.group()), clean)
    clean = _APPLICATION_BUNDLE_RE.sub(_spoken_path, clean)
    clean = _ABSOLUTE_PATH_RE.sub(_spoken_path, clean)
    clean = _REPEAT_COUNTER_RE.sub("", clean)
    clean = _HEADING_RE.sub("", clean)
    lines = clean.splitlines()
    joined = ""
    previous_item = False
    for line in lines:
        is_item = bool(_LIST_MARKER_RE.match(line))
        item = _LIST_MARKER_RE.sub("", line).strip()
        if item:
            separator = ", " if is_item and previous_item and not joined.endswith(tuple(".,;:!?")) else " "
            joined += (separator if joined else "") + item
        previous_item = is_item
    clean = joined
    clean = _EMPHASIS_RE.sub(_spoken_emphasis, clean)
    clean = _NUMERIC_RANGE_RE.sub(" to ", clean)
    clean = re.sub(r"\s*−\s*", " minus ", clean)
    clean = re.sub(r"(?<![\w.])-(?=\d)", "minus ", clean)
    clean = clean.replace("%", " percent")
    clean = clean.replace("&", " and ")
    clean = re.sub(
        r"(?<=\d)°\s*([FCfc])\b",
        lambda match: " degrees " + ("Celsius" if match.group(1).lower() == "c" else "Fahrenheit"),
        clean,
    )
    clean = re.sub(r"(?<=\d)°", " degrees", clean)
    clean = _SPACE_RE.sub(" ", clean).strip(" ,;:")
    clean = re.sub(r"^-(?!\d)", "", clean).rstrip(" -").strip()
    return clean
