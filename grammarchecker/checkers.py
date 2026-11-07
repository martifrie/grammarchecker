"""DETERMINISTIC checkers: client's regex rules and a local LanguageTool server"""
from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from collections import Counter

import httpx

from .models import Edit, Rule, Rulebook

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- regex rules


def _applies(rule: Rule, text: str) -> bool:
    if rule.when and not re.search(rule.when, text):
        return False
    if rule.unless and re.search(rule.unless, text):
        return False
    return True


def _match_case(src: str, new: str) -> str:
    if src.isupper() and len(src) > 1:
        return new.upper()
    if src[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def _compile(rule: Rule) -> re.Pattern:
    flags = re.MULTILINE | (re.IGNORECASE if rule.ignore_case or rule.words else 0)
    if rule.words:
        alts = "|".join(sorted((re.escape(w) for w in rule.words), key=len, reverse=True))
        return re.compile(rf"(?<!\w)({alts})(?!\w)", flags)
    return re.compile(rule.pattern, flags)


def regex_edits(texts: list[str], rulebook: Rulebook, ms=None) -> list[Edit]:
    """`ms` (the Manuscript) is needed for formatting rules, to skip text already in italics"""
    edits = []
    compiled = [(r, _compile(r)) for r in rulebook.by_type("regex")]
    words_lower = {r.id: {k.lower(): v for k, v in r.words.items()} for r, _ in compiled if r.words}
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        for rule, rx in compiled:
            if not _applies(rule, text):
                continue
            for m in rx.finditer(text):
                old = m.group(0)
                if rule.require_italic and (ms is None or not ms.is_italic(i, m.start(), m.end())):
                    continue
                if rule.format:
                    if ms is not None and ms.is_italic(i, m.start(), m.end()):
                        continue
                    edits.append(Edit(i, m.start(), m.end(), old, None, rule.id, rule.description,
                                      "regex", format=rule.format))
                    continue
                if rule.words:
                    new = _match_case(old, words_lower[rule.id][old.lower()])
                elif rule.transform:
                    new = old.lower() if rule.transform == "lower" else old.upper()
                else:
                    new = m.expand(rule.replacement or "")
                if rule.mode == "flag":
                    g = rule.span_group
                    e = Edit(i, m.start(g), m.end(g), m.group(g), new if new != old else None,
                             rule.id, rule.description, "regex")
                    e.to_flag()
                    edits.append(e)
                    continue
                if new == old:
                    continue
                edits.extend(_minimal(i, m.start(), old, new, rule.id, rule.description))
    return edits


# --------------------------------------------------------------------------- quotes

_OPEN = {"«": "»", "“": "”", "‘": "’"}
_CLOSE = {v: k for k, v in _OPEN.items()}
_OPEN_CONTEXT = set(" \t\n([{«“‘—―–-¿¡") | {""}


def quote_edits(texts: list[str], rule: Rule) -> list[Edit]:
    """Normalize quote nesting: level 1 «», level 2 “”, level 3 ‘’ (configurable).
    Paragraphs with unbalanced quotes (e.g. quotations spanning paragraphs) are left alone"""
    levels = rule.levels or [["«", "»"], ["“", "”"], ["‘", "’"]]
    edits = []
    for i, text in enumerate(texts):
        stack: list[str] = []   # original opening chars
        wanted: list[tuple[int, str]] = []
        ok = True
        for k, c in enumerate(text):
            prev = text[k - 1] if k else ""
            if c == '"':
                is_open = not (stack and stack[-1] == '"') and prev in _OPEN_CONTEXT
                if not is_open and not stack:
                    ok = False
                    break
                c_kind = "open" if is_open else "close"
            elif c in _OPEN:
                c_kind = "open"
            elif c in _CLOSE:
                if c == "’" and not (stack and stack[-1] == "‘"):
                    continue  # apostrophe
                if not stack:
                    ok = False
                    break
                c_kind = "close"
            else:
                continue
            if c_kind == "open":
                depth = min(len(stack), len(levels) - 1)
                stack.append(c)
                wanted.append((k, levels[depth][0]))
            else:
                stack.pop()
                depth = min(len(stack), len(levels) - 1)
                wanted.append((k, levels[depth][1]))
        if not ok or stack:
            continue
        for k, w in wanted:
            if text[k] != w:
                e = Edit(i, k, k + 1, text[k], w, rule.id, rule.description, "regex")
                if rule.mode == "flag":
                    e.to_flag()
                edits.append(e)
    return edits


# --------------------------------------------------------------------------- consistency

_NAME = re.compile(r"(?<!\w)[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?!\w)")
_LOWER = re.compile(r"(?<!\w)[a-záéíóúñü]+(?!\w)")


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def consistency_edits(texts: list[str], rule: Rule) -> list[Edit]:
    """Find proper names written in more than one way (Julián / Julian) and mark the
    minority spelling. Words that also appear in lowercase (Él/El, Sólo/Solo) are ignored."""
    counts: Counter = Counter()
    lower: set[str] = set()
    for t in texts:
        counts.update(m.group(0) for m in _NAME.finditer(t))
        lower.update(_LOWER.findall(t))
    groups: dict[str, list[str]] = {}
    for form in counts:
        groups.setdefault(_fold(form), []).append(form)
    fixes = {}
    for forms in groups.values():
        if len(forms) < 2 or any(f.lower() in lower for f in forms):
            continue
        forms.sort(key=lambda f: counts[f], reverse=True)
        main = forms[0]
        for f in forms[1:]:
            if counts[main] > counts[f]:
                fixes[f] = (main, counts[main], counts[f])
    edits = []
    if not fixes:
        return edits
    rx = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, fixes)) + r")(?!\w)")
    for i, t in enumerate(texts):
        for m in rx.finditer(t):
            main, n_main, n_f = fixes[m.group(0)]
            why = f"{rule.description} «{main}» aparece {n_main} veces y «{m.group(0)}» {n_f}."
            e = Edit(i, m.start(), m.end(), m.group(0), main, rule.id, why, "regex", 0.9)
            if rule.mode == "flag":
                e.to_flag()
            edits.append(e)
    return edits


def _minimal(para, start, old, new, rule_id, reason) -> list[Edit]:
    """Split a replacement into the smallest edits (difflib opcodes), so «hola» shows as
    two quote swaps and «basta.» → «basta». as a moved period, not a retyped sentence."""
    if len(old) == 2 and new == old[::-1]:
        # Swap (punctuation before a closing sign): move the first character, keep the sign
        # as is, so other rules (e.g. quote levels) can still edit it.
        return [Edit(para, start, start + 1, old[0], "", rule_id, reason, "regex"),
                Edit(para, start + 2, start + 2, "", old[0], rule_id, reason, "regex")]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        out.append(Edit(para, start + i1, start + i2, old[i1:i2], new[j1:j2], rule_id, reason, "regex"))
    return out


# --------------------------------------------------------------------------- LanguageTool


def _utf16_to_py(text: str, offset16: int) -> int:
    """LanguageTool returns Java (UTF-16) offsets; convert to Python indices."""
    count = 0
    for i, ch in enumerate(text):
        if count >= offset16:
            return i
        count += 2 if ord(ch) > 0xFFFF else 1
    return len(text)


class LanguageTool:
    # Typography and punctuation are owned by the rulebook's regex rules
    DISABLED_CATEGORIES = "TYPOGRAPHY,PUNCTUATION,STYLE,REDUNDANCY"
    # Dictionary misses in fiction are mostly invented words or names, and the first suggestion
    # can be nonsense («morenotes» → «more notes»): always flag, never replace
    FLAG_ONLY_PREFIXES = ("MORFOLOGIK",)

    def __init__(self, url: str, language: str, max_chars: int = 15000):
        self.url = url.rstrip("/") + "/v2/check"
        self.language = language
        self.max_chars = max_chars
        self.client = httpx.Client(timeout=120)

    def available(self) -> bool:
        try:
            r = self.client.post(self.url, data={"text": "hola", "language": self.language})
        except httpx.HTTPError:
            return False
        if r.status_code != 200:
            log.warning("LanguageTool rejected the language %r: %s", self.language, r.text[:200])
            return False
        return True

    def edits(self, texts: list[str], ignore_words: set[str]) -> list[Edit]:
        out: list[Edit] = []
        for batch in self._batches(texts):
            joined = "\n\n".join(texts[i] for i in batch)
            starts, pos = [], 0
            for i in batch:
                starts.append(pos)
                pos += len(texts[i]) + 2
            resp = self.client.post(self.url, data={
                "text": joined,
                "language": self.language,
                "disabledCategories": self.DISABLED_CATEGORIES,
            })
            resp.raise_for_status()
            for m in resp.json().get("matches", []):
                start = _utf16_to_py(joined, m["offset"])
                end = _utf16_to_py(joined, m["offset"] + m["length"])
                k = max(j for j, s in enumerate(starts) if s <= start)
                para, p0 = batch[k], starts[k]
                original = joined[start:end]
                if not original.strip() or original in ignore_words or end - p0 > len(texts[para]):
                    continue
                suggestions = [r["value"] for r in m.get("replacements", [])][:5]
                rule = m["rule"]["id"]
                msg = m.get("message", "")
                if len(suggestions) == 1 and not rule.startswith(self.FLAG_ONLY_PREFIXES):
                    out.append(Edit(para, start - p0, end - p0, original, suggestions[0],
                                    f"LT:{rule}", msg, "languagetool", 0.8))
                else:
                    hint = f" Sugerencias: {', '.join(suggestions)}" if suggestions else ""
                    out.append(Edit(para, start - p0, end - p0, original, None,
                                    f"LT:{rule}", msg + hint, "languagetool", 0.5))
        return out

    def _batches(self, texts: list[str]):
        batch, size = [], 0
        for i, t in enumerate(texts):
            if not t.strip():
                continue
            if batch and size + len(t) > self.max_chars:
                yield batch
                batch, size = [], 0
            batch.append(i)
            size += len(t) + 2
        if batch:
            yield batch
