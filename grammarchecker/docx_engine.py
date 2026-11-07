"""Read paragraphs from a .docx and write edits back as Word tracked changes.

Every correction becomes:
  <w:del>original</w:del><w:ins>replacement</w:ins>  + a Word comment with rule id and reason.
Every mark is highlighted in the color of its rule category (rules/*.yaml `categories`):
the inserted text carries the highlight; deleted text and "flag only" spans get it as a
tracked formatting change, so rejecting everything in Word gives back the exact original.
Formatting fixes (e.g. italics for latinisms) are tracked the same way.

Text model: each paragraph's visible text is the concatenation of its runs. Runs that hold
only <w:t> are "simple" and can be split anywhere; runs with tabs, breaks, fields, images,
etc. are "complex" and an edit that would cut through or delete one is skipped (reported).
"""
from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass

from typing import Callable

from docx import Document
from docx.oxml.ns import qn
from docx.text.run import Run
from lxml import etree

from .models import Category, Edit

W_P, W_R, W_T = qn("w:p"), qn("w:r"), qn("w:t")
W_RPR, W_DEL, W_INS, W_DELTEXT = qn("w:rPr"), qn("w:del"), qn("w:ins"), qn("w:delText")
W_ID, W_AUTHOR, W_DATE = qn("w:id"), qn("w:author"), qn("w:date")
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

# Children that contribute visible characters but make a run "complex"
_CHAR_CHILDREN = {
    qn("w:tab"): "\t",
    qn("w:br"): "\n",
    qn("w:cr"): "\n",
    qn("w:noBreakHyphen"): "‑",
    qn("w:softHyphen"): "",
    qn("w:sym"): "￼",
    qn("w:drawing"): "￼",
    qn("w:pict"): "￼",
    qn("w:object"): "￼",
    qn("w:footnoteReference"): "￼",
    qn("w:endnoteReference"): "￼",
}
# Zero-width children that don't stop a run from being split
_HARMLESS = {W_RPR, qn("w:lastRenderedPageBreak")}

# Color for marks whose rule belongs to no category
DEFAULT_HIGHLIGHT = "yellow"
# <w:rPr> children that must come after <w:highlight> (schema order)
_AFTER_HIGHLIGHT = {qn(f"w:{t}") for t in (
    "u", "effect", "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang",
    "eastAsianLayout", "specVanish", "oMath", "rPrChange")}


def _set_highlight(run: etree._Element, color: str) -> None:
    rpr = run.find(W_RPR)
    if rpr is None:
        rpr = etree.Element(W_RPR)
        run.insert(0, rpr)
    for old in rpr.findall(qn("w:highlight")):
        rpr.remove(old)
    hl = etree.Element(qn("w:highlight"))
    hl.set(qn("w:val"), color)
    after = next((c for c in rpr if c.tag in _AFTER_HIGHLIGHT), None)
    if after is not None:
        after.addprevious(hl)
    else:
        rpr.append(hl)


@dataclass
class RunSpan:
    el: etree._Element
    start: int
    end: int
    simple: bool
    deleted: bool = False
    italic: bool = False


class SkippedEdit(Exception):
    pass


# --------------------------------------------------------------------------- reading


def _excluded(run: etree._Element, p: etree._Element) -> bool:
    """True for runs that are already deleted or belong to a nested paragraph."""
    node = run.getparent()
    while node is not None and node is not p:
        if node.tag in (W_DEL, qn("w:moveFrom")):
            return True
        if node.tag == W_P:  # run of a nested paragraph (e.g. text box)
            return True
        node = node.getparent()
    return False


def _run_text(r: etree._Element) -> tuple[str, bool]:
    text, simple = [], True
    for child in r:
        if child.tag == W_T:
            text.append(child.text or "")
        elif child.tag in _HARMLESS:
            continue
        else:
            simple = False
            text.append(_CHAR_CHILDREN.get(child.tag, ""))
    return "".join(text), simple


def _normalize_simple_run(r: etree._Element) -> None:
    """Collapse several <w:t> into one so splitting is trivial."""
    ts = r.findall(W_T)
    if len(ts) > 1:
        ts[0].text = "".join(t.text or "" for t in ts)
        for t in ts[1:]:
            r.remove(t)
    if ts:
        ts[0].set(XML_SPACE, "preserve")


def _is_italic(r: etree._Element) -> bool:
    rpr = r.find(W_RPR)
    i = rpr.find(qn("w:i")) if rpr is not None else None
    return i is not None and i.get(qn("w:val"), "true") not in ("0", "false", "off")


def build_spans(p: etree._Element) -> tuple[str, list[RunSpan]]:
    spans, parts, pos = [], [], 0
    for r in p.iter(W_R):
        if _excluded(r, p):
            continue
        text, simple = _run_text(r)
        if simple:
            _normalize_simple_run(r)
        spans.append(RunSpan(r, pos, pos + len(text), simple, italic=_is_italic(r)))
        parts.append(text)
        pos += len(text)
    return "".join(parts), spans


class Manuscript:
    def __init__(self, path: str):
        self.doc = Document(path)
        self.p_elements = list(self.doc.element.body.iter(W_P))
        self.texts: list[str] = []
        self.spans: list[list[RunSpan]] = []
        for p in self.p_elements:
            text, spans = build_spans(p)
            self.texts.append(text)
            self.spans.append(spans)
        ids = [int(v) for v in self.doc.element.xpath("//@w:id") if str(v).lstrip("-").isdigit()]
        self._next_rev_id = max(ids, default=0) + 1000

    def is_italic(self, para: int, start: int, end: int) -> bool:
        spans = [s for s in self.spans[para] if s.start < end and start < s.end]
        return bool(spans) and all(s.italic for s in spans)

    # ------------------------------------------------------------------- writing

    def _rev_id(self) -> str:
        self._next_rev_id += 1
        return str(self._next_rev_id)

    def _stamp(self, el: etree._Element, author: str, date: str) -> etree._Element:
        el.set(W_ID, self._rev_id())
        el.set(W_AUTHOR, author)
        el.set(W_DATE, date)
        return el

    @staticmethod
    def _split(spans: list[RunSpan], pos: int) -> None:
        for i, s in enumerate(spans):
            if s.start < pos < s.end:
                if not s.simple:
                    raise SkippedEdit("edit cuts through a run with tabs/fields/images")
                t = s.el.find(W_T)
                k = pos - s.start
                right = copy.deepcopy(s.el)
                for extra in right.findall(qn("w:lastRenderedPageBreak")):
                    right.remove(extra)
                right.find(W_T).text = t.text[k:]
                t.text = t.text[:k]
                s.el.addnext(right)
                spans.insert(i + 1, RunSpan(right, pos, s.end, True, italic=s.italic))
                s.end = pos
                return

    @staticmethod
    def _rpr_copy(span: RunSpan | None) -> etree._Element | None:
        if span is None:
            return None
        rpr = span.el.find(W_RPR)
        return copy.deepcopy(rpr) if rpr is not None else None

    def _new_run(self, text: str, rpr: etree._Element | None) -> etree._Element:
        r = etree.Element(W_R)
        if rpr is not None:
            for change in rpr.findall(qn("w:rPrChange")):
                rpr.remove(change)
            r.append(rpr)
        t = etree.SubElement(r, W_T)
        t.text = text
        t.set(XML_SPACE, "preserve")
        return r

    def _comment(self, first: etree._Element, last: etree._Element, text: str, author: str) -> None:
        comment = self.doc.comments.add_comment(text=text, author=author, initials="GC")
        cid = str(comment.comment_id)
        start = etree.Element(qn("w:commentRangeStart"))
        start.set(W_ID, cid)
        end = etree.Element(qn("w:commentRangeEnd"))
        end.set(W_ID, cid)
        ref_run = etree.Element(W_R)
        rpr = etree.SubElement(ref_run, W_RPR)
        etree.SubElement(rpr, qn("w:rStyle")).set(qn("w:val"), "CommentReference")
        etree.SubElement(ref_run, qn("w:commentReference")).set(W_ID, cid)
        first.addprevious(start)
        last.addnext(end)
        end.addnext(ref_run)

    def apply(self, edits: list[Edit], author: str,
              category_for: Callable[[str], Category | None] = lambda rule_id: None,
              ) -> tuple[list[Edit], list[tuple[Edit, str]]]:
        """Apply non-overlapping edits. Returns (applied, skipped_with_reason)."""
        self._category_for = category_for
        date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        applied, skipped = [], []
        # Descending order keeps earlier offsets valid while we split runs.
        for e in sorted(edits, key=lambda x: (x.para, x.start, x.end), reverse=True):
            try:
                self._apply_one(e, author, date)
                applied.append(e)
            except SkippedEdit as exc:
                skipped.append((e, str(exc)))
        applied.reverse()
        return applied, skipped

    def _apply_one(self, e: Edit, author: str, date: str) -> None:
        spans = self.spans[e.para]
        if self.texts[e.para][e.start:e.end] != e.original:
            raise SkippedEdit("original text not found at the given offsets")

        # Validate before mutating anything.
        for s in spans:
            if not s.simple and s.end > s.start and s.start < e.end and e.start < s.end:
                if not (e.start <= s.start and s.end <= e.end) or not (e.is_flag or e.format):
                    raise SkippedEdit("span touches a run with tabs/fields/images")
        self._split(spans, e.start)
        self._split(spans, e.end)
        covered = [s for s in spans if e.start <= s.start and s.end <= e.end
                   and s.end > s.start and not s.deleted]

        cat = self._category_for(e.rule_id)
        color = cat.color if cat else DEFAULT_HIGHLIGHT
        note = f"[{cat.name} · {e.rule_id}]" if cat else f"[{e.rule_id}]"
        note = f"{note} {e.reason}".strip()
        if e.source != "regex":
            note += f"  (fuente: {e.source}"
            note += f", confianza {e.confidence:.0%})" if e.source == "llm" else ")"

        if e.is_flag:
            self._format(covered, author, date, highlight=color)
            self._comment(covered[0].el, covered[-1].el, "REVISAR: " + note, author)
            return
        if e.format:
            if e.format != "italic":
                raise SkippedEdit(f"unknown format {e.format!r}")
            self._format(covered, author, date, italic=True, highlight=color)
            self._comment(covered[0].el, covered[-1].el, note, author)
            return

        rpr_src = covered[0] if covered else next(
            (s for s in reversed(spans) if s.end <= e.start and s.simple), None
        ) or next((s for s in spans if s.simple), None)

        # Deletions: each run gets its own <w:del> (runs may live under different parents).
        if covered:
            self._format(covered, author, date, highlight=color)
        dels = []
        for s in covered:
            t = s.el.find(W_T)
            t.tag = W_DELTEXT
            wrapper = self._stamp(etree.Element(W_DEL), author, date)
            s.el.addprevious(wrapper)
            wrapper.append(s.el)
            s.deleted = True
            dels.append(wrapper)

        ins = None
        if e.replacement:
            ins = self._stamp(etree.Element(W_INS), author, date)
            new_run = self._new_run(e.replacement, self._rpr_copy(rpr_src))
            _set_highlight(new_run, color)
            ins.append(new_run)
            if dels:
                dels[-1].addnext(ins)
            else:  # pure insertion at e.start
                before = next((s for s in reversed(spans) if s.end <= e.start and not s.deleted), None)
                if before is not None:
                    before.el.addnext(ins)
                else:
                    after = next((s for s in spans if s.start >= e.start), None)
                    if after is not None:
                        after.el.addprevious(ins)
                    else:
                        self.p_elements[e.para].append(ins)

        first = dels[0] if dels else ins
        last = ins if ins is not None else dels[-1]
        self._comment(first, last, note, author)

    def _format(self, covered: list[RunSpan], author: str, date: str,
                italic: bool = False, highlight: str | None = None) -> None:
        """Formatting change recorded as <w:rPrChange>, so rejecting it in Word undoes it."""
        if not covered:
            raise SkippedEdit("empty span cannot be formatted")
        for s in covered:
            old_rpr = copy.deepcopy(s.el.find(W_RPR)) if s.el.find(W_RPR) is not None else etree.Element(W_RPR)
            for change in old_rpr.findall(qn("w:rPrChange")):
                old_rpr.remove(change)
            if italic:
                Run(s.el, None).font.italic = True
                s.italic = True
            if highlight:
                _set_highlight(s.el, highlight)
            rpr = s.el.find(W_RPR)
            for change in rpr.findall(qn("w:rPrChange")):
                rpr.remove(change)
            change = self._stamp(etree.SubElement(rpr, qn("w:rPrChange")), author, date)
            change.append(old_rpr)

    def save(self, path: str) -> None:
        self.doc.save(path)
