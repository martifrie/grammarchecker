"""Shared data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

Source = Literal["regex", "languagetool", "llm"]

# When two edits overlap, the one with the higher priority wins
SOURCE_PRIORITY = {"regex": 3, "languagetool": 2, "llm": 1}

RuleType = Literal["regex", "llm", "quotes", "consistency"]


@dataclass
class Rule:
    """One rule of the rulebook.

    type:
      regex        pattern + replacement, or `words` (map of word -> correct word)
      llm          judged by the corrector agent, verified by the critic
      quotes       built-in checker for quote nesting levels (`levels`)
      consistency  built-in checker for proper-name spelling variants
    mode:   fix  -> tracked change      flag -> highlight + comment only
    format: e.g. "italic": apply formatting instead of changing text
    transform: "lower"/"upper": replacement is the matched text with its case changed
    require_italic: only match text that is already in italics (e.g. double emphasis)
    span_group: for flags, anchor the comment on this regex group instead of the whole match
    when / unless: regex the whole paragraph must (not) match for the rule to apply
    group:  LLM rules are sent to the corrector one group at a time (smaller prompts)
    guard:  for LLM rules, regex the corrected fragment must match where it changed, or the
            edit is rejected (keeps the model from citing a rule for an edit it doesn't allow)
    preserve: for LLM rules, characters whose count the edit must not change (e.g. "—")
    only_punctuation: for LLM rules, the edit may not add, remove or change any letter or digit
    """

    id: str
    type: RuleType
    description: str
    mode: Literal["fix", "flag"] = "fix"
    pattern: str | None = None
    replacement: str | None = None
    words: dict[str, str] | None = None
    format: Literal["italic"] | None = None
    transform: Literal["lower", "upper"] | None = None
    require_italic: bool = False
    span_group: int = 0
    when: str | None = None
    unless: str | None = None
    ignore_case: bool = False
    levels: list[list[str]] | None = None
    group: str = "general"
    examples: dict | list | None = None
    guard: str | None = None
    preserve: str = ""
    only_punctuation: bool = False


HIGHLIGHT_COLORS = {
    "yellow", "green", "cyan", "magenta", "blue", "red", "darkBlue", "darkCyan", "darkGreen",
    "darkMagenta", "darkRed", "darkYellow", "lightGray", "darkGray", "black",
}


@dataclass
class Category:
    """A section of the rulebook. Every mark from its rules is highlighted in `color`."""

    name: str
    color: str
    prefixes: list[str]

    def __post_init__(self) -> None:
        if self.color not in HIGHLIGHT_COLORS:
            raise ValueError(f"Highlight color invalid for «{self.name}»: {self.color!r}. "
                             f"Options: {', '.join(sorted(HIGHLIGHT_COLORS))}")


@dataclass
class Rulebook:
    name: str
    author_tag: str
    languagetool_lang: str
    rules: list[Rule]
    notes: str = ""
    categories: list[Category] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "Rulebook":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data.get("name", ""),
            author_tag=data.get("author_tag", "GrammarChecker"),
            languagetool_lang=data.get("languagetool_lang", "es"),
            rules=[Rule(**r) for r in data["rules"]],
            notes=data.get("notes", ""),
            categories=[Category(**c) for c in data.get("categories", [])],
        )

    def category_for(self, rule_id: str) -> Category | None:
        """Longest matching prefix wins (so "PUN-02" can differ from "PUN")"""
        best = None
        for c in self.categories:
            for prefix in c.prefixes:
                if rule_id.startswith(prefix) and (best is None or len(prefix) > best[0]):
                    best = (len(prefix), c)
        return best[1] if best else None

    def by_type(self, kind: str) -> list[Rule]:
        return [r for r in self.rules if r.type == kind]

    def get(self, rule_id: str) -> Rule | None:
        return next((r for r in self.rules if r.id == rule_id), None)

    def ids(self) -> set[str]:
        return {r.id for r in self.rules}


@dataclass
class Edit:
    """A correction anchored to a character span of one paragraph.

    replacement=None and format=None means "flag only": the span gets a comment +
    highlight but the text is not changed (for doubtful cases).
    format="italic" applies formatting as a tracked change instead of changing text.
    """

    para: int
    start: int
    end: int
    original: str
    replacement: str | None
    rule_id: str
    reason: str
    source: Source
    confidence: float = 1.0
    id: str = field(default="")
    format: str | None = None

    @property
    def is_flag(self) -> bool:
        return self.replacement is None and self.format is None

    def to_flag(self, why: str = "") -> None:
        if self.replacement is not None:
            self.reason = f"Sugerencia: «{self.replacement}». {why}{self.reason}".strip()
        self.replacement = None
        self.format = None

    def overlaps(self, other: "Edit") -> bool:
        if self.para != other.para:
            return False
        a_ins, b_ins = self.start == self.end, other.start == other.end
        if a_ins and b_ins:
            return self.start == other.start
        # An insertion only conflicts with a span it falls strictly inside of
        if a_ins:
            return other.start < self.start < other.end
        if b_ins:
            return self.start < other.start < self.end
        return self.start < other.end and other.start < self.end
