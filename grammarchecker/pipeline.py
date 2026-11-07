"""Orchestrates: read → regex → LanguageTool → book profile → corrector → critic → resolve → write"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agents import BookProfile, BookProfileAgent, CorrectorAgent, CriticAgent, chunks, groups
from .checkers import LanguageTool, consistency_edits, quote_edits, regex_edits
from .docx_engine import Manuscript
from .llm import Ollama
from .models import SOURCE_PRIORITY, Edit, Rulebook

log = logging.getLogger(__name__)


@dataclass
class Settings:
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen3:14b"
    languagetool_url: str = "http://localhost:8010"
    chunk_chars: int = 3500          # text per LLM call (~1k tokens)
    flag_below: float = 0.6          # LLM edits under this confidence become comments only
    use_languagetool: bool = True
    use_llm: bool = True
    use_profile: bool = True
    use_critic: bool = True


@dataclass
class Report:
    paragraphs: int = 0
    proposed: dict = field(default_factory=dict)
    applied: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    llm_rejected: list = field(default_factory=list)
    critic_rejected: list = field(default_factory=list)
    overlaps_dropped: int = 0
    llm_errors: list = field(default_factory=list)
    colors: dict = field(default_factory=dict)  # category name -> Word highlight color
    seconds: float = 0.0


def resolve_overlaps(edits: list[Edit]) -> tuple[list[Edit], int]:
    # Higher source priority first; then real fixes over flags; then confidence; then the wider
    # edit (a context-aware rule like " ―" → "—" subsumes a generic one like "―" → "—")
    ranked = sorted(edits, key=lambda e: (SOURCE_PRIORITY[e.source], not e.is_flag, e.confidence,
                                          e.end - e.start), reverse=True)
    kept: list[Edit] = []
    for e in ranked:
        if not any(e.overlaps(k) for k in kept):
            kept.append(e)
    return kept, len(edits) - len(kept)


def run(src: str, dst: str, rulebook_path: str, settings: Settings,
        progress=lambda msg: None) -> Report:
    t0 = time.time()
    rb = Rulebook.load(rulebook_path)
    ms = Manuscript(src)
    texts = ms.texts
    rep = Report(paragraphs=sum(1 for t in texts if t.strip()))
    all_edits: list[Edit] = []

    progress("Applying deterministic rules…")
    rx = regex_edits(texts, rb, ms)
    for rule in rb.by_type("quotes"):
        rx += quote_edits(texts, rule)
    for rule in rb.by_type("consistency"):
        rx += consistency_edits(texts, rule)
    rep.proposed["regex"] = len(rx)
    all_edits += rx

    profile = BookProfile()
    llm = None
    if settings.use_llm:
        llm = Ollama(settings.ollama_url, settings.model)
        if not llm.available():
            raise SystemExit(f"Ollama is not responding at {settings.ollama_url} or the model "
                             f"{settings.model} is missing. Use --no-llm to run without agents.")
        if settings.use_profile:
            progress("Building the book profile…")
            profile = BookProfileAgent(llm).build(texts, settings.chunk_chars * 3)

    if settings.use_languagetool:
        lt = LanguageTool(settings.languagetool_url, rb.languagetool_lang)
        if lt.available():
            progress("LanguageTool…")
            ignore = set(profile.names)
            ignore |= {w for n in profile.names for w in n.split()}
            lte = lt.edits(texts, ignore)
            rep.proposed["languagetool"] = len(lte)
            all_edits += lte
        else:
            log.warning("LanguageTool not available at %s; skipping it.", settings.languagetool_url)

    if llm is not None and rb.by_type("llm"):
        corrector = CorrectorAgent(llm, rb)
        critic = CriticAgent(llm, rb) if settings.use_critic else None
        parts = chunks(texts, settings.chunk_chars)
        rule_groups = groups(rb)
        llm_count, total, n = 0, len(parts) * len(rule_groups), 0
        for g in rule_groups:
            for idx in parts:
                n += 1
                progress(f"Agents [{g}]: chunk {n}/{total}")
                try:
                    proposed, rejected = corrector.propose(texts, idx, profile, g)
                except RuntimeError as exc:
                    log.warning("Corrector [%s] chunk %d skipped: %s", g, n, exc)
                    rep.llm_errors.append(f"corrector [{g}] P{idx[0]}-P{idx[-1]}: {exc}")
                    continue
                rep.llm_rejected += rejected
                for k, e in enumerate(proposed):
                    e.id = f"c{n}-{k}"
                llm_count += len(proposed)
                if critic:
                    try:
                        proposed, dropped = critic.review(texts, proposed)
                    except RuntimeError as exc:
                        # Without a verdict nothing is applied as a fix: everything becomes a flag.
                        log.warning("Critic [%s] chunk %d failed: %s", g, n, exc)
                        rep.llm_errors.append(f"crítico [{g}] P{idx[0]}-P{idx[-1]}: {exc}")
                        dropped = []
                        for e in proposed:
                            e.to_flag("Sin revisión del crítico. ")
                    rep.critic_rejected += dropped
                for e in proposed:
                    rule = rb.get(e.rule_id)
                    if rule and rule.mode == "flag":
                        e.to_flag()
                    elif e.confidence < settings.flag_below:
                        e.to_flag("Confianza baja. ")
                all_edits += proposed
        rep.proposed["llm"] = llm_count

    progress("Writing tracked changes…")
    final, rep.overlaps_dropped = resolve_overlaps(all_edits)
    applied, skipped = ms.apply(final, rb.author_tag, rb.category_for)
    rep.colors = {c.name: c.color for c in rb.categories}
    ms.save(dst)

    rep.applied = [asdict(e) for e in applied]
    rep.skipped = [{**asdict(e), "why": why} for e, why in skipped]
    rep.seconds = round(time.time() - t0, 1)
    Path(dst).with_suffix(".report.json").write_text(
        json.dumps(asdict(rep), ensure_ascii=False, indent=2), encoding="utf-8")
    return rep
