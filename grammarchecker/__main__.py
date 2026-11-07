"""CLI:  python -m grammarchecker libro.docx [-r rules/my_rules.yaml]"""
from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from pathlib import Path

from .models import Rulebook
from .pipeline import Settings, run

DEFAULT_RULES = "rules/style_rules.yaml"
EXAMPLE_RULES = "rules/example_rules.yaml"


def resolve_rules(path: str | None) -> str:
    """An explicit -r must exist. Without -r, use the private rulebook if present, else the example."""
    if path is not None:
        if not Path(path).is_file():
            raise SystemExit(f"Rulebook not found: {path}")
        return path
    if Path(DEFAULT_RULES).is_file():
        return DEFAULT_RULES
    print(f"· {DEFAULT_RULES} not found, using {EXAMPLE_RULES}", flush=True)
    return EXAMPLE_RULES


def main() -> None:
    ap = argparse.ArgumentParser(prog="grammarchecker", description="Spelling and style corrector with Word tracked changes")
    ap.add_argument("input", help="manuscript .docx")
    ap.add_argument("-o", "--output", help="output .docx (default: <input>.corregido.docx)")
    ap.add_argument("-r", "--rules", help=f"rulebook YAML (default: {DEFAULT_RULES}, else {EXAMPLE_RULES})")
    ap.add_argument("-m", "--model", default=os.getenv("GRAMMARCHECKER_MODEL", "qwen3:14b"),
                    help="Ollama model (default: %(default)s)")
    ap.add_argument("--ollama", default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                    help="Ollama URL (default: %(default)s)")
    ap.add_argument("--languagetool", default=os.getenv("LT_URL", "http://localhost:8010"),
                    help="LanguageTool URL (default: %(default)s)")
    ap.add_argument("--chunk-chars", type=int, default=3500,
                    help="characters of text per LLM call (default: %(default)s)")
    ap.add_argument("--no-llm", action="store_true", help="regex rules + LanguageTool only, no LLM")
    ap.add_argument("--no-lt", action="store_true", help="skip LanguageTool")
    ap.add_argument("--no-profile", action="store_true", help="skip the book profile pass")
    ap.add_argument("--no-critic", action="store_true", help="skip the critic agent (not recommended)")
    ap.add_argument("-v", "--verbose", action="store_true", help="detailed logging")
    a = ap.parse_args()
    a.rules = resolve_rules(a.rules)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s")
    out = a.output or str(Path(a.input).with_suffix("")) + ".corregido.docx"
    s = Settings(ollama_url=a.ollama, model=a.model, languagetool_url=a.languagetool,
                 chunk_chars=a.chunk_chars, use_llm=not a.no_llm, use_languagetool=not a.no_lt,
                 use_profile=not a.no_profile, use_critic=not a.no_critic)

    rep = run(a.input, out, a.rules, s, progress=lambda m: print("·", m, flush=True))

    by_rule = Counter(e["rule_id"] for e in rep.applied)
    flags = sum(1 for e in rep.applied if e["replacement"] is None and not e["format"])
    print(f"\n✔ {out}")
    print(f"  {rep.paragraphs} paragraphs · {len(rep.applied)} marks ({flags} to review) · {rep.seconds}s")
    print(f"  proposed: {rep.proposed} · overlaps dropped: {rep.overlaps_dropped}")
    print(f"  rejected by validation: {len(rep.llm_rejected)} · by the critic: {len(rep.critic_rejected)}"
          f" · not applicable: {len(rep.skipped)}")
    if rep.llm_errors:
        print(f"  LLM chunks skipped after errors: {len(rep.llm_errors)}")
    rb = Rulebook.load(a.rules)
    by_cat = Counter()
    for rule, n in by_rule.items():
        cat = rb.category_for(rule)
        by_cat[(cat.name, cat.color) if cat else ("(no category)", "yellow")] += n
    for (name, color), n in by_cat.most_common():
        print(f"  ■ {name} [{color}]: {n}")
    for rule, n in by_rule.most_common():
        print(f"    {rule}: {n}")
    print(f"  details: {Path(out).with_suffix('.report.json')}")


if __name__ == "__main__":
    main()
