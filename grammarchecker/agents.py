"""LLM agents. Prompts are in Spanish because the manuscripts are

- BookProfileAgent: builds a per-book profile (names, places, register) for consistency
- CorrectorAgent: proposes minimal edits that cite a rule id. Never rewrites paragraphs
- CriticAgent:    rejects edits that aren't backed by the cited rule or change meaning/voice
"""
from __future__ import annotations

import copy
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from .llm import Ollama
from .models import Edit, Rulebook

log = logging.getLogger(__name__)


@dataclass
class BookProfile:
    names: Counter = field(default_factory=Counter)
    register_notes: list[str] = field(default_factory=list)

    def summary(self, top: int = 60) -> str:
        names = ", ".join(n for n, _ in self.names.most_common(top)) or "(ninguno)"
        notes = "\n".join(f"- {n}" for n in self.register_notes[:20]) or "- (sin notas)"
        return f"Nombres propios del libro (no corregir su grafía): {names}\nRegistro y voz:\n{notes}"


def chunks(texts: list[str], max_chars: int) -> list[list[int]]:
    out, cur, size = [], [], 0
    for i, t in enumerate(texts):
        if not t.strip():
            continue
        if cur and size + len(t) > max_chars:
            out.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += len(t)
    if cur:
        out.append(cur)
    return out


def numbered(texts: list[str], idx: list[int]) -> str:
    return "\n".join(f"[P{i}] {texts[i]}" for i in idx)


# --------------------------------------------------------------------------- book profile

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        # maxItems keeps the model from looping inside the list (seen with qwen3 at temperature 0).
        "nombres_propios": {"type": "array", "items": {"type": "string"}, "maxItems": 40},
        "notas_registro": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
    },
    "required": ["nombres_propios", "notas_registro"],
}

PROFILE_SYSTEM = """Sos un editor que prepara el perfil de un libro en español latinoamericano.
Del fragmento, extraé:
- nombres_propios: SOLO nombres propios (con mayúscula inicial): personajes, lugares, instituciones e
  inventos del autor, tal como están escritos, cada uno una sola vez. Nunca sustantivos comunes
  (bruja, padre, iglesia, mercado) ni frases.
- notas_registro: observaciones breves sobre cómo habla cada personaje (vos/tú/usted, modismos regionales).
No corrijas nada. Respondé solo en JSON."""


def _is_proper_name(name: str, texts: list[str], idx: list[int]) -> bool:
    """Capitalized, short, and actually written that way in the chunk (drops common nouns)."""
    return (bool(name) and name[0].isupper() and len(name.split()) <= 4
            and any(name in texts[i] for i in idx))


class BookProfileAgent:
    def __init__(self, llm: Ollama):
        self.llm = llm

    def build(self, texts: list[str], max_chars: int) -> BookProfile:
        profile = BookProfile()
        for idx in chunks(texts, max_chars):
            try:
                data = self.llm.chat_json(PROFILE_SYSTEM, numbered(texts, idx), PROFILE_SCHEMA)
            except RuntimeError as e:
                log.warning("Book profile: skipping chunk (P%d-P%d): %s", idx[0], idx[-1], e)
                continue
            profile.names.update(n for n in (x.strip() for x in data.get("nombres_propios", []))
                               if _is_proper_name(n, texts, idx))
            for note in data.get("notas_registro", []):
                if note not in profile.register_notes:
                    profile.register_notes.append(note)
        return profile


# --------------------------------------------------------------------------- corrector

CORRECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "correcciones": {
            "type": "array",
            "maxItems": 20,  # see PROFILE_SCHEMA: stops runaway lists
            "items": {
                "type": "object",
                "properties": {
                    "parrafo": {"type": "integer"},
                    "original": {"type": "string"},
                    "reemplazo": {"type": "string"},
                    "regla": {"type": "string"},
                    "motivo": {"type": "string"},
                    "confianza": {"type": "number"},
                },
                "required": ["parrafo", "original", "reemplazo", "regla", "motivo", "confianza"],
            },
        }
    },
    "required": ["correcciones"],
}


def _examples(r) -> str:
    ex = r.examples
    if not ex:
        return ""
    if isinstance(ex, dict):
        ex = [ex]
    parts = [f"mal: «{e.get('wrong', '')}» → bien: «{e.get('right', '')}»" for e in ex]
    return " Ejemplos: " + "; ".join(parts)


def llm_rules(rulebook: Rulebook, group: str | None = None, ids: set[str] | None = None):
    return [r for r in rulebook.by_type("llm")
            if (group is None or r.group == group) and (ids is None or r.id in ids)]


def rules_block(rules) -> str:
    lines = []
    for r in rules:
        tag = " [SOLO SEÑALAR: proponé el cambio igual, se mostrará como sugerencia]" if r.mode == "flag" else ""
        lines.append(f"- {r.id}: {r.description}{tag}{_examples(r)}")
    return "\n".join(lines)


def groups(rulebook: Rulebook) -> list[str]:
    seen = []
    for r in rulebook.by_type("llm"):
        if r.group not in seen:
            seen.append(r.group)
    return seen


CORRECTOR_SYSTEM = """Sos corrector de ortografía. El texto está en español latinoamericano.
Aplicá ÚNICAMENTE estas reglas:
{rules}

{profile}

Instrucciones estrictas:
1. Corregí solo lo que viole una de las reglas listadas. No mejores el estilo, no cambies la voz del autor.
2. "original" debe ser un fragmento EXACTO y CORTO del párrafo (copiado carácter por carácter), con las
   palabras mínimas para que sea único dentro del párrafo. "reemplazo" es el mismo fragmento ya corregido.
3. "parrafo" es el número que aparece en [P#].
4. "regla" es el id exacto de una de las reglas listadas arriba. Si un error no corresponde a ninguna
   de ellas, no lo incluyas.
5. "confianza" entre 0 y 1. Si dudás, usá menos de 0.6.
6. Si no hay errores, devolvé una lista vacía.
Respondé solo en JSON."""


def corrector_schema(rule_ids: list[str]) -> dict:
    """CORRECTOR_SCHEMA with `regla` restricted to the rules of the current group."""
    schema = copy.deepcopy(CORRECTOR_SCHEMA)
    schema["properties"]["correcciones"]["items"]["properties"]["regla"]["enum"] = rule_ids
    return schema


def changed_region(original: str, replacement: str) -> tuple[int, int]:
    """[start, end) of the part of `replacement` that differs from `original`."""
    p = 0
    while p < min(len(original), len(replacement)) and original[p] == replacement[p]:
        p += 1
    s = 0
    while (s < min(len(original), len(replacement)) - p
           and original[-1 - s] == replacement[-1 - s]):
        s += 1
    return p, len(replacement) - s


def find_span(text: str, fragment: str) -> tuple[int, int] | None:
    """Unique position of `fragment` in `text`; runs of spaces may differ (models collapse them)."""
    pos = text.find(fragment)
    if pos >= 0:
        return (pos, pos + len(fragment)) if text.find(fragment, pos + 1) < 0 else None
    words = fragment.split()
    if not words:
        return None
    matches = list(re.finditer(r"[ \t]+".join(map(re.escape, words)), text))
    return (matches[0].start(), matches[0].end()) if len(matches) == 1 else None


def _letters(s: str) -> str:
    return re.sub(r"[\W_]+", "", s)


def passes_guard(guard: str, original: str, replacement: str) -> bool:
    """The rule's `guard` regex must match the replacement where it changed.

    Catches edits that cite a rule but don't do what the rule says (e.g. a comma *after*
    «pero» cited as PUN-11, which only allows a comma *before* it).
    """
    start, end = changed_region(original, replacement)
    return any(m.start() <= end and m.end() >= start for m in re.finditer(guard, replacement))


class CorrectorAgent:
    def __init__(self, llm: Ollama, rulebook: Rulebook):
        self.llm, self.rulebook = llm, rulebook

    def propose(self, texts: list[str], idx: list[int], profile: BookProfile,
                group: str | None = None) -> tuple[list[Edit], list[str]]:
        rules = llm_rules(self.rulebook, group)
        notes = f"\nCriterios generales:\n{self.rulebook.notes}\n" if self.rulebook.notes else ""
        system = CORRECTOR_SYSTEM.format(rules=rules_block(rules) + notes, profile=profile.summary())
        data = self.llm.chat_json(system, numbered(texts, idx), corrector_schema([r.id for r in rules]))
        edits, rejected = [], []
        valid_ids, allowed = {r.id for r in rules}, set(idx)
        guards = {r.id: r.guard for r in rules if r.guard}
        preserve = {r.id: r.preserve for r in rules if r.preserve}
        punct_only = {r.id for r in rules if r.only_punctuation}
        for c in data.get("correcciones", []):
            para, orig, new = c.get("parrafo"), c.get("original", ""), c.get("reemplazo", "")
            if para not in allowed:
                rejected.append(f"P{para}: párrafo fuera del fragmento")
                continue
            if c.get("regla") not in valid_ids:
                rejected.append(f"P{para}: regla inexistente {c.get('regla')!r}")
                continue
            if not orig or orig == new:
                continue
            guard = guards.get(c["regla"])
            if guard and not passes_guard(guard, orig, new):
                rejected.append(f"P{para}: «{orig}» → «{new}» no es lo que pide {c['regla']}")
                continue
            if any(orig.count(ch) != new.count(ch) for ch in preserve.get(c["regla"], "")):
                rejected.append(f"P{para}: «{orig}» → «{new}» agrega o quita signos que {c['regla']} no toca")
                continue
            if c["regla"] in punct_only and _letters(orig) != _letters(new):
                rejected.append(f"P{para}: «{orig}» → «{new}» cambia palabras y {c['regla']} solo toca la puntuación")
                continue
            text = texts[para]
            span = find_span(text, orig)
            if span is None:
                why = "es ambiguo (aparece más de una vez)" if orig in text else "no existe literalmente en el párrafo"
                rejected.append(f"P{para}: «{orig}» {why}")
                continue
            pos, end = span
            orig = text[pos:end]
            conf = float(c.get("confianza", 0.5))
            edits.append(Edit(para, pos, end, orig, new, c["regla"],
                              c.get("motivo", ""), "llm", max(0.0, min(conf, 1.0))))
        return edits, rejected


# --------------------------------------------------------------------------- critic

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "veredictos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "aprobar": {"type": "boolean"},
                    "motivo": {"type": "string"},
                },
                "required": ["id", "aprobar", "motivo"],
            },
        }
    },
    "required": ["veredictos"],
}

CRITIC_SYSTEM = """Sos el revisor final de ortografía. Otro corrector propuso cambios.
Reglas vigentes:
{rules}

Aprobá un cambio SOLO si se cumplen las tres condiciones:
a) el texto original realmente viola la regla citada;
b) el reemplazo la corrige sin introducir errores nuevos;
c) no cambia el significado, el registro (voseo, regionalismos) ni la voz del autor.
Ante la duda, rechazá. Devolvé un veredicto por cada id. Respondé solo en JSON."""


class CriticAgent:
    def __init__(self, llm: Ollama, rulebook: Rulebook):
        self.llm, self.rulebook = llm, rulebook

    def review(self, texts: list[str], edits: list[Edit]) -> tuple[list[Edit], list[str]]:
        if not edits:
            return [], []
        rules = rules_block(llm_rules(self.rulebook, ids={e.rule_id for e in edits}))
        items = []
        for e in edits:
            items.append({
                "id": e.id, "parrafo_completo": texts[e.para], "original": e.original,
                "reemplazo": e.replacement, "regla": e.rule_id, "motivo": e.reason,
            })
        data = self.llm.chat_json(CRITIC_SYSTEM.format(rules=rules),
                                  json.dumps(items, ensure_ascii=False, indent=1), CRITIC_SCHEMA)
        verdicts = {v["id"]: v for v in data.get("veredictos", [])}
        kept, dropped = [], []
        for e in edits:
            v = verdicts.get(e.id)
            if v and v.get("aprobar"):
                kept.append(e)
            else:
                why = v.get("motivo", "") if v else "sin veredicto"
                dropped.append(f"{e.id} [{e.rule_id}] «{e.original}» → «{e.replacement}»: {why}")
        return kept, dropped
