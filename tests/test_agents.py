"""LLM-side safety nets that don't need a model: guards, schema, book profile filtering."""
from pathlib import Path

from grammarchecker.agents import BookProfile, BookProfileAgent, CorrectorAgent, corrector_schema, find_span, passes_guard
from grammarchecker.models import Rulebook

RULES = Path(__file__).resolve().parents[1] / "rules" / "example_rules.yaml"


def test_guards_reject_edits_the_rule_does_not_allow():
    rb = Rulebook.load(RULES)
    cases = [
        ("PU-01", "Quería salir pero llovía", "Quería salir, pero llovía", True),
        ("PU-01", "pero llovía mucho", "pero, llovía mucho", False),          # comma after «pero»
        ("PU-01", "se quedó porque llovía", "se quedó, porque llovía", False),  # not pero/sino
        ("PU-02", "Vení Lucía", "Vení, Lucía", True),
        ("PU-02", "empezó a correr", "empezó, a correr", False),              # no vocative
        ("DI-01", "llegué. —Dijo Tomás", "llegué —dijo Tomás", True),
        ("DI-01", "llegué —, dijo Tomás", "llegué— dijo Tomás", False),
    ]
    for rule, orig, new, ok in cases:
        assert passes_guard(rb.get(rule).guard, orig, new) is ok, (rule, orig, new)


def test_corrector_schema_only_allows_listed_rules():
    schema = corrector_schema(["PU-01", "PU-02"])
    regla = schema["properties"]["correcciones"]["items"]["properties"]["regla"]
    assert regla["enum"] == ["PU-01", "PU-02"]


class _LoopingLLM:
    """First chunk works (with common nouns mixed in), second one hits the token cap."""

    def __init__(self):
        self.calls = 0

    def chat_json(self, system, user, schema, retries=2):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("LLM output hit the token cap")
        return {"nombres_propios": ["Lucía", "bruja", "Tomás", "Inventada", "de algún modo"],
                "notas_registro": ["Lucía usa voseo."]}


def test_book_profile_skips_failed_chunk_and_drops_common_nouns():
    texts = ["Lucía vio a la bruja y a Tomás."] * 3 + ["Otro capítulo."] * 3
    profile = BookProfileAgent(_LoopingLLM()).build(texts, max_chars=100)
    assert set(profile.names) == {"Lucía", "Tomás"}
    assert profile.register_notes == ["Lucía usa voseo."]


def test_find_span_tolerates_collapsed_spaces():
    text = "Compró pan     , azúcar      y caramelos."
    start, end = find_span(text, "pan , azúcar y")
    assert text[start:end] == "pan     , azúcar      y"
    assert find_span("uno y uno", "uno") is None  # ambiguous


class _FixedLLM:
    def __init__(self, items):
        self.items = items

    def chat_json(self, system, user, schema, retries=2):
        return {"correcciones": self.items}


def test_punctuation_rules_cannot_change_words():
    rb = Rulebook.load(RULES)
    texts = ["Estaba cansada, ella siguió caminando.", "Quería salir pero llovía."]
    llm = _FixedLLM([
        {"parrafo": 0, "original": "cansada, ella siguió", "reemplazo": "cansada, pero siguió",
         "regla": "PU-01", "motivo": "", "confianza": 0.9},     # inserts a word: only_punctuation
        {"parrafo": 1, "original": "salir pero llovía", "reemplazo": "salir, pero llovía",
         "regla": "PU-01", "motivo": "", "confianza": 0.9},
    ])
    edits, rejected = CorrectorAgent(llm, rb).propose(texts, [0, 1], BookProfile(), "puntuacion")
    assert [(e.original, e.replacement) for e in edits] == [("salir pero llovía", "salir, pero llovía")]
    assert len(rejected) == 1 and "solo toca la puntuación" in rejected[0]


def test_dialogue_rules_cannot_add_or_remove_dashes():
    rb = Rulebook.load(RULES)
    texts = ["—Ya llegué. —Dijo Tomás.", "—Estoy cansada, quiero dormir."]
    llm = _FixedLLM([
        {"parrafo": 0, "original": "llegué. —Dijo", "reemplazo": "llegué —dijo",
         "regla": "DI-01", "motivo": "", "confianza": 0.9},
        {"parrafo": 1, "original": "cansada, quiero", "reemplazo": "cansada —quiero",
         "regla": "DI-01", "motivo": "", "confianza": 0.9},     # adds a dash inside the speech
    ])
    edits, rejected = CorrectorAgent(llm, rb).propose(texts, [0, 1], BookProfile(), "dialogos")
    assert [(e.original, e.replacement) for e in edits] == [("llegué. —Dijo", "llegué —dijo")]
    assert len(rejected) == 1
