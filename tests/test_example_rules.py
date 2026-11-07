"""Deterministic example rules (no Ollama, no LanguageTool needed)."""
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

from grammarchecker import pipeline

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "rules" / "example_rules.yaml"

# (input paragraph, expected text after accepting all changes)
CASES = [
    ("- ¿Venís mañana? -preguntó Tomás.", "—¿Venís mañana? —preguntó Tomás."),   # OT-01, OT-06
    ("Esperó...", "Esperó…"),                                                     # OT-02
    ("Llegó  tarde , como siempre .", "Llegó tarde, como siempre."),              # OT-03, OT-04
    ('Dijo "basta" y se fue.', "Dijo «basta» y se fue."),                         # OT-05
    ("Me dijo: «Ana gritó «basta» y se fue».", "Me dijo: «Ana gritó “basta” y se fue»."),  # OT-07
    ("¡¡¡No puede ser!!!", "¡¡¡No puede ser!!!"),                                 # OT-09: flag only
    ("Lo decidió a priori y de facto.", "Lo decidió a priori y de facto."),       # OT-08: italics only
]
NAMES = ["Julián llegó.", "Julián se fue.", "Julian volvió."]                     # NM-01: flag only


def _run(tmp_path, paragraphs):
    d = Document()
    for text in paragraphs:
        d.add_paragraph(text)
    src, dst = tmp_path / "in.docx", tmp_path / "out.docx"
    d.save(src)
    rep = pipeline.run(str(src), str(dst), str(RULES),
                       pipeline.Settings(use_llm=False, use_languagetool=False))
    return rep, dst


def test_example_rules(tmp_path):
    from test_pipeline import _visible
    rep, dst = _run(tmp_path, [c[0] for c in CASES] + NAMES)
    assert not rep.skipped, rep.skipped

    # Accepting all changes gives the expected text; rejecting all gives back the original.
    accepted = _visible(dst, accept=True)
    rejected = _visible(dst, accept=False)
    for (src_text, expected), got, orig in zip(CASES, accepted, rejected):
        assert got == expected, f"\n  in:  {src_text}\n  got: {got}\n  exp: {expected}"
        assert orig == src_text

    # Latin phrases: a formatting change, the text stays the same
    doc = Document(str(dst))
    italic = [r.text for r in doc.paragraphs[len(CASES) - 1].runs if r.font.italic]
    assert italic == ["a priori", "de facto"]

    # Flags: highlighted and commented, text untouched
    by_rule = {e["rule_id"]: e for e in rep.applied}
    assert by_rule["OT-09"]["replacement"] is None and "«!»" in by_rule["OT-09"]["reason"]
    names = [e for e in rep.applied if e["rule_id"] == "NM-01"]
    assert len(names) == 1 and names[0]["original"] == "Julian" and names[0]["replacement"] is None
    assert "Julián" in names[0]["reason"]


def test_every_mark_is_highlighted_with_its_category_color(tmp_path):
    rep, dst = _run(tmp_path, ["Hola  mundo.", "¡¡¡No!!!", "Lo decidió a priori."] + NAMES)
    assert rep.colors["Nombres"] == "blue"

    body = Document(str(dst)).element.body
    paras = body.findall(qn("w:p"))
    expected = {0: "cyan", 1: "cyan", 2: "cyan", 5: "blue"}   # change, flag, italics, name flag
    for i, color in expected.items():
        p = paras[i]
        marked = ([r for d in p.iter(qn("w:del")) for r in d.iter(qn("w:r"))]
                  + [r for d in p.iter(qn("w:ins")) for r in d.iter(qn("w:r"))]
                  + [r for r in p.iter(qn("w:r")) if r.find(f"{qn('w:rPr')}/{qn('w:rPrChange')}") is not None])
        assert marked, f"paragraph {i} has no marks"
        for r in marked:
            hl = r.find(f"{qn('w:rPr')}/{qn('w:highlight')}")
            assert hl is not None and hl.get(qn("w:val")) == color, (i, color)
    # Highlight is a tracked change on deleted/flagged text: the pre-change formatting has none.
    for change in body.iter(qn("w:rPrChange")):
        assert change.find(f"{qn('w:rPr')}/{qn('w:highlight')}") is None
