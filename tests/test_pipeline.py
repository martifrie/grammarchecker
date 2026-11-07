"""End-to-end test with a fake LLM (no Ollama/LanguageTool needed).

    pip install pytest && pytest -q
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from docx import Document

from grammarchecker import pipeline
from grammarchecker.docx_engine import Manuscript
from grammarchecker.llm import Ollama

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "rules" / "example_rules.yaml"


def _sample(tmp_path) -> Path:
    path = tmp_path / "muestra.docx"
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_sample.py"), str(path)], check=True)
    return path


class FakeOllama(Ollama):
    """Answers like a decent model would; includes one hallucination and one bad edit."""

    def available(self):
        return True

    def chat_json(self, system, user, schema, retries=2):
        props = schema["properties"]
        if "nombres_propios" in props:
            return {"nombres_propios": ["Ramona", "Rosario", "Julián"],
                    "notas_registro": ["Ramona usa voseo rioplatense."]}
        if "correcciones" in props:
            para = {int(line.split("]")[0][2:]): line for line in user.splitlines()}

            def find(snippet):
                return next((k for k, v in para.items() if snippet in v), None)

            out = []
            for orig, new, rule, conf in [
                ("sólo", "solo", "GR-03", 0.95),
                ("ésta casa", "esta casa", "GR-03", 0.9),
                ("Tenía 23 años", "Tenía veintitrés años", "GR-06", 0.9),
                ("¿Vosotros venís", "¿Ustedes vienen", "GR-01", 0.95),
                ("Vos tienes", "Vos tenés", "GR-05", 0.9),
                ("Pienso de que", "Pienso que", "GR-02", 0.95),
                ("estaba vacío", "estaban vacías", "GR-04", 0.55),   # low confidence → flag
                ("Ramona llegó", "Ramona arribó", "GR-02", 0.9),      # critic must reject
                ("frase inventada", "otra", "GR-02", 0.9),            # not in text → rejected
            ]:
                k = find(orig)
                if orig == "frase inventada":
                    k = next(iter(para))  # hallucinated span pointed at a real paragraph
                if k is not None:
                    out.append({"parrafo": k, "original": orig, "reemplazo": new,
                                "regla": rule, "motivo": "test", "confianza": conf})
            return {"correcciones": out}
        if "veredictos" in props:
            items = json.loads(user)
            return {"veredictos": [{"id": i["id"], "aprobar": i["reemplazo"] != "Ramona arribó",
                                    "motivo": "cambia la voz" if i["reemplazo"] == "Ramona arribó" else "ok"}
                                   for i in items]}
        raise AssertionError("unexpected schema")


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "Ollama", FakeOllama)
    src = _sample(tmp_path)
    dst = tmp_path / "out.docx"
    rep = pipeline.run(str(src), str(dst), str(RULES), pipeline.Settings(use_languagetool=False))

    rules = {e["rule_id"] for e in rep.applied}
    assert {"OT-01", "OT-02", "OT-03", "OT-04", "OT-05", "GR-01", "GR-02", "GR-03", "GR-05", "GR-06"} <= rules
    assert any("arribó" in r for r in rep.critic_rejected)
    assert any("frase inventada" in r for r in rep.llm_rejected)
    flag = next(e for e in rep.applied if e["rule_id"] == "GR-04")
    assert flag["replacement"] is None

    xml = zipfile.ZipFile(dst).read("word/document.xml").decode()
    assert "<w:ins " in xml and "<w:del " in xml and "w:delText" in xml
    assert "commentReference" in xml and 'w:highlight w:val="yellow"' in xml

    # Accepting all changes must yield the corrected text; rejecting must give back the original.
    accepted = _visible(dst, accept=True)
    rejected = _visible(dst, accept=False)
    assert rejected == Manuscript(str(src)).texts
    joined = "\n".join(accepted)
    for expected in ["solo quería", "esta casa", "veintitrés", "—¿Ustedes vienen", "—Vos tenés que venir, dijo",
                     "Pienso que", "«Se vende»", "letras rojas", "vendieran…", "runs…"]:
        assert expected in joined, expected
    assert "\tTexto con tabulación que no se toca." in joined
    assert "Ramona llegó" in joined


def _visible(path, accept: bool) -> list[str]:
    """Text of each paragraph after accepting or rejecting every tracked change."""
    import copy
    from docx.oxml.ns import qn
    doc = Document(str(path))
    body = copy.deepcopy(doc.element.body)
    drop, keep = (qn("w:del"), qn("w:ins")) if accept else (qn("w:ins"), qn("w:del"))
    for el in list(body.iter(drop)):
        if el.getparent() is not None and el.getparent().tag != qn("w:rPr"):
            el.getparent().remove(el)
    for el in list(body.iter(keep)):
        if el.getparent() is not None and el.getparent().tag != qn("w:rPr"):
            for t in el.iter(qn("w:delText")):
                t.tag = qn("w:t")
            idx = el.getparent().index(el)
            for child in list(el):
                el.getparent().insert(idx, child)
                idx += 1
            el.getparent().remove(el)
    from grammarchecker.docx_engine import build_spans
    return [build_spans(p)[0] for p in body.iter(qn("w:p"))]
