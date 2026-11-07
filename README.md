# GrammarChecker

A proofreading pipeline for Spanish (Latin American) manuscripts. It takes a `.docx` and returns **the same `.docx` with Word tracked changes**: every correction shows up as a deletion/insertion with a comment naming the rule that was applied, and every mark is **highlighted in the color of its rule category**. Doubtful cases are not changed: they are highlighted with a «REVISAR» (review) comment so a human decides.

Everything runs locally and for free: Ollama (LLM), LanguageTool (spelling/grammar) and Python.

## How it works

```
.docx ─► paragraphs ─┬─► regex rules (typography, dialogue dashes, spacing)   deterministic
                     ├─► LanguageTool (spelling, grammar)                      deterministic
                     └─► LLM agents (Ollama)
                           1. Book profile: proper names + register (vos/tú/usted)
                           2. Corrector: minimal edits, each citing one rule id
                           3. Critic: rejects edits the cited rule doesn't justify
        ─► validation (fragment exists, rule exists, guards, no overlaps)
        ─► .docx with <w:del>/<w:ins> + comments + category highlights + JSON report
```

The agents never rewrite paragraphs. They return `{original, replacement, rule, reason, confidence}` through a JSON schema, and the code then checks every proposal before it can touch the document:

- `original` must exist in the paragraph exactly once (runs of spaces may differ, since models collapse them).
- `rule` is restricted by the schema to the ids of the rule group being checked, so the model can't invent rules.
- Rules can declare a `guard` (a regex the corrected text must match where it changed), `preserve` (characters whose count may not change, e.g. dialogue dashes) and `only_punctuation` (the edit may not change any letter). For example, the comma rule for «pero» only accepts a comma *before* «pero», never after it.
- Edits below 0.6 confidence, or from rules marked `mode: flag`, become suggestions: highlighted and commented, text untouched.

When two sources propose overlapping edits, the winner is regex > LanguageTool > LLM.

## Setup

```bash
docker compose up -d                      # LanguageTool (+ Ollama, if you don't run it natively)
ollama pull qwen3:14b                     # ~9 GB; on smaller machines: qwen3:8b
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m grammarchecker manuscript.docx     # rules/style_rules.yaml, else rules/example_rules.yaml
# → manuscript.corregido.docx + manuscript.corregido.report.json
```

Useful options:

| Flag | Purpose |
|---|---|
| `-r rules/other.yaml` | use a different rulebook |
| `-m qwen3:8b` | lighter model |
| `--no-llm` | regex + LanguageTool only (seconds; good for iterating on rules) |
| `--no-lt` | skip LanguageTool |
| `--no-profile` | skip the book profile pass |
| `--chunk-chars 2500` | smaller chunks if the model loses track |

The input file is never modified; output always goes to a new file.

## Highlight colors

Every mark (text change, italics change or «REVISAR» suggestion) is highlighted with the color of its rule's category, and its comment starts with the category name, e.g. `[Puntuación · PUN-12] …`. Categories and colors are defined under `categories` in the rulebook (Word's highlight palette):

| Category | Color | Rule ids |
|---|---|---|
| Cursivas, redondas y comillas (italics & quotes) | turquoise | COM, CUR |
| Puntuación (punctuation) | pink | PUN, ACL |
| Diálogos (dialogue) | bright green | DIA, RAY |
| Tildación (accents) | dark yellow | TIL |
| Mayúsculas y minúsculas (capitalization) | teal | MAY |
| Otras cuestiones gramaticales (other grammar) | violet | PRE, CUE |
| Recomendaciones (name consistency) | blue | NOM |
| Cuestiones de estilo (style) | yellow | EST |
| Ortografía y gramática (spelling & grammar) | red | ORT, LanguageTool |
| Errores de tipeo (typos) | light gray | TIP |

Inserted text carries the highlight directly. On deleted text and suggestions the highlight is a *tracked* formatting change, so "Reject all" in Word returns the exact original. After "Accept all", the new text stays highlighted; clear it with Select all → Highlight → No color.

## Rulebooks

Rules live in a YAML rulebook passed with `-r`. Without `-r`, the corrector uses `rules/style_rules.yaml` if it exists (a private rulebook, not part of this repository) and otherwise falls back to `rules/example_rules.yaml`: 20 generic rules that show every feature and are used by the tests. Rule types:

- `regex`: `pattern` + `replacement`, or `words` (word → correction map). Fast, exact, no hallucinations.
- `quotes`: quote nesting levels (« » → “ ” → ‘ ’).
- `consistency`: proper names spelled more than one way (Julián / Julian).
- `llm`: `description` + `examples`, sent to the model in `group`s so each prompt stays short. The corrector may only cite these ids; the critic checks the citation.

Per-rule options: `mode: flag` (suggest only), `format: italic` (tracked formatting change), `when` / `unless` (apply only to dialogue paragraphs, for example), `require_italic`, `transform: lower`, and for LLM rules `guard`, `preserve`, `only_punctuation`.

## Tests

```bash
.venv/bin/pip install pytest && .venv/bin/python -m pytest -q
```

- `test_pipeline.py`: end-to-end with a fake LLM (no Ollama needed), including a hallucinated fragment and a voice-changing edit that must be rejected.
- `test_example_rules.py`: cases for every deterministic example rule; accepting all changes must give the expected text and rejecting them the exact original; every mark must carry its category's highlight.
- `test_agents.py`: LLM safety nets: guards, schema-restricted rule ids, punctuation-only and dash-preserving rules, whitespace-tolerant matching, and a book profile pass that survives a failing chunk.

## Performance

On an Apple M4 Pro (48 GB) with `qwen3:14b`, a 5,400-word sample takes about 39 minutes with the LLM (40 corrector calls + 40 critic calls); a 35,000-word book about 4 hours. With `--no-llm` the whole book takes under a second.

## Current limitations

- Edits that cut through a run containing tabs, fields or images are skipped (listed under `skipped` in the report).
- Only the document body is checked: footnotes and headers are not yet.
- The book profile is built per run; it isn't saved between books.
- Not yet automated (they need to know what is already in italics or the book's genre): titles of works in italics, foreign words in italics (realism) or unmarked (fantasy), and names that take no emphasis.

## License

MIT. See [LICENSE](LICENSE).
