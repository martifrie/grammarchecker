"""Generate a small Spanish (LATAM) test manuscript with deliberate errors."""
import sys

from docx import Document

out = sys.argv[1] if len(sys.argv) > 1 else "tests/muestra.docx"
d = Document()
d.add_heading("Capítulo 1: La casa de Ramona", level=1)

p = d.add_paragraph("Ramona llegó a Rosario un martes. Tenía 23 años y ")
p.add_run("sólo").bold = True
p.add_run(" quería ver ésta casa antes de que la vendieran...")

d.add_paragraph("- ¿Vosotros venís mañana? -preguntó Julián.")
d.add_paragraph("-Vos tienes que venir , dijo ella. Pienso de que es importante.")
d.add_paragraph('El cartel decía "Se vende" en letras  rojas.')

p = d.add_paragraph("Las casas del barrio estaba vacío.")
p.add_run("\tTexto con tabulación que no se toca.")

p = d.add_paragraph("Una frase con ")
p.add_run("cursiva partida").italic = True
p.add_run(" en varios runs...")
d.save(out)
print(out)
