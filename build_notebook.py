"""Build and execute a notebook from a percent-format script: python notebooks/_build.py chaos_classification.py"""
import re, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
src = Path(sys.argv[1])
cells = []
for chunk in re.split(r"^# %%", src.read_text(), flags=re.M)[1:]:
    if chunk.startswith(" [markdown]"):
        body = chunk[len(" [markdown]"):].strip("\n")
        cells.append(nbformat.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())))
    else:
        cells.append(nbformat.v4.new_code_cell(chunk.strip("\n")))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "non_linear_analysis", "display_name": "Python (non_linear_analysis)", "language": "python"}
NotebookClient(nb, timeout=-1, kernel_name="non_linear_analysis", resources={"metadata": {"path": str(src.parent)}}).execute()
out = src.with_suffix(".ipynb")
nbformat.write(nb, out)
errs = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
print(out.name, len(cells), "cells; errors:", len(errs))
for e in errs:
    print(e["ename"], e["evalue"])
