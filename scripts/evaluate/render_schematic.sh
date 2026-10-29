#!/usr/bin/env bash
# Compile docs/figures/model_schematic.tex and render PNG + SVG next to it (used by README and the report).
set -euo pipefail
cd "$(dirname "$0")/../docs/figures"
../../.venv/bin/tectonic -X compile model_schematic.tex >/dev/null 2>&1 || ../../.venv/bin/tectonic -X compile model_schematic.tex
../../.venv/bin/python - <<'PY'
import fitz
doc = fitz.open("model_schematic.pdf"); page = doc[0]
page.get_pixmap(dpi=200).save("model_schematic.png")
open("model_schematic.svg", "w").write(page.get_svg_image())
print("rendered model_schematic.{pdf,png,svg}")
PY
