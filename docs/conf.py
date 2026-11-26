"""Sphinx config: the guide pages are MyST Markdown (with $...$ math); api.md is autodoc over src/, importing the
package with its real dependencies (uv run --group docs sphinx-build -b html docs docs/_build/html)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

project = "shared-delta-moe"
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax", "sphinx.ext.viewcode"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3
autodoc_member_order = "bysource"
autodoc_typehints = "description"
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "shared-delta-moe"
