"""Sphinx config: the guide pages are MyST Markdown (with $...$ math); api.md is autodoc over src/, importing the
package with its real dependencies (uv run --group docs sphinx-build -b html docs docs/_build/html)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

project = "shared-delta-moe"
copyright = "2026, Anke Tang"
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon", "sphinx.ext.mathjax", "sphinx.ext.viewcode",
              "sphinxcontrib.mermaid"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_fence_as_directive = ["mermaid"]  # ```mermaid blocks render as diagrams (also on GitHub)
mermaid_init_config = {"startOnLoad": False, "flowchart": {"useMaxWidth": False},  # natural size, not stretched
                       "deterministicIds": True}  # else ids are Date.now(): two diagrams in one ms collide
myst_heading_anchors = 3
toc_object_entries_show_parents = "hide"  # contents list: "forward_param", not "DeltaMoE.forward_param"
autodoc_member_order = "bysource"
autosummary_generate = True  # api.md: one page per module under docs/_api/ (generated, git-ignored)
templates_path = ["_templates"]
autodoc_typehints = "description"
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "shared-delta-moe"
html_static_path = ["_static"]
html_css_files = ["compact.css"]


def _skip_namedtuple_fields(app, what, name, obj, skip, options):
    """NamedTuple fields get the docstring "Alias for field number N"; the class docstring already lists them
    (``:ivar:``), so documenting them again would duplicate every field."""
    return True if (getattr(obj, "__doc__", None) or "").startswith("Alias for field number") else None


def setup(app):
    app.connect("autodoc-skip-member", _skip_namedtuple_fields)
