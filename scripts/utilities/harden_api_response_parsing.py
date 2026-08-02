#!/usr/bin/env python3

"""Patch API experiment runners with robust source extraction.

Models occasionally return a filename followed by a fenced C++ block despite
being instructed to return source only. The previous parser required the fence
to span the entire response, so labels such as ``src/bicg.cpp`` were written
into the source file. This utility installs a parser that extracts the first
C/C++ fenced block wherever it occurs, removes a standalone filename label,
and rejects remaining Markdown fences.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    ROOT / "scripts" / "experiments" / "run_api_experiment.py",
    ROOT / "scripts" / "experiments" / "run_iterative_api_experiment.py",
]

OLD = '''def clean_source(text: str) -> str:\n    text = text.strip()\n    fenced = re.fullmatch(r"```(?:cpp|c\\+\\+|c)?\\s*(.*?)\\s*```", text, re.DOTALL)\n    if fenced:\n        text = fenced.group(1).strip()\n    if not text.endswith("\\n"):\n        text += "\\n"\n    return text\n'''

NEW = '''def clean_source(text: str) -> str:\n    text = text.strip()\n\n    # Prefer an explicit C/C++ fenced block even when the model prepends a\n    # filename such as ``src/bicg.cpp``.\n    fenced = re.search(\n        r"```(?:cpp|c\\+\\+|cc|cxx|c)?\\s*\\n?(.*?)\\n?```",\n        text,\n        re.DOTALL | re.IGNORECASE,\n    )\n    if fenced:\n        text = fenced.group(1).strip()\n    else:\n        lines = text.splitlines()\n        if lines and re.fullmatch(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\\.(?:c|cc|cpp|cxx)", lines[0].strip()):\n            text = "\\n".join(lines[1:]).strip()\n\n    if "```" in text:\n        raise ValueError("Model response still contains Markdown fences after parsing")\n    if not text.strip():\n        raise ValueError("Model response did not contain source code")\n    if not text.endswith("\\n"):\n        text += "\\n"\n    return text\n'''


def main() -> None:
    changed = 0
    for path in TARGETS:
        source = path.read_text(encoding="utf-8")
        if NEW in source:
            print(f"Already hardened: {path.relative_to(ROOT)}")
            continue
        if OLD not in source:
            raise SystemExit(f"Expected clean_source implementation not found in {path}")
        path.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
        changed += 1
        print(f"Hardened: {path.relative_to(ROOT)}")
    print(f"Updated {changed} runner(s).")


if __name__ == "__main__":
    main()
