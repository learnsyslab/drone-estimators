"""Generate the API reference pages and their navigation.

This script is executed by the gen-files plugin during ``properdocs build`` or ``properdocs serve``.
"""

from pathlib import Path

import mkdocs_gen_files

SKIP_PARTS = {"_typing", "__init__", "__main__"}

summary = ["* [Overview](index.md)\n"]
section = None
for path in sorted(Path("drone_estimators").rglob("*.py")):
    parts = path.with_suffix("").parts
    if any(part in SKIP_PARTS for part in parts):
        continue
    doc_path = Path("reference", path.with_suffix(".md"))
    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(f"::: {'.'.join(parts)}\n")
    mkdocs_gen_files.set_edit_path(doc_path, path)

    link = path.with_suffix(".md").as_posix()
    if len(parts) == 2:  # Top level module
        summary.append(f"* [{parts[-1]}]({link})\n")
        continue
    if parts[1] != section:  # First module of a subpackage
        section = parts[1]
        summary.append(f"* {section}\n")
    summary.append(f"    * [{'.'.join(parts[1:])}]({link})\n")

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(summary)
