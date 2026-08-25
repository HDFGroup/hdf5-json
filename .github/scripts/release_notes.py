#!/usr/bin/env python3
"""Extract the CHANGELOG.md section for the version being released.

Writes release-notes.md and sets the `has_notes` step output. A missing
CHANGELOG, or a CHANGELOG with no section for this version, is not an error -
the release job falls back to GitHub's generated notes. That keeps the release
pipeline usable before CHANGELOG.md exists (it is still a TBD in the release
plan) without silently publishing an empty release body.

Recognized heading forms, at any heading level:
    ## 2.0.0
    ## v2.0.0
    ## [2.0.0]
    ## [2.0.0] - 2026-08-25
"""
import os
import re
import sys
from pathlib import Path

CHANGELOG = Path("CHANGELOG.md")
OUT = Path("release-notes.md")


def set_output(name, value):
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def find_section(text, version):
    """Return the body of the heading matching `version`, or None."""
    # Escape the version so dots don't act as wildcards.
    v = re.escape(version)
    heading = re.compile(
        rf"^(?P<level>#{{1,6}})\s*\[?v?{v}\]?\s*(?:[-–—]\s*.*)?$",
        re.MULTILINE,
    )
    match = heading.search(text)
    if not match:
        return None

    start = match.end()
    level = len(match.group("level"))
    # The section ends at the next heading of the same or a shallower level.
    nxt = re.compile(rf"^#{{1,{level}}}\s", re.MULTILINE)
    end_match = nxt.search(text, start)
    end = end_match.start() if end_match else len(text)
    return text[start:end].strip()


def main():
    version = os.environ.get("VERSION", "").strip()
    if not version:
        print("::error::VERSION is not set", file=sys.stderr)
        return 1

    if not CHANGELOG.is_file():
        print(f"::warning::{CHANGELOG} not found - release will use generated notes")
        set_output("has_notes", "false")
        return 0

    text = CHANGELOG.read_text(encoding="utf-8")
    body = find_section(text, version)
    if not body:
        print(
            f"::warning::no {CHANGELOG} section found for version {version} "
            "- release will use generated notes"
        )
        set_output("has_notes", "false")
        return 0

    OUT.write_text(body + "\n", encoding="utf-8")
    set_output("has_notes", "true")
    print(f"Extracted {len(body.splitlines())} lines of notes for {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
