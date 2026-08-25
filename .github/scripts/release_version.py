#!/usr/bin/env python3
"""Resolve the release version from the repo-local single source of truth.

SSOT: the `version` field of [project] in pyproject.toml.

Everything else in the release pipeline keys off what this prints, so a tag
that disagrees with the source tree stops the run before anything is built or
published.
"""
import os
import re
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path("pyproject.toml")
DIST_NAME = "h5json"

# PEP 440 pre-release / dev-release markers.
PRERELEASE_RE = re.compile(r"(a|b|rc|alpha|beta|dev)\d*$", re.IGNORECASE)


def set_output(name, value):
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")
    print(f"{name}={value}")


def fail(msg):
    print(f"::error::{msg}", file=sys.stderr)
    raise SystemExit(1)


def read_pyproject_version():
    if not PYPROJECT.is_file():
        fail(f"{PYPROJECT} not found")
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    if "version" not in project:
        fail(
            f"{PYPROJECT} [project] has no static `version`. If the project has "
            "moved to a dynamic version, point this script at the new SSOT."
        )
    return str(project["version"]).strip()


def check_dunder_version(version):
    """Soft check: warn if h5json exposes a __version__ that disagrees.

    docs/conf.py reads `h5json.__version__`, so once that attribute is
    restored it becomes a second place the version lives and must agree with
    pyproject.toml. Warn rather than fail so this does not block a release
    while the attribute is still missing.
    """
    init = Path("src/h5json/__init__.py")
    if not init.is_file():
        return
    match = re.search(
        r"^__version__\s*=\s*[\"']([^\"']+)[\"']",
        init.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        print(
            "::warning::src/h5json/__init__.py does not define __version__, but "
            "docs/conf.py reads h5json.__version__ - the docs build will fail "
            "until it is restored"
        )
        return
    if match.group(1) != version:
        fail(
            f"version mismatch: pyproject.toml says {version}, "
            f"src/h5json/__init__.py __version__ says {match.group(1)}"
        )


def main():
    version = read_pyproject_version()
    check_dunder_version(version)

    tag = f"v{version}"
    if os.environ.get("EVENT_NAME") == "push":
        ref = os.environ.get("GITHUB_REF_NAME", "")
        if ref != tag:
            fail(
                f"tag {ref!r} does not match the version in {PYPROJECT} "
                f"({version}, expected tag {tag!r}). Bump the SSOT and re-tag."
            )

    set_output("version", version)
    set_output("tag", tag)
    set_output("dist_name", DIST_NAME)
    set_output("prerelease", "true" if PRERELEASE_RE.search(version) else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
