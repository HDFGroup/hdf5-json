#!/usr/bin/env python3
"""Resolve the release version from the repo-local single source of truth.

SSOT: the `version` field of [project] in pyproject.toml.

Everything else in the release pipeline keys off what this prints, so a
malformed version, or one that disagrees with h5json.__version__, stops the
run before anything is built or published.
"""
import ast
import os
import re
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path("pyproject.toml")
DIST_NAME = "h5json"

# PEP 440 pre-release / dev-release markers.
PRERELEASE_RE = re.compile(r"(a|b|rc|alpha|beta|dev)\d*$", re.IGNORECASE)

# PEP 440 version syntax, from the spec's Appendix B (the same pattern
# `packaging.version` uses). `packaging` is not installed in the job that runs
# this script, so the pattern is inlined. Always applied with fullmatch().
PEP440_RE = re.compile(
    r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
    """,
    re.VERBOSE | re.IGNORECASE,
)


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
    return str(project["version"])


def validate_version(version):
    """Fail unless the version is a well-formed PEP 440 version with no whitespace."""
    if any(c.isspace() for c in version) or not PEP440_RE.fullmatch(version):
        fail(
            f"{PYPROJECT} [project] version {version!r} is not a valid PEP 440 "
            "version (whitespace and newlines are not allowed)"
        )


def check_dunder_version(version):
    """Check that h5json.__version__ agrees with pyproject.toml.

    docs/conf.py reads `h5json.__version__`, so it is a second place the
    version is exposed and must not disagree with the SSOT.

    Two forms are accepted. Deriving it from installed distribution metadata
    (`importlib.metadata.version("h5json")`) is the preferred one: there is
    only one version in the tree, so it cannot drift, and nothing is checked.
    A hard-coded literal is a second source of truth, so it is compared and
    fails on a mismatch. Note the metadata form usually carries a literal
    fallback for uninstalled source trees - that fallback is deliberately not
    the declared version and must not be compared against.
    """
    init = Path("src/h5json/__init__.py")
    if not init.is_file():
        return
    try:
        tree = ast.parse(init.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        fail(f"could not parse {init}: {exc}")

    literals = []
    derived = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
        ):
            continue
        if isinstance(node.value, ast.Call):
            derived = True
        elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            literals.append(node.value.value)

    if derived:
        # Single source of truth already - nothing can disagree.
        return
    if not literals:
        print(
            f"::warning::{init} does not define __version__, but docs/conf.py "
            "reads h5json.__version__ - the docs build will fail until it is "
            "restored"
        )
        return
    for found in literals:
        if found != version:
            fail(
                f"version mismatch: {PYPROJECT} says {version}, "
                f"{init} __version__ says {found}"
            )


def main():
    version = read_pyproject_version()
    validate_version(version)
    check_dunder_version(version)

    tag = f"v{version}"
    set_output("version", version)
    set_output("tag", tag)
    set_output("dist_name", DIST_NAME)
    set_output("prerelease", "true" if PRERELEASE_RE.search(version) else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
