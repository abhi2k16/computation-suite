# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
test_release_metadata.py -- the version, changelog and license metadata must agree.

Catches the usual release slip: bumping pyproject.toml but not __version__ (or the reverse), or
shipping a version the CHANGELOG never mentions. Skipped when the repository files are not
next to the tests (e.g. tests run from an installed copy).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"

import pathlib
import re

import pytest

import fea_engine

PKG_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO = PKG_DIR.parent
PYPROJECT = PKG_DIR / "pyproject.toml"


def _pyproject_version():
    m = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M)
    assert m, "no version line in pyproject.toml"
    return m.group(1)


pytestmark = pytest.mark.skipif(not PYPROJECT.exists(), reason="repository files not available")


def test_version_matches_pyproject():
    assert fea_engine.__version__ == _pyproject_version()


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", fea_engine.__version__)


def test_changelog_mentions_this_version():
    log = REPO / "CHANGELOG.md"
    if not log.exists():
        pytest.skip("no CHANGELOG.md next to the package")
    text = log.read_text(encoding="utf-8")
    assert re.search(r"^## fea_engine " + re.escape(fea_engine.__version__) + r"\b", text, re.M), (
        f"CHANGELOG.md has no '## fea_engine {fea_engine.__version__}' section")


def test_package_ships_a_license_matching_the_root_one():
    lic = PKG_DIR / "LICENSE"
    assert lic.exists(), "package folder needs its own LICENSE so builds include it"
    root = REPO / "LICENSE"
    if root.exists():
        assert lic.read_text(encoding="utf-8") == root.read_text(encoding="utf-8")


def test_pyproject_declares_license_file():
    assert 'license-files = ["LICENSE"]' in PYPROJECT.read_text(encoding="utf-8")
