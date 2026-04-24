"""
Regression test: every image referenced from README.md must
(a) exist on disk AND (b) be tracked by git.

We had one incident where README pointed at a file under
`papers/figures/`, which is gitignored — GitHub then rendered a broken
image.  This test prevents that from happening again.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"

MARKDOWN_IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+?)(?:\s+\"[^\"]*\")?\)")
HTML_IMG = re.compile(r"<img[^>]*\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)


def _extract_image_refs(markdown: str) -> list[str]:
    refs = MARKDOWN_IMG.findall(markdown) + HTML_IMG.findall(markdown)
    return [r for r in refs if not r.startswith(("http://", "https://"))]


def _git_tracked_files(repo_root: Path) -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


IMAGE_REFS = _extract_image_refs(README_PATH.read_text(encoding="utf-8"))
TRACKED = _git_tracked_files(REPO_ROOT)


def test_readme_has_image_refs():
    assert IMAGE_REFS, "README.md contains no local image references — unexpected"


@pytest.mark.parametrize("ref", IMAGE_REFS)
def test_readme_image_exists_on_disk(ref):
    path = (REPO_ROOT / ref).resolve()
    assert path.exists(), (
        f"README.md references {ref!r}, but {path} does not exist on disk."
    )


@pytest.mark.parametrize("ref", IMAGE_REFS)
def test_readme_image_is_tracked_by_git(ref):
    normalized = ref.replace("\\", "/").lstrip("./")
    assert normalized in TRACKED, (
        f"README.md references {ref!r}, but this path is NOT tracked by git "
        f"(probably matched by .gitignore).  GitHub will render a broken "
        f"image.  Either move the file to a tracked location, or add an "
        f"explicit exception to .gitignore."
    )
