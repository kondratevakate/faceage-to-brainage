"""Install licensed FLAME assets into local avatar checkouts.

This script does not download FLAME. Download FLAME2020 from the official FLAME
site after accepting the license, then pass either the zip file or the extracted
`generic_model.pkl` here. The script copies the model to the DECA and MICA
locations expected by the local CPU runners.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


DEFAULT_DECA = Path(r"D:\projects\02_academia\_external\avatars\DECA")
DEFAULT_MICA = Path(r"D:\projects\02_academia\_external\avatars\MICA")


def _extract_generic_model(zip_path: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    tmp = Path(tempfile.mkdtemp(prefix="flame2020_"))
    with zipfile.ZipFile(zip_path) as zf:
        matches = [name for name in zf.namelist() if name.replace("\\", "/").endswith("generic_model.pkl")]
        if not matches:
            raise FileNotFoundError("generic_model.pkl not found inside FLAME zip")
        member = matches[0]
        zf.extract(member, tmp)
        return tmp / member


def _resolve_source(args) -> Path:
    if args.generic_model:
        source = args.generic_model.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        return source
    if args.flame_zip:
        return _extract_generic_model(args.flame_zip.resolve())
    raise ValueError("Provide --generic-model or --flame-zip")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic-model", type=Path, help="Path to extracted generic_model.pkl")
    parser.add_argument("--flame-zip", type=Path, help="Path to downloaded FLAME2020 zip")
    parser.add_argument("--deca-repo", type=Path, default=DEFAULT_DECA)
    parser.add_argument("--mica-repo", type=Path, default=DEFAULT_MICA)
    args = parser.parse_args()

    source = _resolve_source(args)
    targets = [
        args.deca_repo.resolve() / "data" / "generic_model.pkl",
        args.mica_repo.resolve() / "data" / "FLAME2020" / "generic_model.pkl",
    ]

    copied = []
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(target))

    print(json.dumps({"source": str(source), "copied": copied}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
