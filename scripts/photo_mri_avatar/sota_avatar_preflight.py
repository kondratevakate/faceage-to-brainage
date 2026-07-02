"""Preflight checks for stronger one-shot avatar baselines.

This script does not upload images or run inference. It verifies whether the
local machine has enough code, weights, and runtime support to run each
candidate under the photo-to-MRI evaluation contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MethodStatus:
    method: str
    role: str
    status: str
    runnable_now: bool
    blockers: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    next_command: str | None = None


def exists(path: Path) -> bool:
    return path.exists()


def has_any(root: Path, patterns: list[str]) -> list[str]:
    found: list[str] = []
    if not root.exists():
        return found
    for pattern in patterns:
        found.extend(str(path) for path in root.glob(pattern))
    return sorted(found)


def run_text(command: list[str], timeout: int = 10) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired as exc:
        return 124, (exc.stdout or "") + (exc.stderr or "")
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def import_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def torch_status() -> dict[str, Any]:
    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        info["error"] = repr(exc)
        return info
    info["installed"] = True
    info["version"] = getattr(torch, "__version__", None)
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    info["device_count"] = int(torch.cuda.device_count())
    return info


def nvidia_status() -> dict[str, Any]:
    code, output = run_text(["nvidia-smi"], timeout=10)
    return {
        "nvidia_smi_found": code == 0,
        "return_code": code,
        "summary": output[:1600],
    }


def status_from_blockers(method: str, role: str, blockers: list[str], evidence: dict[str, Any], next_command: str | None) -> MethodStatus:
    return MethodStatus(
        method=method,
        role=role,
        status="ready" if not blockers else "blocked",
        runnable_now=not blockers,
        blockers=blockers,
        evidence=evidence,
        next_command=next_command,
    )


def check_3ddfa(external: Path, work: Path, py: str) -> MethodStatus:
    repo = external / "3DDFA_V2"
    outputs = work / "photo_avatar_crops_3subjects_3ddfa_v2"
    blockers: list[str] = []
    if not repo.exists():
        blockers.append(f"missing repo: {repo}")
    if not (repo / "configs" / "bfm_noneck_v3.pkl").exists():
        blockers.append("missing 3DDFA BFM config")
    existing_outputs = has_any(outputs, ["*.ply"])
    evidence = {
        "repo": str(repo),
        "bfm": str(repo / "configs" / "bfm_noneck_v3.pkl"),
        "existing_mesh_count": len(existing_outputs),
    }
    cmd = (
        f"{py} scripts/photo_mri_avatar/run_3ddfa_v2.py "
        f"--repo \"{repo}\" --input-dir \"{work / 'photo_crops_3subjects_3ddfa_1024'}\" "
        f"--output-dir \"{outputs}\" --subject faceage3 --session crops_3subjects_1024"
    )
    return status_from_blockers("3DDFA_V2", "dense BFM face mesh calibration baseline", blockers, evidence, cmd)


def check_deca(external: Path, work: Path, py: str, torch_info: dict[str, Any]) -> MethodStatus:
    repo = external / "DECA"
    data = repo / "data"
    blockers: list[str] = []
    if not repo.exists():
        blockers.append(f"missing repo: {repo}")
    if not torch_info.get("installed"):
        blockers.append("torch is not installed in the active Python environment")
    if not (data / "generic_model.pkl").exists():
        blockers.append("missing FLAME generic_model.pkl; requires FLAME license download")
    if not (data / "deca_model.tar").exists():
        blockers.append("missing DECA trained model tar")
    evidence = {
        "repo": str(repo),
        "has_deca_model": (data / "deca_model.tar").exists(),
        "has_flame_generic": (data / "generic_model.pkl").exists(),
        "has_support_files": {
            "landmark_embedding": (data / "landmark_embedding.npy").exists(),
            "fixed_displacement": (data / "fixed_displacement_256.npy").exists(),
            "head_template": (data / "head_template.obj").exists(),
        },
    }
    cmd = (
        f"{py} scripts/photo_mri_avatar/run_deca_cpu_geometry.py "
        f"--deca-repo \"{repo}\" --input-dir \"{work / 'photo_crops_3subjects_3ddfa_1024'}\" "
        f"--pattern \"1_1*\" --output-dir \"{work / 'photo_avatar_deca_cpu_case_a'}\" "
        f"--device cpu --subject case_a --session crops_1024"
    )
    return status_from_blockers("DECA", "FLAME mesh geometry baseline via CPU geometry-only runner", blockers, evidence, cmd)


def check_mica(external: Path, work: Path, torch_info: dict[str, Any]) -> MethodStatus:
    repo = external / "MICA"
    flame = repo / "data" / "FLAME2020"
    pretrained = repo / "data" / "pretrained" / "mica.tar"
    blockers: list[str] = []
    if not repo.exists():
        blockers.append(f"missing repo: {repo}")
    if not torch_info.get("installed"):
        blockers.append("torch is not installed in the active Python environment")
    if not pretrained.exists():
        blockers.append("missing MICA data/pretrained/mica.tar")
    if not (flame / "generic_model.pkl").exists():
        blockers.append("missing MICA FLAME2020/generic_model.pkl; requires FLAME license download")
    if not (flame / "landmark_embedding.npy").exists():
        blockers.append("missing MICA FLAME2020/landmark_embedding.npy")
    if not (flame / "head_template.obj").exists():
        blockers.append("missing MICA FLAME2020/head_template.obj")
    if not import_available("insightface"):
        blockers.append("missing insightface Python package for MICA preprocessing")
    insight_root = Path.home() / ".insightface" / "models"
    if not (insight_root / "antelopev2").exists():
        blockers.append("missing InsightFace antelopev2 model directory")
    if not (insight_root / "buffalo_l").exists():
        blockers.append("missing InsightFace buffalo_l model directory")
    candidates = has_any(repo, ["**/*.tar", "**/*.pth", "**/*.ckpt", "**/*.pt"])
    evidence = {
        "repo": str(repo),
        "has_mica_tar": pretrained.exists(),
        "has_flame2020": {
            "generic_model": (flame / "generic_model.pkl").exists(),
            "landmark_embedding": (flame / "landmark_embedding.npy").exists(),
            "head_template": (flame / "head_template.obj").exists(),
        },
        "insightface_installed": import_available("insightface"),
        "insightface_model_root": str(insight_root),
        "candidate_weight_files": candidates[:10],
        "expected_output": str(work / "photo_avatar_mica"),
    }
    cmd = (
        f"{sys.executable} scripts/photo_mri_avatar/run_mica_cpu_geometry.py "
        f"--mica-repo \"{repo}\" --input-dir \"{work / 'photo_crops_3subjects_3ddfa_1024'}\" "
        f"--pattern \"1_1*\" --output-dir \"{work / 'photo_avatar_mica_cpu_case_a'}\" "
        f"--subject case_a --session crops_1024"
    )
    return status_from_blockers("MICA", "metric FLAME mesh geometry baseline via CPU runner", blockers, evidence, cmd)


def check_emoca(external: Path, work: Path, torch_info: dict[str, Any]) -> MethodStatus:
    repo = external / "emoca"
    blockers: list[str] = []
    if not repo.exists():
        blockers.append(f"missing repo: {repo}")
    if not torch_info.get("installed"):
        blockers.append("torch is not installed in the active Python environment")
    candidates = has_any(repo, ["**/*.ckpt", "**/*.pth", "**/*.pt"])
    if not candidates:
        blockers.append("missing EMOCA checkpoint/model assets")
    evidence = {
        "repo": str(repo),
        "candidate_weight_files": candidates[:10],
        "expected_output": str(work / "photo_avatar_emoca"),
    }
    return status_from_blockers("EMOCA", "expression-aware FLAME face reconstruction", blockers, evidence, None)


def check_lam(external: Path, work: Path, torch_info: dict[str, Any], nvidia: dict[str, Any]) -> MethodStatus:
    repo = external / "LAM"
    blockers: list[str] = []
    if not repo.exists():
        blockers.append(f"missing repo: {repo}")
    if not torch_info.get("installed"):
        blockers.append("torch is not installed in the active Python environment")
    if not torch_info.get("cuda_available"):
        blockers.append("CUDA is not available to torch")
    if not nvidia.get("nvidia_smi_found"):
        blockers.append("nvidia-smi not found; local NVIDIA GPU/CUDA runtime is not visible")
    weights = has_any(repo, ["model_zoo/lam_models/releases/lam/lam-20k/step_045500/**/*"])
    if not weights:
        blockers.append("missing LAM-20K model weights")
    assets = [repo / "assets", repo / "external"]
    evidence = {
        "repo": str(repo),
        "has_assets_dir": all(path.exists() for path in assets),
        "weight_file_count": len(weights),
        "expected_output": str(work / "photo_avatar_lam"),
    }
    cmd = (
        f"cd /d \"{repo}\" && python app_lam.py "
        "# or: bash scripts/inference.sh <CONFIG> <MODEL_NAME> <IMAGE_PATH_OR_FOLDER> <MOTION_SEQ>"
    )
    return status_from_blockers("LAM", "one-shot animatable Gaussian head", blockers, evidence, cmd)


def check_gagavatar(external: Path, work: Path, torch_info: dict[str, Any], nvidia: dict[str, Any]) -> MethodStatus:
    repo = external / "GAGAvatar"
    blockers: list[str] = []
    if not repo.exists():
        blockers.append(f"missing repo: {repo}")
    if not torch_info.get("installed"):
        blockers.append("torch is not installed in the active Python environment")
    if not torch_info.get("cuda_available"):
        blockers.append("CUDA is not available to torch")
    if not nvidia.get("nvidia_smi_found"):
        blockers.append("nvidia-smi not found; local NVIDIA GPU/CUDA runtime is not visible")
    weights = has_any(repo, ["**/*.pth", "**/*.pt", "**/*.ckpt", "**/*.safetensors"]) if repo.exists() else []
    if repo.exists() and not weights:
        blockers.append("missing GAGAvatar checkpoint/model assets")
    evidence = {
        "repo": str(repo),
        "candidate_weight_files": weights[:10],
        "expected_output": str(work / "photo_avatar_gagavatar"),
    }
    cmd = f"cd /d \"{repo}\" && conda env create -f environment.yml && conda activate GAGAvatar"
    return status_from_blockers("GAGAvatar", "one-shot animatable Gaussian head comparator", blockers, evidence, cmd)


def check_meshlam(external: Path, work: Path, torch_info: dict[str, Any]) -> MethodStatus:
    repo = external / "MeshLAM"
    blockers: list[str] = []
    if not repo.exists():
        blockers.append("no local MeshLAM checkout found")
    if repo.exists() and not torch_info.get("installed"):
        blockers.append("torch is not installed in the active Python environment")
    weights = has_any(repo, ["**/*.pth", "**/*.pt", "**/*.ckpt", "**/*.safetensors"]) if repo.exists() else []
    if repo.exists() and not weights:
        blockers.append("missing MeshLAM weights")
    evidence = {
        "repo": str(repo),
        "candidate_weight_files": weights[:10],
        "expected_output": str(work / "photo_avatar_meshlam"),
    }
    return status_from_blockers("MeshLAM", "one-shot animatable textured mesh head", blockers, evidence, None)


def write_markdown(report: dict[str, Any], out: Path) -> None:
    lines = [
        "# SOTA Avatar Preflight",
        "",
        f"- Python: `{report['python']}`",
        f"- Workbench: `{report['workbench']}`",
        f"- External avatar repos: `{report['external_avatar_root']}`",
        f"- Torch installed: `{report['torch']['installed']}`",
        f"- Torch CUDA available: `{report['torch']['cuda_available']}`",
        f"- nvidia-smi found: `{report['nvidia']['nvidia_smi_found']}`",
        "",
        "## Method Status",
        "",
        "| Method | Runnable now | Status | Blockers |",
        "|---|---:|---|---|",
    ]
    for method in report["methods"]:
        blockers = "<br>".join(method["blockers"]) if method["blockers"] else ""
        lines.append(f"| {method['method']} | {method['runnable_now']} | {method['status']} | {blockers} |")
    lines.extend(["", "## Next Commands", ""])
    for method in report["methods"]:
        if method["next_command"]:
            lines.extend([f"### {method['method']}", "", "```powershell", method["next_command"], "```", ""])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--external-avatar-root", type=Path, default=Path(r"D:\projects\02_academia\_external\avatars"))
    parser.add_argument("--workbench", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo.resolve()
    work = (args.workbench or repo / "data" / "avatar_2026_work").resolve()
    output_dir = (args.output_dir or work / "reports").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    torch_info = torch_status()
    nvidia = nvidia_status()
    external = args.external_avatar_root.resolve()

    methods = [
        check_3ddfa(external, work, py),
        check_deca(external, work, py, torch_info),
        check_mica(external, work, torch_info),
        check_emoca(external, work, torch_info),
        check_lam(external, work, torch_info, nvidia),
        check_gagavatar(external, work, torch_info, nvidia),
        check_meshlam(external, work, torch_info),
    ]

    report = {
        "python": py,
        "repo": str(repo),
        "workbench": str(work),
        "external_avatar_root": str(external),
        "torch": torch_info,
        "nvidia": nvidia,
        "tools": {
            "git": shutil.which("git"),
            "hf": shutil.which("hf") or shutil.which("huggingface-cli"),
        },
        "methods": [asdict(method) for method in methods],
    }

    json_path = output_dir / "sota_avatar_preflight.json"
    md_path = output_dir / "SOTA_AVATAR_PREFLIGHT.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print(md_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
