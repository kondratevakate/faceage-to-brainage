from __future__ import annotations

import csv
import math
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(r"D:\YandexDisk\kondratevakate\01_insidekatesbrain\01_my_brain_years\faceage_brainage\avatar_2026")
PAGE = ROOT / "project_page"
ASSETS = PAGE / "assets"

CROPS = ROOT / "photo_crops_3subjects_3ddfa_1024"
DDDFA = ROOT / "photo_avatar_crops_3subjects_3ddfa_v2"
MEDIAPIPE = ROOT / "photo_avatar_crops_3subjects_mediapipe"
MRI = ROOT / "mri_surfaces"
ALIGN = ROOT / "landmark_alignment" / "crops_3ddfa_v2"
REPORTS = ROOT / "reports"
CONSISTENCY = ROOT / "subject_consistency" / "crops_3subjects_3ddfa_1024"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_16 = font(16)
FONT_18 = font(18)
FONT_20 = font(20)
FONT_22_B = font(22, bold=True)
FONT_28_B = font(28, bold=True)


def safe_session(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", stem)


def fit_image(path: Path, size: tuple[int, int], fill: str = "#f3f4f6") -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    bg = Image.new("RGB", size, fill)
    bg.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return bg


def overlay_for_crop(crop: Path, method: str) -> Path:
    if method == "3ddfa":
        return DDDFA / f"faceage3_crops_3subjects_1024_{crop.stem}_3ddfa_v2_face1_overlay.jpg"
    if method == "mediapipe":
        return MEDIAPIPE / f"faceage3_{safe_session(crop.stem)}_landmarks_overlay.jpg"
    raise ValueError(method)


def subject_group(path: Path) -> str:
    match = re.match(r"^(\d+_\d+)_", path.name)
    return match.group(1) if match else "unknown"


def grouped_crops() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for crop in sorted(CROPS.glob("*_facecrop.jpg")):
        groups.setdefault(subject_group(crop), []).append(crop)
    return groups


def copy_asset(src: Path, name: str) -> str:
    dst = ASSETS / name
    shutil.copy2(src, dst)
    return f"assets/{name}"


def make_pipeline_gif() -> str:
    groups = grouped_crops()
    selected: list[Path] = []
    for group in ["1_1", "2_1", "3_1"]:
        if groups.get(group):
            selected.append(groups[group][min(1, len(groups[group]) - 1)])

    frames: list[Image.Image] = []
    w, h = 360, 360
    pad = 18
    label_h = 54
    canvas_size = (pad * 4 + w * 3, pad * 2 + label_h + h)

    for crop in selected:
        canvas = Image.new("RGB", canvas_size, "#f8fafc")
        draw = ImageDraw.Draw(canvas)
        group = subject_group(crop)
        title = f"known folder {group}: crop -> dense face fit -> landmarks"
        draw.text((pad, 16), title, fill="#111827", font=FONT_28_B)
        items = [
            ("Input crop", crop),
            ("3DDFA_V2 fit", overlay_for_crop(crop, "3ddfa")),
            ("MediaPipe mesh", overlay_for_crop(crop, "mediapipe")),
        ]
        for idx, (label, path) in enumerate(items):
            x = pad + idx * (w + pad)
            y = pad + label_h
            img = fit_image(path, (w, h), "#ffffff")
            canvas.paste(img, (x, y))
            draw.rounded_rectangle((x, y, x + w, y + h), radius=10, outline="#cbd5e1", width=2)
            draw.text((x + 8, y + 8), label, fill="#0f172a", font=FONT_20)
        frames.extend([canvas] * 4)

    out = ASSETS / "teaser_pipeline.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=360, loop=0, optimize=True)
    return "assets/teaser_pipeline.gif"


def make_subject_mosaic() -> str:
    groups = grouped_crops()
    thumb = (170, 170)
    pad = 16
    max_cols = max(len(v) for v in groups.values())
    width = pad * (max_cols + 2) + thumb[0] * max_cols + 150
    height = pad * (len(groups) + 1) + thumb[1] * len(groups)
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)

    y = pad
    for group in sorted(groups):
        draw.text((pad, y + 44), group, fill="#111827", font=FONT_28_B)
        draw.text((pad, y + 76), f"{len(groups[group])} photos", fill="#64748b", font=FONT_18)
        for idx, crop in enumerate(groups[group]):
            x = pad + 150 + idx * (thumb[0] + pad)
            img = fit_image(crop, thumb)
            canvas.paste(img, (x, y))
            draw.rounded_rectangle((x, y, x + thumb[0], y + thumb[1]), radius=8, outline="#cbd5e1", width=2)
            badge = f"{group}.{idx + 1}"
            draw.rounded_rectangle((x + 8, y + 8, x + 66, y + 34), radius=6, fill="#ffffff", outline="#cbd5e1")
            draw.text((x + 16, y + 11), badge, fill="#111827", font=FONT_16)
        y += thumb[1] + pad

    out = ASSETS / "known_folder_mosaic.jpg"
    canvas.save(out, quality=92)
    return "assets/known_folder_mosaic.jpg"


def make_alignment_strip() -> str | None:
    paths = sorted(ALIGN.glob("*_landmark_constrained_alignment.png"))[:6]
    if not paths:
        return None
    thumb = (300, 230)
    pad = 14
    cols = 3
    rows = math.ceil(len(paths) / cols)
    canvas = Image.new("RGB", (pad + cols * (thumb[0] + pad), pad + rows * (thumb[1] + pad)), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    for i, path in enumerate(paths):
        x = pad + (i % cols) * (thumb[0] + pad)
        y = pad + (i // cols) * (thumb[1] + pad)
        canvas.paste(fit_image(path, thumb), (x, y))
        draw.rounded_rectangle((x, y, x + thumb[0], y + thumb[1]), radius=8, outline="#cbd5e1", width=2)
    out = ASSETS / "mri_alignment_strip.jpg"
    canvas.save(out, quality=92)
    return "assets/mri_alignment_strip.jpg"


def read_ascii_ply_vertices(path: Path) -> np.ndarray:
    vertex_count = None
    with path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.rsplit(" ", 1)[1])
            elif line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"no vertex count in {path}")
        points = np.loadtxt(f, dtype=np.float32, max_rows=vertex_count, usecols=(0, 1, 2))
    return points


def first_ply_for_group(group: str) -> Path:
    paths = sorted(DDDFA.glob(f"faceage3_crops_3subjects_1024_{group}_*_3ddfa_v2_face1.ply"))
    if not paths:
        raise FileNotFoundError(group)
    return paths[0]


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points[np.isfinite(points).all(axis=1)]
    center = points.mean(axis=0)
    points = points - center
    scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    return points / max(scale, 1e-6)


def make_mesh_turntable() -> str:
    rng = np.random.default_rng(42)
    meshes = []
    colors = ["#2563eb", "#16a34a", "#db2777"]
    for group, color in zip(["1_1", "2_1", "3_1"], colors):
        points = normalize_points(read_ascii_ply_vertices(first_ply_for_group(group)))
        if len(points) > 2600:
            points = points[rng.choice(len(points), size=2600, replace=False)]
        meshes.append((group, points, color))

    frames: list[Image.Image] = []
    for angle in range(0, 360, 18):
        fig = plt.figure(figsize=(9, 3.2), dpi=120)
        fig.patch.set_facecolor("#0b1020")
        for i, (group, points, color) in enumerate(meshes, start=1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            ax.set_facecolor("#0b1020")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.7, c=color, alpha=0.72)
            ax.view_init(elev=8, azim=angle)
            ax.set_title(f"folder {group}", color="white", fontsize=11, pad=0)
            ax.set_axis_off()
            ax.set_xlim(-0.28, 0.28)
            ax.set_ylim(-0.28, 0.28)
            ax.set_zlim(-0.28, 0.28)
        fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0.02, wspace=0)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE))
        plt.close(fig)

    out = ASSETS / "mesh_turntable_3folders.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=95, loop=0, optimize=True)
    return "assets/mesh_turntable_3folders.gif"


def make_consistency_chart() -> str:
    rows = []
    with (CONSISTENCY / "subject_consistency_summary.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=160)
    labels = [f"{r['method']}\n{r['metric'].replace('_', ' ')}" for r in rows]
    genuine_p90 = [float(r["genuine_p90"]) for r in rows]
    impostor_p10 = [float(r["impostor_p10"]) for r in rows]
    x = np.arange(len(rows))
    width = 0.36
    ax.bar(x - width / 2, genuine_p90, width, label="genuine p90", color="#ef4444")
    ax.bar(x + width / 2, impostor_p10, width, label="impostor p10", color="#14b8a6")
    ax.axhline(0, color="#111827", linewidth=0.7)
    ax.set_title("Strict identity-separation check: genuine p90 should be below impostor p10", fontsize=12, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("distance / normalized units")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = ASSETS / "consistency_separation_chart.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return "assets/consistency_separation_chart.png"


def make_methods_diagram() -> str:
    w, h = 1360, 430
    canvas = Image.new("RGB", (w, h), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    boxes = [
        ("Known folders", "1_1 / 2_1 / 3_1\nlabels only", "#2563eb"),
        ("Face crops", "3DDFA detector\n1024 px crops", "#16a34a"),
        ("One-photo meshes", "MediaPipe + 3DDFA\nrough geometry", "#db2777"),
        ("MRI bridge", "outer-head surface\nlandmark alignment", "#f97316"),
        ("Metrics", "genuine vs impostor\nMRI distance + QC", "#0f766e"),
    ]
    x0 = 36
    box_w = 230
    gap = 28
    y = 92
    for i, (title, body, color) in enumerate(boxes):
        x = x0 + i * (box_w + gap)
        draw.rounded_rectangle((x, y, x + box_w, y + 220), radius=18, fill="#ffffff", outline="#d1d5db", width=2)
        draw.rectangle((x, y, x + box_w, y + 10), fill=color)
        draw.text((x + 18, y + 34), title, fill="#111827", font=FONT_22_B)
        draw.multiline_text((x + 18, y + 82), body, fill="#475569", font=FONT_16, spacing=7)
        if i < len(boxes) - 1:
            ax = x + box_w + 6
            ay = y + 110
            draw.line((ax, ay, ax + gap - 12, ay), fill="#64748b", width=3)
            draw.polygon([(ax + gap - 12, ay - 7), (ax + gap - 12, ay + 7), (ax + gap, ay)], fill="#64748b")
    draw.text((36, 32), "Current evaluation pipeline", fill="#0f172a", font=FONT_28_B)
    draw.text((36, 348), "The page separates visual plausibility, identity consistency, and biological-age validity.", fill="#334155", font=FONT_20)
    out = ASSETS / "method_pipeline_diagram.png"
    canvas.save(out, quality=94)
    return "assets/method_pipeline_diagram.png"


def html_page(assets: dict[str, str]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FaceAge-to-BrainAge Avatar Lab</title>
  <style>
    :root {{
      --ink: #111827;
      --muted: #64748b;
      --line: #d7dde7;
      --paper: #ffffff;
      --soft: #f6f8fb;
      --blue: #2563eb;
      --green: #16a34a;
      --rose: #db2777;
      --orange: #f97316;
      --teal: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      line-height: 1.55;
    }}
    a {{ color: var(--blue); text-decoration-thickness: 1px; text-underline-offset: 3px; }}
    .nav {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(255,255,255,0.92);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--line);
    }}
    .nav-inner {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 12px 22px;
      display: flex;
      gap: 18px;
      align-items: center;
      justify-content: space-between;
    }}
    .brand {{ font-weight: 760; letter-spacing: 0; }}
    .nav-links {{ display: flex; gap: 14px; flex-wrap: wrap; font-size: 14px; }}
    .nav-links a {{ color: #334155; text-decoration: none; }}
    .hero {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 58px 22px 26px;
    }}
    .eyebrow {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      color: var(--teal);
      font-weight: 700;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    h1 {{
      margin: 10px 0 12px;
      font-size: clamp(44px, 7vw, 86px);
      line-height: .95;
      letter-spacing: 0;
      max-width: 980px;
    }}
    .subtitle {{
      max-width: 860px;
      font-size: 21px;
      color: #334155;
      margin: 0 0 26px;
    }}
    .button-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 24px 0 30px; }}
    .button {{
      border: 1px solid #cbd5e1;
      color: #0f172a;
      background: #fff;
      border-radius: 8px;
      padding: 10px 14px;
      text-decoration: none;
      font-weight: 650;
      font-size: 14px;
    }}
    .button.primary {{ background: #0f172a; color: #fff; border-color: #0f172a; }}
    .hero-media {{
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      background: var(--soft);
    }}
    .hero-media img {{ display: block; width: 100%; height: auto; }}
    section {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 48px 22px;
    }}
    .band {{
      max-width: none;
      background: var(--soft);
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }}
    .band > .inner {{ max-width: 1180px; margin: 0 auto; padding: 48px 22px; }}
    h2 {{ font-size: 34px; line-height: 1.1; margin: 0 0 12px; letter-spacing: 0; }}
    h3 {{ font-size: 19px; margin: 0 0 6px; }}
    .lead {{ color: #475569; max-width: 900px; font-size: 18px; margin: 0 0 24px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 26px 0 0;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      background: #fff;
    }}
    .metric .num {{ font-size: 32px; font-weight: 800; line-height: 1; }}
    .metric .label {{ color: var(--muted); margin-top: 6px; font-size: 14px; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 24px;
      align-items: start;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }}
    .panel-body {{ padding: 18px; }}
    .panel img {{ width: 100%; display: block; height: auto; }}
    .callout {{
      border-left: 5px solid var(--rose);
      background: #fff;
      padding: 18px 20px;
      margin-top: 22px;
      color: #334155;
    }}
    .three {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }}
    .mini {{
      border-top: 4px solid var(--blue);
      padding: 16px;
      background: #fff;
      border-radius: 8px;
      border-left: 1px solid var(--line);
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }}
    .mini:nth-child(2) {{ border-top-color: var(--green); }}
    .mini:nth-child(3) {{ border-top-color: var(--orange); }}
    .caption {{ color: var(--muted); font-size: 14px; padding: 10px 2px 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin-top: 18px;
    }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px; text-align: left; }}
    th {{ color: #334155; background: #f8fafc; }}
    .footer {{
      border-top: 1px solid var(--line);
      background: #0f172a;
      color: #cbd5e1;
      padding: 34px 22px;
    }}
    .footer .inner {{ max-width: 1180px; margin: 0 auto; }}
    .footer a {{ color: #93c5fd; }}
    @media (max-width: 820px) {{
      .metrics, .grid-2, .three {{ grid-template-columns: 1fr; }}
      .nav-inner {{ align-items: flex-start; flex-direction: column; }}
      h1 {{ font-size: 44px; }}
      .subtitle {{ font-size: 18px; }}
    }}
  </style>
</head>
<body>
  <nav class="nav">
    <div class="nav-inner">
      <div class="brand">FaceAge-to-BrainAge Avatar Lab</div>
      <div class="nav-links">
        <a href="#data">Data</a>
        <a href="#method">Method</a>
        <a href="#geometry">Geometry</a>
        <a href="#metrics">Metrics</a>
        <a href="#narrative">Narrative</a>
      </div>
    </div>
  </nav>

  <header class="hero">
    <div class="eyebrow">Private research page / visual draft</div>
    <h1>One-photo face avatars meet MRI-grounded biological age.</h1>
    <p class="subtitle">A visual project page for the current FaceAge-to-BrainAge experiment: face crops, one-photo 3D baselines, MRI alignment, and identity-consistency checks across known folders.</p>
    <div class="button-row">
      <a class="button primary" href="#metrics">Current result</a>
      <a class="button" href="../reports/METRICS_AND_LABELS.md">Metrics note</a>
      <a class="button" href="../reports/TWIN_FACEAGE_LITERATURE_CONTEXT.md">Twin/FaceAge context</a>
    </div>
    <div class="hero-media">
      <img src="{assets['teaser']}" alt="Animated pipeline from crop to 3D face fitting overlays">
    </div>
    <div class="metrics">
      <div class="metric"><div class="num">14</div><div class="label">local face photos currently visible</div></div>
      <div class="metric"><div class="num">3</div><div class="label">known folders used as supervised labels</div></div>
      <div class="metric"><div class="num">2</div><div class="label">one-photo geometry baselines</div></div>
      <div class="metric"><div class="num">0</div><div class="label">strict identity-separation metrics passed</div></div>
    </div>
  </header>

  <section id="data">
    <h2>Known-folder photo cohort</h2>
    <p class="lead">The page uses folder labels only. It does not infer identity from faces. Locally available folders are 1_1, 2_1, and 3_1; 1_2 and 1_3/photos are currently empty on disk.</p>
    <div class="panel"><img src="{assets['mosaic']}" alt="Mosaic of face crops grouped by folder"></div>
  </section>

  <section class="band" id="method">
    <div class="inner">
      <h2>Pipeline</h2>
      <p class="lead">This is a baseline pipeline, not yet a final avatar method. It turns photos into standardized crops, estimates rough meshes, aligns them to an MRI-derived outer-head surface, then checks consistency across repeated photos.</p>
      <div class="panel"><img src="{assets['diagram']}" alt="Pipeline diagram"></div>
      <div class="three" style="margin-top: 18px;">
        <div class="mini"><h3>Visual plausibility</h3><p>Does the overlay look reasonable on the input image?</p></div>
        <div class="mini"><h3>Identity consistency</h3><p>Are same-folder avatars closer than different-folder avatars?</p></div>
        <div class="mini"><h3>Biological-age validity</h3><p>Does a face signal track aging, lifestyle, MRI, or outcomes?</p></div>
      </div>
    </div>
  </section>

  <section id="geometry">
    <h2>Geometry baseline</h2>
    <p class="lead">3DDFA and MediaPipe give usable detection and rough facial geometry. They are not yet a MeshLAM/MATCH-level high-fidelity avatar and should be treated as the floor for better methods.</p>
    <div class="grid-2">
      <div class="panel">
        <img src="{assets['turntable']}" alt="Animated 3D point-cloud turntable for three known folders">
        <div class="panel-body"><h3>3D mesh turntable</h3><p class="caption">Point-cloud render from 3DDFA meshes for one representative photo per known folder.</p></div>
      </div>
      <div class="panel">
        <img src="{assets['quicklook']}" alt="Full quicklook of crop, 3DDFA and MediaPipe overlays">
        <div class="panel-body"><h3>Visual QC sheet</h3><p class="caption">All 14 crops with 3DDFA and MediaPipe overlays for rapid inspection.</p></div>
      </div>
    </div>
  </section>

  <section class="band">
    <div class="inner">
      <h2>MRI bridge</h2>
      <p class="lead">MRI comparison is useful, but posture and soft tissue matter. Supine MRI and upright photos should not be treated as the same facial surface in eyelids, cheeks, jawline, or submental regions.</p>
      <div class="grid-2">
        <div class="panel">
          <img src="{assets['mri_qc']}" alt="MRI outer-head surface quality control">
          <div class="panel-body"><h3>MRI outer-head surface</h3><p class="caption">Current MRI-derived surface used for coarse face-to-head alignment.</p></div>
        </div>
        <div class="panel">
          <img src="{assets['alignment']}" alt="MRI alignment previews">
          <div class="panel-body"><h3>Landmark-constrained alignment</h3><p class="caption">Preview panels from the current automatic alignment pipeline.</p></div>
        </div>
      </div>
      <div class="callout">The earlier ~2.5 mm value is not a validated anatomical accuracy claim. It is a surface-distance baseline after landmark-seeded alignment and needs manual MRI landmarks or a controlled 3D face scan.</div>
    </div>
  </section>

  <section id="metrics">
    <h2>Current consistency result</h2>
    <p class="lead">A Face-ID-style geometric constraint is framed as a supervised separation test: same-folder pairs are genuine, different-folder pairs are impostor. The strict acceptance rule is genuine p90 below impostor p10.</p>
    <div class="panel"><img src="{assets['chart']}" alt="Consistency separation chart"></div>
    <table>
      <tr><th>Finding</th><th>Interpretation</th></tr>
      <tr><td>26 genuine pairs and 65 impostor pairs per method</td><td>Enough for a first diagnostic distribution, not enough for a general identity benchmark.</td></tr>
      <tr><td>No current metric passes genuine_p90 &lt; impostor_p10</td><td>The baseline is not identity-separable enough for Face ID-grade avatar claims.</td></tr>
      <tr><td>MediaPipe/3DDFA still detect all crop photos</td><td>They are useful as preprocessing and QC baselines before stronger avatar methods.</td></tr>
    </table>
  </section>

  <section class="band" id="narrative">
    <div class="inner">
      <h2>FaceAge narrative</h2>
      <p class="lead">Twin literature supports the premise that perceived facial age is biologically meaningful, but modern AI FaceAge models still need twin-controlled validation.</p>
      <div class="three">
        <div class="mini"><h3>Anchor</h3><p>Perceived age in Danish twins predicts survival and function, even under within-pair designs.</p></div>
        <div class="mini"><h3>Caveat</h3><p>Facial age and methylation age may capture distinct aging axes; do not collapse them into one clock.</p></div>
        <div class="mini"><h3>Gap</h3><p>No deep-learning FaceAge model is established as validated on MZ/DZ twin cohorts.</p></div>
      </div>
    </div>
  </section>

  <section>
    <h2>Next visual upgrade</h2>
    <p class="lead">The next page iteration should replace the baseline mesh with a stronger one-photo avatar method such as MICA/DECA/EMOCA or MeshLAM-style textured mesh reconstruction, then repeat the same consistency and MRI checks.</p>
  </section>

  <footer class="footer">
    <div class="inner">
      <p>This is a private project page draft containing face images. Do not publish or commit it without explicit review.</p>
      <p>Visual-page references: <a href="https://malteprinzler.github.io/projects/match/">MATCH project page</a> and <a href="https://meshlam.github.io/">MeshLAM project page</a>.</p>
    </div>
  </footer>
</body>
</html>
"""


def build() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    assets = {
        "teaser": make_pipeline_gif(),
        "mosaic": make_subject_mosaic(),
        "diagram": make_methods_diagram(),
        "turntable": make_mesh_turntable(),
        "chart": make_consistency_chart(),
        "quicklook": copy_asset(REPORTS / "quicklook_3subjects_crops_mediapipe_3ddfa.jpg", "quicklook_3subjects_crops_mediapipe_3ddfa.jpg"),
        "mri_qc": copy_asset(MRI / "kate_2018_qc.png", "kate_2018_qc.png"),
        "alignment": make_alignment_strip() or copy_asset(MRI / "kate_2018_qc.png", "mri_alignment_fallback.png"),
    }
    (PAGE / "index.html").write_text(html_page(assets), encoding="utf-8")
    print(PAGE / "index.html")


if __name__ == "__main__":
    build()
