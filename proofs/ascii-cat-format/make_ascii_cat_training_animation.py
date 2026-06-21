#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path("/tmp/ascii-cat-llm-finetuning")
DATA_ROOT = REPO / "src/dataset/ascii_art/animals/cat"
OUT_DIR = Path("/home/ath/ascii-cat-format-proof")
FRAMES_DIR = OUT_DIR / "frames"
MP4_PATH = OUT_DIR / "ascii_cat_training_data_format_proof_45s.mp4"
REPORT_PATH = OUT_DIR / "format_proof_report.txt"

WIDTH = 1280
HEIGHT = 720
FPS = 30
DURATION_SECONDS = 45
TOTAL_FRAMES = FPS * DURATION_SECONDS
PANELS = 4

PROMPT_TEMPLATE = """Generate ascii art that matches the following description.

### description:
cat

### ascii visualization:
<ascii>
{ascii_art}
</ascii>
"""


@dataclass(frozen=True)
class Sample:
    index: int
    path: Path
    text: str
    lines: list[str]
    width: int
    height: int
    raw_sha: str
    prompt_sha: str


def numeric_sort_key(path: Path) -> int:
    return int(path.parent.name)


def load_samples() -> list[Sample]:
    paths = sorted(DATA_ROOT.glob("*/content.txt"), key=numeric_sort_key)
    samples: list[Sample] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        prompt = PROMPT_TEMPLATE.format(ascii_art=text)
        samples.append(
            Sample(
                index=numeric_sort_key(path),
                path=path,
                text=text,
                lines=lines,
                width=max((len(line) for line in lines), default=0),
                height=len(lines),
                raw_sha=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
                prompt_sha=hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
            )
        )
    return samples


def fit_mono_font(sample: Sample, max_w: int, max_h: int, min_size: int = 9) -> ImageFont.FreeTypeFont:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    for size in range(24, min_size - 1, -1):
        font = ImageFont.truetype(font_path, size)
        bbox = font.getbbox("M")
        char_w = bbox[2] - bbox[0]
        line_h = int((bbox[3] - bbox[1]) * 1.35)
        if sample.width * char_w <= max_w and max(1, sample.height) * line_h <= max_h:
            return font
    return ImageFont.truetype(font_path, min_size)


def draw_panel(draw: ImageDraw.ImageDraw, sample: Sample, box: tuple[int, int, int, int], font_regular: ImageFont.FreeTypeFont) -> None:
    x0, y0, x1, y1 = box
    panel_w = x1 - x0
    panel_h = y1 - y0
    draw.rounded_rectangle(box, radius=8, fill=(18, 23, 29), outline=(82, 95, 108), width=1)

    title = f"sample {sample.index:03d}  {sample.width}x{sample.height}  raw {sample.raw_sha}  prompt {sample.prompt_sha}"
    draw.text((x0 + 18, y0 + 14), title, fill=(225, 232, 238), font=font_regular)

    art_x = x0 + 24
    art_y = y0 + 48
    art_w = panel_w - 48
    art_h = panel_h - 76
    mono = fit_mono_font(sample, art_w, art_h - 54)
    bbox = mono.getbbox("M")
    line_h = int((bbox[3] - bbox[1]) * 1.35)

    draw.rectangle((art_x - 8, art_y - 8, art_x + art_w + 8, art_y + art_h + 8), fill=(8, 11, 15), outline=(40, 48, 56))
    draw.text((art_x, art_y), "<ascii>", fill=(107, 190, 255), font=font_regular)
    y = art_y + 25
    for line in sample.lines:
        draw.text((art_x, y), line, fill=(236, 238, 240), font=mono)
        y += line_h
    draw.text((art_x, art_y + art_h - 18), "</ascii>", fill=(107, 190, 255), font=font_regular)


def draw_frame(samples: list[Sample], frame_no: int, title_font: ImageFont.FreeTypeFont, ui_font: ImageFont.FreeTypeFont) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), (10, 14, 18))
    draw = ImageDraw.Draw(img)

    t = frame_no / max(1, TOTAL_FRAMES - 1)
    exact_pos = t * (len(samples) - 1)
    center = int(round(exact_pos))
    active = [(center + offset) % len(samples) for offset in range(PANELS)]

    draw.rectangle((0, 0, WIDTH, 72), fill=(15, 22, 29))
    draw.text((28, 18), "ASCII cat training data format proof", fill=(245, 248, 250), font=title_font)
    sub = f"{len(samples)} raw content.txt files rendered through the training prompt wrapper | 45 seconds | frame {frame_no + 1}/{TOTAL_FRAMES}"
    draw.text((29, 50), sub, fill=(159, 176, 190), font=ui_font)

    progress_w = WIDTH - 56
    draw.rectangle((28, 682, 28 + progress_w, 692), fill=(39, 47, 55))
    draw.rectangle((28, 682, 28 + int(progress_w * t), 692), fill=(91, 196, 146))
    draw.text((28, 650), f"dataset position: {exact_pos:06.2f} / {len(samples) - 1}", fill=(176, 188, 199), font=ui_font)

    margin = 28
    gap = 18
    top = 92
    panel_w = (WIDTH - margin * 2 - gap) // 2
    panel_h = 260
    boxes = [
        (margin, top, margin + panel_w, top + panel_h),
        (margin + panel_w + gap, top, WIDTH - margin, top + panel_h),
        (margin, top + panel_h + gap, margin + panel_w, top + panel_h * 2 + gap),
        (margin + panel_w + gap, top + panel_h + gap, WIDTH - margin, top + panel_h * 2 + gap),
    ]
    for sample_index, box in zip(active, boxes):
        draw_panel(draw, samples[sample_index], box, ui_font)

    return img


def write_report(samples: list[Sample]) -> None:
    widths = [sample.width for sample in samples]
    heights = [sample.height for sample in samples]
    exact_newline_count = sum(1 for sample in samples if sample.text.endswith("\n"))
    weird_chars = sorted(
        {
            ch
            for sample in samples
            for ch in sample.text
            if ch not in "\n\r\t" and (ord(ch) < 32 or ord(ch) > 126)
        }
    )
    report = [
        "ASCII cat training data format proof",
        f"source repo: {REPO}",
        f"source data: {DATA_ROOT}",
        f"samples: {len(samples)}",
        f"width range: {min(widths)}..{max(widths)} chars",
        f"height range: {min(heights)}..{max(heights)} lines",
        f"files ending with newline: {exact_newline_count}/{len(samples)}",
        f"non-basic-ascii visible chars: {''.join(weird_chars) if weird_chars else 'none'}",
        "",
        "Training formatting contract observed in repo:",
        "1. create_dataset.py reads every content.txt with f.read() and stores it unchanged in the ascii column.",
        "2. train.py inserts that ascii value inside <ascii>...</ascii> in OUTPUT_ASCII_PROMPT.",
        "3. The animation renders the same raw text in a monospace <ascii> block and shows raw/prompt SHA-256 prefixes.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    samples = load_samples()
    if not samples:
        raise SystemExit(f"No samples found under {DATA_ROOT}")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAMES_DIR.glob("frame_*.png"):
        old.unlink()

    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    ui_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)

    for frame_no in range(TOTAL_FRAMES):
        frame = draw_frame(samples, frame_no, title_font, ui_font)
        frame.save(FRAMES_DIR / f"frame_{frame_no:04d}.png")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(FRAMES_DIR / "frame_%04d.png"),
            "-t",
            str(DURATION_SECONDS),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(MP4_PATH),
        ],
        check=True,
    )
    write_report(samples)
    print(MP4_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
