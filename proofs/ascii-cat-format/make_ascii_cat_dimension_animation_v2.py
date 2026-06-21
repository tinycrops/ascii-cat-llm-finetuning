#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import html
import math
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("/home/ath/ascii-cat-format-proof")
FRAMES_DIR = OUT_DIR / "frames_v2"
MP4_PATH = OUT_DIR / "ascii_cat_dimension_proof_v2_45s.mp4"
REPORT_PATH = OUT_DIR / "dimension_proof_v2_report.txt"
XMISSION_HTML = OUT_DIR / "xmission_ascii_cats.html"
ASCIIART_EU_HTML = OUT_DIR / "asciiart_eu_cats.html"
REPO_DATA_ROOT = Path("/tmp/ascii-cat-llm-finetuning/src/dataset/ascii_art/animals/cat")

WIDTH = 1280
HEIGHT = 720
FPS = 30
DURATION_SECONDS = 45
TOTAL_FRAMES = FPS * DURATION_SECONDS
PANELS = 6

MONO_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
UI_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

SOURCE_COLORS = {
    "repo": (103, 205, 156),
    "asciiart.eu": (111, 184, 255),
    "apehex": (250, 194, 87),
    "xmission": (236, 116, 129),
}

CAT_TERMS = (
    "cat",
    "cats",
    "kitten",
    "kitty",
    "feline",
    "lion",
    "tiger",
    "panther",
    "leopard",
    "cheetah",
    "lynx",
    "puma",
    "cougar",
)


@dataclass(frozen=True)
class ArtItem:
    source: str
    source_index: str
    title: str
    text: str
    width: int
    height: int
    digest: str
    renderable: bool = True


def ensure_download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 10_000:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        path.write_bytes(response.read())


def normalize_art(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", "        ")
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def make_item(source: str, source_index: str, title: str, text: str, renderable: bool = True) -> ArtItem | None:
    text = normalize_art(text)
    if not text or "\n" not in text:
        return None
    lines = text.split("\n")
    width = max((len(line) for line in lines), default=0)
    height = len(lines)
    if width < 2 or height < 2:
        return None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return ArtItem(source, source_index, title, text, width, height, digest, renderable)


def tag_text_with_br(tag) -> str:
    raw = "".join(str(child) for child in tag.contents)
    raw = re.sub(r"<br\s*/?>", "@@BR@@", raw, flags=re.I)
    raw = raw.replace("\r", "").replace("\n", "")
    raw = raw.replace("@@BR@@", "\n")
    return BeautifulSoup(raw, "html.parser").get_text("")


def load_repo_items() -> list[ArtItem]:
    items: list[ArtItem] = []
    for path in sorted(REPO_DATA_ROOT.glob("*/content.txt"), key=lambda p: int(p.parent.name)):
        item = make_item("repo", path.parent.name, "local training content.txt", path.read_text(encoding="utf-8"))
        if item:
            items.append(item)
    return items


def load_asciiart_eu_items() -> list[ArtItem]:
    ensure_download("https://www.asciiart.eu/animals/cats", ASCIIART_EU_HTML)
    soup = BeautifulSoup(ASCIIART_EU_HTML.read_text(encoding="utf-8", errors="replace"), "html.parser")
    items: list[ArtItem] = []
    for idx, card in enumerate(soup.select(".art-card")):
        art = card.select_one(".art-card__ascii")
        if not art:
            continue
        title = card.get("data-title") or "cat"
        item = make_item("asciiart.eu", str(idx), title, art.get_text(""))
        if item:
            items.append(item)
    return items


def load_xmission_items() -> tuple[list[ArtItem], list[ArtItem]]:
    ensure_download("https://user.xmission.com/~emailbox/ascii_cats.htm", XMISSION_HTML)
    soup = BeautifulSoup(XMISSION_HTML.read_text(encoding="iso-8859-1", errors="replace"), "html.parser")
    all_items: list[ArtItem] = []
    for idx, tag in enumerate(soup.find_all(["tt", "pre"])):
        item = make_item("xmission", str(idx), "CatStuff archive item", tag_text_with_br(tag), renderable=False)
        if item and len(item.text) >= 12:
            all_items.append(item)

    # Dimension-stratified representative subset. The full XMission dimension
    # catalog is still plotted and reported; this avoids republishing the archive.
    ordered = sorted(all_items, key=lambda item: (item.width * item.height, item.height, item.width))
    if len(ordered) <= 60:
        chosen = ordered
    else:
        chosen = []
        used = set()
        for n in range(60):
            pos = round(n * (len(ordered) - 1) / 59)
            while pos in used and pos + 1 < len(ordered):
                pos += 1
            used.add(pos)
            chosen.append(ordered[pos])
    chosen = [
        ArtItem(item.source, item.source_index, item.title, item.text, item.width, item.height, item.digest, True)
        for item in chosen
    ]
    return all_items, chosen


def load_apehex_items() -> list[ArtItem]:
    from datasets import load_dataset

    ds = load_dataset("apehex/ascii-art", data_files="asciiart/train/animals.parquet", split="train")
    items: list[ArtItem] = []
    for idx, row in enumerate(ds):
        haystack = f"{row.get('labels', '')} {row.get('caption', '')}".lower()
        if not any(term in haystack for term in CAT_TERMS):
            continue
        title = row.get("caption") or row.get("labels") or "ApeHex cat-family item"
        item = make_item("apehex", str(idx), str(title), row["content"])
        if item:
            items.append(item)
    return items


def fit_font(item: ArtItem, max_w: int, max_h: int) -> ImageFont.FreeTypeFont:
    for size in range(24, 5, -1):
        font = ImageFont.truetype(MONO_FONT, size)
        box = font.getbbox("M")
        char_w = box[2] - box[0]
        line_h = max(8, int((box[3] - box[1]) * 1.28))
        if item.width * char_w <= max_w and item.height * line_h <= max_h:
            return font
    return ImageFont.truetype(MONO_FONT, 6)


def draw_scatter(draw: ImageDraw.ImageDraw, items: list[ArtItem], current: list[ArtItem], box: tuple[int, int, int, int], ui_font: ImageFont.FreeTypeFont) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill=(14, 18, 23), outline=(61, 72, 84))
    draw.text((x0 + 14, y0 + 10), "dimension map: width x height", fill=(220, 228, 235), font=ui_font)
    plot = (x0 + 18, y0 + 38, x1 - 18, y1 - 22)
    px0, py0, px1, py1 = plot
    draw.rectangle(plot, fill=(7, 10, 14), outline=(40, 48, 58))
    max_w = max(item.width for item in items)
    max_h = max(item.height for item in items)
    for item in items:
        color = SOURCE_COLORS.get(item.source, (180, 180, 180))
        x = px0 + int((item.width / max_w) * (px1 - px0))
        y = py1 - int((item.height / max_h) * (py1 - py0))
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    for item in current:
        x = px0 + int((item.width / max_w) * (px1 - px0))
        y = py1 - int((item.height / max_h) * (py1 - py0))
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 255, 255), width=2)
    draw.text((px0, py1 + 4), f"max observed: {max_w} cols x {max_h} lines", fill=(145, 158, 170), font=ui_font)


def draw_panel(draw: ImageDraw.ImageDraw, item: ArtItem, box: tuple[int, int, int, int], ui_font: ImageFont.FreeTypeFont) -> None:
    x0, y0, x1, y1 = box
    color = SOURCE_COLORS.get(item.source, (180, 180, 180))
    draw.rounded_rectangle(box, radius=8, fill=(17, 22, 28), outline=color, width=1)
    label = f"{item.source}:{item.source_index}  {item.width}x{item.height}  {item.digest}"
    draw.text((x0 + 12, y0 + 9), label, fill=(230, 236, 241), font=ui_font)
    art_box = (x0 + 12, y0 + 33, x1 - 12, y1 - 12)
    ax0, ay0, ax1, ay1 = art_box
    draw.rectangle(art_box, fill=(6, 9, 12), outline=(42, 50, 58))
    font = fit_font(item, ax1 - ax0 - 10, ay1 - ay0 - 10)
    font_box = font.getbbox("M")
    line_h = max(8, int((font_box[3] - font_box[1]) * 1.28))
    y = ay0 + 5
    for line in item.text.split("\n"):
        draw.text((ax0 + 5, y), line, fill=(239, 241, 243), font=font)
        y += line_h


def draw_frame(render_items: list[ArtItem], dimension_items: list[ArtItem], frame_no: int, title_font: ImageFont.FreeTypeFont, ui_font: ImageFont.FreeTypeFont) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (9, 12, 16))
    draw = ImageDraw.Draw(image)
    t = frame_no / max(1, TOTAL_FRAMES - 1)
    center = int(round(t * (len(render_items) - 1)))
    active = [render_items[(center + offset) % len(render_items)] for offset in range(PANELS)]

    draw.rectangle((0, 0, WIDTH, 76), fill=(14, 20, 26))
    draw.text((28, 16), "ASCII cat dimension proof v2", fill=(245, 248, 250), font=title_font)
    subtitle = f"{len(render_items)} rendered representatives + {len(dimension_items)} measured source items | adaptive monospace fit | 45 seconds"
    draw.text((29, 50), subtitle, fill=(158, 174, 187), font=ui_font)

    draw_scatter(draw, dimension_items, active, (28, 92, 412, 265), ui_font)

    legend_x = 435
    for name, color in SOURCE_COLORS.items():
        draw.rectangle((legend_x, 103, legend_x + 12, 115), fill=color)
        draw.text((legend_x + 18, 99), name, fill=(190, 202, 213), font=ui_font)
        legend_x += 130

    margin = 28
    gap = 14
    top = 286
    panel_w = (WIDTH - margin * 2 - gap * 2) // 3
    panel_h = 166
    boxes = []
    for row in range(2):
        for col in range(3):
            x0 = margin + col * (panel_w + gap)
            y0 = top + row * (panel_h + gap)
            boxes.append((x0, y0, x0 + panel_w, y0 + panel_h))
    for item, box in zip(active, boxes):
        draw_panel(draw, item, box, ui_font)

    progress_w = WIDTH - 56
    draw.rectangle((28, 682, 28 + progress_w, 692), fill=(39, 47, 55))
    draw.rectangle((28, 682, 28 + int(progress_w * t), 692), fill=(109, 205, 156))
    current = render_items[center]
    draw.text((28, 653), f"current: {current.source}:{current.source_index}  {current.width} columns x {current.height} lines", fill=(176, 188, 199), font=ui_font)
    return image


def source_summary(items: list[ArtItem]) -> list[str]:
    lines = []
    for source in sorted({item.source for item in items}):
        source_items = [item for item in items if item.source == source]
        widths = [item.width for item in source_items]
        heights = [item.height for item in source_items]
        lines.append(
            f"{source}: {len(source_items)} items, width {min(widths)}..{max(widths)}, height {min(heights)}..{max(heights)}"
        )
    return lines


def write_report(render_items: list[ArtItem], dimension_items: list[ArtItem], xmission_all: list[ArtItem]) -> None:
    report = [
        "ASCII cat dimension proof v2",
        "",
        "Measured sources:",
        *source_summary(dimension_items),
        "",
        "Rendered representative sources:",
        *source_summary(render_items),
        "",
        f"XMission full measured catalog: {len(xmission_all)} items; rendered dimension-stratified subset: {sum(1 for i in render_items if i.source == 'xmission')}",
        "ApeHex source: apehex/ascii-art, asciiart/train/animals.parquet, cat-family labels/captions.",
        "Renderer change from v1: adaptive multi-panel fit plus global width/height scatterplot.",
        "Original v1 script intentionally left unchanged.",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    repo_items = load_repo_items()
    asciiart_items = load_asciiart_eu_items()
    xmission_all, xmission_subset = load_xmission_items()
    apehex_items = load_apehex_items()

    dimension_items = repo_items + asciiart_items + xmission_all + apehex_items
    render_items = repo_items + asciiart_items + xmission_subset + apehex_items
    render_items = sorted(render_items, key=lambda item: (item.width * item.height, item.height, item.width, item.source, item.source_index))

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAMES_DIR.glob("frame_*.png"):
        old.unlink()

    title_font = ImageFont.truetype(UI_FONT, 27)
    ui_font = ImageFont.truetype(UI_FONT, 14)
    for frame_no in range(TOTAL_FRAMES):
        frame = draw_frame(render_items, dimension_items, frame_no, title_font, ui_font)
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
    write_report(render_items, dimension_items, xmission_all)
    print(MP4_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
