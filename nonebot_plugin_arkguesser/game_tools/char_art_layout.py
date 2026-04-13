"""精二立绘在 new_base 面板中的 SVG 变换：CSV 锚点与 PRTS 视口换算。"""

from __future__ import annotations

import csv
from pathlib import Path

# 须与 resources/templates/new_theme.css 中 --char-art-* 一致
CHAR_ART_PRTS_VIEWPORT_PX = 512.0
CHAR_ART_PANEL_PX = 540.0
# 旧版固定 2.5；现改为按绿框在 slice 后视口内跨度动态计算，仅作文档/兼容参照
CHAR_E2_HEAD_ZOOM = 2.5
# 希望绿框在 512 视口里呈现的目标特征跨度（px）；zoom ≈ TARGET / max(rw*slice, rh*slice)
# 略小则整体放大倍率降低，头肩与背景留出更多（曾用 280 偏「怼脸」）
CHAR_E2_HEAD_TARGET_VP_SPAN = 100.0
CHAR_E2_HEAD_ZOOM_MIN = 1.6
CHAR_E2_HEAD_ZOOM_MAX = 12.0


def svg_chip_width_px(
    text: object,
    *,
    cjk_per_em: float = 15.0,
    latin_per_em: float = 7.2,
    padding: float = 14.0,
    min_w: float = 22.0,
) -> int:
    """
    new_base.html 中势力/标签 chip 的 rect 宽度（字号约 15 的 SVG 文本）。

    cjk_per_em 与 font-size 15 对齐，避免「罗德岛-精英干员」等混排时绿底左右偏窄；
    纯拉丁仍用 latin_per_em，避免 Ave Mujica 等词 chip 过宽。
    连接符 - – — · 介于两者之间，略增宽以免与汉字挤在一起。
    """
    s = "" if text is None else str(text).strip()
    if not s:
        return int(min_w)
    total = 0.0
    for ch in s:
        o = ord(ch)
        if ch.isspace():
            total += latin_per_em * 0.35
        elif ch in "-–—·":
            total += (cjk_per_em * 0.45 + latin_per_em * 0.55)
        elif o < 128:
            total += latin_per_em
        else:
            total += cjk_per_em
    return int(max(min_w, round(total + padding)))


def parse_elite2_csv_number(value: object, default: float) -> float:
    if value is None:
        return default
    s = str(value).strip()
    if not s or s in ("未知", "-", "—", "N/A", "n/a"):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def char_art_e2_svg_transform(
    elite2_scale: float,
    elite2_coord_x: float,
    elite2_coord_y: float,
    *,
    panel_px: float = CHAR_ART_PANEL_PX,
    prts_ref_px: float = CHAR_ART_PRTS_VIEWPORT_PX,
) -> tuple[float, float, float]:
    """返回 (translate_x, translate_y, scale)，用于 SVG ``translate(tx ty) scale(s)``（与 PRTS 矩阵顺序一致）。"""
    k = panel_px / prts_ref_px
    return elite2_coord_x * k, elite2_coord_y * k, elite2_scale * k


def _parse_align_float(row: dict[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _png_pixel_size(path: Path) -> tuple[int, int] | None:
    """读取 PNG IHDR 宽高，避免依赖 Pillow；非 PNG 或损坏时返回 None。"""
    try:
        with path.open("rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return None
            length = int.from_bytes(f.read(4), "big")
            ctype = f.read(4)
            if ctype != b"IHDR" or length < 8:
                return None
            data = f.read(length)
            w = int.from_bytes(data[0:4], "big")
            h = int.from_bytes(data[4:8], "big")
            if w <= 0 or h <= 0:
                return None
            return (w, h)
    except OSError:
        return None


def load_e2_head_align_row(align_csv: Path, char_id: str) -> dict[str, str] | None:
    """读取 char_e2_head_align.csv 中 char_id 匹配的首行（表中仅存成功对齐记录）。"""
    if not align_csv.is_file():
        return None
    cid = (char_id or "").strip()
    if not cid:
        return None
    try:
        with align_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("char_id") or "").strip() != cid:
                    continue
                return dict(row)
    except OSError:
        return None
    return None


def char_e2_head_inner_svg_transform(
    head_cx: float,
    head_cy: float,
    art_w: float,
    art_h: float,
    slice_scale: float,
    *,
    zoom: float,
    panel_px: float = CHAR_ART_PANEL_PX,
    vp_px: float = CHAR_ART_PRTS_VIEWPORT_PX,
) -> str:
    """
    512 逻辑视口内、在 ``scale(panel/vp)`` 之前应用的 ``<g transform>`` 字符串。

    假定 <image> 为 width=height=vp_px 且 preserveAspectRatio=xMidYMid slice；
    将立绘中 (head_cx, head_cy) 先映射到视口坐标 (vx,vy)，再 ``translate(T) scale(zoom) translate(-vx,-vy)``
    使头部落在「面板左上象限中心」在视口内的对应点 T = (panel/4)*(vp/panel) = vp/4。

    ``zoom`` 须由调用方按绿框 ``refine_w/h`` 与 ``slice_scale`` 计算：全身立绘 slice 后绿框在视口里往往很小，
    固定 2.5 会导致脸过小；动态 zoom ≈ TARGET_VP_SPAN / max(rw*slice, rh*slice)。
    """
    if art_w <= 0 or art_h <= 0 or slice_scale <= 0 or zoom <= 0:
        return ""
    vx = 256.0 + slice_scale * (head_cx - art_w / 2.0)
    vy = 256.0 + slice_scale * (head_cy - art_h / 2.0)
    # 540 均分四块，左上块中心在面板 (panel/4, panel/4) —— 注意 quadrant 宽为 panel/2，中心为 panel/4
    tx = (panel_px / 4.0) * (vp_px / panel_px)
    ty = tx
    return (
        f"translate({tx:.4f},{ty:.4f}) "
        f"scale({zoom:.4f}) "
        f"translate({-vx:.4f},{-vy:.4f})"
    )


def char_e2_inner_transform_from_row(
    row: dict[str, str],
    *,
    data_root: Path,
    char_id: str,
    elite_level: int = 2,
) -> str | None:
    """
    从 char_e2_head_align.csv 一行生成内层 transform。

    依赖 ``refine_x/y/w/h``（绿框）与本地 ``char_arts/{char_id}_{elite}.png`` 的像素尺寸。
    缩放：绿框在 slice 后的视口跨度 max(rw,rh)*slice_scale 小时自动提高 ``zoom``，
    使脸部在左半幅占比稳定；CSV 中 ``scale`` 与几何一致时可作校验，渲染以绿框边长为准。
    """
    art_path = data_root / "char_arts" / f"{char_id}_{int(elite_level)}.png"
    wh = _png_pixel_size(art_path)
    if wh is None:
        return None

    rw = _parse_align_float(row, "refine_w")
    if rw is None or rw <= 0:
        return None
    rx = _parse_align_float(row, "refine_x")
    ry = _parse_align_float(row, "refine_y")
    rh = _parse_align_float(row, "refine_h")
    if None in (rx, ry, rh) or rh <= 0:
        return None

    art_w, art_h = float(wh[0]), float(wh[1])
    head_cx = rx + rw / 2.0
    head_cy = ry + rh / 2.0
    slice_s = max(CHAR_ART_PRTS_VIEWPORT_PX / art_w, CHAR_ART_PRTS_VIEWPORT_PX / art_h)
    # 与 vx 公式一致：立绘内一段长度 L 映射到 512 用户空间约为 L * slice_s
    head_w_vp = rw * slice_s
    head_h_vp = rh * slice_s
    head_span_vp = max(head_w_vp, head_h_vp)
    zoom = CHAR_E2_HEAD_TARGET_VP_SPAN / max(head_span_vp, 1.0)
    zoom = min(max(zoom, CHAR_E2_HEAD_ZOOM_MIN), CHAR_E2_HEAD_ZOOM_MAX)
    s = char_e2_head_inner_svg_transform(
        head_cx, head_cy, art_w, art_h, slice_s, zoom=zoom
    )
    return s or None


def resolve_char_e2_inner_transform(char_id: str, data_dirs: tuple[Path, ...]) -> str:
    """在多个数据根目录下查找 char_e2_head_align.csv，返回内层 transform 或空串。"""
    cid = (char_id or "").strip()
    if not cid:
        return ""
    for root in data_dirs:
        row = load_e2_head_align_row(root / "char_e2_head_align.csv", cid)
        if row:
            t = char_e2_inner_transform_from_row(
                row, data_root=root, char_id=cid, elite_level=2
            )
            if t:
                return t
    return ""


def char_e2_left_half_center_inner_transform(
    *,
    panel_px: float = CHAR_ART_PANEL_PX,
    vp_px: float = CHAR_ART_PRTS_VIEWPORT_PX,
) -> str:
    """
    非六星精二立绘：在面板「左侧」区域水平垂直居中。

    默认无内层 transform 时，512 视口 xMidYMid slice 的中心落在全屏 (270,270)；
    左半区中心为 (panel/4, panel/2)=(135,270)，在视口内等价于将内容水平平移 -vp/4。
    """
    tx = -(panel_px / 4.0) * (vp_px / panel_px)
    return f"translate({tx:.4f},0)"


def resolve_char_e2_inner_transform_for_rarity(
    char_id: str,
    rarity: int,
    data_dirs: tuple[Path, ...],
) -> str:
    """六星：头部锚点（CSV）；非六星：左侧区域居中。"""
    if int(rarity or 0) == 6:
        return resolve_char_e2_inner_transform(char_id, data_dirs)
    return char_e2_left_half_center_inner_transform()
