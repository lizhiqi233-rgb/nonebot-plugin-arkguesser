from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from nonebot_plugin_htmlrender import html_to_pic
import nonebot_plugin_localstore as store

from .char_art_layout import (
    svg_chip_width_px,
    resolve_char_e2_inner_transform_for_rarity,
)
from .config import get_plugin_config

_TEMPLATES_DIR = Path(__file__).parent.parent / "resources" / "templates"
_RESOURCES_DIR = Path(__file__).parent.parent / "resources"
_PANEL_FONT_TTF = _RESOURCES_DIR / "fonts" / "HarmonyOS_Sans_Medium.ttf"


def _panel_font_ttf_uri() -> str:
    """
    面板自定义字体 file:// URI，供 new_base 内联 @font-face 使用。
    相对路径写在 new_theme.css 里在 html_to_pic/setContent 下常无法解析，故由 Python 传绝对路径。
    """
    try:
        if _PANEL_FONT_TTF.is_file() and _PANEL_FONT_TTF.stat().st_size > 0:
            return _PANEL_FONT_TTF.resolve().as_uri()
    except OSError:
        pass
    return ""


# 设置Jinja2环境
env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=True,
    enable_async=True
)
env.filters["svg_chip_width"] = svg_chip_width_px

_PANEL_BASE_PX = 540


def _panel_render_pixels() -> tuple[int, int, float]:
    """设计坐标系固定 540×540（viewBox），输出像素 = 基准 × render_scale。"""
    cfg = get_plugin_config()
    scale = float(cfg.render_scale)
    w = max(1, int(round(_PANEL_BASE_PX * scale)))
    return w, w, scale


def _new_theme_css_uri() -> str:
    return (_TEMPLATES_DIR / "new_theme.css").resolve().as_uri()


def _char_art_elite_level(rarity: int) -> int:
    """整身 char_arts：4 星及以上用精二 _2.png，3 星及以下用精一 _1.png（与 game._get_char_art_url 一致）。"""
    return 1 if int(rarity or 0) < 4 else 2


def _char_arts_base_uri(char_id: str, rarity: int) -> str:
    """本地 char_arts/{id}_1.png 或 _2.png 存在时返回该目录 file:// URI。"""
    cid = (char_id or "").strip()
    if not cid:
        return ""
    root = store.get_plugin_data_dir()
    d = (root / "char_arts").resolve()
    lvl = _char_art_elite_level(rarity)
    p = d / f"{cid}_{lvl}.png"
    try:
        if p.is_file() and p.stat().st_size > 0:
            return d.as_uri()
    except OSError:
        pass
    return ""


async def render_guess_result(
    guessed_operator: dict[str, Any] | None,
    comparison: dict[str, Any],
    attempts_left: int,
    mode: str = "大头"
) -> bytes:
    width, height, _scale = _panel_render_pixels()
    op = guessed_operator or {}
    char_id = (op.get("enName") or "").strip()
    rarity_i = int(op.get("rarity") or 0)
    arts_uri = _char_arts_base_uri(char_id, rarity_i)
    char_e2_tf = (
        resolve_char_e2_inner_transform_for_rarity(char_id, rarity_i, (store.get_plugin_data_dir(),))
        if char_id
        else ""
    )
    # 无本地文件时用 charArt（已按星级为精一/精二整身图），勿用 charArtE2（恒为 _2）
    _fallback_art = (op.get("charArt") or op.get("charArtE2") or "")

    template = env.get_template("new_base.html")
    html = await template.render_async(
        new_theme_css_uri=_new_theme_css_uri(),
        panel_font_ttf_uri=_panel_font_ttf_uri(),
        char_art_elite_level=_char_art_elite_level(rarity_i),
        operator_name=op.get("name", ""),
        attempts_left=attempts_left,
        profession=op.get("profession", "未知"),
        profession_correct=comparison.get("profession", False),
        subProfession=op.get("subProfession", "未知"),
        subProfession_correct=comparison.get("subProfession", False),
        rarity=op.get("rarity", 1),
        rarity_class=comparison.get("rarity", "same"),
        origin=op.get("origin", "未知"),
        origin_correct=comparison.get("origin", False),
        race=op.get("race", "未知"),
        race_correct=comparison.get("race", False),
        gender=op.get("gender", ""),
        gender_correct=comparison.get("gender", False),
        position=op.get("position", "未知"),
        position_correct=comparison.get("position", False),
        faction=op.get("faction", "无"),
        parent_faction=op.get("parentFaction", "") or "无",
        faction_comparison=comparison.get("faction") or {},
        tags=op.get("tags", []),
        tags_comparison=comparison.get("tags") or {},
        char_operator_id=char_id,
        char_arts_base_uri=arts_uri,
        char_art_e2_url=("" if arts_uri else _fallback_art),
        char_e2_inner_transform=char_e2_tf,
        reveal_all=False,
        name=op.get("name", ""),
        width=width,
        height=height,
    )

    return await html_to_pic(
        html, viewport={"width": width, "height": height}, device_scale_factor=1
    )

async def render_correct_answer(operator: dict[str, Any], mode: str = "大头") -> bytes:
    width, height, _scale = _panel_render_pixels()
    char_id = (operator.get("enName") or "").strip()
    rarity_i = int(operator.get("rarity") or 0)
    arts_uri = _char_arts_base_uri(char_id, rarity_i)
    char_e2_tf = (
        resolve_char_e2_inner_transform_for_rarity(char_id, rarity_i, (store.get_plugin_data_dir(),))
        if char_id
        else ""
    )
    _fallback_art = operator.get("charArt") or operator.get("charArtE2") or ""

    template = env.get_template("new_base.html")
    html = await template.render_async(
        new_theme_css_uri=_new_theme_css_uri(),
        panel_font_ttf_uri=_panel_font_ttf_uri(),
        char_art_elite_level=_char_art_elite_level(rarity_i),
        operator_name=operator.get("name", "未知干员"),
        attempts_left=0,
        profession=operator.get("profession", "未知"),
        profession_correct=True,
        subProfession=operator.get("subProfession", "未知"),
        subProfession_correct=True,
        rarity=operator.get("rarity", 1),
        rarity_class="same",
        origin=operator.get("origin", "未知"),
        origin_correct=True,
        race=operator.get("race", "未知"),
        race_correct=True,
        gender=operator.get("gender", ""),
        gender_correct=True,
        position=operator.get("position", "未知"),
        position_correct=True,
        faction=operator.get("faction", "无"),
        parent_faction=operator.get("parentFaction", "") or "无",
        faction_comparison={},
        tags=operator.get("tags", []),
        tags_comparison={},
        char_operator_id=char_id,
        char_arts_base_uri=arts_uri,
        char_art_e2_url=("" if arts_uri else _fallback_art),
        char_e2_inner_transform=char_e2_tf,
        reveal_all=True,
        name=operator.get("name", "未知干员"),
        width=width,
        height=height,
    )

    return await html_to_pic(
        html, viewport={"width": width, "height": height}, device_scale_factor=1
    )