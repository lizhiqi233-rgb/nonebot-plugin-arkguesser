#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用与 ``scripts/test_resource_tools_pipeline.py`` 相同的数据目录（NoneBot localstore 下
``nonebot_plugin_arkguesser``）中的 ``characters.csv``、本地 ``char_arts``、
``char_e2_head_align.csv``，批量生成**六星干员「公布答案」面板图**（与游戏内 ``render_correct_answer`` 一致）。

**前置**：先跑通资源流水线，例如::

    pip install -e ".[head-align]"
    python scripts/test_resource_tools_pipeline.py full

**用法**（仓库根目录）::

    python scripts/render_six_star_answer_panels.py
    python scripts/render_six_star_answer_panels.py --out ./out_answers
    python scripts/render_six_star_answer_panels.py --limit 5

**说明**：须先 ``nonebot.init()`` + ``load_plugin("nonebot_plugin_arkguesser")``。数据根目录与 ``render_correct_answer`` 一致
（``game_tools.render._plugin_data_dir``，即 ``data_update.DATA_DIR``，含权限回退到 ``~/.arkguesser/data`` 的情形）。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import re
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_repo_on_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _bootstrap_nonebot() -> None:
    import nonebot

    nonebot.init()
    nonebot.load_plugin("nonebot_plugin_arkguesser")


def _parse_attack_speed(speed_str: str) -> float:
    try:
        if speed_str and isinstance(speed_str, str):
            m = re.search(r"(\d+\.?\d*)", speed_str)
            if m:
                return float(m.group(1))
    except (ValueError, TypeError):
        pass
    return 0.0


def _char_art_urls(char_id: str, rarity: int) -> tuple[str, str]:
    level = 2 if int(rarity or 0) >= 4 else 1
    char_art = f"https://torappu.prts.wiki/assets/char_arts/{char_id}_{level}.png"
    char_art_e2 = f"https://torappu.prts.wiki/assets/char_arts/{char_id}_2.png"
    return char_art, char_art_e2


def _operator_from_csv_row(row: dict[str, str], idx: int) -> dict[str, Any]:
    """与 ``game_tools.game.OperatorGuesser._load_data`` 字段一致，供 ``render_correct_answer`` 使用。"""
    operator_name = (row.get("name") or "").strip()
    char_id = (row.get("id") or "").strip()
    rarity_int = int((row.get("rarity") or "0").strip())
    tags_raw = [
        row.get("tag1", ""),
        row.get("tag2", ""),
        row.get("tag3", ""),
        row.get("tag4", ""),
    ]
    tags = [t.strip() for t in tags_raw if (t or "").strip()]
    char_art, char_art_e2 = _char_art_urls(char_id, rarity_int)
    return {
        "id": idx,
        "name": operator_name,
        "enName": char_id,
        "profession": (row.get("career") or "未知").strip(),
        "subProfession": (row.get("subcareer") or "未知").strip(),
        "rarity": rarity_int,
        "origin": (row.get("birthplace") or "未知").strip(),
        "race": (row.get("race") or "未知").strip(),
        "gender": (row.get("gender") or "").strip(),
        "parentFaction": (row.get("camp") or "无").strip() or "无",
        "faction": (row.get("subcamp") or "无").strip() or "无",
        "position": (row.get("position") or "未知").strip(),
        "tags": tags,
        "charArt": char_art,
        "charArtE2": char_art_e2,
        "attack": int(row.get("max_atk") or 0),
        "defense": int(row.get("max_def") or 0),
        "hp": int(row.get("max_hp") or 0),
        "res": int(row.get("max_magic_res") or 0),
        "interval": _parse_attack_speed(row.get("attack_speed", "0") or "0"),
        "cost": int(row.get("deploy_cost") or 0),
    }


def _load_six_star_operators(characters_csv: Path) -> list[dict[str, Any]]:
    if not characters_csv.is_file():
        raise FileNotFoundError(f"未找到 {characters_csv}，请先运行 test_resource_tools_pipeline.py full")
    out: list[dict[str, Any]] = []
    with characters_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            try:
                if int((row.get("rarity") or "0").strip()) != 6:
                    continue
            except ValueError:
                continue
            cid = (row.get("id") or "").strip()
            if not cid:
                continue
            out.append(_operator_from_csv_row(row, idx))
    return out


async def _render_one(
    op: dict[str, Any],
    out_dir: Path,
    sem: asyncio.Semaphore,
    index: int,
    total: int,
) -> tuple[str, bool, str]:
    from nonebot_plugin_arkguesser.game_tools.render import render_correct_answer

    cid = str(op.get("enName") or "unknown")
    async with sem:
        try:
            png = await render_correct_answer(op, mode="大头")
            safe = cid.replace("/", "_")
            path = out_dir / f"{safe}_answer.png"
            path.write_bytes(png)
            print(f"[{index}/{total}] OK {cid} -> {path.name}")
            return (cid, True, "")
        except Exception as e:
            err = str(e)
            print(f"[{index}/{total}] FAIL {cid}: {err}")
            return (cid, False, err)


async def _amain(out_dir: Path, limit: int | None, concurrency: int) -> int:
    # 与游戏内面板一致：须与 ``game_tools.render._plugin_data_dir`` 相同（含 data_update 权限回退目录）
    from nonebot_plugin_arkguesser.game_tools.render import _plugin_data_dir

    data_root = _plugin_data_dir()
    characters_csv = data_root / "characters.csv"
    ops = _load_six_star_operators(characters_csv)
    if not ops:
        print("未找到六星干员（characters.csv 为空或无 rarity==6）")
        return 1
    if limit is not None and limit > 0:
        ops = ops[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"数据目录: {data_root}")
    print(f"共 {len(ops)} 名六星，输出: {out_dir.resolve()}")

    sem = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        _render_one(op, out_dir, sem, i + 1, len(ops))
        for i, op in enumerate(ops)
    ]
    results = await asyncio.gather(*tasks)
    ok_c = sum(1 for _, ok, _ in results if ok)
    print(f"完成: 成功 {ok_c} / {len(results)}")
    return 0 if ok_c == len(results) else 1


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_on_path()
    parser = argparse.ArgumentParser(
        description="批量生成六星「公布答案」面板 PNG（使用 test_resource_tools_pipeline 同款数据）"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录（默认：仓库下 output/six_star_answer_panels）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅渲染前 N 名（0 表示全部）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="并发数（html_to_pic 较重，默认 2）",
    )
    args = parser.parse_args(argv)

    out_dir = args.out
    if out_dir is None:
        out_dir = _repo_root() / "output" / "six_star_answer_panels"

    lim = args.limit if args.limit and args.limit > 0 else None

    _bootstrap_nonebot()
    return asyncio.run(_amain(out_dir, lim, args.concurrency))


if __name__ == "__main__":
    raise SystemExit(main())
