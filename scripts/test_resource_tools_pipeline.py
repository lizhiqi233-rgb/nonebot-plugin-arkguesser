#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地测试：按与机器人内相同的调用链执行 resource_tools（可选完整更新或仅头像对齐）。

不修改 ``nonebot_plugin_arkguesser/resource_tools`` 内任何文件；本脚本放在仓库 ``scripts/`` 下。

**默认跳过数据获取**（不拉 PRTS、不下载六星立绘/头像）：无参数时等价 ``head-align``，需已有
``characters.csv`` 与 ``char_arts`` / ``char_avatar`` 等本地数据。完整联网更新请显式传 ``full``。

**依赖**（与插件一致）::

    pip install -e ".[head-align]"

**必须先** ``nonebot.init()`` **再** ``load_plugin("nonebot_plugin_arkguesser")``，
否则 ``nonebot_plugin_localstore`` 在导入 ``game_tools.pool_manager`` 时会报
``Cannot detect caller plugin``。

**模式说明**

- ``full``：调用 ``data_update.update_data()``，串联 PRTS 拉取 CSV、六星精二立绘/头像下载、
  ``char_art_match.rebuild_char_e2_head_align_csv``（与插件内「arkstart 更新」一致）。
- ``download-images``：仅 ``char_art_update.sync_six_star_elite2_*``，需已有 ``characters.csv``。
- ``head-align``：仅 ``char_art_match.rebuild_char_e2_head_align_csv``，需已有 CSV 与本地图片。

本脚本**默认开启**头像对齐调试图（未设置时等价 ``ARKGUESSER_HEAD_ALIGN_DEBUG=1``），输出在
``<DATA_DIR>/char_avatar_align_debug/``。若需关闭，运行前设置 ``ARKGUESSER_HEAD_ALIGN_DEBUG=0``。

**头像对齐（本脚本默认）**：**不复用**已有 ``char_e2_head_align.csv`` 行，全员重跑 OpenCV 并**覆盖**对齐表
（等价默认 ``ARKGUESSER_HEAD_ALIGN_REBUILD_ALL=1``）。若需增量复用旧行，传 ``--reuse-align`` 或自行设
``ARKGUESSER_HEAD_ALIGN_REBUILD_ALL=0``。

**注意**：调试图上**黄框**为粗候选细 ROI **交叠合并**后的各块外接矩形（每块一次细扫）；**绿框**为最终**整颗头**范围。
PNG 上会叠加 ``scheme2 … e2_zoom=…``，与面板内归一化一致。最终效果仍以 ``render_guess_result`` / 游戏内为准。

用法（在仓库根目录）::

    python scripts/test_resource_tools_pipeline.py
    python scripts/test_resource_tools_pipeline.py full
    python scripts/test_resource_tools_pipeline.py download-images
    python scripts/test_resource_tools_pipeline.py head-align
    python scripts/test_resource_tools_pipeline.py --reuse-align

上表第一行与 ``head-align`` 相同（默认本地对齐）；``full`` 才会执行维基拉取与资源下载。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_repo_on_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _bootstrap_nonebot() -> None:
    """初始化 NoneBot 并加载本插件，使 localstore 能解析数据目录。"""
    import nonebot

    nonebot.init()
    nonebot.load_plugin("nonebot_plugin_arkguesser")


async def _run_full_update() -> bool:
    from nonebot_plugin_arkguesser.resource_tools.data_update import update_data

    return bool(await update_data())


async def _run_download_images_only() -> None:
    import httpx
    from nonebot_plugin_arkguesser.resource_tools.char_art_update import (
        sync_six_star_elite2_arts,
        sync_six_star_elite2_avatars,
    )
    from nonebot_plugin_arkguesser.resource_tools.data_update import (
        CHARACTERS_FILE,
        DATA_DIR,
    )

    if not CHARACTERS_FILE.is_file():
        raise FileNotFoundError(
            f"未找到 {CHARACTERS_FILE}，请先执行 full 模式或自行准备 characters.csv"
        )

    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        await sync_six_star_elite2_arts(CHARACTERS_FILE, DATA_DIR, client)
        await sync_six_star_elite2_avatars(CHARACTERS_FILE, DATA_DIR, client)


async def _run_head_align_only() -> tuple[int, int, int, int]:
    from nonebot_plugin_arkguesser.resource_tools.char_art_match import (
        rebuild_char_e2_head_align_csv,
    )
    from nonebot_plugin_arkguesser.resource_tools.char_art_update import (
        CHAR_ART_LOCAL_MIN_BYTES,
        CHAR_AVATAR_LOCAL_MIN_BYTES,
    )
    from nonebot_plugin_arkguesser.resource_tools.data_update import (
        CHARACTERS_FILE,
        CHAR_E2_HEAD_ALIGN_FILE,
        DATA_DIR,
    )

    _head_dbg = os.environ.get("ARKGUESSER_HEAD_ALIGN_DEBUG", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    _head_align_full = os.environ.get(
        "ARKGUESSER_HEAD_ALIGN_REBUILD_ALL", "0"
    ).strip().lower() in ("1", "true", "yes")

    def _sync_rebuild() -> tuple[int, int, int, int]:
        return rebuild_char_e2_head_align_csv(
            DATA_DIR,
            CHARACTERS_FILE,
            CHAR_E2_HEAD_ALIGN_FILE,
            min_art_bytes=CHAR_ART_LOCAL_MIN_BYTES,
            min_avatar_bytes=CHAR_AVATAR_LOCAL_MIN_BYTES,
            max_workers=4,
            write_debug_images=_head_dbg,
            reuse_existing_rows=not _head_align_full,
        )

    return await asyncio.to_thread(_sync_rebuild)


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_on_path()

    parser = argparse.ArgumentParser(
        description="测试 resource_tools：数据下载与头像对齐（OpenCV）"
    )
    parser.add_argument(
        "--reuse-align",
        action="store_true",
        help="复用已有 char_e2_head_align.csv 中已存在的 char_id 行（增量）；默认为本脚本全量重算并覆盖对齐表",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="head-align",
        choices=("full", "download-images", "head-align"),
        help="默认 head-align（仅本地对齐，不联网拉数据）；full=维基+下载+对齐；download-images=仅六星立绘/头像",
    )
    args = parser.parse_args(argv)

    # 默认全量重算头像对齐（与机器人内 update_data 默认增量相反，避免测试时沿用旧位置）
    if args.reuse_align:
        os.environ["ARKGUESSER_HEAD_ALIGN_REBUILD_ALL"] = "0"
    else:
        os.environ.setdefault("ARKGUESSER_HEAD_ALIGN_REBUILD_ALL", "1")

    # 与 data_update 中逻辑一致：默认开启 char_avatar_align_debug（未设置环境变量时）
    os.environ.setdefault("ARKGUESSER_HEAD_ALIGN_DEBUG", "1")

    _bootstrap_nonebot()

    if args.mode == "full":
        ok = asyncio.run(_run_full_update())
        print("update_data:", "成功" if ok else "失败")
        if ok:
            from nonebot_plugin_arkguesser.resource_tools.char_art_match import (
                CHAR_AVATAR_ALIGN_DEBUG_REL_DIR,
            )
            from nonebot_plugin_arkguesser.resource_tools.data_update import DATA_DIR

            print(f"头像对齐调试图: {DATA_DIR / CHAR_AVATAR_ALIGN_DEBUG_REL_DIR}")
        return 0 if ok else 1

    if args.mode == "download-images":
        try:
            asyncio.run(_run_download_images_only())
        except Exception as e:
            print("download-images 失败:", e)
            return 1
        print("download-images: 完成（六星精二立绘与头像已尝试同步）")
        return 0

    if args.mode == "head-align":
        try:
            ok_c, skip_c, fail_c, reused_c = asyncio.run(_run_head_align_only())
        except ImportError as e:
            print(
                "head-align 需要 OpenCV/NumPy，请: pip install \"nonebot-plugin-arkguesser[head-align]\""
            )
            print("ImportError:", e)
            return 1
        except Exception as e:
            print("head-align 失败:", e)
            return 1
        from nonebot_plugin_arkguesser.resource_tools.char_art_match import (
            CHAR_AVATAR_ALIGN_DEBUG_REL_DIR,
        )
        from nonebot_plugin_arkguesser.resource_tools.data_update import (
            CHAR_E2_HEAD_ALIGN_FILE,
            DATA_DIR,
        )

        print(
            f"head-align: 匹配成功={ok_c}, 跳过={skip_c}, 失败={fail_c}, 复用已有行={reused_c}"
        )
        print(f"DATA_DIR: {DATA_DIR}")
        print(f"对齐表: {CHAR_E2_HEAD_ALIGN_FILE}")
        print(f"头像对齐调试图: {DATA_DIR / CHAR_AVATAR_ALIGN_DEBUG_REL_DIR}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
