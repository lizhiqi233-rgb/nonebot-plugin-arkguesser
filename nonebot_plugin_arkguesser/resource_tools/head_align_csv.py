"""六星精二：根据 char_arts 与 char_avatar 匹配头像在立绘中的位置，写入 char_e2_head_align.csv。"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .cv_head_match import compute_e2_head_alignment, write_head_align_debug_png
except ImportError:
    from cv_head_match import compute_e2_head_alignment, write_head_align_debug_png

# 与 char_avatar 同级：整身立绘 + 匹配矩形 + ref 头像缩略图
CHAR_AVATAR_ALIGN_DEBUG_REL_DIR = "char_avatar_align_debug"

CSV_FIELDS = [
    "char_id",
    "name",
    "art_w",
    "art_h",
    "avatar_x",
    "avatar_y",
    "avatar_w",
    "avatar_h",
    "head_cx",
    "head_cy",
    "slice_scale_512",
    "norm_head_cx",
    "norm_head_cy",
    "match_mse",
    "status",
    "error",
]


def _load_existing_align_by_char_id(out_csv: Path) -> Dict[str, Dict[str, str]]:
    """读取已有 char_e2_head_align.csv，按 char_id 索引（用于增量跳过识别）。"""
    out: Dict[str, Dict[str, str]] = {}
    if not out_csv.is_file():
        return out
    try:
        with out_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = (row.get("char_id") or "").strip()
                if not cid:
                    continue
                out[cid] = {field: (row.get(field) or "").strip() for field in CSV_FIELDS}
    except OSError:
        pass
    return out


def _row_from_existing(existing: Dict[str, str], cid: str, name: str) -> Dict[str, Any]:
    """复用已有表行，仅刷新 name（干员显示名可能随 CSV 更新）。"""
    base: Dict[str, Any] = {field: existing.get(field, "") for field in CSV_FIELDS}
    base["char_id"] = cid
    base["name"] = name
    return base


def _load_six_star_rows(csv_path: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not csv_path.is_file():
        return out
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int((row.get("rarity") or "0").strip()) != 6:
                    continue
            except ValueError:
                continue
            cid = (row.get("id") or "").strip()
            name = (row.get("name") or "").strip()
            if cid and name:
                out.append((cid, name))
    return out


def _clear_align_debug_pngs(debug_dir: Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for p in debug_dir.glob("*_align_debug.png"):
        try:
            p.unlink()
        except OSError:
            pass


def _emit_align_debug(
    debug_dir: Optional[Path],
    art_path: Path,
    avatar_path: Path,
    cid: str,
    base: Dict[str, Any],
) -> None:
    if debug_dir is None:
        return
    out_dbg = debug_dir / f"{cid}_align_debug.png"
    st = str(base.get("status") or "")
    err = str(base.get("error") or "")
    try:
        if st == "skip_no_art":
            return
        if st == "ok":
            ax = int(base["avatar_x"])
            ay = int(base["avatar_y"])
            aw = int(base["avatar_w"])
            ah = int(base["avatar_h"])
            mse = float(base["match_mse"])
            write_head_align_debug_png(
                art_path,
                out_dbg,
                char_id=cid,
                status="ok",
                match_rect=(ax, ay, aw, ah),
                match_mse=mse,
                error="",
                avatar_path=avatar_path if avatar_path.is_file() else None,
            )
            return
        if st == "match_failed":
            write_head_align_debug_png(
                art_path,
                out_dbg,
                char_id=cid,
                status="match_failed",
                match_rect=None,
                match_mse=None,
                error=err,
                avatar_path=avatar_path if avatar_path.is_file() else None,
            )
            return
        if st == "skip_no_avatar":
            write_head_align_debug_png(
                art_path,
                out_dbg,
                char_id=cid,
                status="skip_no_avatar",
                match_rect=None,
                match_mse=None,
                error=err,
                avatar_path=None,
            )
            return
    except Exception:
        pass


def _process_one(
    cid: str,
    name: str,
    art_path: Path,
    avatar_path: Path,
    min_art_bytes: int,
    min_avatar_bytes: int,
    debug_dir: Optional[Path],
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "char_id": cid,
        "name": name,
        "art_w": "",
        "art_h": "",
        "avatar_x": "",
        "avatar_y": "",
        "avatar_w": "",
        "avatar_h": "",
        "head_cx": "",
        "head_cy": "",
        "slice_scale_512": "",
        "norm_head_cx": "",
        "norm_head_cy": "",
        "match_mse": "",
        "status": "",
        "error": "",
    }
    if not art_path.is_file() or art_path.stat().st_size < min_art_bytes:
        base["status"] = "skip_no_art"
        base["error"] = "missing_or_small_char_art"
        _emit_align_debug(debug_dir, art_path, avatar_path, cid, base)
        return base
    if not avatar_path.is_file() or avatar_path.stat().st_size < min_avatar_bytes:
        base["status"] = "skip_no_avatar"
        base["error"] = "missing_or_small_char_avatar"
        _emit_align_debug(debug_dir, art_path, avatar_path, cid, base)
        return base
    try:
        r = compute_e2_head_alignment(art_path, avatar_path)
        base.update(
            {
                "art_w": r["art_w"],
                "art_h": r["art_h"],
                "avatar_x": r["avatar_x"],
                "avatar_y": r["avatar_y"],
                "avatar_w": r["avatar_w"],
                "avatar_h": r["avatar_h"],
                "head_cx": r["head_cx"],
                "head_cy": r["head_cy"],
                "slice_scale_512": r["slice_scale_512"],
                "norm_head_cx": r["norm_head_cx"],
                "norm_head_cy": r["norm_head_cy"],
                "match_mse": r["match_mse"],
                "status": "ok",
                "error": "",
            }
        )
    except Exception as e:
        base["status"] = "match_failed"
        base["error"] = str(e)[:500]
    _emit_align_debug(debug_dir, art_path, avatar_path, cid, base)
    return base


def rebuild_char_e2_head_align_csv(
    data_dir: Path,
    characters_csv: Path,
    out_csv: Path,
    *,
    min_art_bytes: int = 512,
    min_avatar_bytes: int = 128,
    max_workers: int = 4,
    write_debug_images: bool = False,
    reuse_existing_rows: bool = True,
) -> Tuple[int, int, int, int]:
    """
    重写 char_e2_head_align.csv（仅包含当前 characters.csv 中的六星干员）。

    返回 (ok_count, skip_count, fail_count, reused_existing_count)：
    - skip：本次执行识别时因缺立绘/缺头像跳过；
    - reused_existing：out_csv 中已有该 char_id 时直接复用行、不跑图像识别（增量）。

    write_debug_images=True 时在 data_dir/char_avatar_align_debug/ 下生成调试图（默认关闭）。
    reuse_existing_rows=False 时忽略已有表，对全员重新识别（全量重算）。
    """
    char_arts = data_dir / "char_arts"
    char_avatar = data_dir / "char_avatar"
    debug_dir: Optional[Path] = None
    if write_debug_images:
        debug_dir = data_dir / CHAR_AVATAR_ALIGN_DEBUG_REL_DIR
        _clear_align_debug_pngs(debug_dir)

    rows_in = _load_six_star_rows(characters_csv)
    if not rows_in:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
        return (0, 0, 0, 0)

    existing_by_id = (
        _load_existing_align_by_char_id(out_csv) if reuse_existing_rows else {}
    )

    tasks: List[Tuple[str, str, Path, Path]] = []
    reused_rows: List[Dict[str, Any]] = []
    for cid, name in rows_in:
        if cid in existing_by_id:
            reused_rows.append(_row_from_existing(existing_by_id[cid], cid, name))
            continue
        tasks.append(
            (
                cid,
                name,
                char_arts / f"{cid}_2.png",
                char_avatar / f"{cid}_2.png",
            )
        )

    results: List[Dict[str, Any]] = list(reused_rows)
    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futs = [
                ex.submit(
                    _process_one,
                    cid,
                    name,
                    ap,
                    avp,
                    min_art_bytes,
                    min_avatar_bytes,
                    debug_dir,
                )
                for cid, name, ap, avp in tasks
            ]
            for fu in as_completed(futs):
                results.append(fu.result())

    results.sort(key=lambda r: r["char_id"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    ok_c = sum(1 for r in results if r["status"] == "ok")
    skip_c = sum(1 for r in results if str(r["status"]).startswith("skip"))
    fail_c = len(results) - ok_c - skip_c
    reused_c = len(reused_rows)
    return (ok_c, skip_c, fail_c, reused_c)
