"""立绘匹配：精二头像在整身立绘中的定位（OpenCV）与 char_e2_head_align.csv 生成。
精修阶段在可行域内用 BGR 掩膜 TM_SQDIFF（RGB 色彩误差）确定缩放与位置，粗匹配与特征排序仍为灰度/融合代价。
精修默认将缩放后头像均分 3×3，取正中格 + 中下格两小块分别匹配，代价取二者掩膜内均方和的平均（ARKGUESSER_HEAD_REFINE_GRID33）。
精修默认仅对粗 ROI 的紧包围子图做 matchTemplate（ARKGUESSER_HEAD_REFINE_TIGHT_ROI），显著快于整幅 ROI。
对齐表仅含 char_id、name、refine_x/y/w/h、scale（相对头像原图宽高的平均缩放，两位小数）。
个别干员可走 ``HEAD_ALIGN_FORCE_COARSE_CHAR_IDS`` 强制写入粗扫矩形（scale=1.00）。
"""
from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


def imread_unicode(path: str | Path) -> np.ndarray:
    p = Path(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(str(p))
    return img


def bgra_flatten_to_bgr(
    img: np.ndarray,
    bg_bgr: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    将带 alpha 的图合成到实色背景（默认黑底）。
    透明区未定义的 BGR 不再参与 matchTemplate，也避免调试图里出现竖条/色块伪影。
    """
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] < 3:
        raise ValueError("expected HxW or HxWx3+ image")
    bgr = img[:, :, :3].astype(np.float32)
    if img.shape[2] == 4:
        a = np.clip(img[:, :, 3].astype(np.float32) / 255.0, 0.0, 1.0)
        bg = np.array(bg_bgr, dtype=np.float32).reshape(1, 1, 3)
        out = a[..., np.newaxis] * bgr + (1.0 - a[..., np.newaxis]) * bg
        return np.clip(out, 0, 255).astype(np.uint8)
    return np.clip(bgr, 0, 255).astype(np.uint8)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)).strip())
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)).strip())
    except ValueError:
        return default


def _topk_separated_low_cost(
    cost: np.ndarray,
    k: int,
    min_sep: int,
) -> list[tuple[int, int]]:
    """
    按代价升序贪心选取至多 k 个位置，Chebyshev 间距 >= min_sep，覆盖多个「看起来像」的局部，
    便于后续用 ORB 等特征区分真脸与特效/背景块。
    """
    Rh, Rw = cost.shape
    work = cost.astype(np.float64)
    work[~np.isfinite(work)] = np.inf
    order = np.argsort(work, axis=None)
    picked: list[tuple[int, int]] = []
    for flat_i in order:
        ly, lx = divmod(int(flat_i), Rw)
        v = cost[ly, lx]
        if not np.isfinite(v):
            continue
        if all(
            max(abs(ly - py), abs(lx - px)) >= min_sep for py, px in picked
        ):
            picked.append((ly, lx))
        if len(picked) >= k:
            break
    if not picked:
        ly, lx = np.unravel_index(int(np.nanargmin(cost)), cost.shape)
        picked.append((ly, lx))
    return picked


def _orb_good_match_count(
    tpl_gray: np.ndarray,
    patch_gray: np.ndarray,
    mask_tpl: np.ndarray | None,
    *,
    ratio: float = 0.70,
) -> int:
    """模板与候选块之间 ORB + ratio test 合格匹配数；真头像与 char_avatar 同源时通常明显更高。"""
    if patch_gray.shape != tpl_gray.shape:
        return 0
    orb = cv2.ORB_create(
        nfeatures=720,
        scaleFactor=1.18,
        nlevels=7,
        edgeThreshold=6,
        fastThreshold=6,
    )
    kp1, d1 = orb.detectAndCompute(tpl_gray, mask_tpl)
    kp2, d2 = orb.detectAndCompute(patch_gray, None)
    if d1 is None or d2 is None or len(d1) < 4 or len(d2) < 4:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        pairs = bf.knnMatch(d1, d2, k=2)
    except cv2.error:
        return 0
    good = 0
    for m_n in pairs:
        if len(m_n) < 2:
            if len(m_n) == 1:
                good += 1
            continue
        m, n = m_n[0], m_n[1]
        if m.distance < ratio * n.distance:
            good += 1
    return good


def _akaze_good_match_count(
    tpl_gray: np.ndarray,
    patch_gray: np.ndarray,
    mask_tpl: np.ndarray | None,
    *,
    ratio: float = 0.70,
) -> int:
    """AKAZE 与 ORB 互补，对部分灰阶/线条主导的皮肤稿更稳。"""
    if patch_gray.shape != tpl_gray.shape:
        return 0
    try:
        ak = cv2.AKAZE_create()
    except AttributeError:
        return 0
    kp1, d1 = ak.detectAndCompute(tpl_gray, mask_tpl)
    kp2, d2 = ak.detectAndCompute(patch_gray, None)
    if d1 is None or d2 is None or len(d1) < 4 or len(d2) < 4:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        pairs = bf.knnMatch(d1, d2, k=2)
    except cv2.error:
        return 0
    good = 0
    for m_n in pairs:
        if len(m_n) < 2:
            if len(m_n) == 1:
                good += 1
            continue
        m, n = m_n[0], m_n[1]
        if m.distance < ratio * n.distance:
            good += 1
    return good


def _merge_spatial_dedupe(
    lists: list[list[tuple[int, int]]],
    min_sep: int,
    max_out: int,
) -> list[tuple[int, int]]:
    """合并多路候选，去掉过近重复点，最多保留 max_out 个（先出现的优先）。"""
    merged: list[tuple[int, int]] = []
    thr = max(3, min_sep // 2)
    for lst in lists:
        for ly, lx in lst:
            if any(max(abs(ly - py), abs(lx - px)) < thr for py, px in merged):
                continue
            merged.append((ly, lx))
            if len(merged) >= max_out:
                return merged
    return merged


def _refine_range_center_pixel(
    lx_r: int,
    ly_r: int,
    w: int,
    h: int,
    Wr: int,
    Hr: int,
) -> tuple[int, int] | None:
    """
    精修范围（九宫格 3w×3h 与 ROI 的交集）的轴对齐包围盒，取其**中心像素** (cx, cy)。
    若交集退化则返回 None。
    """
    bx0 = max(0, lx_r - w)
    bx1 = min(Wr - 1, lx_r + 2 * w - 1)
    by0 = max(0, ly_r - h)
    by1 = min(Hr - 1, ly_r + 2 * h - 1)
    if bx0 > bx1 or by0 > by1:
        return None
    return ((bx0 + bx1) // 2, (by0 + by1) // 2)


def _megabox_tl_bounds_scaled(
    lx_r: int,
    ly_r: int,
    w: int,
    h: int,
    ws: int,
    hs: int,
    Wr: int,
    Hr: int,
) -> tuple[int, int, int, int] | None:
    """
    粗匹配为九宫格中心格（每格 w×h）时，缩放模板 ws×hs 在 ROI 内允许的左上角范围。
    大框为 3w×3h，与粗匹配阶段搜索语义一致。
    """
    lx_min = max(0, lx_r - w)
    lx_max = min(Wr - ws, lx_r + 2 * w - ws)
    ly_min = max(0, ly_r - h)
    ly_max = min(Hr - hs, ly_r + 2 * h - hs)
    if lx_min > lx_max or ly_min > ly_max:
        return None
    return (lx_min, lx_max, ly_min, ly_max)


def _refine_tight_roi_bounds(
    lx_r: int,
    ly_r: int,
    w: int,
    h: int,
    Wr: int,
    Hr: int,
    scale_ratios: tuple[float, ...],
) -> tuple[int, int, int, int]:
    """
    精修有效搜索仅为粗位置附近九宫格及多档缩放模板覆盖区；对整块粗扫描 ROI 做 matchTemplate 浪费极大。
    返回 (ox, oy, cw, ch)，使 sub = roi[oy : oy + ch, ox : ox + cw] 覆盖：
    - 中心像素约束所用 3w×3h 与 ROI 的交集；
    - 各档缩放下 megabox 内任意 (lx, ly) 对应的模板覆盖像素。
    """
    bx0 = max(0, lx_r - w)
    bx1 = min(Wr - 1, lx_r + 2 * w - 1)
    by0 = max(0, ly_r - h)
    by1 = min(Hr - 1, ly_r + 2 * h - 1)
    xs0, ys0 = bx0, by0
    xe, ye = bx1 + 1, by1 + 1
    for s in scale_ratios:
        ws = max(8, int(round(w * float(s))))
        hs = max(8, int(round(h * float(s))))
        if ws > Wr or hs > Hr:
            continue
        box = _megabox_tl_bounds_scaled(lx_r, ly_r, w, h, ws, hs, Wr, Hr)
        if box is None:
            continue
        lx_min, lx_max, ly_min, ly_max = box
        xs0 = min(xs0, lx_min)
        ys0 = min(ys0, ly_min)
        xe = max(xe, lx_max + ws)
        ye = max(ye, ly_max + hs)
    xs0 = max(0, xs0)
    ys0 = max(0, ys0)
    xe = min(Wr, xe)
    ye = min(Hr, ye)
    cw = xe - xs0
    ch = ye - ys0
    if cw < w + 4 or ch < h + 4:
        return (0, 0, Wr, Hr)
    return (xs0, ys0, cw, ch)


def _avatar_grid33_center_bottom(
    tpl_s: np.ndarray,
    mask_bin: np.ndarray,
    ws: int,
    hs: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, int, int],
] | None:
    """
    将 ws×hs 头像均分 3×3，取正中格 (1,1) 与中下格 (2,1)（0-based 行、列）。
    返回 ((center_bgr, center_mask>127, off_x, off_y), (bottom_bgr, bottom_mask, off_x, off_y))，
    off 为子块在整幅缩放头像内的左上角坐标。
    """
    if ws < 9 or hs < 9:
        return None
    x1 = ws // 3
    x2 = (2 * ws) // 3
    y1 = hs // 3
    y2 = (2 * hs) // 3
    if x2 <= x1 or y2 <= y1 or hs <= y2:
        return None
    c_bgr = tpl_s[y1:y2, x1:x2]
    c_m = mask_bin[y1:y2, x1:x2]
    b_bgr = tpl_s[y2:hs, x1:x2]
    b_m = mask_bin[y2:hs, x1:x2]
    if c_bgr.size == 0 or b_bgr.size == 0:
        return None
    if c_bgr.shape[0] < 4 or c_bgr.shape[1] < 4 or b_bgr.shape[0] < 4:
        return None
    return ((c_bgr, c_m, x1, y1), (b_bgr, b_m, x1, y2))


def _match_template_sqdiff_masked(
    bgr_roi: np.ndarray,
    tpl: np.ndarray,
    mask_bin: np.ndarray,
) -> np.ndarray | None:
    """BGR TM_SQDIFF，mask 为 bool 与同尺寸 tpl。"""
    if tpl.shape[0] < 2 or tpl.shape[1] < 2:
        return None
    m_u8 = (mask_bin.astype(np.uint8) * 255)
    mpx = int(np.count_nonzero(mask_bin))
    if mpx < 8:
        return None
    try:
        return cv2.matchTemplate(bgr_roi, tpl, cv2.TM_SQDIFF, mask=m_u8)
    except cv2.error:
        try:
            return cv2.matchTemplate(bgr_roi, tpl, cv2.TM_SQDIFF)
        except cv2.error:
            return None


def _refine_by_scaled_template_match(
    bgr_roi: np.ndarray,
    bgr_tpl: np.ndarray,
    mask_u8: np.ndarray,
    ly_r: int,
    lx_r: int,
    h: int,
    w: int,
) -> tuple[int, int, int | None, int | None, int | None, int | None]:
    """
    精修：头像 **BGR 三通道** 多档等比例缩放，在九宫格可行域内用 **TM_SQDIFF**（掩膜内未归一化匹配和）搜位。
    OpenCV 对多通道模板逐像素对 B、G、R 求平方差并求和，再对掩膜内求和；**sum/mask_px** 即掩膜内
    「每像素 BGR 平方误差之和」的平均（与灰度版形式一致，改为 RGB 色彩取样）。

    档位、可行域、中心像素约束、缩放下限等与原先一致（环境变量 ARKGUESSER_HEAD_REFINE_*）。
    ARKGUESSER_HEAD_REFINE_GRID33=1（默认）：缩放后头像 3×3 均分，用正中 + 中下两格分别 TM_SQDIFF，代价为二者掩膜内均方之和的一半。
    ARKGUESSER_HEAD_REFINE_TIGHT_ROI=1（默认）：先裁出精修紧包围子图再 matchTemplate，语义与整幅 ROI 一致。

    返回：(ly, lx) 为原模板尺寸 ROI 左上角；(lx_s, ly_s, ws, hs) 为胜出缩放模板在 ROI 内（供绿框）。
    """
    if bgr_roi.ndim != 3 or bgr_roi.shape[2] < 3:
        raise ValueError("refine ROI expected HxWx3 BGR")
    if bgr_tpl.ndim != 3 or bgr_tpl.shape[2] < 3:
        raise ValueError("refine template expected HxWx3 BGR")
    Hr, Wr = bgr_roi.shape[:2]

    # 过小缩放易在局部平纹区得到「假低 MSE」；默认 0.82～2.0 共 6 档（仍覆盖大头到略放大）
    _s_lo = _env_float("ARKGUESSER_HEAD_REFINE_MIN_SCALE", 0.82)
    _s_hi = _env_float("ARKGUESSER_HEAD_REFINE_MAX_SCALE", 2.5)
    _n_st = max(2, _env_int("ARKGUESSER_HEAD_REFINE_SCALE_STEPS", 10))
    scale_ratios = tuple(
        float(x) for x in np.linspace(_s_lo, min(_s_hi, 3.0), _n_st)
    )

    off_x, off_y = 0, 0
    _tight = os.environ.get("ARKGUESSER_HEAD_REFINE_TIGHT_ROI", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if _tight:
        ox, oy, cw, ch = _refine_tight_roi_bounds(lx_r, ly_r, w, h, Wr, Hr, scale_ratios)
        if cw < Wr or ch < Hr:
            lx2, ly2 = lx_r - ox, ly_r - oy
            if _refine_range_center_pixel(lx2, ly2, w, h, cw, ch) is not None:
                bgr_roi = bgr_roi[oy : oy + ch, ox : ox + cw]
                lx_r, ly_r = lx2, ly2
                Hr, Wr = bgr_roi.shape[:2]
                off_x, off_y = ox, oy

    ctr = _refine_range_center_pixel(lx_r, ly_r, w, h, Wr, Hr)
    if ctr is None:
        return ly_r + off_y, lx_r + off_x, None, None, None, None
    cx, cy = ctr

    use_grid33 = os.environ.get("ARKGUESSER_HEAD_REFINE_GRID33", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    best_mean_sq = float("inf")
    best_ly, best_lx = ly_r, lx_r
    best_lx_s: int | None = None
    best_ly_s: int | None = None
    best_ws: int | None = None
    best_hs: int | None = None
    found = False

    for s in scale_ratios:
        ws = max(8, int(round(w * s)))
        hs = max(8, int(round(h * s)))
        if ws > Wr or hs > Hr:
            continue
        box = _megabox_tl_bounds_scaled(lx_r, ly_r, w, h, ws, hs, Wr, Hr)
        if box is None:
            continue
        lx_min, lx_max, ly_min, ly_max = box

        if ws == w and hs == h:
            tpl_s = bgr_tpl
            mask_bin = mask_u8 > 127
        else:
            interp = cv2.INTER_AREA if s < 1.0 - 1e-6 else cv2.INTER_LINEAR
            tpl_s = cv2.resize(bgr_tpl, (ws, hs), interpolation=interp)
            mask_bin = cv2.resize(mask_u8, (ws, hs), interpolation=cv2.INTER_NEAREST) > 127
        mask_s = (mask_bin.astype(np.uint8) * 255)
        mask_px = int(np.count_nonzero(mask_bin))
        if mask_px < 25:
            continue

        grid_packs = _avatar_grid33_center_bottom(tpl_s, mask_bin, ws, hs) if use_grid33 else None
        res: np.ndarray | None = None
        res_c: np.ndarray | None = None
        res_b: np.ndarray | None = None
        mpx_c = mpx_b = 1
        y1 = y2 = x1 = 0

        if grid_packs is not None:
            (c_bgr, c_m_s, x1, y1), (b_bgr, b_m_s, _xb, y2) = grid_packs
            c_bool = np.asarray(c_m_s, dtype=bool)
            b_bool = np.asarray(b_m_s, dtype=bool)
            mpx_c = int(np.count_nonzero(c_bool))
            mpx_b = int(np.count_nonzero(b_bool))
            if mpx_c >= 8 and mpx_b >= 8:
                res_c = _match_template_sqdiff_masked(bgr_roi, c_bgr, c_bool)
                res_b = _match_template_sqdiff_masked(bgr_roi, b_bgr, b_bool)
                if res_c is None or res_b is None:
                    grid_packs = None
            else:
                grid_packs = None

        if grid_packs is not None and res_c is not None and res_b is not None:
            # megabox 约束的是整幅缩放模板左上角 (ly, lx)；子块在图中的左上角为 (lx+x1, ly+y1)、(lx+x1, ly+y2)
            r0 = max(0, ly_min + y1)
            r1 = min(res_c.shape[0], ly_max + y1 + 1)
            c0 = max(0, lx_min + x1)
            c1 = min(res_c.shape[1], lx_max + x1 + 1)
            if r0 >= r1 or c0 >= c1:
                continue
            rr = np.arange(r0, r1, dtype=np.int32)[:, np.newaxis]
            cc = np.arange(c0, c1, dtype=np.int32)[np.newaxis, :]
            ly_c = rr
            lx_c = cc
            ly_b = rr + (y2 - y1)
            lx_b = cc
            ok = (
                (ly_b < res_b.shape[0])
                & (lx_b < res_b.shape[1])
                & (ly_c >= 0)
                & (lx_c >= 0)
            )
            sub = (
                res_c[ly_c, lx_c].astype(np.float64) / float(mpx_c)
                + res_b[ly_b, lx_b].astype(np.float64) / float(mpx_b)
            ) * 0.5
            sub = np.where(ok, sub, np.inf)
        else:
            try:
                res = cv2.matchTemplate(bgr_roi, tpl_s, cv2.TM_SQDIFF, mask=mask_s)
            except cv2.error:
                try:
                    res = cv2.matchTemplate(bgr_roi, tpl_s, cv2.TM_SQDIFF)
                except cv2.error:
                    continue
            if res is None or res.size == 0:
                continue

            r0, r1 = max(0, ly_min), min(res.shape[0], ly_max + 1)
            c0, c1 = max(0, lx_min), min(res.shape[1], lx_max + 1)
            if r0 >= r1 or c0 >= c1:
                continue
            sub = res[r0:r1, c0:c1].astype(np.float64)
        if sub.size == 0:
            continue

        ii, jj = np.indices(sub.shape)
        ly_i = r0 + ii.astype(np.int32)
        lx_i = c0 + jj.astype(np.int32)
        if grid_packs is not None and res_c is not None and res_b is not None:
            # ly_i,lx_i 为正中子块在 ROI 内的左上角；整幅模板左上角需平移
            ft_lx_i = lx_i - x1
            ft_ly_i = ly_i - y1
        else:
            ft_lx_i = lx_i
            ft_ly_i = ly_i
        cx_m = ft_lx_i.astype(np.float64) + ws * 0.5
        cy_m = ft_ly_i.astype(np.float64) + hs * 0.5
        lx_orig = np.rint(cx_m - w * 0.5).astype(np.int32)
        ly_orig = np.rint(cy_m - h * 0.5).astype(np.int32)
        lx_orig = np.clip(lx_orig, 0, Wr - w)
        ly_orig = np.clip(ly_orig, 0, Hr - h)
        covers = (
            (lx_orig <= cx)
            & (cx < lx_orig + w)
            & (ly_orig <= cy)
            & (cy < ly_orig + h)
        )
        masked = np.where(covers & np.isfinite(sub), sub, np.inf)
        min_sum = float(np.min(masked))
        if not np.isfinite(min_sum):
            continue

        arg = int(np.argmin(masked))
        rel_y, rel_x = divmod(arg, sub.shape[1])
        ly = int(r0 + rel_y)
        lx = int(c0 + rel_x)
        if grid_packs is not None and res_c is not None and res_b is not None:
            # ly,lx：正中子块在 res_c 中的行列 = ROI 内子块左上角
            mean_sq = (
                float(res_c[ly, lx]) / float(mpx_c)
                + float(res_b[ly + (y2 - y1), lx]) / float(mpx_b)
            ) * 0.5
            ft_lx = lx - x1
            ft_ly = ly - y1
        else:
            assert res is not None
            mean_sq = float(res[ly, lx]) / float(mask_px)
            ft_lx, ft_ly = lx, ly
        if not np.isfinite(mean_sq) or mean_sq >= best_mean_sq:
            continue

        cx_m1 = ft_lx + ws * 0.5
        cy_m1 = ft_ly + hs * 0.5
        lx_o = int(np.rint(cx_m1 - w * 0.5))
        ly_o = int(np.rint(cy_m1 - h * 0.5))
        lx_o = max(0, min(lx_o, Wr - w))
        ly_o = max(0, min(ly_o, Hr - h))
        best_mean_sq = mean_sq
        best_ly, best_lx = ly_o, lx_o
        best_lx_s, best_ly_s = ft_lx, ft_ly
        best_ws, best_hs = ws, hs
        found = True

    if not found:
        return ly_r + off_y, lx_r + off_x, None, None, None, None
    return (
        best_ly + off_y,
        best_lx + off_x,
        None if best_lx_s is None else best_lx_s + off_x,
        None if best_ly_s is None else best_ly_s + off_y,
        best_ws,
        best_hs,
    )


def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _scharr_mag_u8(gray: np.ndarray) -> np.ndarray:
    """
    Scharr 梯度幅值（CLAHE 后），归一化到 uint8。
    火焰/晶体/烟雾等多为平滑渐变，人脸与发饰边缘走向更复杂，可削弱「同色不同形」的假匹配。
    """
    g = _clahe_gray(gray)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    p99 = float(np.percentile(mag, 99.0)) + 1e-6
    return np.clip(mag / p99 * 255.0, 0, 255).astype(np.uint8)


def _gray_edges_for_match(gray: np.ndarray) -> np.ndarray:
    """CLAHE 提亮暗部后再 Canny，弱化「整图偏暗、仅靠灰度」的假匹配。"""
    g = _clahe_gray(gray)
    med = float(np.median(g))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, max(lo + 1, 1.33 * med)))
    if hi <= lo + 5:
        lo, hi = 40, 120
    return cv2.Canny(g, lo, hi)


def find_by_match_template(
    src_bgr: np.ndarray,
    tpl_bgr: np.ndarray,
    alpha: np.ndarray,
    x_range: tuple[int, int] | None = None,
    y_range: tuple[int, int] | None = None,
) -> tuple[
    float,
    tuple[int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int] | None,
]:
    """
    融合多项 matchTemplate 代价后取全局最优位置（越小越好）：
    - 颜色/灰度 TM_SQDIFF_NORMED（权重可调低，减轻「同色不同轮廓」）
    - Canny 二值边 TM_SQDIFF_NORMED
    - Scharr 梯度幅值 TM_SQDIFF_NORMED（连续边缘强度，与色块/柔光区分更明显）
    - TM_CCOEFF_NORMED → (1 - cc)

    返回：
    - match_mse：最终位置上颜色 TM_SQDIFF_NORMED（供调用方/调试；对齐表 CSV 不写此项）；
    - 最终模板左上角 (x, y)（整图坐标，**原头像尺寸 w×h**）；
    - first_round_match_rect (x, y, w, h)：第一轮粗匹配后模板矩形，供调试图黄框；
    - refine_scaled_rect (x, y, ws, hs) 或 None：精修胜出时**缩放模板**在整图上的矩形（调试图绿框）；
      未开精修或精修未更新则 None。

    环境变量：ARKGUESSER_HEAD_MATCH_W_COLOR（默认 0.26）、W_EDGE（0.84）、
    W_GRAD（0.74）、W_CC（0.29）；任一为 0 可关闭对应项。

    另：TOPK / TOPK_COLOR / MERGE_MAX / min_sep / dual_pool / res_c 的 matchTemplate 回退链均保持不变。
    ORB 排序在特征数相同后比较该点的 TM_CCOEFF_NORMED（越大越好），减轻剑柄/特效等误检。
    CENTER_REFINE=1（默认）：精修在九宫格内对 **BGR 模板** 多档缩放，每档用 **TM_SQDIFF**（mask，未归一化和）
    在可行域内取最小（掩膜内 RGB 平方误差和 / mask 像素数）；档位间用该均值比较。
    默认缩放下限约 **0.82×**（ARKGUESSER_HEAD_REFINE_MIN_SCALE）、上限 2.0、6 档（可调）。
    中心像素约束与 refine_scaled_rect 绿框语义同前。设 0 关闭精修。
    ORB/AKAZE/RATIO 开关同前。
    """
    h, w = tpl_bgr.shape[:2]
    mask_u8 = ((alpha > 10).astype(np.uint8) * 255)
    if int(cv2.countNonZero(mask_u8)) < 50:
        raise ValueError("template mask too small")

    H, W = src_bgr.shape[:2]
    x0_lo, x0_hi = (0, W - w) if x_range is None else x_range
    y0_lo, y0_hi = (0, H - h) if y_range is None else y_range

    x0_lo = max(0, min(int(x0_lo), W - w))
    x0_hi = max(x0_lo, min(int(x0_hi), W - w))
    y0_lo = max(0, min(int(y0_lo), H - h))
    y0_hi = max(y0_lo, min(int(y0_hi), H - h))

    roi_w = x0_hi - x0_lo + w
    roi_h = y0_hi - y0_lo + h
    roi = src_bgr[y0_lo : y0_lo + roi_h, x0_lo : x0_lo + roi_w]

    if roi.shape[0] < h or roi.shape[1] < w:
        raise ValueError("search ROI smaller than template")

    def try_sqdiff(
        img: np.ndarray, templ: np.ndarray, m: np.ndarray | None
    ) -> np.ndarray | None:
        try:
            if m is not None:
                return cv2.matchTemplate(img, templ, cv2.TM_SQDIFF_NORMED, mask=m)
            return cv2.matchTemplate(img, templ, cv2.TM_SQDIFF_NORMED)
        except cv2.error:
            return None

    def try_ccoeff(
        img: np.ndarray, templ: np.ndarray, m: np.ndarray | None
    ) -> np.ndarray | None:
        try:
            if m is not None:
                return cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED, mask=m)
            return cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return None

    def first_good_map(
        fn: Callable[
            [np.ndarray, np.ndarray, np.ndarray | None], np.ndarray | None
        ],
        alist: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    ) -> np.ndarray | None:
        for img, templ, m in alist:
            out = fn(img, templ, m)
            if out is not None and np.any(np.isfinite(out)):
                return out.astype(np.float64)
        return None

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_tpl = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    attempts: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = [
        (roi, tpl_bgr, mask_u8),
        (gray_roi, gray_tpl, mask_u8),
        (roi, tpl_bgr, None),
        (gray_roi, gray_tpl, None),
    ]

    res_c = first_good_map(try_sqdiff, attempts)
    if res_c is None:
        raise RuntimeError("matchTemplate failed for all fallbacks")

    res_cc = first_good_map(try_ccoeff, attempts)

    er = _gray_edges_for_match(gray_roi)
    et = _gray_edges_for_match(gray_tpl)
    res_e = first_good_map(try_sqdiff, [(er, et, mask_u8), (er, et, None)])

    gm_roi = _scharr_mag_u8(gray_roi)
    gm_tpl = _scharr_mag_u8(gray_tpl)
    res_g = first_good_map(try_sqdiff, [(gm_roi, gm_tpl, mask_u8), (gm_roi, gm_tpl, None)])

    w_color = _env_float("ARKGUESSER_HEAD_MATCH_W_COLOR", 0.26)
    w_e = _env_float("ARKGUESSER_HEAD_MATCH_W_EDGE", 0.84)
    w_grad = _env_float("ARKGUESSER_HEAD_MATCH_W_GRAD", 0.74)
    w_cc = _env_float("ARKGUESSER_HEAD_MATCH_W_CC", 0.29)

    cost = w_color * res_c
    if w_e > 0 and res_e is not None and res_e.shape == cost.shape:
        cost = cost + w_e * res_e
    if w_grad > 0 and res_g is not None and res_g.shape == cost.shape:
        cost = cost + w_grad * res_g
    if w_cc > 0 and res_cc is not None and res_cc.shape == cost.shape:
        cost = cost + w_cc * (1.0 - np.clip(res_cc, -1.0, 1.0))

    use_orb = os.environ.get("ARKGUESSER_HEAD_MATCH_ORB", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    use_akaze = os.environ.get("ARKGUESSER_HEAD_MATCH_AKAZE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    dual_pool = os.environ.get("ARKGUESSER_HEAD_MATCH_DUAL_POOL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    topk = max(1, _env_int("ARKGUESSER_HEAD_MATCH_TOPK", 26))
    topk_color = max(0, _env_int("ARKGUESSER_HEAD_MATCH_TOPK_COLOR", 20))
    merge_max = max(
        topk + topk_color,
        _env_int("ARKGUESSER_HEAD_MATCH_MERGE_MAX", 64),
    )
    min_sep = max(5, max(h, w) // 4)
    feat_ratio = _env_float("ARKGUESSER_HEAD_MATCH_RATIO", 0.70)
    feat_ratio = min(0.95, max(0.55, feat_ratio))

    if use_orb and topk > 1:
        c_fused = _topk_separated_low_cost(cost, topk, min_sep)
        c_lists: list[list[tuple[int, int]]] = [c_fused]
        if dual_pool and topk_color > 0:
            c_lists.append(_topk_separated_low_cost(res_c, topk_color, min_sep))
        cands = _merge_spatial_dedupe(c_lists, min_sep, merge_max)
        if not cands:
            cands = c_fused

        def _rank(ij: tuple[int, int]) -> tuple[int, float, float, float]:
            ly, lx = ij
            pg = gray_roi[ly : ly + h, lx : lx + w]
            o = _orb_good_match_count(
                gray_tpl, pg, mask_u8, ratio=feat_ratio
            )
            a = (
                _akaze_good_match_count(
                    gray_tpl, pg, mask_u8, ratio=feat_ratio
                )
                if use_akaze
                else 0
            )
            feat = o + a
            rc = float(res_c[ly, lx])
            c = float(cost[ly, lx])
            if res_cc is not None:
                cc = float(res_cc[ly, lx])
                if not np.isfinite(cc):
                    cc = -1.0
            else:
                cc = -1.0
            return (feat, cc, -rc, -c)

        ly, lx = max(cands, key=_rank)
    else:
        ly, lx = np.unravel_index(int(np.nanargmin(cost)), cost.shape)

    # 第一轮粗匹配：ROI 内左上角；后续仅精修坐标，不改变模板尺寸语义
    ly_round1, lx_round1 = int(ly), int(lx)

    use_center = os.environ.get("ARKGUESSER_HEAD_MATCH_CENTER_REFINE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    rlx_s = rly_s = rws = rhs = None
    if use_center:
        ly, lx, rlx_s, rly_s, rws, rhs = _refine_by_scaled_template_match(
            roi,
            tpl_bgr,
            mask_u8,
            ly_round1,
            lx_round1,
            h,
            w,
        )

    color_sq = float(res_c[ly, lx])
    if not np.isfinite(color_sq):
        raise RuntimeError("color SQDIFF at best fused position is not finite")
    x0, y0 = x0_lo + int(lx), y0_lo + int(ly)
    fr_x = x0_lo + lx_round1
    fr_y = y0_lo + ly_round1
    first_round_match_rect = (fr_x, fr_y, w, h)
    refine_scaled_full: tuple[int, int, int, int] | None = None
    if rlx_s is not None and rly_s is not None and rws is not None and rhs is not None:
        refine_scaled_full = (x0_lo + int(rlx_s), y0_lo + int(rly_s), int(rws), int(rhs))
    return color_sq, (x0, y0), first_round_match_rect, refine_scaled_full


PRTS_CHAR_ART_VIEWPORT = 512


def coarse_scan_rect_wh(
    W: int,
    H: int,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    """
    粗扫描范围：与 find_by_match_template 内 ROI 一致，整图坐标系下矩形 (x, y, rw, rh)。
    模板左上角 (x0,y0) 落在 [x0_lo,x0_hi]×[y0_lo,y0_hi] 时，覆盖区域即该矩形。
    """
    margin_x = int(W * 0.18)
    margin_y_top = int(H * 0.08)
    margin_y_bot = int(H * 0.35)
    x0_lo, x0_hi = margin_x, W - w - margin_x
    y0_lo, y0_hi = margin_y_top, H - h - margin_y_bot
    if x0_hi < x0_lo:
        x0_lo, x0_hi = 0, max(0, W - w)
    if y0_hi < y0_lo:
        y0_lo, y0_hi = 0, max(0, H - h)
    roi_w = x0_hi - x0_lo + w
    roi_h = y0_hi - y0_lo + h
    return (x0_lo, y0_lo, roi_w, roi_h)


def compute_e2_head_alignment(
    art_path: str | Path,
    avatar_path: str | Path,
) -> dict:
    """
    在精二立绘中匹配精二头像图位置；写入 CSV 的字段由调用方从返回值中选取（见 ``CSV_FIELDS``）。

    语义约定（与调试图一致）：
    - **黄框** ``first_round_match_rect``：粗扫描融合代价最终选定的 **原头像尺寸 w×h** 位置（第一轮）。
    - **绿框** ``refine_scaled_rect``：精修阶段多档缩放后锁定的 **实际匹配块** 在整图上的 (x,y,w,h)。
    - CSV 仅导出绿框 ``refine_x/y/w/h`` 及 **scale** = (绿框宽/头像原宽 + 绿框高/头像原高)/2，保留两位小数。
      无绿框时回退为粗对齐矩形，``scale``=1.00。

    ``match_mse``（粗匹配颜色 TM_SQDIFF_NORMED）仅用于调试图，不写入 CSV。
    """
    src = imread_unicode(art_path)
    tpl = imread_unicode(avatar_path)
    alpha = tpl[:, :, 3] if tpl.shape[2] == 4 else np.full(tpl.shape[:2], 255, np.uint8)
    # 立绘与头像均先按 alpha 铺底，避免透明区垃圾 RGB 导致误匹配；mask 仍用原始 alpha
    src_bgr = bgra_flatten_to_bgr(src)
    tpl_bgr = bgra_flatten_to_bgr(tpl)

    H, W = src_bgr.shape[:2]
    h, w = tpl_bgr.shape[:2]
    sx, sy, srw, srh = coarse_scan_rect_wh(W, H, w, h)
    x0_hi = sx + srw - w
    y0_hi = sy + srh - h
    x_range = (sx, x0_hi)
    y_range = (sy, y0_hi)

    tm_sqdiff_norm, (x0, y0), first_round_match_rect, refine_scaled_rect = find_by_match_template(
        src_bgr, tpl_bgr, alpha, x_range=x_range, y_range=y_range
    )
    # 绿框像素与相对头像原尺寸（w×h）的比例；无绿框时用粗对齐矩形，比例为 1
    orig_w_f = float(w)
    orig_h_f = float(h)
    if refine_scaled_rect is not None:
        gx, gy, gw, gh = refine_scaled_rect
        refine_x = int(gx)
        refine_y = int(gy)
        refine_w = int(gw)
        refine_h = int(gh)
        sx = float(gw) / orig_w_f if orig_w_f > 0 else 1.0
        sy = float(gh) / orig_h_f if orig_h_f > 0 else 1.0
        scale = round((sx + sy) / 2.0, 2)
    else:
        refine_x = int(x0)
        refine_y = int(y0)
        refine_w = w
        refine_h = h
        scale = 1.0

    return {
        "refine_x": refine_x,
        "refine_y": refine_y,
        "refine_w": refine_w,
        "refine_h": refine_h,
        "scale": scale,
        "match_mse": round(float(tm_sqdiff_norm), 4),
        "first_round_match_rect": first_round_match_rect,
        "refine_scaled_rect": refine_scaled_rect,
    }


def write_head_align_debug_png(
    art_path: str | Path,
    out_path: str | Path,
    *,
    char_id: str,
    status: str,
    match_rect: tuple[int, int, int, int] | None = None,
    match_mse: float | None = None,
    error: str = "",
    avatar_path: str | Path | None = None,
    first_round_match_rect: tuple[int, int, int, int] | None = None,
    refined_match_rect: tuple[int, int, int, int] | None = None,
) -> None:
    """
    在整身立绘上绘制匹配矩形与说明文字，并可选贴上 char_avatar 缩略图便于肉眼核对。
    first_round_match_rect：第一轮粗匹配模板框，黄色线框。
    refined_match_rect：精修时**实际参与匹配的缩放模板**在整图上的矩形 (x,y,ws,hs)，绿色线框；
    若提供则优先用它画绿框，以反映 50%～200% 档位尺度；否则用 match_rect（原 w×h）。
    文字仅 ASCII（char_id / sqdiff_n / status），避免 OpenCV 无法绘制中文。
    """
    src = imread_unicode(art_path)
    vis = bgra_flatten_to_bgr(src)
    H, W = vis.shape[:2]
    thick = max(2, min(H, W) // 400)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.6, min(H, W) / 900.0)
    lh = int(28 * fs + 8)

    yellow = (0, 255, 255)
    if first_round_match_rect is not None:
        cx, cy, crw, crh = first_round_match_rect
        cx = max(0, min(int(cx), W - 1))
        cy = max(0, min(int(cy), H - 1))
        x2 = min(W - 1, cx + max(1, int(crw)) - 1)
        y2 = min(H - 1, cy + max(1, int(crh)) - 1)
        cv2.rectangle(vis, (cx, cy), (x2, y2), yellow, thick)

    green_rect = refined_match_rect if refined_match_rect is not None else match_rect
    if green_rect is not None:
        x0, y0, rw, rh = green_rect
        x1, y1 = x0 + rw - 1, y0 + rh - 1
        if status == "ok":
            color = (0, 255, 0)
        else:
            color = (0, 165, 255)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, thick)

    lines = [char_id, f"status={status}"]
    if match_mse is not None:
        lines.append(f"sqdiff_n={match_mse:.4f}")
    if error:
        lines.append(f"err={error[:120]}")

    y_text = lh
    for line in lines:
        cv2.putText(
            vis,
            line,
            (thick * 2, y_text),
            font,
            fs,
            (0, 0, 0),
            thick + 2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            line,
            (thick * 2, y_text),
            font,
            fs,
            (255, 255, 255),
            thick,
            lineType=cv2.LINE_AA,
        )
        y_text += lh

    if avatar_path is not None:
        ap = Path(avatar_path)
        if ap.is_file():
            try:
                av = imread_unicode(ap)
                ab = bgra_flatten_to_bgr(av)
                ah, aw = ab.shape[:2]
                tw = min(220, max(aw, 1))
                th = max(1, int(ah * tw / aw))
                small = cv2.resize(ab, (tw, th), interpolation=cv2.INTER_AREA)
                px = W - tw - thick * 2
                py = thick * 2
                if px >= 0 and py >= 0 and px + tw <= W and py + th <= H:
                    vis[py : py + th, px : px + tw] = small
                    cv2.rectangle(
                        vis,
                        (px, py),
                        (px + tw - 1, py + th - 1),
                        (255, 255, 255),
                        max(1, thick // 2),
                    )
                    if py > lh:
                        cv2.putText(
                            vis,
                            "ref_avatar",
                            (px, py - 4),
                            font,
                            fs * 0.75,
                            (255, 255, 255),
                            thick,
                            lineType=cv2.LINE_AA,
                        )
            except Exception:
                pass

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", vis)
    if not ok:
        raise RuntimeError("imencode failed")
    buf.tofile(str(out_p))


# --- char_e2_head_align CSV ---
# 仅导出：char_id、name、refine_x/y/w/h、scale（无 status/error；失败/跳过不写行）。
# 调试图：黄框 = first_round_match_rect；绿框 = refine 块。

# 与 char_avatar 同级：整身立绘 + 匹配矩形 + ref 头像缩略图
CHAR_AVATAR_ALIGN_DEBUG_REL_DIR = "char_avatar_align_debug"

# 精修绿框易偏时，CSV 与调试图强制采用粗扫矩形（原头像尺寸），scale 固定 1.00
HEAD_ALIGN_FORCE_COARSE_CHAR_IDS: frozenset[str] = frozenset(
    (
        "char_4145_ulpia",  # 乌尔比安
        "char_4138_narant",  # 娜仁图亚
    )
)

CSV_FIELDS = [
    "char_id",
    "name",
    "refine_x",
    "refine_y",
    "refine_w",
    "refine_h",
    "scale",
]


def _fmt_csv_scale(v: float) -> str:
    """scale 写入 CSV，固定两位小数。"""
    return f"{float(v):.2f}"


def _coerce_align_row_from_reader(row: dict[str, str]) -> dict[str, str]:
    """按 CSV_FIELDS 取列；兼容旧表头 scale_x/scale_y → scale。"""
    d = {field: (row.get(field) or "").strip() for field in CSV_FIELDS}
    if not d.get("scale"):
        sx = (row.get("scale_x") or "").strip()
        sy = (row.get("scale_y") or "").strip()
        if sx or sy:
            try:
                fx = float(sx or "1")
                fy = float(sy or "1")
                d["scale"] = f"{(fx + fy) / 2.0:.2f}"
            except ValueError:
                pass
    return d


def _load_existing_align_by_char_id(out_csv: Path) -> dict[str, dict[str, str]]:
    """读取已有 char_e2_head_align.csv，按 char_id 索引（用于增量跳过识别）。"""
    out: dict[str, dict[str, str]] = {}
    if not out_csv.is_file():
        return out
    try:
        with out_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = (row.get("char_id") or "").strip()
                if not cid:
                    continue
                out[cid] = _coerce_align_row_from_reader(row)
    except OSError:
        pass
    return out


def _row_from_existing(existing: dict[str, str], cid: str, name: str) -> dict[str, Any]:
    """复用已有表行，仅刷新 name（干员显示名可能随 CSV 更新）。"""
    base: dict[str, Any] = {field: existing.get(field, "") for field in CSV_FIELDS}
    base["char_id"] = cid
    base["name"] = name
    base["status"] = "ok"  # 进程内计数用，不写入 CSV
    return base


def _load_six_star_rows(csv_path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
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
    debug_dir: Path | None,
    art_path: Path,
    avatar_path: Path,
    cid: str,
    base: dict[str, Any],
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
            ax = int(base["refine_x"])
            ay = int(base["refine_y"])
            aw = int(base["refine_w"])
            ah = int(base["refine_h"])
            mse = float(base.get("match_mse", 0.0))
            fr = base.get("first_round_match_rect")
            if (
                isinstance(fr, tuple)
                and len(fr) == 4
                and all(isinstance(x, (int, float)) for x in fr)
            ):
                fr_rect = (int(fr[0]), int(fr[1]), int(fr[2]), int(fr[3]))
            else:
                fr_rect = None
            rs = base.get("refine_scaled_rect")
            if (
                isinstance(rs, tuple)
                and len(rs) == 4
                and all(isinstance(x, (int, float)) for x in rs)
            ):
                refined_r = (int(rs[0]), int(rs[1]), int(rs[2]), int(rs[3]))
            else:
                refined_r = None
            write_head_align_debug_png(
                art_path,
                out_dbg,
                char_id=cid,
                status="ok",
                match_rect=(ax, ay, aw, ah),
                match_mse=mse,
                error="",
                avatar_path=avatar_path if avatar_path.is_file() else None,
                first_round_match_rect=fr_rect,
                refined_match_rect=refined_r,
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
                first_round_match_rect=None,
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
                first_round_match_rect=None,
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
    debug_dir: Path | None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "char_id": cid,
        "name": name,
        "refine_x": "",
        "refine_y": "",
        "refine_w": "",
        "refine_h": "",
        "scale": "",
        "status": "",  # 进程内：ok / skip_* / match_failed，不写入 CSV
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
        if cid in HEAD_ALIGN_FORCE_COARSE_CHAR_IDS:
            fr = r.get("first_round_match_rect")
            if isinstance(fr, tuple) and len(fr) == 4:
                rx, ry, rw_, rh_ = int(fr[0]), int(fr[1]), int(fr[2]), int(fr[3])
                r["refine_x"] = rx
                r["refine_y"] = ry
                r["refine_w"] = rw_
                r["refine_h"] = rh_
                r["scale"] = 1.0
                r["refine_scaled_rect"] = None
        base.update(
            {
                "refine_x": r["refine_x"],
                "refine_y": r["refine_y"],
                "refine_w": r["refine_w"],
                "refine_h": r["refine_h"],
                "scale": _fmt_csv_scale(r["scale"]),
                "match_mse": r["match_mse"],
                "first_round_match_rect": r["first_round_match_rect"],
                "refine_scaled_rect": r.get("refine_scaled_rect"),
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
) -> tuple[int, int, int, int]:
    """
    重写 char_e2_head_align.csv（仅包含当前 characters.csv 中的六星干员）。
    列为：char_id、name、refine_x/y/w/h、scale；仅成功对齐的行会写入。

    返回 (ok_count, skip_count, fail_count, reused_existing_count)：
    - skip：本次执行识别时因缺立绘/缺头像跳过；
    - reused_existing：out_csv 中已有该 char_id 时直接复用行、不跑图像识别（增量）。

    write_debug_images=True 时在 data_dir/char_avatar_align_debug/ 下生成调试图（默认关闭）。
    reuse_existing_rows=False 时忽略已有表，对全员重新识别（全量重算）。
    """
    char_arts = data_dir / "char_arts"
    char_avatar = data_dir / "char_avatar"
    debug_dir: Path | None = None
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

    tasks: list[tuple[str, str, Path, Path]] = []
    reused_rows: list[dict[str, Any]] = []
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

    results: list[dict[str, Any]] = list(reused_rows)
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
            if row.get("status") != "ok":
                continue
            try:
                if int(row.get("refine_w") or 0) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})

    ok_c = sum(1 for r in results if r.get("status") == "ok")
    skip_c = sum(1 for r in results if str(r["status"]).startswith("skip"))
    fail_c = len(results) - ok_c - skip_c
    reused_c = len(reused_rows)
    return (ok_c, skip_c, fail_c, reused_c)
