"""立绘匹配：精二头像在整身立绘中的定位（OpenCV）与 char_e2_head_align.csv 生成。"""
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


def _refine_top_left_by_res_c(
    res_c: np.ndarray,
    ly: int,
    lx: int,
    radius: int,
) -> tuple[int, int]:
    """
    在已选左上角附近小窗口内，按颜色通道 res_c（TM_SQDIFF_NORMED）取最小值。
    纠正粗定位后框仍偏上/偏下/偏侧（在半径内扫 res_c 最小）；对完全选错区域无效。
    """
    if radius <= 0:
        return ly, lx
    Rh, Rw = res_c.shape
    best_y, best_x = ly, lx
    best_v = float(res_c[ly, lx])
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = ly + dy, lx + dx
            if ny < 0 or nx < 0 or ny >= Rh or nx >= Rw:
                continue
            v = float(res_c[ny, nx])
            if np.isfinite(v) and v < best_v:
                best_v = v
                best_y, best_x = ny, nx
    return best_y, best_x


def _parse_center_scales(default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get("ARKGUESSER_HEAD_MATCH_CENTER_SCALES", "").strip()
    if not raw:
        return default
    out: list[float] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            pass
    return tuple(out) if out else default


def _center_masked_mse(
    gray_roi: np.ndarray,
    tg: np.ndarray,
    mk: np.ndarray,
    cx: float,
    cy: float,
    nw: int,
    nh: int,
    Wr: int,
    Hr: int,
) -> float | None:
    nl = int(round(cx - nw / 2.0))
    nt = int(round(cy - nh / 2.0))
    if nl < 0 or nt < 0 or nl + nw > Wr or nt + nh > Hr:
        return None
    patch = gray_roi[nt : nt + nh, nl : nl + nw]
    d = patch.astype(np.float32) - tg.astype(np.float32)
    return float(np.mean((d[mk]) ** 2))


def _refine_center_scale_masked_mse(
    gray_roi: np.ndarray,
    gray_tpl: np.ndarray,
    mask_u8: np.ndarray,
    ly: int,
    lx: int,
    h: int,
    w: int,
    *,
    scales: tuple[float, ...],
    off_px: int,
) -> tuple[int, int]:
    """
    以粗定位框中心为锚，在 ±off 像素内平移并尝试多尺度模板，对 mask 内灰度做 MSE；
    输出仍为**原模板尺寸 (h,w)** 的左上角，不改变 CSV 中 avatar 尺寸语义。
    不修改 res_c 的 matchTemplate 流程，仅精修坐标；利于尺度假设略有偏差（如空弦）与小幅错位。

    速度：默认两阶段——先在各尺度下仅算中心点 MSE 排序，再只对最优的 K 个尺度做全平移搜索
   （约 K/len(scales) × 原耗时）。环境变量 ARKGUESSER_HEAD_MATCH_CENTER_SCALE_TOPK：
    默认 4；设为 0 或大于等于尺度个数则与「每个尺度都做全平移」等价（最稳、最慢）。
    """
    if off_px <= 0 or not scales:
        return ly, lx
    Hr, Wr = gray_roi.shape[:2]
    cx0 = lx + w * 0.5
    cy0 = ly + h * 0.5
    best_mse = 1e18
    best_lx, best_ly = lx, ly

    topk = _env_int("ARKGUESSER_HEAD_MATCH_CENTER_SCALE_TOPK", 4)

    # 每个尺度：resize 一次，算中心点 MSE 供排序；保留 tg/mk 供二阶段复用
    entries: list[
        tuple[float, np.ndarray, np.ndarray, int, int]
    ] = []  # (mse_center, tg, mk, nw, nh)
    for s in scales:
        if s <= 0.05:
            continue
        nw = max(12, int(round(w * s)))
        nh = max(12, int(round(h * s)))
        interp = cv2.INTER_AREA if s < 1.0 - 1e-6 else cv2.INTER_LINEAR
        tg = cv2.resize(gray_tpl, (nw, nh), interpolation=interp)
        mk = cv2.resize(mask_u8, (nw, nh), interpolation=cv2.INTER_NEAREST) > 127
        if int(np.count_nonzero(mk)) < 25:
            continue
        mse_c = _center_masked_mse(gray_roi, tg, mk, cx0, cy0, nw, nh, Wr, Hr)
        if mse_c is None:
            continue
        entries.append((mse_c, tg, mk, nw, nh))

    if not entries:
        return ly, lx

    if topk <= 0 or topk >= len(entries):
        ranked = entries
    else:
        ranked = sorted(entries, key=lambda t: t[0])[:topk]

    for _mse_c, tg, mk, nw, nh in ranked:
        for dy in range(-off_px, off_px + 1):
            for dx in range(-off_px, off_px + 1):
                cx = cx0 + dx
                cy = cy0 + dy
                mse = _center_masked_mse(gray_roi, tg, mk, cx, cy, nw, nh, Wr, Hr)
                if mse is None:
                    continue
                if mse < best_mse:
                    best_mse = mse
                    best_lx = int(round(cx - w / 2.0))
                    best_ly = int(round(cy - h / 2.0))

    best_lx = max(0, min(best_lx, Wr - w))
    best_ly = max(0, min(best_ly, Hr - h))
    return best_ly, best_lx


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
) -> tuple[float, tuple[int, int]]:
    """
    融合多项 matchTemplate 代价后取全局最优位置（越小越好）：
    - 颜色/灰度 TM_SQDIFF_NORMED（权重可调低，减轻「同色不同轮廓」）
    - Canny 二值边 TM_SQDIFF_NORMED
    - Scharr 梯度幅值 TM_SQDIFF_NORMED（连续边缘强度，与色块/柔光区分更明显）
    - TM_CCOEFF_NORMED → (1 - cc)

    返回的 match_mse（CSV）仍为「最终位置上」的颜色 TM_SQDIFF_NORMED。

    环境变量：ARKGUESSER_HEAD_MATCH_W_COLOR（默认 0.26）、W_EDGE（0.84）、
    W_GRAD（0.74）、W_CC（0.29）；任一为 0 可关闭对应项。

    另：TOPK / TOPK_COLOR / MERGE_MAX / min_sep / dual_pool / res_c 的 matchTemplate 回退链均保持不变。
    ORB 排序在特征数相同后比较该点的 TM_CCOEFF_NORMED（越大越好），减轻剑柄/特效等误检。
    CENTER_REFINE=1（默认）：在 res_c 像素精修前，以当前框中心 ±CENTER_OFF 像素、
    多尺度（默认 0.90～1.10 每 0.02 一档）做 mask 内灰度 MSE，输出仍固定为原模板宽高。
    环境变量 ARKGUESSER_HEAD_MATCH_CENTER_SCALES 可设为逗号分隔比例覆盖默认。
    CENTER_SCALE_TOPK（默认 4）：先按中心点 MSE 只保留最优 K 个尺度再做平移搜索；0 表示不裁剪（最慢最稳）。
    REFINE_RADIUS（默认 8）仍为 res_c 图上的平移精修；设 0 关闭。
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

    use_center = os.environ.get("ARKGUESSER_HEAD_MATCH_CENTER_REFINE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if use_center:
        center_off = max(0, _env_int("ARKGUESSER_HEAD_MATCH_CENTER_OFF", 6))
        scales_t = _parse_center_scales(
            (0.90,0.96,1.02,1.08)
        )
        ly, lx = _refine_center_scale_masked_mse(
            gray_roi,
            gray_tpl,
            mask_u8,
            ly,
            lx,
            h,
            w,
            scales=scales_t,
            off_px=center_off,
        )

    refine_r = max(0, _env_int("ARKGUESSER_HEAD_MATCH_REFINE_RADIUS", 8))
    # 防止误配环境变量过大：不超过模板短边约一半
    refine_cap = max(8, min(h, w) // 2)
    if refine_r > refine_cap:
        refine_r = refine_cap
    ly, lx = _refine_top_left_by_res_c(res_c, ly, lx, refine_r)

    color_sq = float(res_c[ly, lx])
    if not np.isfinite(color_sq):
        raise RuntimeError("color SQDIFF at best fused position is not finite")
    x0, y0 = x0_lo + int(lx), y0_lo + int(ly)
    return color_sq, (x0, y0)


PRTS_CHAR_ART_VIEWPORT = 512


def compute_e2_head_alignment(
    art_path: str | Path,
    avatar_path: str | Path,
) -> dict:
    """
    在精二立绘中匹配精二头像图位置，返回后续 SVG/CSS 定位用字段。

    slice_scale_512：立绘按 512×512、xMidYMid slice 铺满时的缩放系数 max(512/art_w, 512/art_h)。
    match_mse：融合定位后，该点上的颜色 TM_SQDIFF_NORMED（越小越好），列名保留以兼容 CSV。
    """
    src = imread_unicode(art_path)
    tpl = imread_unicode(avatar_path)
    alpha = tpl[:, :, 3] if tpl.shape[2] == 4 else np.full(tpl.shape[:2], 255, np.uint8)
    # 立绘与头像均先按 alpha 铺底，避免透明区垃圾 RGB 导致误匹配；mask 仍用原始 alpha
    src_bgr = bgra_flatten_to_bgr(src)
    tpl_bgr = bgra_flatten_to_bgr(tpl)

    H, W = src_bgr.shape[:2]
    h, w = tpl_bgr.shape[:2]
    # 精二立绘头部常在画布中上、且水平偏移大；略放宽左右边距以免真解落在 ROI 外
    margin_x = int(W * 0.18)
    margin_y_top = int(H * 0.08)
    # 下边距：从画布底向上预留，缩小头像框左上角可出现的最低 y（模板底不越过 H-30%H）
    margin_y_bot = int(H * 0.35)
    x0_lo, x0_hi = margin_x, W - w - margin_x
    y0_lo, y0_hi = margin_y_top, H - h - margin_y_bot
    if x0_hi < x0_lo:
        x0_lo, x0_hi = 0, max(0, W - w)
    if y0_hi < y0_lo:
        y0_lo, y0_hi = 0, max(0, H - h)
    x_range = (x0_lo, x0_hi)
    y_range = (y0_lo, y0_hi)

    tm_sqdiff_norm, (x0, y0) = find_by_match_template(
        src_bgr, tpl_bgr, alpha, x_range=x_range, y_range=y_range
    )
    head_cx = x0 + w / 2.0
    head_cy = y0 + h / 2.0
    vp = float(PRTS_CHAR_ART_VIEWPORT)
    slice_scale = max(vp / float(W), vp / float(H))

    return {
        "art_w": W,
        "art_h": H,
        "avatar_x": x0,
        "avatar_y": y0,
        "avatar_w": w,
        "avatar_h": h,
        "head_cx": round(head_cx, 2),
        "head_cy": round(head_cy, 2),
        "slice_scale_512": round(slice_scale, 6),
        "norm_head_cx": round(head_cx / W, 6) if W else 0.0,
        "norm_head_cy": round(head_cy / H, 6) if H else 0.0,
        "match_mse": round(float(tm_sqdiff_norm), 4),
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
) -> None:
    """
    在整身立绘上绘制匹配矩形与说明文字，并可选贴上 char_avatar 缩略图便于肉眼核对。
    文字仅 ASCII（char_id / sqdiff_n / status），避免 OpenCV 无法绘制中文。
    """
    src = imread_unicode(art_path)
    vis = bgra_flatten_to_bgr(src)
    H, W = vis.shape[:2]
    thick = max(2, min(H, W) // 400)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.6, min(H, W) / 900.0)
    lh = int(28 * fs + 8)

    if match_rect is not None:
        x0, y0, rw, rh = match_rect
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
                out[cid] = {field: (row.get(field) or "").strip() for field in CSV_FIELDS}
    except OSError:
        pass
    return out


def _row_from_existing(existing: dict[str, str], cid: str, name: str) -> dict[str, Any]:
    """复用已有表行，仅刷新 name（干员显示名可能随 CSV 更新）。"""
    base: dict[str, Any] = {field: existing.get(field, "") for field in CSV_FIELDS}
    base["char_id"] = cid
    base["name"] = name
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
    debug_dir: Path | None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
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
) -> tuple[int, int, int, int]:
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
            writer.writerow(row)

    ok_c = sum(1 for r in results if r["status"] == "ok")
    skip_c = sum(1 for r in results if str(r["status"]).startswith("skip"))
    fail_c = len(results) - ok_c - skip_c
    reused_c = len(reused_rows)
    return (ok_c, skip_c, fail_c, reused_c)
