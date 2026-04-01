"""六星精二立绘与头像资源同步（torappu 优先，PRTS imageinfo 备用）。"""

from __future__ import annotations

import asyncio
import csv
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
from nonebot import logger

# 维基 API（PRTS 立绘文件查询）
WIKI_API = "https://prts.wiki/api.php"

TORAPPU_CHAR_ART_URL = "https://torappu.prts.wiki/assets/char_arts/{char_id}_2.png"
CHAR_ARTS_REL_DIR = "char_arts"
TORAPPU_CHAR_AVATAR_URL = "https://torappu.prts.wiki/assets/char_avatar/{char_id}_2.png"
CHAR_AVATAR_REL_DIR = "char_avatar"
ART_DOWNLOAD_USER_AGENT = (
    "Mozilla/5.0 (compatible; nonebot-plugin-arkguesser/1.0; +prts-wiki-art-dl)"
)
# 与 char_art_match.rebuild_char_e2_head_align_csv 默认阈值一致（供 data_update 传入）
CHAR_ART_LOCAL_MIN_BYTES = 512
CHAR_AVATAR_LOCAL_MIN_BYTES = 128


def _local_download_target_exists(dest: Path) -> bool:
    """目标路径已有同名文件则跳过下载（不校验体积；需强制更新请先删除本地文件）。"""
    return dest.is_file()


PRTS_ART_QUERY_BATCH = 8
PRTS_ART_QUERY_SLEEP = 0.35
ART_DOWNLOAD_CONCURRENCY = 4


def _prts_illustration_file_title(operator_name: str) -> str:
    return f"文件:立绘 {operator_name.strip()} 2.png"


def load_six_star_rows_from_csv(csv_path: Path) -> list[tuple[str, str]]:
    """(char_id, name)，仅 rarity==6。char_id 为 CSV 中完整 id（如 char_4123_ela）。"""
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


async def _http_retry_get_json(
    client: httpx.AsyncClient,
    params: dict[str, str],
    timeout: float,
    max_retries: int = 4,
) -> dict[str, Any]:
    q = urllib.parse.urlencode(params)
    url = f"{WIKI_API}?{q}"
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = await client.get(
                url,
                timeout=timeout,
                headers={"User-Agent": ART_DOWNLOAD_USER_AGENT},
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            last_err = e
            code = e.response.status_code
            if code in (502, 503, 504) and attempt + 1 < max_retries:
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            raise
        except OSError as e:
            last_err = e
            if attempt + 1 < max_retries:
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            raise
    assert last_err is not None
    raise last_err


async def query_prts_image_urls(
    client: httpx.AsyncClient,
    names: list[str],
    thumb_width: int,
    timeout: float,
) -> dict[str, str | None]:
    """operator_name -> 下载 URL；PRTS 无文件则为 None。"""
    result: dict[str, str | None] = {n: None for n in names}
    batch_size = max(1, PRTS_ART_QUERY_BATCH)
    for i in range(0, len(names), batch_size):
        chunk = names[i : i + batch_size]
        titles = "|".join(_prts_illustration_file_title(n) for n in chunk)
        params: dict[str, str] = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url",
            "titles": titles,
        }
        if thumb_width > 0:
            params["iiurlwidth"] = str(thumb_width)

        data = await _http_retry_get_json(client, params, timeout=timeout)
        pages = (data.get("query") or {}).get("pages") or {}
        for _pid, page in pages.items():
            title = page.get("title") or ""
            prefix = "文件:立绘 "
            suffix = " 2.png"
            if not title.startswith(prefix) or not title.endswith(suffix):
                continue
            op_name = title[len(prefix) : -len(suffix)]
            if "missing" in page or "imageinfo" not in page:
                result[op_name] = None
                continue
            infos = page["imageinfo"]
            if not infos:
                result[op_name] = None
                continue
            info = infos[0]
            if thumb_width > 0:
                url = info.get("thumburl") or info.get("url")
            else:
                url = info.get("url")
            result[op_name] = url

        if PRTS_ART_QUERY_SLEEP > 0 and i + batch_size < len(names):
            await asyncio.sleep(PRTS_ART_QUERY_SLEEP)
    return result


async def torappu_resource_exists(
    client: httpx.AsyncClient, url: str, timeout: float = 30.0
) -> bool:
    """HEAD torappu 资源；不支持 HEAD 时用 Range GET 探测。"""
    headers = {"User-Agent": ART_DOWNLOAD_USER_AGENT}
    try:
        r = await client.head(url, follow_redirects=True, timeout=timeout, headers=headers)
        if r.status_code == 200:
            return True
        if r.status_code in (405, 501):
            r2 = await client.get(
                url,
                headers={**headers, "Range": "bytes=0-0"},
                follow_redirects=True,
                timeout=timeout,
            )
            return r2.status_code in (200, 206)
        return False
    except Exception:
        return False


async def torappu_char_art_exists(
    client: httpx.AsyncClient, char_id: str, timeout: float = 30.0
) -> bool:
    """HEAD torappu char_arts/{char_id}_2.png。"""
    return await torappu_resource_exists(
        client, TORAPPU_CHAR_ART_URL.format(char_id=char_id), timeout=timeout
    )


async def download_art_to_path(
    client: httpx.AsyncClient, url: str, dest: Path, timeout: float = 120.0
) -> str | None:
    """成功返回 None，失败返回错误说明。dest 已存在则不再请求网络。"""
    try:
        if dest.is_file():
            return None
        r = await client.get(
            url,
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": ART_DOWNLOAD_USER_AGENT},
        )
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        data = r.content
        if not data:
            return "empty body"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return None
    except Exception as e:
        return str(e)


async def sync_six_star_elite2_arts(
    csv_path: Path, data_dir: Path, client: httpx.AsyncClient
) -> None:
    """
    数据写入后：对六星检测 torappu 是否存在精二立绘；本地尚无同名文件时优先 torappu 下载，
    否则按 PRTS MediaWiki imageinfo 解析真实地址下载。
    输出目录：{data_dir}/char_arts/{char_id}_2.png（已存在 {char_id}_2.png 则跳过该干员下载）
    """
    rows = load_six_star_rows_from_csv(csv_path)
    if not rows:
        logger.debug("无六星干员，跳过立绘同步")
        return

    out_dir = data_dir / CHAR_ARTS_REL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    to_process: list[tuple[str, str]] = []
    for cid, name in rows:
        dest = out_dir / f"{cid}_2.png"
        if _local_download_target_exists(dest):
            continue
        to_process.append((cid, name))

    if not to_process:
        logger.debug(f"六星立绘目录下均已存在同名文件，跳过下载 {out_dir}")
        return

    logger.info(f"六星精二立绘：待下载 {len(to_process)} 个 → {out_dir}")

    sem = asyncio.Semaphore(max(1, ART_DOWNLOAD_CONCURRENCY))
    torappu_ok = 0

    async def try_torappu(cid: str, name: str) -> tuple[str, str] | None:
        nonlocal torappu_ok
        dest = out_dir / f"{cid}_2.png"
        if _local_download_target_exists(dest):
            torappu_ok += 1
            logger.debug(f"立绘本地已存在，跳过 torappu {cid} ({name})")
            return None
        async with sem:
            ok_remote = await torappu_char_art_exists(client, cid)
            if not ok_remote:
                return (cid, name)
            url = TORAPPU_CHAR_ART_URL.format(char_id=cid)
            err = await download_art_to_path(client, url, dest)
            if err is None:
                torappu_ok += 1
                logger.debug(f"立绘 OK torappu {cid} ({name})")
                return None
            logger.warning(f"立绘 torappu 下载失败 {cid} ({name}): {err}，将尝试 PRTS")
            return (cid, name)

    torappu_results = await asyncio.gather(
        *(try_torappu(cid, name) for cid, name in to_process)
    )
    seen_cid: set = set()
    prts_rows: list[tuple[str, str]] = []
    for item in torappu_results:
        if not item:
            continue
        cid, name = item
        if cid in seen_cid:
            continue
        seen_cid.add(cid)
        prts_rows.append((cid, name))

    if not prts_rows:
        logger.success(f"六星立绘同步完成（{len(to_process)} 个，来源 torappu）")
        return

    unique_names = list(dict.fromkeys(name for _, name in prts_rows))
    logger.debug(f"PRTS 查询立绘 URL（{len(unique_names)} 个干员名）…")
    try:
        url_map = await query_prts_image_urls(client, unique_names, thumb_width=0, timeout=120.0)
    except Exception as e:
        logger.error(f"PRTS 立绘 API 失败: {e}")
        return

    prts_ok = 0

    async def dl_prts(cid: str, name: str) -> None:
        nonlocal prts_ok
        dest = out_dir / f"{cid}_2.png"
        if _local_download_target_exists(dest):
            prts_ok += 1
            logger.debug(f"立绘本地已存在，跳过 PRTS {cid} ({name})")
            return
        url = url_map.get(name)
        if not url:
            logger.warning(f"PRTS 无立绘文件: {name} ({cid})")
            return
        async with sem:
            err = await download_art_to_path(client, url, dest)
            if err is None:
                prts_ok += 1
                logger.debug(f"立绘 OK PRTS {cid} ({name})")
            else:
                logger.warning(f"立绘 PRTS 下载失败 {cid} ({name}): {err}")

    await asyncio.gather(*(dl_prts(cid, name) for cid, name in prts_rows))
    n_ok = torappu_ok + prts_ok
    n_fail = len(to_process) - n_ok
    logger.success(
        f"六星立绘同步完成（{len(to_process)} 个：成功 {n_ok}，未写入 {n_fail}；torappu {torappu_ok}，PRTS {prts_ok}）"
    )


async def sync_six_star_elite2_avatars(
    csv_path: Path, data_dir: Path, client: httpx.AsyncClient
) -> None:
    """
    六星精二头像：torappu char_avatar/{char_id}_2.png，无备用链接。
    输出目录：{data_dir}/char_avatar/{char_id}_2.png（已存在同名文件则跳过）
    """
    rows = load_six_star_rows_from_csv(csv_path)
    if not rows:
        logger.debug("无六星干员，跳过精二头像同步")
        return

    out_dir = data_dir / CHAR_AVATAR_REL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    to_process: list[tuple[str, str]] = []
    for cid, name in rows:
        dest = out_dir / f"{cid}_2.png"
        if _local_download_target_exists(dest):
            continue
        to_process.append((cid, name))

    if not to_process:
        logger.debug(f"六星精二头像目录下均已存在同名文件，跳过下载 {out_dir}")
        return

    logger.info(f"六星精二头像：待下载 {len(to_process)} 个 → {out_dir}")
    sem = asyncio.Semaphore(max(1, ART_DOWNLOAD_CONCURRENCY))

    async def one(cid: str, name: str) -> str:
        dest = out_dir / f"{cid}_2.png"
        if _local_download_target_exists(dest):
            logger.debug(f"头像本地已存在，跳过 {cid} ({name})")
            return "ok"
        url = TORAPPU_CHAR_AVATAR_URL.format(char_id=cid)
        async with sem:
            if not await torappu_resource_exists(client, url):
                logger.debug(f"头像 torappu 无资源 {cid} ({name})")
                return "skip_remote"
            err = await download_art_to_path(client, url, dest)
            if err is None:
                logger.debug(f"头像 OK {cid} ({name})")
                return "ok"
            logger.warning(f"精二头像下载失败 {cid} ({name}): {err}")
            return "fail"

    results = await asyncio.gather(*(one(cid, name) for cid, name in to_process))
    n_ok = sum(1 for x in results if x == "ok")
    n_skip = sum(1 for x in results if x == "skip_remote")
    n_fail = sum(1 for x in results if x == "fail")
    logger.success(
        f"六星精二头像同步完成（{len(to_process)} 个：成功 {n_ok}，远端无 {n_skip}，失败 {n_fail}）"
    )
