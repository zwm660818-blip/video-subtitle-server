"""Video-to-subtitle backend.

POST /transcribe starts a background job and returns a job_id immediately
(long videos take 2-4 minutes, past the platform proxy's ~2 minute limit
for one request). Poll GET /jobs/{id} until status is done or error.

Pipeline: parse (zacao, hk0.cc fallback) -> download -> ffmpeg audio
extract -> Groq transcribe (chunked if the mp3 is over 25 MB) -> LLM
proofread of homophone errors.

The Groq API key comes ONLY from the GROQ_API_KEY environment variable.
No secret is stored in this code.
"""

import json
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
# Failing to even open the connection (Errno 101 Network is unreachable,
# refused, connect timeout). Retrying the same host never helps, so the
# downloader gives up on that address at once and moves to the next one.
_CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)
ZACAO_URL = "https://video.zacao.top/api/parse/v2"
HK0_BASE = "https://api.hk0.cc/api"
# hk0.cc fallback parser (no key needed). zacao's WeChat parser broke on
# 2026-11-18 (HTTP 401 for every WeChat link; Douyin/Kuaishou still fine),
# while hk0.cc parses WeChat/Douyin/Kuaishou/Bilibili. Response shape:
# {"code":200,"msg":"解析成功","data":{"url": direct mp4, "title": ...}}.
HK0_ENDPOINTS = [
    ("weixin.qq.com", "wxsph"),
    ("channels.weixin.qq.com", "wxsph"),
    ("douyin.com", "douyin"),
    ("v.douyin.com", "douyin"),
    ("kuaishou.com", "ksjx"),
    ("v.kuaishou.com", "ksjx"),
    ("bilibili.com", "bilibili"),
    ("xiaohongshu.com", "xhs"),
    ("weibo.com", "weibo"),
]
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
# Proofreading model: Qwen is the strongest Chinese model on Groq's free tier.
# Tested 2026-11-16 on the exact failing game-video lines: fixed 8/8 homophone
# errors (键位/趴键/右探头) with zero new errors. gpt-oss-120b was tried and
# rejected: it over-corrected and invented new errors (秒蹲->秒抢, 左探头->左键).
# llama-3.3-70b-versatile no longer exists on this account (404).
PROOFREAD_MODEL = "qwen/qwen3.8-27b"
PROOFREAD_BATCH = 100  # segments per LLM call
TRANSLATE_BATCH = 50  # segments per LLM call (translation is heavier)
TRANSLATE_SYSTEM = (
    "你是英中翻译。用户给你一段英文语音转写的分段文本，每行前面有编号。"
    "你的任务：把每行翻译成中文，保持原意，不增删内容、不润色、不解释；"
    "人名、地名、品牌、型号、专有名词保留英文原文；"
    "日期、数字按中文习惯写（例如 October 12 -> 10月12日）；"
    "只返回你翻译过的行，每行格式“编号. 翻译后的文本”；没有翻译的行不要返回；"
    "不要输出任何多余内容。"
)
PROOFREAD_SYSTEM = (
    "你是中文语音转写校对员。用户给你一段中文语音转写的分段文本，每行前面有编号。"
    "你的任务：只修正其中听错的同音字、错别字（例如“键位”被听成“箭位”、“趴”被听成“啪”），"
    "结合上下文和整段内容判断。规则：只改错字，不改句式、不增删内容、不润色、不解释；"
    "如果某句话按字面讲不通，或与前后文的意思对不上（比如前面刚提到左探头，后面出现意思不明的防探头），"
    "按上下文推断最合理的词来修正；"
    "注意检查漏字：转写有时候会漏掉一个字，特别是重复的字、叠词（如“东东”被写成“东”），"
    "或句子读起来少了半截；如果某个词在同一段里别处出现过更完整的写法，"
    "按更完整的写法补全（如别处出现“东东ID”，这里的“东电竞”就该是“东东电竞”）；"
    "注意常见的固定说法：说话里常用“这步棋”“这一招”“这一手”这类讲策略的固定说法；"
    "如果某行读起来别扭、又和某个固定说法发音接近，优先按固定说法修正"
    "（例如“这不岂是多妙”很可能是“这步棋有多妙”）；"
    "讲学习、看辩论、长见识的语境里，“找脑子”应为“长脑子”"
    "（长和找发音接近，Whisper 常把“长脑子”听成“找脑子”，"
    "而“找脑子”字面是“寻找脑子”，讲不通）；"
    "如果某行里的型号、编号与上面列出的反复出现的型号只差一两个字母，优先按反复出现的那个修正"
    "（例如视频里 CT5 出现多次，某行出现 CP5 很可能是把 T 听成了 P）；"
    "注意中英夹杂：说话人有时会说英文单词，转写可能把英文听成了中文同音字"
    "（例如“畅选”其实是 chance、“来味”其实是 level）；"
    "如果某行中文读起来别扭、又和某个英文单词发音接近、且按英文理解更符合上下文，就改成那个英文单词；"
    "特别注意“趴”要看上下文：讲游戏操作时的“趴键”“趴下”是原意，不要改；"
    "但“参加个趴”“开个趴”“去个趴”这种搭配里，“趴”是 party（派对、聚会）的音译，"
    "要改成 party（例如前后在讲论坛、合影、社交场合时，“我去参加个趴”应改为“我去参加个party”）；"
    "已经写对的英文单词不要改成中文，也不要改动；"
    "输出必须是简体中文，不要出现繁体字；"
    "只返回你改动过的行，每行格式“编号. 修正后的文本”；没有改动的行不要返回；"
    "不要输出任何多余内容。"
)
# English-gloss step: runs AFTER proofreading (so misheard English like 趴 has
# already been turned back into party) and only on lines that contain English.
ANNOTATE_BATCH = 60  # lines per LLM call; only English-bearing lines are sent
ANNOTATE_SYSTEM = (
    "你是字幕编辑。用户给你一段视频字幕里的若干行（已经校对过），每行前面有编号。"
    "你的任务：给每行里出现的英文单词或英文短语，在它后面紧跟一个全角括号，写上它在这句话里的中文意思，"
    "例如“你去寻找自己的chance”改成“你去寻找自己的chance（机会）”，"
    "“我去参加个party”改成“我去参加个party（派对）”。"
    "规则：中文意思要简短（一般 2 到 6 个字），按这句话的上下文取最贴切的意思；"
    "整个英文短语（如 face to face）作为一个整体只标一个意思，不要拆开；"
    "人名、品牌名、型号编号、缩写、字母加数字的写法（如 CT5、iPhone、ID、GPT-6、OK）不用加释义，保持原样；"
    "常见科技名词、公司名、产品名也不用加释义，例如 GPU、CPU、AI、GPT、Claude、Gemini、ChatGPT、Meet、API、WiFi、USB、Windows、Android、Google、Microsoft、Apple、OpenAI，这些词大家都知道是什么意思；"
    "括号里的中文意思用简体中文，不要用繁体字；"
    "已经带有括号解释的英文词不要重复加；"
    "除了加括号释义，不要改动这一行的任何其他字，不要改标点，不要改中文，不要润色；"
    "只返回你加过释义的行，每行格式“编号. 加好释义的文本”；没有英文词需要加释义的行不要返回；"
    "不要输出任何多余内容。"
)
MAX_GROQ_BYTES = 25 * 1024 * 1024          # Groq rejects files over 25 MB
DOUYIN_REFERER = "https://www.douyin.com/"

MODELS = {
    "fast": "whisper-large-v3-turbo",
    "accurate": "whisper-large-v3",
}

# Traditional -> Simplified: Whisper (language="zh") often outputs Traditional
# Chinese, and the proofreader only rewrites lines it thinks are wrong, so
# Traditional slips through untouched. Normalize everything to Simplified at
# the end of the pipeline. opencc-python-reimplemented is pure Python, no
# system dependencies.
try:
    from opencc import OpenCC
    _T2S = OpenCC("t2s")
except Exception:
    _T2S = None


def _to_simplified(text: str) -> str:
    """Convert Traditional Chinese to Simplified; no-op when unavailable."""
    if _T2S is None:
        return text
    try:
        return _T2S.convert(text)
    except Exception:
        return text

app = FastAPI(title="video-to-subtitle")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # personal trial backend
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineError(Exception):
    """A failure in one pipeline step, with a plain-language message."""

    def __init__(self, step: str, message: str, status: int = 502):
        super().__init__(message)
        self.step = step
        self.message = message
        self.status = status


def _run(cmd: list[str], step: str, what: str) -> str:
    """Run a subprocess command; raise PipelineError on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise PipelineError(step, f"{what}失败: {detail[-1] if detail else '未知错误'}")
    return (proc.stdout or "").strip()


# ---------------------------------------------------------------- pipeline

def _parse_via_zacao(url: str) -> tuple[str, str, list[str], bool] | None:
    """Try zacao. Returns (video_url, title, backup_urls, is_huge) or None."""
    try:
        resp = httpx.get(
            ZACAO_URL,
            params={"url": url},
            headers={"User-Agent": CHROME_UA},
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=True,
        )
    except httpx.RequestError:
        return None
    if resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        return None
    if payload.get("code") != 200:
        return None
    data = payload.get("data") or {}
    video_url = (
        data.get("source_video_url")
        or data.get("sourceURL")
        or data.get("video_url")
    )
    if not video_url:
        return None
    # The other URL field, when different, is a second address for the
    # same video (different CDN host) -- a free fallback if the first
    # host is unreachable from the server (Errno 101, seen 2026-11-18).
    backups = []
    for key in ("video_url", "sourceURL", "source_video_url"):
        alt = data.get(key)
        if alt and alt != video_url and alt not in backups:
            backups.append(alt)
    title = (
        data.get("title")
        or data.get("desc")
        or data.get("aweme_desc")
        or "未命名视频"
    )
    return video_url, str(title), backups, False


def _parse_via_hk0(url: str) -> tuple[str, str, list[str], bool] | None:
    """Try hk0.cc (no key). Returns (video_url, title, backups, is_huge)."""
    endpoint = None
    for host, ep in HK0_ENDPOINTS:
        if host in url:
            endpoint = ep
            break
    if not endpoint:
        return None
    try:
        resp = httpx.get(
            f"{HK0_BASE}/{endpoint}",
            params={"url": url},
            headers={"User-Agent": CHROME_UA},
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=True,
        )
    except httpx.RequestError:
        return None
    if resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        return None
    if payload.get("code") != 200:
        return None
    data = payload.get("data") or {}
    video_url = data.get("url") or data.get("video_url")
    if not video_url:
        return None
    title = data.get("title") or data.get("desc") or "未命名视频"
    # hk0 returns video_backup: same video on other CDN hosts. Its primary
    # URL is sometimes the huge original (e.g. Douyin 751 MB for a 7-min
    # video); the backups are usually the compressed ~60 MB version, so a
    # huge primary goes last in the candidate order.
    backups = []
    for item in data.get("video_backup") or []:
        if isinstance(item, str) and item and item != video_url and item not in backups:
            backups.append(item)
    for key in ("video_url",):
        alt = data.get(key)
        if alt and alt != video_url and alt not in backups:
            backups.append(alt)
    try:
        size = int(data.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    is_huge = size > 200 * 1024 * 1024
    return video_url, str(title), backups, is_huge


def parse_candidates(url: str) -> tuple[str, list[str]]:
    """Step 1: resolve the short link to ALL direct video URLs we can get.

    Runs both parsers (zacao first, hk0.cc second; WeChat reversed) and
    collects every candidate address, deduplicated, best first. Previously
    only the first parser's single URL was kept, so when that one CDN host
    was unreachable from the server (Errno 101 Network is unreachable,
    2026-11-18 18:26 on a Douyin link) the whole job failed even though
    the other parser had a working address for the same video.
    Returns (title, [url, ...])."""
    is_wechat = "weixin.qq.com" in url or "channels.weixin.qq.com" in url
    attempts = (
        [_parse_via_hk0, _parse_via_zacao] if is_wechat
        else [_parse_via_zacao, _parse_via_hk0]
    )
    results = []
    for fn in attempts:
        try:
            result = fn(url)
        except Exception:
            # A parser must never crash the job; try the next one.
            continue
        if result:
            results.append(result)
    if not results:
        raise PipelineError("parse", "这个链接解析不出来，换个链接试试")
    title = results[0][1]
    primaries: list[str] = []
    huge_primaries: list[str] = []
    backups: list[str] = []
    for video_url, _t, backup_urls, is_huge in results:
        (huge_primaries if is_huge else primaries).append(video_url)
        backups.extend(backup_urls)
    candidates: list[str] = []
    for u in primaries + backups + huge_primaries:
        if u and u not in candidates:
            candidates.append(u)
    return title, candidates


def parse_link(url: str) -> tuple[str, str]:
    """Back-compat wrapper: first candidate only."""
    title, candidates = parse_candidates(url)
    return candidates[0], title


def download_video(video_url: str, dest: Path) -> None:
    """Step 2: download the video file. For files over 20 MB, fetch 4 byte
    ranges in parallel (several CDNs serve one connection slowly but allow
    concurrent ranges); small files and servers without Range support fall
    back to a single stream. Retries up to 3 times, resuming with HTTP Range
    from where the last attempt stopped: a dropped connection on a large
    file (e.g. Bilibili ~245 MB, peer closed at 82%) picks up the remaining
    bytes instead of starting over."""
    headers = {"User-Agent": CHROME_UA}
    if "douyin" in video_url:
        headers["Referer"] = DOUYIN_REFERER
    # Find the total size with a 1-byte Range probe; fall back to single
    # stream if the server won't tell us (or won't do ranges).
    total = 0
    try:
        with httpx.stream(
            "GET", video_url,
            headers={**headers, "Range": "bytes=0-0"},
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        ) as resp:
            if resp.status_code == 206:
                cr = resp.headers.get("Content-Range", "")
                total = int(cr.split("/")[-1]) if "/" in cr else 0
    except _CONNECT_ERRORS as exc:
        # Can't open a connection at all: don't fall into the 3-retry loop
        # below (it would hit the same wall three times, ~10 s each for a
        # timeout). Let the caller move on to the next candidate address.
        raise PipelineError("download", f"连不上: {exc}")
    except httpx.RequestError:
        total = 0
    if total > 20 * 1024 * 1024:
        try:
            _download_parallel(video_url, dest, total, headers)
            if dest.stat().st_size == total:
                return
        except Exception:
            pass  # fall through to the single-stream path
    last_exc: Exception | None = None
    for _attempt in range(3):
        downloaded = dest.stat().st_size if dest.exists() else 0
        req_headers = dict(headers)
        if downloaded > 0:
            req_headers["Range"] = f"bytes={downloaded}-"
        try:
            with httpx.stream(
                "GET",
                video_url,
                headers=req_headers,
                timeout=httpx.Timeout(600.0, connect=10.0),
                follow_redirects=True,
            ) as resp:
                if resp.status_code not in (200, 206):
                    raise PipelineError("download", f"视频下载失败，HTTP {resp.status_code}")
                # Server ignored Range and answered 200: restart from scratch.
                mode = "ab" if (resp.status_code == 206 and downloaded > 0) else "wb"
                with open(dest, mode) as f:
                    for chunk in resp.iter_bytes(1024 * 256):
                        f.write(chunk)
        except _CONNECT_ERRORS as exc:
            if downloaded == 0:
                # Never got a connection: same fail-fast rule as the probe.
                raise PipelineError("download", f"连不上: {exc}")
            # A resume after a mid-stream drop could not reconnect: that is
            # a flaky host rather than an unreachable one, so keep trying.
            last_exc = exc
            time.sleep(2)
            continue
        except httpx.RequestError as exc:
            last_exc = exc
            time.sleep(2)
            continue
        if dest.stat().st_size == 0:
            raise PipelineError("download", "视频下载下来是空文件")
        return
    raise PipelineError("download", f"下载中断，重试3次仍失败: {last_exc}")


def download_video_any(candidates: list[str], dest: Path) -> str:
    """Try each candidate URL in order until one downloads.

    A candidate that fails (e.g. its CDN host is unreachable from this
    server) must not poison the next one: the partial file and any
    .part* chunks are deleted first, because download_video resumes from
    the existing file size -- resuming a different video's bytes would
    corrupt the file. Returns the URL that worked."""
    failures: list[str] = []
    for video_url in candidates:
        for suffix in ("", ".part0", ".part1", ".part2", ".part3"):
            try:
                (dest.with_suffix(suffix) if suffix else dest).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            download_video(video_url, dest)
        except PipelineError as exc:
            reason = exc.message
        except Exception as exc:  # pragma: no cover - safety net
            reason = f"{type(exc).__name__}: {exc}"
        else:
            if dest.exists() and dest.stat().st_size > 0:
                print(f"[download] ok {_describe_host(video_url)}", flush=True)
                return video_url
            reason = "视频下载下来是空文件"
        # Record WHICH host failed and what addresses it resolves to: when a
        # job fails, this is the only clue to why (e.g. an IPv6-only CDN host
        # gives Errno 101 on a server with no IPv6 route).
        failures.append(f"{_describe_host(video_url)} {reason}")
        print(f"[download] fail {failures[-1]}", flush=True)
    if not failures:
        raise PipelineError("download", "没有可用的下载地址")
    raise PipelineError(
        "download",
        f"视频下载失败，试了{len(candidates)}个地址都不行，请过一会儿再试。详情：" + "；".join(failures[:4]),
    )


def _describe_host(video_url: str) -> str:
    """Short 'host（IPv4×n，IPv6×m）' label for logs and error messages.

    Never raises and never blocks long: it only runs on the failure path."""
    try:
        host = urlsplit(video_url).hostname or video_url[:40]
    except ValueError:
        return video_url[:40]
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return f"{host}（域名解析失败）"
    v4 = len({i[4][0] for i in infos if i[0] == socket.AF_INET})
    v6 = len({i[4][0] for i in infos if i[0] == socket.AF_INET6})
    return f"{host}（IPv4×{v4}，IPv6×{v6}）"


def download_with_reparse(url: str, candidates: list[str], dest: Path) -> str:
    """Download from the candidates; if every one fails, parse the link again
    and try the fresh addresses once more.

    The CDN hosts and signed URLs change with every parse (e.g.
    v11/v13-default.365yg.com), so a second parse often yields an address
    that works, and it also gives a transient failure a second chance."""
    try:
        return download_video_any(candidates, dest)
    except PipelineError as first:
        try:
            _title, fresh = parse_candidates(url)
        except PipelineError:
            raise first
        # Addresses we have not tried come first, then re-tries of the old
        # ones from the new parse (their signed URLs differ).
        retry = [u for u in fresh if u not in candidates] + [u for u in fresh if u in candidates]
        if not retry:
            raise first
        time.sleep(3)
        return download_video_any(retry, dest)


def _download_parallel(video_url: str, dest: Path, total: int, headers: dict) -> None:
    """Download a file in 4 concurrent Range chunks, then concatenate."""
    import concurrent.futures

    parts = 4
    chunk_size = (total + parts - 1) // parts
    part_paths = [dest.with_suffix(f".part{i}") for i in range(parts)]

    def fetch(i: int) -> None:
        start = i * chunk_size
        end = min(total - 1, start + chunk_size - 1)
        with httpx.stream(
            "GET", video_url,
            headers={**headers, "Range": f"bytes={start}-{end}"},
            timeout=httpx.Timeout(600.0, connect=10.0),
            follow_redirects=True,
        ) as resp:
            if resp.status_code != 206:
                raise PipelineError("download", f"并行下载失败，HTTP {resp.status_code}")
            with open(part_paths[i], "wb") as f:
                for chunk in resp.iter_bytes(1024 * 256):
                    f.write(chunk)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parts) as pool:
        list(pool.map(fetch, range(parts)))
    with open(dest, "wb") as out:
        for p in part_paths:
            with open(p, "rb") as f:
                out.write(f.read())
            p.unlink()


def probe_duration(video_path: Path) -> float:
    """ffprobe: video duration in seconds."""
    out = _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        "audio",
        "读取视频时长",
    )
    try:
        return float(out)
    except ValueError:
        raise PipelineError("audio", "读取视频时长失败")


def extract_audio(video_path: Path, tmpdir: Path) -> Path:
    """Step 3: get the audio track. First try a stream copy (no re-encode,
    nearly instant when the video's audio is already AAC); fall back to
    re-encoding to 128k MP3. Returns the audio file path (may be .m4a)."""
    m4a_path = tmpdir / "audio.m4a"
    try:
        _run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(video_path),
                "-vn", "-c:a", "copy",
                str(m4a_path),
            ],
            "audio",
            "提取音频",
        )
        if m4a_path.stat().st_size > 0:
            return m4a_path
    except (PipelineError, OSError):
        pass
    mp3_path = tmpdir / "audio.mp3"
    _run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video_path),
            "-vn", "-c:a", "libmp3lame", "-b:a", "128k",
            str(mp3_path),
        ],
        "audio",
        "提取音频",
    )
    if mp3_path.stat().st_size == 0:
        raise PipelineError("audio", "提取出来的音频是空的")
    return mp3_path


def split_audio(mp3_path: Path, duration: float, tmpdir: Path) -> list[tuple[Path, float]]:
    """Split the audio into chunks under the Groq size limit.

    Returns a list of (chunk_path, time_offset_seconds). Each chunk is cut
    with a stream copy (no re-encode, nearly instant) when the source
    allows it; falls back to re-encoding to 128k MP3. Chunk length is
    sized by the MP3 output bitrate (23 MB / 16 KB/s = 1507s), which is
    safe for both AAC and MP3 sources.
    """
    size = mp3_path.stat().st_size
    if size <= MAX_GROQ_BYTES:
        return [(mp3_path, 0.0)]
    chunk_seconds = max(60, (23 * 1024 * 1024) // 16000)
    chunks: list[tuple[Path, float]] = []
    start = 0.0
    index = 0
    while start < duration:
        chunk_path = tmpdir / f"chunk{index}.m4a"
        copied = False
        try:
            _run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-ss", str(start), "-t", str(chunk_seconds),
                    "-i", str(mp3_path),
                    "-vn", "-c:a", "copy",
                    str(chunk_path),
                ],
                "audio",
                "切分长音频",
            )
            copied = chunk_path.stat().st_size > 0
        except (PipelineError, OSError):
            copied = False
        if not copied:
            chunk_path = tmpdir / f"chunk{index}.mp3"
            _run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-ss", str(start), "-t", str(chunk_seconds),
                    "-i", str(mp3_path),
                    "-vn", "-c:a", "libmp3lame", "-b:a", "128k",
                    str(chunk_path),
                ],
                "audio",
                "切分长音频",
            )
        chunks.append((chunk_path, start))
        start += chunk_seconds
        index += 1
    return chunks


# Whisper verbose_json returns the detected language as a name
# ("english", "chinese", ...). Groq's `language` parameter wants an
# ISO 639-1 code, so map the common names back to codes. Anything
# unmapped falls back to auto-detect (omit the parameter).
_LANG_NAME_TO_CODE = {
    "english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
    "spanish": "es", "french": "fr", "german": "de", "italian": "it",
    "portuguese": "pt", "russian": "ru", "arabic": "ar", "hindi": "hi",
    "indonesian": "id", "vietnamese": "vi", "thai": "th", "turkish": "tr",
    "dutch": "nl", "polish": "pl", "ukrainian": "uk", "swedish": "sv",
}


def _detect_lang_code(result: dict) -> str:
    """Normalize Whisper's detected language to an ISO 639-1 code (or "")."""
    lang = (result.get("language") or "").strip().lower()
    if not lang:
        return ""
    if len(lang) == 2 and lang.isalpha():
        return lang
    return _LANG_NAME_TO_CODE.get(lang, "")


def _cjk_ratio(text: str) -> float:
    """Fraction of CJK characters among letters in text (0 when no text)."""
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in text if ch.isalpha())
    total = cjk + letters
    return (cjk / total) if total else 0.0


def transcribe_file(audio_path: Path, model: str, api_key: str, language: str = None) -> dict:
    """Step 4: one Groq transcription call (verbose_json -> segments).

    language: ISO 639-1 code to force, or None to let Whisper auto-detect.
    Forcing "zh" on an English video makes Whisper translate the English
    into garbled Chinese (e.g. "December 12, 2026" -> "在12月12日2026年"),
    so callers auto-detect from the first chunk instead.
    """
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/mpeg")}
        data = {
            "model": model,
            # NOTE: do NOT add a `prompt` here. Tested 2026-11-17: an
            # instructional bilingual prompt ("中文写中文，英文写英文…")
            # makes Whisper echo the prompt itself ("中文写中文，") instead
            # of transcribing the audio. English words misheard as Chinese
            # ("畅选" for chance) are fixed downstream by the proofreader,
            # which has the full context to tell them apart.
            "response_format": "verbose_json",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            # Cloudflare in front of api.groq.com blocks non-browser UAs (403/1010)
            "User-Agent": CHROME_UA,
        }
        try:
            resp = httpx.post(
                GROQ_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=httpx.Timeout(600.0, connect=10.0),
            )
        except httpx.RequestError as exc:
            raise PipelineError("transcribe", f"转写服务连不上: {exc}")
    if resp.status_code == 401:
        raise PipelineError("transcribe", "Groq API key 无效，请检查服务器上的 GROQ_API_KEY")
    if resp.status_code != 200:
        raise PipelineError("transcribe", f"转写失败，Groq 返回 HTTP {resp.status_code}")
    try:
        return resp.json()
    except json.JSONDecodeError:
        raise PipelineError("transcribe", "转写服务返回了无法识别的内容")


def _extract_anchors(segments: list[dict]) -> str:
    """Find model/version codes that recur across the whole transcript.

    Whisper often mishears one letter of a code once (CT5 -> CP5) while
    getting it right everywhere else. A proofreader that only sees 100
    segments at a time cannot spot that inconsistency, so the frequent
    forms are collected up front and handed to every batch as anchors.
    Only letter+digit tokens count (CT5, 4S, GPT-6); plain words and bare
    numbers are too common to be useful anchors. Tokens seen at least
    twice, most frequent first, capped so the prompt stays small.
    Returns "" when there is nothing worth anchoring.
    """
    counts: Counter[str] = Counter()
    token_re = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
    for seg in segments:
        for token in token_re.findall(seg.get("text", "")):
            has_letter = any(c.isalpha() for c in token)
            has_digit = any(c.isdigit() for c in token)
            if has_letter and has_digit and len(token) >= 2:
                counts[token] += 1
    frequent = [(tok, n) for tok, n in counts.most_common(12) if n >= 2]
    if not frequent:
        return ""
    return "、".join(f"{tok}（{n}次）" for tok, n in frequent)


def _proofread_batch(
    batch: list[dict],
    title: str,
    api_key: str,
    anchors: str = "",
    system: str = PROOFREAD_SYSTEM,
    extra_chars: int = 20,
) -> dict[int, str]:
    """Proofread one batch; returns {1-based index: corrected text}.

    Returns only the lines the model changed (sparse output). On any failure
    returns {} (caller keeps the originals). Retries on 429: the free tier
    for this model allows only 8000 tokens/min, so a long video's batches
    will hit the limit -- wait out the minute window and try again.

    `system` and `extra_chars` let the English-gloss step reuse this same
    call/parse/retry logic with its own prompt and a bigger length allowance
    (a line can legitimately grow by a few "（中文意思）" groups).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": CHROME_UA,
        "Content-Type": "application/json",
    }
    numbered = "\n".join(f"{i + 1}. {s.get('text', '')}" for i, s in enumerate(batch))
    user_msg = f"视频标题：{title}\n"
    if anchors:
        user_msg += f"视频中反复出现的型号/编号：{anchors}\n"
    user_msg += f"转写分段：\n{numbered}"
    for attempt in range(4):
        try:
            resp = httpx.post(
                GROQ_CHAT_URL,
                headers=headers,
                json={
                    "model": PROOFREAD_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                    "max_tokens": 4096,
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        except Exception:
            return {}
        if resp.status_code == 429:
            wait = 65
            try:
                retry_after = int(resp.headers.get("retry-after", "0"))
                if 0 < retry_after <= 120:
                    wait = retry_after
            except ValueError:
                pass
            if attempt < 3:
                time.sleep(wait)
                continue
            return {}
        if resp.status_code != 200:
            return {}
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return {}
        parsed: dict[int, str] = {}
        for line in content.strip().splitlines():
            line = line.strip().strip("`").strip()
            if not line or "." not in line:
                continue
            num, _, text = line.partition(".")
            num = num.strip()
            text = text.strip()
            if not num.isdigit() or not text:
                continue
            idx = int(num)
            if not 1 <= idx <= len(batch):
                continue
            original = batch[idx - 1].get("text", "")
            # Guard against a runaway "correction" that rewrites the line.
            if len(text) > max(3 * len(original), len(original) + extra_chars):
                continue
            parsed[idx] = text
        return parsed
    return {}


# Deterministic fixed-phrase corrections, applied after the LLM proofreader.
# These are Whisper homophone traps the proofreader has already missed in
# real videos, so they are enforced in code instead of relying on the model:
# - "找脑子" -> "长脑子": Whisper hears 长 (zhang) as 找 (zhao) in both
#   fast and accurate modes; user reported 2026-11-18 on the debate video
#   ("长脑子最累的方式就是多看辩论赛"), proofreader changed 6 other lines
#   in that job but left this one. "长脑子" is the idiom for growing one's
#   brain; "找脑子" (searching for a brain) makes no sense in context.
FIXED_PHRASES = {
    "找脑子": "长脑子",
}


def _apply_fixed_phrases(segments: list[dict]) -> int:
    """Replace known misheard fixed phrases; returns number of lines changed."""
    changed = 0
    for seg in segments:
        text = seg.get("text", "")
        new_text = text
        for wrong, right in FIXED_PHRASES.items():
            new_text = new_text.replace(wrong, right)
        if new_text != text:
            seg["text"] = new_text
            changed += 1
    return changed


def proofread_segments(segments: list[dict], title: str, api_key: str) -> tuple[list[dict], dict]:
    """Fix Whisper homophone errors with an LLM, keeping timestamps.

    Whisper hears sounds and guesses characters, so fast game slang comes out
    wrong in both fast and accurate modes (键位->箭位, 趴键->啪箭, 右探头->防探头).
    A language model reading the whole passage can fix these from context.
    Never raises: on any failure the original segments come back untouched.
    Returns (segments, stats) where stats counts changed lines and batches.
    """
    if not segments:
        return segments, {"changed": 0, "batches": 0}
    fixed = [dict(s) for s in segments]
    changed = 0
    batches = 0
    # Anchor terms are computed once over the whole transcript and given to
    # every batch, so a one-off mishearing (CP5) can be caught against the
    # form that recurs (CT5) even when the two never share a batch.
    anchors = _extract_anchors(segments)
    for base in range(0, len(segments), PROOFREAD_BATCH):
        batch = segments[base:base + PROOFREAD_BATCH]
        batches += 1
        try:
            parsed = _proofread_batch(batch, title, api_key, anchors)
        except Exception:
            parsed = {}
        for idx, text in parsed.items():
            if fixed[base + idx - 1]["text"] != text:
                fixed[base + idx - 1]["text"] = text
                changed += 1
        # A short pause between batches: the free tier is 8000 tokens/min and
        # a full batch is ~1500 tokens, so back-to-back batches risk a 429
        # (which would cost a 65s retry and blow the proxy timeout). Two
        # seconds is enough to stay under the limit without stalling a
        # long video past the ~2 minute proxy ceiling.
        if base + PROOFREAD_BATCH < len(segments):
            time.sleep(2)
    return fixed, {"changed": changed, "batches": batches}


_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_GLOSS_RE = re.compile(r"（[^（）]*）")


def _only_glosses_added(original: str, new_text: str) -> bool:
    """True if new_text equals original once the （…） glosses are removed.

    This is the safety net for the gloss step: the model may only INSERT
    parenthesised meanings. If it also rewrote the Chinese, fixed the
    punctuation or dropped words, the line is rejected and the original is
    kept. ASCII parentheses are normalised first, and whitespace is ignored
    (Whisper segments often start with a space).
    """
    def norm(s: str) -> str:
        s = s.replace("(", "（").replace(")", "）")
        return re.sub(r"\s+", "", s)

    new_norm = norm(new_text)
    if "（" not in new_norm:
        return False
    return _GLOSS_RE.sub("", new_norm) == _GLOSS_RE.sub("", norm(original))


# Words everyone already knows -- never gloss these, even if the model does.
# The prompt tells the model to skip them, but it still glosses some (GPU,
# GPT, Claude, meet), so this is a code-level safety net.
NO_GLOSS = {
    # common tech terms
    "gpu", "cpu", "ai", "api", "wifi", "usb", "5g", "4g", "vr", "ar",
    "pc", "tv", "app", "os", "ios", "id", "ok", "vip", "diy",
    # companies / products (user: these need no explanation)
    "gpt", "claude", "gemini", "chatgpt", "meet", "google", "microsoft",
    "apple", "openai", "anthropic", "meta", "amazon", "tesla", "nvidia",
    "intel", "amd", "iphone", "ipad", "mac", "macbook", "windows",
    "android", "chrome", "safari", "firefox", "edge", "zoom", "teams",
    "slack", "discord", "telegram", "whatsapp", "wechat", "douyin",
    "tiktok", "youtube", "instagram", "facebook", "twitter", "reddit",
    "github", "git", "python", "java", "javascript", "html", "css",
    "sql", "linux", "docker",
    # AI model names (also need no explanation)
    "grok", "llama", "astra", "opus", "sonnet", "haiku", "mistral",
    "qwen", "deepseek", "kimi", "doubao", "glm", "ernie", "hunyuan",
    # units
    "kg", "km", "cm", "mm", "ml", "gb", "mb", "tb", "hz",
}
_GLOSS_WORD_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)?)\s*（[^（）]*）")


def _strip_denylisted_glosses(text: str) -> str:
    """Remove （中文意思） after words everyone already knows.

    Only the gloss is removed; the English word itself is kept.
    """
    def repl(m):
        token = m.group(1)
        base = token.lower().split("-")[0]
        if token.lower() in NO_GLOSS or base in NO_GLOSS:
            return token
        return m.group(0)
    return _GLOSS_WORD_RE.sub(repl, text)


def annotate_segments(segments: list[dict], title: str, api_key: str) -> tuple[list[dict], int]:
    """Add （中文意思） after English words, keeping timestamps.

    Only lines that contain an English word are sent to the LLM, so this
    costs few tokens (usually a handful of lines). Never raises: on any
    failure the segments come back unchanged. Returns (segments, count of
    lines that got glosses).
    """
    fixed = [dict(s) for s in segments]
    idxs = [i for i, s in enumerate(fixed) if _ENGLISH_WORD_RE.search(s.get("text", ""))]
    if not idxs:
        return fixed, 0
    annotated = 0
    for base in range(0, len(idxs), ANNOTATE_BATCH):
        sub_idx = idxs[base:base + ANNOTATE_BATCH]
        sub = [fixed[i] for i in sub_idx]
        try:
            parsed = _proofread_batch(
                sub, title, api_key, system=ANNOTATE_SYSTEM, extra_chars=60
            )
        except Exception:
            parsed = {}
        for k, text in parsed.items():
            i = sub_idx[k - 1]
            original = fixed[i].get("text", "")
            if not _only_glosses_added(original, text):
                continue
            # Keep Whisper's leading space, if any, so spacing stays consistent.
            lead = original[: len(original) - len(original.lstrip())]
            glossed = _strip_denylisted_glosses(
                text.strip().replace("(", "（").replace(")", "）")
            )
            new_text = lead + glossed
            # If every gloss was on a denylisted word, the line is unchanged
            # and doesn't count as annotated.
            if new_text != original:
                fixed[i]["text"] = new_text
                annotated += 1
        if base + ANNOTATE_BATCH < len(idxs):
            time.sleep(2)
    return fixed, annotated


_SENTENCE_END = set(".!?。！？…")
_TRAILING_QUOTE = "\"”’）】})]"


def merge_segments(segments: list[dict], max_chars: int = 220, max_gap: float = 3.0) -> list[dict]:
    """Merge Whisper segments that are continuations of the same sentence.

    Whisper splits at pauses, so a sentence like "over a trillion dollars"
    can land as "...over a trillion" in one segment and "dollars ..." in the
    next -- the sentence reads as broken, and translating segment by segment
    keeps the break (the reader hits "超过一万亿" at 06:04 and only meets
    "美元" at 06:09). Merge a segment into the previous one when the
    previous does not end with sentence-ending punctuation, the gap is
    short, and the merged text stays readable. The merged segment keeps the
    first segment's start and the last one's end. Fewer, coherent segments
    also mean fewer tokens, so /translate is faster and less likely to hit
    the Groq rate limit.
    """
    if not segments:
        return segments
    merged: list[dict] = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        prev_text = (prev.get("text") or "").strip()
        cur_text = (seg.get("text") or "").strip()
        if not cur_text:
            continue
        if not prev_text:
            prev.update(seg)
            continue
        stripped = prev_text.rstrip(_TRAILING_QUOTE)
        last_char = stripped[-1] if stripped else ""
        gap = float(seg.get("start", 0)) - float(prev.get("end", 0))
        # Join with a space unless both sides are CJK (Chinese needs none).
        joiner = "" if ('\u4e00' <= last_char <= '\u9fff' and '\u4e00' <= cur_text[0] <= '\u9fff') else " "
        combined = prev_text + joiner + cur_text
        if (last_char not in _SENTENCE_END
                and gap <= max_gap
                and len(combined) <= max_chars):
            prev["text"] = combined
            prev["end"] = seg.get("end", prev.get("end"))
        else:
            merged.append(dict(seg))
    return merged


def do_transcribe(
    url: str, mode: str, api_key: str, on_stage=None, annotate: bool = False
) -> dict:
    """Run the whole pipeline in a worker thread. Returns the response dict.

    on_stage, when given, is called with the current stage name so a caller
    running this in the background can report real progress: parse, download,
    audio, transcribe, proofread.
    """
    def stage(name: str):
        if on_stage is not None:
            try:
                on_stage(name)
            except Exception:
                pass

    model = MODELS.get(mode, MODELS["fast"])
    with tempfile.TemporaryDirectory(prefix="vts-") as tmp:
        tmpdir = Path(tmp)
        stage("parse")
        title, candidates = parse_candidates(url)
        stage("download")
        video_path = tmpdir / "video.mp4"
        # Tries every candidate address; if all fail, re-parses the link for
        # fresh addresses and tries those too.
        download_with_reparse(url, candidates, video_path)
        stage("audio")
        duration = probe_duration(video_path)
        audio_path = extract_audio(video_path, tmpdir)
        chunks = split_audio(audio_path, duration, tmpdir)

        stage("transcribe")
        segments: list[dict] = []
        detected_lang = ""
        for chunk_path, offset in chunks:
            # First chunk auto-detects the language; the rest reuse it so
            # a mixed-language video doesn't flip languages mid-way.
            result = transcribe_file(chunk_path, model, api_key, language=detected_lang or None)
            if not detected_lang:
                detected_lang = _detect_lang_code(result)
            for seg in result.get("segments", []) or []:
                segments.append({
                    "start": round(float(seg.get("start", 0)) + offset, 2),
                    "end": round(float(seg.get("end", 0)) + offset, 2),
                    "text": seg.get("text", ""),
                })
        # Merge Whisper's pause-splits back into whole sentences before
        # proofreading and translation, so a sentence is never cut in half
        # across two segments (e.g. "over a trillion" / "dollars").
        segments = merge_segments(segments)
        # The Chinese proofreader, the English-gloss annotator and the
        # Traditional->Simplified pass all assume a Chinese transcript.
        # On a non-Chinese video (e.g. an English documentary) they would
        # mangle the English, so skip them and return the raw transcript.
        # is_chinese: detected language is Chinese, or the transcript is
        # mostly CJK (covers a first chunk that was silent/music).
        full_text = "".join(s.get("text", "") for s in segments)
        is_chinese = (detected_lang in ("zh", "chinese")) or _cjk_ratio(full_text) > 0.3
        # Proofread: fix homophone errors from context. Never breaks the
        # transcription itself -- on failure the raw segments are returned.
        stage("proofread")
        proofread_stats = {"changed": 0, "batches": 0}
        if is_chinese:
            try:
                segments, proofread_stats = proofread_segments(segments, title, api_key)
            except Exception:
                pass
            # Optional: put （中文意思） after English words. Runs after proofreading
            # so misheard English has already been restored. It reuses the
            # "proofread" stage name on purpose: the page only knows the stages
            # parse/download/audio/transcribe/proofread, and an unknown name would
            # make its progress steps jump back to step 1.
            if annotate:
                try:
                    segments, annotated = annotate_segments(segments, title, api_key)
                    proofread_stats["annotated"] = annotated
                except Exception:
                    pass
            # Fixed-phrase corrections run after the LLM proofreader (and
            # after annotating, so a gloss never lands on a wrong phrase):
            # the model has missed these in real jobs, code enforces them.
            proofread_stats["fixed_phrases"] = _apply_fixed_phrases(segments)
            # Normalize everything to Simplified Chinese: Whisper (language="zh")
            # often outputs Traditional, and the proofreader doesn't rewrite lines
            # it thinks are already right, so Traditional slips through. This is
            # the final guarantee -- it runs after proofreading and annotating.
            for s in segments:
                s["text"] = _to_simplified(s.get("text", ""))
        return {
            "title": title,
            "duration": round(duration, 1),
            "language": detected_lang or "",
            "transcript": "".join(s["text"] for s in segments),
            "segments": segments,
            "proofread": proofread_stats,
        }


# ---------------------------------------------------------------- jobs
# Long videos take 2-4 minutes end to end (download + transcribe +
# proofread), but the platform proxy kills synchronous requests after
# ~2 minutes (HTTP 524, seen twice on a 20-minute video). So /transcribe
# only starts a background job and returns immediately; the page polls
# /jobs/{id} until the job is done. Jobs live in memory only -- a
# container that sleeps mid-job loses it, and the page must retry.

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
JOB_TTL_SECONDS = 3600


def _run_job(job_id: str, url: str, mode: str, api_key: str, annotate: bool = False) -> None:
    """Background worker: run the pipeline and store the outcome."""
    def on_stage(name: str):
        with jobs_lock:
            job = jobs.get(job_id)
            if job is not None:
                job["stage"] = name

    try:
        result = do_transcribe(url, mode, api_key, on_stage=on_stage, annotate=annotate)
        with jobs_lock:
            jobs[job_id] = {"status": "done", "stage": "done", **result}
    except PipelineError as exc:
        with jobs_lock:
            jobs[job_id] = {
                "status": "error", "stage": "error",
                "error": exc.message, "step": exc.step,
            }
    except Exception as exc:  # pragma: no cover - safety net
        with jobs_lock:
            jobs[job_id] = {
                "status": "error", "stage": "error",
                "error": f"服务器内部出错: {exc}", "step": "server",
            }


def _cleanup_jobs() -> None:
    now = time.time()
    with jobs_lock:
        stale = [
            jid for jid, job in jobs.items()
            if now - job.get("created_at", now) > JOB_TTL_SECONDS
        ]
        for jid in stale:
            del jobs[jid]


# ---------------------------------------------------------------- routes

@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/transcribe")
async def transcribe(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "请求不是有效的 JSON", "step": "input"},
        )
    url = (body.get("url") or "").strip()
    mode = (body.get("mode") or "fast").strip().lower()
    if not url:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "请先粘贴视频链接", "step": "input"},
        )
    if not (url.startswith("http://") or url.startswith("https://")):
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "链接不对，要以 http 开头", "step": "input"},
        )
    if mode not in MODELS:
        mode = "fast"
    # Optional switch from the page: add （中文意思） after English words.
    # Absent or anything else means off, so older pages behave as before.
    annotate = body.get("annotate") in (True, 1, "1", "true", "True")

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "服务器没有设置 GROQ_API_KEY，请在部署平台的环境变量里添加后再试",
                "step": "config",
            },
        )

    _cleanup_jobs()
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {
            "status": "processing", "stage": "parse",
            "created_at": time.time(),
        }
    thread = threading.Thread(
        target=_run_job, args=(job_id, url, mode, api_key, annotate), daemon=True
    )
    thread.start()
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "processing"},
    )


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "error": "任务不存在或已过期，请重新提交",
                "step": "input",
            },
        )
    with jobs_lock:
        snapshot = dict(jobs[job_id])
    snapshot.pop("created_at", None)
    return JSONResponse(status_code=200, content=snapshot)


# ---------------------------------------------------------------- translate
def _translate_batch(batch: list[dict], api_key: str) -> dict[int, str]:
    """Translate one batch; returns {1-based index: translated text}.

    Same sparse-output + 429-retry shape as _proofread_batch: on any
    failure returns {} and the caller keeps the original English lines.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": CHROME_UA,
        "Content-Type": "application/json",
    }
    numbered = "\n".join(f"{i + 1}. {s.get('text', '')}" for i, s in enumerate(batch))
    for attempt in range(4):
        try:
            resp = httpx.post(
                GROQ_CHAT_URL,
                headers=headers,
                json={
                    "model": PROOFREAD_MODEL,
                    "messages": [
                        {"role": "system", "content": TRANSLATE_SYSTEM},
                        {"role": "user", "content": f"英文分段：\n{numbered}"},
                    ],
                    "temperature": 0,
                    "max_tokens": 4096,
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        except Exception:
            return {}
        if resp.status_code == 429:
            wait = 65
            try:
                retry_after = int(resp.headers.get("retry-after", "0"))
                if 0 < retry_after <= 120:
                    wait = retry_after
            except ValueError:
                pass
            if attempt < 3:
                time.sleep(wait)
                continue
            return {}
        if resp.status_code != 200:
            return {}
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return {}
        parsed: dict[int, str] = {}
        for line in content.strip().splitlines():
            line = line.strip().strip("`").strip()
            if not line or "." not in line:
                continue
            num, _, text = line.partition(".")
            num = num.strip()
            text = text.strip()
            if not num.isdigit() or not text:
                continue
            idx = int(num)
            if not 1 <= idx <= len(batch):
                continue
            original = batch[idx - 1].get("text", "")
            if len(text) > max(3 * len(original), len(original) + 20):
                continue
            parsed[idx] = text
        return parsed
    return {}


@app.post("/translate")
def translate(body: dict):
    """Translate English segments to Chinese, one-to-one, keeping timestamps.

    Body: {"segments": [{"start": float, "end": float, "text": str}, ...]}.
    Returns {"segments": [...] } with the same start/end and translated text.
    Lines the model doesn't return keep their original English.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "服务器没有设置 GROQ_API_KEY"},
        )
    segments = body.get("segments") or []
    if not isinstance(segments, list) or not segments:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "没有可翻译的字幕段"},
        )
    if len(segments) > 2000:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "字幕段太多（超过 2000 段），请分段翻译"},
        )
    out: list[dict] = []
    for base in range(0, len(segments), TRANSLATE_BATCH):
        batch = segments[base:base + TRANSLATE_BATCH]
        try:
            parsed = _translate_batch(batch, api_key)
        except Exception:
            parsed = {}
        for i, seg in enumerate(batch):
            text = parsed.get(i + 1, seg.get("text", ""))
            out.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": text,
            })
        if base + TRANSLATE_BATCH < len(segments):
            time.sleep(2)
    return JSONResponse(status_code=200, content={"segments": out})
