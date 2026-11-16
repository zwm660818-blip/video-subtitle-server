"""Video-to-subtitle backend.

POST /transcribe starts a background job and returns a job_id immediately
(long videos take 2-4 minutes, past the platform proxy's ~2 minute limit
for one request). Poll GET /jobs/{id} until status is done or error.

Pipeline: zacao parse -> download -> ffmpeg audio extract -> Groq
transcribe (chunked if the mp3 is over 25 MB) -> LLM proofread of
homophone errors.

The Groq API key comes ONLY from the GROQ_API_KEY environment variable.
No secret is stored in this code.
"""

import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
ZACAO_URL = "https://video.zacao.top/api/parse/v2"
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
# Proofreading model: Qwen is the strongest Chinese model on Groq's free tier.
# Tested 2026-11-16 on the exact failing game-video lines: fixed 8/8 homophone
# errors (键位/趴键/右探头) with zero new errors. gpt-oss-120b was tried and
# rejected: it over-corrected and invented new errors (秒蹲->秒抢, 左探头->左键).
# llama-3.3-70b-versatile no longer exists on this account (404).
PROOFREAD_MODEL = "qwen/qwen3.8-27b"
PROOFREAD_BATCH = 100  # segments per LLM call
PROOFREAD_SYSTEM = (
    "你是中文语音转写校对员。用户给你一段中文语音转写的分段文本，每行前面有编号。"
    "你的任务：只修正其中听错的同音字、错别字（例如“键位”被听成“箭位”、“趴”被听成“啪”），"
    "结合上下文和整段内容判断。规则：只改错字，不改句式、不增删内容、不润色、不解释；"
    "如果某句话按字面讲不通，或与前后文的意思对不上（比如前面刚提到左探头，后面出现意思不明的防探头），"
    "按上下文推断最合理的词来修正；"
    "注意检查漏字：转写有时候会漏掉一个字，特别是重复的字、叠词（如“东东”被写成“东”），"
    "或句子读起来少了半截；如果某个词在同一段里别处出现过更完整的写法，"
    "按更完整的写法补全（如别处出现“东东ID”，这里的“东电竞”就该是“东东电竞”）；"
    "如果某行里的型号、编号与上面列出的反复出现的型号只差一两个字母，优先按反复出现的那个修正"
    "（例如视频里 CT5 出现多次，某行出现 CP5 很可能是把 T 听成了 P）；"
    "只返回你改动过的行，每行格式“编号. 修正后的文本”；没有改动的行不要返回；"
    "不要输出任何多余内容。"
)
MAX_GROQ_BYTES = 25 * 1024 * 1024          # Groq rejects files over 25 MB
CHUNK_TARGET_BYTES = 24 * 1024 * 1024      # stay safely under the limit
DOUYIN_REFERER = "https://www.douyin.com/"

MODELS = {
    "fast": "whisper-large-v3-turbo",
    "accurate": "whisper-large-v3",
}

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

def parse_link(url: str) -> tuple[str, str]:
    """Step 1: resolve the short link to a direct video URL via zacao."""
    try:
        resp = httpx.get(
            ZACAO_URL,
            params={"url": url},
            headers={"User-Agent": CHROME_UA},
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=True,
        )
    except httpx.RequestError as exc:
        raise PipelineError("parse", f"解析服务连不上: {exc}")
    if resp.status_code != 200:
        raise PipelineError("parse", f"解析服务返回 HTTP {resp.status_code}")
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        raise PipelineError("parse", "解析服务返回了无法识别的内容")
    if payload.get("code") != 200:
        raise PipelineError("parse", "这个链接解析不出来，换个链接试试")
    data = payload.get("data") or {}
    video_url = (
        data.get("source_video_url")
        or data.get("sourceURL")
        or data.get("video_url")
    )
    if not video_url:
        raise PipelineError("parse", "解析成功了，但没拿到视频地址")
    title = (
        data.get("title")
        or data.get("desc")
        or data.get("aweme_desc")
        or "未命名视频"
    )
    return video_url, str(title)


def download_video(video_url: str, dest: Path) -> None:
    """Step 2: download the video file."""
    headers = {"User-Agent": CHROME_UA}
    if "douyin" in video_url:
        headers["Referer"] = DOUYIN_REFERER
    try:
        with httpx.stream(
            "GET",
            video_url,
            headers=headers,
            timeout=httpx.Timeout(600.0, connect=10.0),
            follow_redirects=True,
        ) as resp:
            if resp.status_code != 200:
                raise PipelineError("download", f"视频下载失败，HTTP {resp.status_code}")
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(1024 * 256):
                    f.write(chunk)
    except httpx.RequestError as exc:
        raise PipelineError("download", f"视频下载失败: {exc}")
    if dest.stat().st_size == 0:
        raise PipelineError("download", "视频下载下来是空文件")


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


def extract_audio(video_path: Path, mp3_path: Path) -> None:
    """Step 3: ffmpeg -> 128k mp3 (about 1 MB per minute)."""
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


def split_audio(mp3_path: Path, duration: float, tmpdir: Path) -> list[tuple[Path, float]]:
    """Split the mp3 into chunks under the Groq size limit.

    Returns a list of (chunk_path, time_offset_seconds).
    """
    size = mp3_path.stat().st_size
    if size <= MAX_GROQ_BYTES:
        return [(mp3_path, 0.0)]
    # chunk length so each chunk stays under the target size
    chunk_seconds = max(60, int(duration * CHUNK_TARGET_BYTES / size))
    chunks: list[tuple[Path, float]] = []
    start = 0.0
    index = 0
    while start < duration:
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


def transcribe_file(audio_path: Path, model: str, api_key: str) -> dict:
    """Step 4: one Groq transcription call (verbose_json -> segments)."""
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/mpeg")}
        data = {
            "model": model,
            "language": "zh",
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
    batch: list[dict], title: str, api_key: str, anchors: str = ""
) -> dict[int, str]:
    """Proofread one batch; returns {1-based index: corrected text}.

    Returns only the lines the model changed (sparse output). On any failure
    returns {} (caller keeps the originals). Retries on 429: the free tier
    for this model allows only 8000 tokens/min, so a long video's batches
    will hit the limit -- wait out the minute window and try again.
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
                        {"role": "system", "content": PROOFREAD_SYSTEM},
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
            if len(text) > max(3 * len(original), len(original) + 20):
                continue
            parsed[idx] = text
        return parsed
    return {}


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


def do_transcribe(url: str, mode: str, api_key: str, on_stage=None) -> dict:
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
        video_url, title = parse_link(url)
        stage("download")
        video_path = tmpdir / "video.mp4"
        download_video(video_url, video_path)
        stage("audio")
        duration = probe_duration(video_path)
        mp3_path = tmpdir / "audio.mp3"
        extract_audio(video_path, mp3_path)
        chunks = split_audio(mp3_path, duration, tmpdir)

        stage("transcribe")
        segments: list[dict] = []
        for chunk_path, offset in chunks:
            result = transcribe_file(chunk_path, model, api_key)
            for seg in result.get("segments", []) or []:
                segments.append({
                    "start": round(float(seg.get("start", 0)) + offset, 2),
                    "end": round(float(seg.get("end", 0)) + offset, 2),
                    "text": seg.get("text", ""),
                })
        # Proofread: fix homophone errors from context. Never breaks the
        # transcription itself -- on failure the raw segments are returned.
        stage("proofread")
        proofread_stats = {"changed": 0, "batches": 0}
        try:
            segments, proofread_stats = proofread_segments(segments, title, api_key)
        except Exception:
            pass
        return {
            "title": title,
            "duration": round(duration, 1),
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


def _run_job(job_id: str, url: str, mode: str, api_key: str) -> None:
    """Background worker: run the pipeline and store the outcome."""
    def on_stage(name: str):
        with jobs_lock:
            job = jobs.get(job_id)
            if job is not None:
                job["stage"] = name

    try:
        result = do_transcribe(url, mode, api_key, on_stage=on_stage)
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
        target=_run_job, args=(job_id, url, mode, api_key), daemon=True
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
