"""Video-to-subtitle backend.

POST /transcribe: takes a Douyin/Kuaishou video link and returns a Chinese
transcript (Groq Whisper). Pipeline: zacao parse -> download -> ffmpeg audio
extract -> Groq transcribe (chunked if the mp3 is over 25 MB).

The Groq API key comes ONLY from the GROQ_API_KEY environment variable.
No secret is stored in this code.
"""

import asyncio
import json
import os
import subprocess
import tempfile
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
    "正确的行原样返回；按原编号逐行返回，格式与输入完全一致（“编号. 文本”）；"
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


def proofread_segments(segments: list[dict], title: str, api_key: str) -> list[dict]:
    """Fix Whisper homophone errors with an LLM, keeping timestamps.

    Whisper hears sounds and guesses characters, so fast game slang comes out
    wrong in both fast and accurate modes (键位->箭位, 趴键->啪箭, 右探头->防探头).
    A language model reading the whole passage can fix these from context.
    Never raises: on any failure the original segments come back untouched.
    """
    if not segments:
        return segments
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": CHROME_UA,
        "Content-Type": "application/json",
    }
    fixed = [dict(s) for s in segments]
    for base in range(0, len(segments), PROOFREAD_BATCH):
        batch = segments[base:base + PROOFREAD_BATCH]
        numbered = "\n".join(f"{i + 1}. {s.get('text', '')}" for i, s in enumerate(batch))
        user_msg = f"视频标题：{title}\n转写分段：\n{numbered}"
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
                    "max_tokens": 8192,
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
            if resp.status_code != 200:
                continue
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            continue
        try:
            parsed: dict[int, str] = {}
            for line in content.strip().splitlines():
                line = line.strip().strip("`").strip()
                if not line or "." not in line:
                    continue
                num, _, text = line.partition(".")
                num = num.strip()
                if num.isdigit():
                    parsed[int(num)] = text.strip()
            # Only trust the batch if most lines came back; otherwise keep originals.
            if len(parsed) < len(batch) // 2:
                continue
            for i, seg in enumerate(batch, 1):
                if i in parsed and parsed[i]:
                    fixed[base + i - 1]["text"] = parsed[i]
        except Exception:
            continue
    return fixed


def do_transcribe(url: str, mode: str, api_key: str) -> dict:
    """Run the whole pipeline in a worker thread. Returns the response dict."""
    model = MODELS.get(mode, MODELS["fast"])
    with tempfile.TemporaryDirectory(prefix="vts-") as tmp:
        tmpdir = Path(tmp)
        video_url, title = parse_link(url)
        video_path = tmpdir / "video.mp4"
        download_video(video_url, video_path)
        duration = probe_duration(video_path)
        mp3_path = tmpdir / "audio.mp3"
        extract_audio(video_path, mp3_path)
        chunks = split_audio(mp3_path, duration, tmpdir)

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
        try:
            segments = proofread_segments(segments, title, api_key)
        except Exception:
            pass
        return {
            "title": title,
            "duration": round(duration, 1),
            "transcript": "".join(s["text"] for s in segments),
            "segments": segments,
        }


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

    try:
        result = await asyncio.to_thread(do_transcribe, url, mode, api_key)
    except PipelineError as exc:
        return JSONResponse(
            status_code=exc.status,
            content={"ok": False, "error": exc.message, "step": exc.step},
        )
    except Exception as exc:  # pragma: no cover - safety net
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"服务器内部出错: {exc}", "step": "server"},
        )
    return JSONResponse(status_code=200, content=result)
