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

        transcript_parts: list[str] = []
        segments: list[dict] = []
        for chunk_path, offset in chunks:
            result = transcribe_file(chunk_path, model, api_key)
            transcript_parts.append(result.get("text", ""))
            for seg in result.get("segments", []) or []:
                segments.append({
                    "start": round(float(seg.get("start", 0)) + offset, 2),
                    "end": round(float(seg.get("end", 0)) + offset, 2),
                    "text": seg.get("text", ""),
                })
        return {
            "title": title,
            "duration": round(duration, 1),
            "transcript": "".join(transcript_parts),
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
