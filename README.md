# 视频转字幕 · 后端

给静态网页版用的转写后端：收到一个抖音 / 快手视频链接，返回中文转写全文和带时间戳的分段。

## 接口

- `GET /health` → `{"ok": true}`
- `POST /transcribe`，JSON body：
  ```json
  {"url": "https://v.douyin.com/xxxx", "mode": "fast"}
  ```
  `mode` 是 `fast`（whisper-large-v3-turbo，约 20 秒）或 `accurate`（whisper-large-v3，更准）。
  成功返回：
  ```json
  {"title": "视频标题", "duration": 544.6,
   "transcript": "全文……",
   "segments": [{"start": 0.0, "end": 4.2, "text": "……"}]}
  ```
  失败返回 `{"ok": false, "error": "一句话说明", "step": "parse|download|audio|transcribe|config|input"}`。

## 流程

1. 解析链接：`GET https://video.zacao.top/api/parse/v2?url=<链接>`（不用 key，抖音和快手都实测过），取 `data.source_video_url`。
2. 下载视频：Chrome UA，抖音再加 `Referer: https://www.douyin.com/`。
3. `ffprobe` 读时长，`ffmpeg` 提音频为 128k mp3（约 1 MB/分钟）。
4. Groq 转写：`POST https://api.groq.com/openai/v1/audio/transcriptions`，`verbose_json`（带时间戳）。**必须发浏览器 UA**，否则 Cloudflare 拦 403（error 1010）。
   mp3 超过 25 MB 时按时间切块再逐块转写，时间戳按偏移合并。

Groq 的 key 只从环境变量 `GROQ_API_KEY` 读，不在代码里出现。没设 key 时 `/transcribe` 直接返回 500，提示去部署平台的环境变量里加。

## 本地运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=你的key
uvicorn server:app --host 0.0.0.0 --port 8000
```

## 部署到 Render（免费档）

1. 把这个目录推到一个 GitHub 仓库。
2. Render → New → Web Service → 选仓库 → Runtime 选 **Docker**。
3. Environment Variables 里加 `GROQ_API_KEY`（Groq 官网免费注册就有，每天 28,800 秒音频额度）。
4. 部署完拿到 `https://xxx.onrender.com`，静态网页把转写请求发到这个地址的 `/transcribe`。

免费档注意：
- 每月 750 小时额度，一个服务一直开着刚好够用。
- 15 分钟没人用就休眠，下一次请求要等约 1 分钟唤醒（第一次点转写会慢，之后就正常了）。
- Render 官方说免费档不用绑卡，但有实测说部署时可能要求卡验证，到时看。

## 已知限制

- 解析服务 zacao 免费档约每小时 30 次（按 IP 计），个人试用够用。
- Groq 免费档每天 28,800 秒音频（8 小时），超了要按量付费（turbo 约 $0.04/音频小时，v3 约 $0.111/音频小时）。
- 长视频先下载再转写，20 分钟视频全程约 1 分钟左右。
- 这是个个人试用后端，CORS 全开，没有做鉴权——别把地址公开到处发。
