# 视频转字幕 · 后端

给静态网页版用的转写后端：收到一个抖音 / 快手视频链接，返回中文转写全文和带时间戳的分段。

## 接口

- `GET /health` → `{"ok": true}`
- `POST /transcribe`，JSON body：
  ```json
  {"url": "https://v.douyin.com/xxxx", "mode": "fast", "annotate": true}
  ```
  `mode` 是 `fast`（whisper-large-v3-turbo，约 20 秒）或 `accurate`（whisper-large-v3，更准）。
  `annotate` 可选，`true` 时校对之后再给字幕里的英文单词后面加全角括号写中文意思（如 `chance（机会）`）；不传或传别的值就是关，旧版页面不受影响。
  长视频全程要 2–4 分钟，超过平台代理约 2 分钟的单次请求上限（20 分钟视频实测两次 524 超时），所以这个接口只创建一个后台任务、立刻返回 `{"job_id": "...", "status": "processing"}`（HTTP 202），再用下面的接口轮询：
- `GET /jobs/{job_id}`，每 5 秒查一次，直到 `status` 变成 `done` 或 `error`：
  - 进行中：`{"status": "processing", "stage": "parse|download|audio|transcribe|proofread"}`（真实进度，页面按这个显示步骤）
  - 成功：`{"status": "done", "title": "视频标题", "duration": 544.6, "transcript": "全文……", "segments": [{"start": 0.0, "end": 4.2, "text": "……"}], "proofread": {"changed": 8, "batches": 1, "annotated": 2}}`（`annotated` 只有请求里 `annotate` 为 `true` 时才有，是加了中文释义的行数）
  - 失败：`{"status": "error", "error": "一句话说明", "step": "parse|download|audio|transcribe|config|input"}`
  - 任务不存在：HTTP 404，`{"status": "error", "error": "任务不存在或已过期，请重新提交"}`

  任务只存在内存里，容器休眠会丢任务，页面遇到这种情况请提示重新提交。任务 1 小时后自动清理。

## 流程

1. 解析链接：`GET https://video.zacao.top/api/parse/v2?url=<链接>`（不用 key，抖音和快手都实测过），取 `data.source_video_url`。
2. 下载视频：Chrome UA，抖音再加 `Referer: https://www.douyin.com/`。
3. `ffprobe` 读时长，`ffmpeg` 提音频为 128k mp3（约 1 MB/分钟）。
4. Groq 转写：`POST https://api.groq.com/openai/v1/audio/transcriptions`，`verbose_json`（带时间戳）。**必须发浏览器 UA**，否则 Cloudflare 拦 403（error 1010）。
   mp3 超过 25 MB 时按时间切块再逐块转写，时间戳按偏移合并。
5. 校对：转写完再调 Groq 的语言模型 `qwen/qwen3.8-27b` 按上下文修正同音字（Whisper 听音猜字，快慢两种模式在游戏黑话上都会成片错，例如“键位”听成“箭位”、“趴键”听成“啪箭”；2026-11-16 在一条和平精英视频的 16 行字幕上实测，8 处错全修好、没改错一句）。按 100 段一批、只返回改动过的行，时间戳不动；校对失败（模型挂了、返回格式不对）时直接返回原始字幕，转写本身不受影响。注意：这个模型免费档只有 8000 token/分钟，长视频的校对会分多批、遇到 429 就等一分钟再重试，20 分钟视频校对要多花 2–4 分钟，网页端的请求超时要相应放宽到 10 分钟。校对前先从全文提取反复出现的型号/编号（字母+数字，如 CT5、4S、GPT-6，出现 2 次以上）当“锚点”写进每一批的提示词：Whisper 偶尔会把某个型号听错一次（CT5 听成 CP5，T/P 都是清塞音、快了容易混），而校对一次只看 100 段，两处不在同一批就对不上；锚点让每一批都知道全视频反复出现的正确写法，单次出现的相似写法优先按锚点修正（2026-11-16 一条买车视频实测：03:28 的 CP5 漏网，04:24 的 CT5 正确，修法由此而来）。校对提示词里还有一条“查漏字”规则：转写有时会漏掉一个字、特别是重复的字（“东东”写成“东”），如果同一段里别处出现过更完整的写法就按完整的补全（2026-11-16 一条东东电竞陪玩视频实测：首段“东电竞”靠同批里“东东ID”补回了“东东电竞”）；还有一条“固定说法”规则：读起来别扭、又和某个常见固定说法发音接近的，优先按固定说法修正（2026-11-16 一条 Muse 开源硬件视频实测：00:08 的“这不岂是多妙”靠这条改回了“这步棋有多妙”）；但“右探头”这类靠成对推理的修正不太稳定，同一提示词有时修好、有时漏掉，不是每次都行。

6. 英文释义（可选，请求里 `annotate: true` 才跑）：校对之后，只把含英文单词的行挑出来交给同一个模型，让它在英文词后面加（中文意思）。人名、品牌、型号编号不加；常见科技名词、公司名、产品名也不加（GPU、CPU、AI、GPT、Claude、Gemini、ChatGPT、Meet、API、WiFi、USB、Windows、Android 等，这些词大家都知道是什么意思）。提示词里写了不加，但模型有时还是会加，所以代码里还有一道名单过滤：返回的行里，名单上的词后面的（中文意思）会被直接去掉、只留英文词本身；如果一行里所有释义都被去掉了，这行就不计入 `annotated`。代码里还有一道保险：模型返回的行去掉所有括号释义后，必须和原句一模一样才采用，模型顺手改了中文、标点或漏了字的行一律丢弃、保留原句；失败时也是原样返回。这一步沿用 `proofread` 这个阶段名，因为页面只认识 parse/download/audio/transcribe/proofread 五个阶段名，出现不认识的阶段名页面进度会跳回第 1 步。
7. 中英夹杂的校对（2026-11-17）：Whisper 固定 `language=zh`，遇到夹在中文里的英文会写成汉字（party 写成“趴”、chance 写成“畅选”）。试过给 Whisper 加 `prompt` 提示中英混说，结果 Whisper 把提示词本身当字幕念出来，所以不加；改成在校对提示词里写明这类音译要改回英文，并专门写了“趴”要看上下文（游戏里的“趴键”不改，“参加个趴”改成 party），避免误伤。这条是靠大模型判断，不保证每次都对。
8. 简体中文归一（2026-11-17）：Whisper 的 `language=zh` 经常输出繁体中文，而且校对只改它认为错的行、繁体字本身没错就原样留下来，所以字幕会简繁混杂（前几行简体、后面突然变繁体）。修法：流程最后一步用 OpenCC（`opencc-python-reimplemented`，纯 Python）把全部字幕从繁体转简体；校对和释义的提示词里也写明了输出必须是简体中文。转简体是最后一步，释义括号里的中文意思也会一起转。

Groq 的 key 只从环境变量 `GROQ_API_KEY` 读，不在代码里出现。没设 key 时 `/transcribe` 直接返回 500，提示去部署平台的环境变量里加。

## 本地运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=你的key
uvicorn server:app --host 0.0.0.0 --port 8000
```

## 部署到 Render（免费档）

Render 免费档部署时会要求绑卡（$1 临时预授权），没卡走不通。2026-11-16 已改用 SnapDeploy，流程一样（GitHub 仓库 → Deploy from GitHub → 加 `GROQ_API_KEY` 环境变量），注册和部署全程不要卡，免费档 4 个容器、每月 100 小时、15 分钟不用休眠、唤醒约 1 分钟；部署地址形如 `https://video-subtitle-server-db002.containers.snapdeploy.app`。SnapDeploy 的部署弹窗状态会滞后，以实际访问 `/health` 为准。

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

<!-- redeploy 2026-11-18 07:38: trigger SnapDeploy auto-deploy for language auto-detect + /translate (main 9467d3f) -->
