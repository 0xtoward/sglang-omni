# OmniTyper

**Local voice typing, powered by SGLang-Omni.** 面向 Apple Silicon Mac 的开源语音输入应用。原生 SwiftUI / AppKit 界面，使用 **SGLang-Omni 的原生 MLX Qwen3-ASR 服务**识别语音，通过可配置的 **OpenAI 兼容 API** 完成整理、翻译、语音编辑和问答，默认连接本机 Ollama。

使用本地 Ollama 模型时无需云端推理 API；也可自行配置其他兼容服务。独立项目，与 Typeless 无关联，不使用其商标素材或私有代码。

## 快速开始

需要 macOS 14+、Apple Silicon、Xcode Command Line Tools（Swift 6 工具链用于测试）、Homebrew。建议至少 16 GB 内存，并为 Python 运行环境及模型预留数 GB 磁盘空间。

在 `sglang-omni` 根目录执行：

```bash
bash OmniTyper/scripts/setup.sh
open OmniTyper/dist/OmniTyper.app
```

安装脚本复用仓库根目录的 `install.sh`：创建 `OmniTyper/.venv`，安装 SGLang `v0.5.19` 的 Apple Silicon 依赖、当前 SGLang-Omni，以及 `ffmpeg@7`。不会安装 CUDA 包或替换系统 Python。Homebrew 和 Command Line Tools 需预先安装。

首次打开：

1. 在首页允许 **Microphone** 和 **Accessibility**。macOS 的隐私授权必须由用户在系统界面授予，应用无法自行批准。
2. 在 **Settings → Local speech model → Download & prepare ASR** 准备识别模型。首次需要访问 Hugging Face；缓存后可离线识别。
3. 在 **Settings → Text API** 填写 API 地址和服务端模型名；默认地址为 `http://127.0.0.1:11434/v1`。点击 **Connect & load models** 读取模型列表，也可以直接输入自己创建的模型名。
4. 在任意支持辅助功能的文本框放好光标，按 **Control + Option + Space**，等悬浮窗显示 **Listening** 后开始说话；录音时可看到实时转写，再按一次结束。结果写入原来的位置。
5. **Esc** 取消。设置中可以录制自己的快捷键，或切换为按住说话、松开完成。

从主窗口直接点击 Start speaking 时，结果显示在应用内供复制。跨应用写入请在目标应用中使用全局快捷键。

应用关闭主窗口后保留菜单栏图标；菜单中的 Quit 会退出应用并关闭其模型进程。启动登录项需要先将构建出的应用放在固定位置，建议 `~/Applications`，再开启 **Open at login**。

从旧名 OpenTypeless 升级时，请先退出旧应用。首次启动会将旧数据目录迁移到 `~/Library/Application Support/OmniTyper`，保留设置、词典、历史和音频；已有新目录时不会覆盖它。项目目录移动后，已失效的旧 Python 路径会在新路径可执行时自动更新。应用标识现为 `org.sglang.OmniTyper`，需要为 OmniTyper 重新授予麦克风、辅助功能权限，并按需重新开启登录项。

## Ollama 与 OmniTyper 的关系

- **SGLang-Omni**：负责本机 ASR，接收麦克风音频并输出转写。
- **Ollama / 兼容 API 服务**：负责文本模型的下载、加载、配置和推理。
- **OmniTyper**：负责录音、上下文、提示词和文字写入，只向文本 API 发送转写、写作偏好及编辑/问答所需的选中文字。

先启动你自己的 Ollama 服务，并用 `ollama list` 查看已安装的模型。把模型名称原样填入 Text API 的 **Model name**；自己用 Modelfile 创建的别名也可以使用。OmniTyper 不会安装、启动、停止 Ollama，也不会替你下载或锁定文本模型。服务端可以随时修改模型实现；更换名称时只需同步更新应用设置。

API 使用 `GET <Base URL>/models` 和 `POST <Base URL>/chat/completions`。Base URL 应包含服务前缀（通常为 `/v1`），不要填写完整的 `/chat/completions` 路径。依据 [Ollama 官方 OpenAI 兼容接口说明](https://docs.ollama.com/api/openai-compatibility)。没有模型列表接口的服务可跳过连接按钮，直接填模型名。

**Request options (JSON)** 默认 `{}`，请求只指定 `model`、`messages` 和 `stream: false`，不覆盖温度或采样默认值。需要时可传入 `{"temperature": 0.2, "max_tokens": 2048}` 等服务端支持的字段；上下文窗口、模型权重和 Modelfile 参数由 Ollama 管理。不同服务支持的字段不同，错误会显示在应用中。

本机 Ollama 通常不需要 API Key。可选 Key 仅保存在应用内存，退出或更改地址即清除，不写入历史、设置文件或日志。远程服务应使用 HTTPS；调用不跟随重定向。API 地址决定文本发送位置，本地 Ollama 也可能代理云模型，因此是否离线取决于你的模型和服务配置。

不需要润色时，在 **Writing style** 选择 **verbatim**：普通 Dictate 仅使用 ASR，无需配置或运行文本 API。API 不可用时，普通听写保留未润色原文并提示；翻译、编辑和问答返回错误并保留原始转写供复制/重试，不把失败输出自动当作成功结果写入。

## 已实现的使用流程

| 功能 | 行为 |
| --- | --- |
| Dictate | 录音 → 本机 ASR → 去口头语、整理标点 → 原光标写入；支持逐字模式跳过文本 API |
| Translate | 自动或指定语音语言，输出指定目标语言 |
| Voice edit | 在目标应用选中文字，说明如何修改，替换原选择；不保存选中文本 |
| Ask | 对选择的内容或一般问题提问；答案显示在应用中，不替换选择；不联网检索 |
| 全局快捷键 | 自定义组合键、切换或按住录音、Esc 取消、非抢焦点浮动录音条 |
| 输入设备 | 选择麦克风、实时音量、起止提示音、5 分钟录音上限 |
| Dictionary | 识别词汇提示、指定拼写替换、CSV 导入导出、从历史纠错添加词条 |
| Writing style | 全局及按应用设置 clean / verbatim / casual / formal / concise 风格和偏好 |
| History | 搜索、模式过滤、原文对照、复制、纠错、导出、删除和保留期限 |
| 音频保留 | 默认关闭；开启后可重试普通听写/翻译和导出 WAV；删除记录同步删除音频 |
| 系统设置 | 隐私权限入口、开机启动、深色/浅色主题、模型预加载和卸载 |

功能参照 [Typeless Quickstart](https://www.typeless.com/help/quickstart)、[语音编辑及问答](https://www.typeless.com/help/quickstart/ask-anything)、[历史与词典](https://www.typeless.com/help/quickstart/history-and-dictionary) 的公开交互，核对日期 2026-09-17。此版本不承诺相同的模型质量；不包含云同步、移动端键盘、跨应用被动学习、联网搜索与自动网页操作。纠错学习只发生在用户明确保存的词条上。

## 模型与运行边界

上游基线为 `27a8293c2d1e91077a48e79926868d6dc039dd3d`。该版本已经包含 `sglang_omni/models/qwen3_asr/mlx/`、MLX scheduler/runner 及 Apple Silicon 安装支持，OmniTyper 直接复用它们，**没有重复实现 ASR 或引入 mlx-audio**。

```text
SwiftUI / AppKit
  ├─ AVAudioEngine → 16 kHz 单声道 PCM16 → /v1/realtime WebSocket
  │    └─ 悬浮窗实时转写 + 同步保存 WAV 供失败重试
  ├─ 全局快捷键、Accessibility、条件恢复剪贴板
  └─ 私有 stdin/stdout JSON-lines worker
       ├─ 自主管理 SGLANG_USE_MLX=1 sgl-omni serve
       │    └─ 原生 Qwen3-ASR MLX（实时 /v1/realtime，重试 /v1/audio/transcriptions）
       └─ OpenAI 兼容 HTTP API → Ollama / 其他文本模型服务
            └─ 文本整理 / 翻译 / 编辑 / 问答
```

ASR 使用固定版本的 `mlx-community/Qwen3-ASR-0.6B-4bit`。文本模型完全由 API 服务管理，应用没有固定文本模型，也不在 worker 内加载 MLX-LM。录音期间通过原生 `/v1/realtime?intent=transcription` 持续发送音频，在悬浮窗和主窗口显示实时转写；停止录音后取得最终转写，再调用文本 API。ASR 保持加载，文本模型的生命周期由服务端决定。


### 边说边看转写

录音前会启动或复用 ASR，等悬浮窗显示 **Listening** 后再说话。首次加载较慢，可先在 Settings 点击 **Download & prepare ASR**；后续录音复用已加载服务。按住说话模式下，加载期间松开快捷键会取消本次启动。

上游目前默认每约 2 秒音频触发一次部分识别，实际显示还取决于推理耗时。部分结果可能修正前面的字词，客户端按片段替换，不重复追加。30 秒边界由上游自动分段；最终插入以服务端确认的完整转写为准。短于刷新间隔的录音可能直接得到最终结果。这是上游已有的周期性音频窗口识别，不是逐 token 的持续解码缓存。

停止录音才会润色并插入文字，实时预览不会写入目标输入框。流式连接失败或发送积压时会提示，并在停止后通过完整 WAV 重新识别；录音与文本处理的失败重试仍可用。实时接口尚不接收词典热词提示，词典替换仍在最终转写后生效；完整 WAV 重试会使用上游支持的热词提示。

worker 为应用私有进程，不对外暴露控制 API。原生 SGLang-Omni 服务绑定随机 `127.0.0.1` 端口，仅用于本机 ASR，不暴露到局域网；当前上游推理接口无身份验证，同机进程可以访问该服务。退出、取消和 worker 终止会清理所属服务进程组。首次准备模型最多等待 30 分钟，普通 worker 请求最多等待 10 分钟；文本 API 连接超时 10 秒、读取超时 180 秒，超时可以重试。

选中文本仅在 Voice edit / Ask 所需的请求中发送到配置的文本 API。为防止写入错误位置，应用会在内存中比较目标字段的辅助功能内容、窗口和选择状态；某些编辑器的字段内容可能包含整篇文档，这些用于校验的完整字段内容不保存、不发送给模型；明确选中的编辑/问答上下文除外。不读取屏幕截图或浏览历史。辅助功能识别为密码框时拒绝录音；若无法识别目标、光标或选择发生变化，保留结果供手动复制。部分网页、自绘编辑器、远程桌面可能不暴露可靠的 Accessibility 信息，不能保证自动插入。

历史和设置保存于：

```text
~/Library/Application Support/OmniTyper/library.json
~/Library/Application Support/OmniTyper/Audio/
```

目录权限 `0700`，数据文件 `0600`，采用原子写入，不提供额外的磁盘加密。最多保留 1,000 条记录，支持 24 小时、7 天、30 天、1 年、永久；关闭历史会删除已有记录，关闭音频保留会删除已保留的音频。失败录音仅在当前会话暂存用于重试，退出时删除。崩溃或强制退出可能留下系统临时文件。模型下载缓存位于 Hugging Face 的标准缓存目录。应用不含分析埋点。

## 开发与构建

```bash
# 已有 Python 运行环境，只编译应用
OMNITYPER_PYTHON=/absolute/path/to/python bash OmniTyper/scripts/build.sh

# 单元与集成测试（无需下载模型、无需麦克风权限）
bash OmniTyper/scripts/test.sh

# 真实流式 ASR：合成语音按实时速度送入，验证录音结束前出现部分结果（不需要文本 API）
OmniTyper/.venv/bin/python OmniTyper/backend/smoke_stream.py

# 真实 ASR + 文本 API：先运行服务，用它列出的模型名替换 my-model
OmniTyper/.venv/bin/python OmniTyper/backend/smoke.py --model my-model

# 使用自己的 WAV 做真实识别测试
OmniTyper/.venv/bin/python OmniTyper/backend/smoke.py --model my-model --audio /absolute/path/to/audio.wav

# 自定义兼容服务；如需认证，可在本次命令环境中设置 OMNITYPER_API_KEY
OmniTyper/.venv/bin/python OmniTyper/backend/smoke.py --base-url http://127.0.0.1:8080/v1 --model my-model
```

开发模式可用 `CONFIGURATION=debug` 构建。应用 bundle 内携带 worker 源码，Info.plist 记录 Python 环境的绝对路径；当前构建是源码开发分发，不是携带全部 Python 和模型的独立安装包。迁移至另一台 Mac 时运行 setup，或者在设置里指定该机已安装的运行环境。不要移动或删除正在使用的仓库/虚拟环境。

`build.sh` 默认使用 ad-hoc 签名供本地运行。公开分发需要设置 `CODE_SIGN_IDENTITY`，使用自己的 Developer ID 签名并完成 Apple notarization。这里没有上传或发布任何版本。

测试覆盖改名时的数据和运行环境路径迁移、协议分帧、退出/取消、音频重采样和时长上限、CSV 格式、词典、历史保留、损坏数据保护、后端请求校验、静音和失败恢复。辅助功能权限、麦克风硬件以及各个第三方应用的插入兼容性需要在授权后的真实桌面上验收。

### 本机验证记录

2026-09-17，Apple M4 / 16 GB、Swift 6.4、Python 3.12.13：

- 15 项 Swift 测试及 16 项 Python 测试通过，覆盖实时片段修正、WebSocket 音频顺序与结束排空、断线/积压/取消、PCM 与 WAV 样本数一致、旧配置读取、API Key 不落盘、HTTP 参数透传及失败恢复。
- 原生 MLX 流式检查在音频仍在发送时得到 6 次非空部分结果，首个在 2.573 秒，最终完整保留最后一句；这是该机器上一次合成语音测试结果，不代表所有录音的延迟。
- 原生 ASR + 独立本地 MLX-LM HTTP 服务通过 6 项真实模型测试：语音识别、中英文整理、法语翻译、选区编辑和问答。Ollama 未安装在验证机器上，因此尚未完成 Ollama 本身的实机测试。
- 真实麦克风、快捷键及第三方应用插入仍需系统授权后的桌面验收。

缓存 ASR 权重、且文本服务使用本地缓存模型后，可为测试命令设置 `HF_HUB_OFFLINE=1`。该变量仅控制 Hugging Face 下载行为，不会禁止你配置的 API 访问网络。

## 故障定位

- **没有快捷键响应**：在系统设置允许 OmniTyper 的辅助功能访问，再重启应用；开发中重新签名可能需要移除旧授权后重新添加。
- **麦克风不可用**：允许麦克风访问，确认设置中选定设备仍连接。系统默认设备会在下一次录音时读取。
- **模型启动失败**：先运行 `scripts/setup.sh`；确认 Python 为 3.12、`ffmpeg@7` 可用，及 Hugging Face 可访问。代理环境需支持 HTTPX 的 SOCKS 依赖，setup 已包含。
- **文本处理失败**：确认 Ollama / API 服务已启动，Base URL 包含正确的 `/v1` 前缀，模型名与服务端一致；远程服务检查 API Key 和 HTTPS。可清空自定义请求参数后重试。翻译/编辑失败不会自动写入原始识别结果，仍可复制原始转写。
- **历史文件损坏**：应用会保留原文件并停止覆盖，显示具体路径。备份该文件后再手工修复或迁走它。

许可证：[Apache-2.0](../LICENSE)。模型权重和上游依赖遵循各自许可证。
