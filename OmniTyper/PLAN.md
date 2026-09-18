# OmniTyper implementation plan

Goal: a usable, open-source, local macOS voice typing application with Typeless-style dictation, translation, selected-text editing, and questions.

Architecture: SwiftUI/AppKit application streams PCM to the owned native SGLang-Omni MLX WebSocket server; a private JSON-lines subprocess manages this loopback server and calls a configurable OpenAI-compatible text API. The app process owns and cleans up the model server. Existing serving remains unchanged.

Constraints: macOS 14+, Apple Silicon for local inference, Swift Package Manager, Python 3.12 isolated environment. User authorized implementation without confirmation. All new files live here; native ASR code is reused from upstream. The ASR model downloads from Hugging Face on first preparation; text models are managed independently by the configured API server. No telemetry. No silent network inference fallback.

## Scope and decisions

- Native app rather than Electron/Tauri: system microphone, accessibility, keyboard and menu bar APIs without a web runtime.
- Initial checkout `89d3b711` was stale. After user correction, successfully fetched and fast-forwarded to `27a8293c`, which includes native Qwen3-ASR MLX. Reuse its documented installation and `SGLANG_USE_MLX=1 sgl-omni serve` command; no duplicate ASR implementation or core changes.
- Modes: dictation (verbatim or polished), translation, explicit selection replacement, question answering displayed without inserting. No autonomous web actions or cloud sync in the first local release.
- Global configurable shortcut, toggle/hold modes, Escape cancel, floating audio meter, menu bar, microphone choice, permission onboarding, launch at login.
- Local searchable history, retention controls, optional audio retention/retry, dictionary replacements and imports, per-app tone rules, export/delete, explicit local correction learning.
- Password fields are rejected. Target/focus changes prevent automatic insertion; result remains copyable. Restore clipboard only when another application/user has not changed it.
- Cancellation terminates the owned model worker when necessary, closes audio resources, and never inserts stale results.

## Task 1: MLX model and worker

Files: `backend/server.py`, `backend/worker.py`, `backend/requirements.txt`, `backend/test_worker.py`.

Contract: one JSON object per stdin line with `id` string and `op` (`prepare`, `transcribe`, `process`, `models`). Fields: `audio_path`, `asr_model`, `text_model`, `text_api_url`, `text_api_key`, `text_api_options`, `mode` (`dictate`, `translate`, `edit`, `ask`), `language` (empty means auto), `target_language`, `style` (`clean`, `verbatim`, `casual`, `formal`, `concise`), `instructions`, `dictionary` (objects with `spoken` and `written`), `selected_text`, `app_name`, and `text` for process-only. Progress responses: `{id,event:"progress",message}`. Final response: `{id,ok:true,text,raw_text,warning,duration}` or `{id,ok:false,error}`. Preparation returns `{id,ok:true,realtime_url}`; model discovery returns `{id,ok:true,models}`. stdout is protocol only. Validate all boundary data and WAV duration before model work. Empty/silent audio must not hallucinate output. One in-flight inference, bounded text/audio, graceful EOF.

- [x] Deploy upstream native ASR with real MLX inference and canonical language handling.
- [x] Implement API-based text cleanup/translation/edit/question prompts and dictionary substitutions.
- [x] Verify invalid requests, silence, model failure, protocol recovery, and real audio inference.

## Task 2: macOS system services

Files: `Sources/OmniTyper/SystemServices.swift`, `Sources/OmniTyper/WorkerClient.swift`.

Contract: `@MainActor AudioRecorder: ObservableObject` with published `level: Double`, `elapsed: Double`, static `devices() -> [MicrophoneDevice]` (id/name), `start(deviceUID: String) async throws`, `stop() throws -> URL`, `cancel()`. `@MainActor GlobalShortcut` with `start(keyCode: UInt16, modifiers: UInt64, hold: Bool, onStart: @escaping () -> Void, onStop: @escaping () -> Void, onCancel: @escaping () -> Void)`, `stop()`. CGEvent modifier bitmask; empty input UID = default. Main controller handles toggle via onStart; hold emits onStart/onStop.

`@MainActor TextInsertion` with static `isTrusted: Bool`, `requestPermission()`, `capture() throws -> InsertionTarget` (applicationName, bundleID, selectedText), `insert(_ text: String, into: InsertionTarget) async throws`, `copy(_ text: String)`.

`@MainActor WorkerClient: ObservableObject` published `status: String`, `isRunning: Bool`; `request(_ payload: [String: Any], python: String) async throws -> [String: Any]`; `stop()`. Worker path from `Bundle.main.resourceURL/backend/worker.py`, fallback `OMNITYPER_WORKER`; model stderr to bounded log, cancellation/timeout/crash resume continuation once. No shell invocation.

- [x] Implement recording, input device selection and meter.
- [x] Implement key event monitoring, cancel, secure field rejection and target-safe paste.
- [x] Implement subprocess transport and lifecycle checks.

## Task 3: application and distribution

Files: `Package.swift`, `Sources/OmniTyper/{OmniTyperApp,AppModel,Views,Store}.swift`, `scripts/*.sh`, `Resources/Info.plist`, `Tests/`, `README.md`.

- [x] Build dashboard, voice panel, history, dictionary, app rules and settings using native controls.
- [x] Connect recorder/worker/insertion with explicit idle/recording/processing state, stale response protection and recoverable errors.
- [x] Package `.app`, isolated dependency setup, local model preparation and documented launch command.
- [x] Swift build/test, Python tests, actual MLX smoke, launch inspection and independent review.

## Verification and parity

Public reference: https://www.typeless.com/help/quickstart and its dictate, translate, ask-anything, settings, personalization, history-and-dictionary guides, checked 2026-09-17. Implement original visuals/branding, no copied assets. Record actual command results and remaining system-permission limitations in README; do not claim untested cross-app behavior or model quality parity.

## Text API integration

ASR stays on native SGLang-Omni MLX. Text processing calls OpenAI-compatible HTTP chat completions, defaulting to local Ollama. The server owns model loading and inference defaults; users configure URL, model and optional JSON request fields. API keys are session-only and cleared on endpoint changes. Verbatim dictation and ASR preparation do not call the text API. No automatic remote fallback. Documented privacy follows the selected endpoint/provider rather than claiming all inference is always local.

## Live transcription

- [x] Enable the existing native realtime endpoint; do not duplicate Qwen decoding.
- [x] Wait for server/session readiness, send the recorder's 16 kHz PCM16 through a bounded WebSocket queue, and keep the WAV for recovery.
- [x] Replace partial segment hypotheses in the nonactivating voice panel and main window. Only the final transcription proceeds to text processing and insertion.
- [x] Drain audio before `transcription.done`; on transport failure fall back to complete-WAV ASR. Close the socket on cancellation or completion.
- [x] Verify Swift transport revisions, ordering, backpressure and cancellation; verify real MLX partial events before audio ingestion finishes.
