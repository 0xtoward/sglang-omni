// SPDX-License-Identifier: Apache-2.0
import AppKit
import AVFoundation
import Combine
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    enum Phase: String { case idle, starting, recording, processing, preparing }
    let store: AppStore
    let recorder = AudioRecorder()
    let worker = WorkerClient()
    private let shortcut = GlobalShortcut()
    @Published var phase: Phase = .idle
    @Published var mode: VoiceMode = .dictate
    @Published var resultText = ""
    @Published var rawText = ""
    @Published var liveText = ""
    @Published var liveStatus = ""
    @Published var notice = ""
    @Published var error = ""
    @Published var lastApp = ""
    // ponytail: keys stay in memory for this session; use Keychain if persistence is needed.
    @Published var textAPIKey = ""
    @Published var textModels: [String] = []
    private var sessionAPIKey = ""
    @Published var microphoneAllowed = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
    @Published var accessibilityAllowed = false
    var showMainWindow: (() -> Void)?
    var showVoicePanel: (() -> Void)?
    var hideVoicePanel: (() -> Void)?
    private var target: InsertionTarget?
    private var task: Task<Void, Never>?
    private var speechStream: ASRStream?
    private var generation = UUID()
    private var timer: Timer?
    private var preferencesSubscription: AnyCancellable?
    private struct FailedRecording {
        let url: URL
        let duration: Double
        let mode: VoiceMode
        let target: InsertionTarget?
        let appName: String
    }
    private var retryRecording: FailedRecording?
    private var sessionPreferences = Preferences()

    init(store: AppStore? = nil) {
        let store = store ?? AppStore()
        self.store = store
        if store.preferences.pythonExecutable.isEmpty {
            store.preferences.pythonExecutable = Bundle.main.object(forInfoDictionaryKey: "OmniTyperPython") as? String
                ?? FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent("Library/Application Support/OmniTyper/runtime/bin/python").path
        }
        refreshPermissions()
        preferencesSubscription = store.$preferences.dropFirst().removeDuplicates().sink { [weak self] preferences in
            if preferences.textSettings.baseURL != self?.store.preferences.textSettings.baseURL {
                self?.textAPIKey = ""
                self?.textModels = []
            }
            // Published emits before storage changes; use the supplied value for shortcut settings.
            Task { @MainActor in self?.configureShortcut(preferences) }
        }
        configureShortcut(store.preferences)
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                if self.phase == .recording && self.recorder.elapsed >= 300 { self.finish() }
                self.refreshPermissions()
                self.store.prune()
            }
        }
    }

    var isBusy: Bool { phase != .idle }
    var shortcutLabel: String {
        let p = store.preferences
        var label = ""
        if p.shortcutModifiers & CGEventFlags.maskControl.rawValue != 0 { label += "⌃" }
        if p.shortcutModifiers & CGEventFlags.maskAlternate.rawValue != 0 { label += "⌥" }
        if p.shortcutModifiers & CGEventFlags.maskShift.rawValue != 0 { label += "⇧" }
        if p.shortcutModifiers & CGEventFlags.maskCommand.rawValue != 0 { label += "⌘" }
        return label + ([UInt16(49): "Space", 63: "Fn", 96: "F5", 97: "F6", 100: "F8", 101: "F9"][p.shortcutKeyCode] ?? "Key \(p.shortcutKeyCode)")
    }

    private func configureShortcut(_ preferences: Preferences) {
        shortcut.start(keyCode: preferences.shortcutKeyCode, modifiers: preferences.shortcutModifiers,
                       hold: preferences.holdToTalk,
                       onStart: { [weak self] in self?.toggle() },
                       onStop: { [weak self] in
                           if self?.phase == .recording { self?.finish() }
                           else if self?.phase == .starting { self?.cancel() }
                       },
                       onCancel: { [weak self] in if self?.isBusy == true { self?.cancel() } })
    }

    func refreshPermissions() {
        let previous = accessibilityAllowed
        accessibilityAllowed = TextInsertion.isTrusted
        microphoneAllowed = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        if accessibilityAllowed && !previous { configureShortcut(store.preferences) }
    }

    func requestMicrophone() {
        Task { _ = await AVCaptureDevice.requestAccess(for: .audio); refreshPermissions() }
    }

    func requestAccessibility() { TextInsertion.requestPermission(); refreshPermissions() }

    func toggle(_ requestedMode: VoiceMode? = nil) {
        if phase == .recording { finish(); return }
        guard phase == .idle else { return }
        if let requestedMode { mode = requestedMode }
        start()
    }

    private func start() {
        error = ""; notice = ""; target = nil; liveText = ""; liveStatus = "Loading speech model…"
        sessionPreferences = store.preferences
        sessionAPIKey = textAPIKey
        do {
            let captured = try TextInsertion.capture()
            if captured.bundleID != Bundle.main.bundleIdentifier { target = captured }
        } catch TextInsertionError.secureField {
            error = "Password and secure fields cannot be used for voice typing."
            showMainWindow?()
            return
        } catch {
            // Unsupported controls still allow a copyable transcript.
            notice = "The current field is not accessible. Your result will be ready to copy."
        }
        if mode == .edit && (target?.selectedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true) {
            error = "Select the text you want to change in another app, then use the shortcut. Accessibility access is required."
            showMainWindow?()
            return
        }
        lastApp = target?.applicationName ?? "OmniTyper"
        do { _ = try payload(audio: nil) }
        catch { self.error = error.localizedDescription; showMainWindow?(); return }
        phase = .starting
        showVoicePanel?()
        let token = UUID(); generation = token
        task = Task { [self] in
            do {
                let response = try await worker.request(["op": "prepare", "asr_model": sessionPreferences.asrModel],
                                                        python: sessionPreferences.pythonExecutable)
                guard generation == token, !Task.isCancelled else { return }
                do {
                    let stream = try ASRStream(url: response["realtime_url"] as? String ?? "", onPartial: { [weak self] text in
                        guard let self, self.generation == token else { return }
                        self.liveText = text
                    }, onFailure: { [weak self] in
                        guard let self, self.generation == token else { return }
                        self.liveStatus = "Live preview unavailable · recording saved for transcription"
                    })
                    speechStream = stream
                    try await stream.connect(language: sessionPreferences.language)
                    liveStatus = "Listening · live words may change"
                } catch {
                    guard generation == token, !Task.isCancelled else { return }
                    speechStream?.cancel(); speechStream = nil
                    liveStatus = "Live preview unavailable · transcription runs when you finish"
                }
                try await recorder.start(deviceUID: sessionPreferences.microphoneUID, onPCM: speechStream?.audioInput)
                guard generation == token, !Task.isCancelled else { recorder.cancel(); return }
                discardRetryRecording()
                phase = .recording; showVoicePanel?()
                if sessionPreferences.sounds { NSSound(named: "Tink")?.play() }
            } catch {
                guard generation == token else { return }
                speechStream?.cancel(); speechStream = nil
                target = nil; phase = .idle; hideVoicePanel?(); self.error = error.localizedDescription; refreshPermissions(); showMainWindow?()
            }
        }
    }

    func finish() {
        guard phase == .recording else { return }
        do {
            let duration = recorder.elapsed
            let audio = try recorder.stop()
            if sessionPreferences.sounds { NSSound(named: "Pop")?.play() }
            run(audio: audio, duration: duration, allowInsertion: true)
        } catch {
            speechStream?.cancel(); speechStream = nil
            target = nil; phase = .idle; self.error = error.localizedDescription; hideVoicePanel?(); showMainWindow?()
        }
    }

    private func payload(audio: URL?, text: String? = nil) throws -> [String: Any] {
        let preferences = sessionPreferences
        let rule = store.rules.first { $0.bundleID == target?.bundleID }
        let instructions = try Preferences.combinedInstructions(preferences.instructions, rule?.instructions ?? "")
        guard store.dictionary.count <= 200, store.dictionary.allSatisfy(\.isValid) else {
            throw AppError.message("Fix or delete invalid Dictionary entries before recording. Each phrase must contain 1–120 characters without NUL, with at most 200 entries.")
        }
        let selectedText = (mode == .edit || mode == .ask) ? (target?.selectedText ?? "") : ""
        guard selectedText.unicodeScalars.count <= 12_000, !selectedText.contains("\0") else {
            throw AppError.message("Select at most 12,000 characters without NUL before using Voice edit or Ask.")
        }
        var request: [String: Any] = [
            "op": audio == nil ? "process" : "transcribe",
            "asr_model": preferences.asrModel,
            "mode": mode.rawValue, "language": preferences.language,
            "target_language": preferences.targetLanguage, "style": rule?.style ?? preferences.style,
            "instructions": instructions,
            "dictionary": store.dictionary.map { ["spoken": $0.spoken, "written": $0.written] },
            "selected_text": selectedText, "app_name": lastApp
        ]
        if mode != .dictate || (rule?.style ?? preferences.style) != "verbatim" {
            request.merge(try preferences.textSettings.payload(apiKey: sessionAPIKey)) { _, new in new }
        }
        if let audio { request["audio_path"] = audio.path }
        if let text { request["text"] = text }
        return request
    }

    private func run(audio: URL, duration: Double, allowInsertion: Bool) {
        phase = .processing; error = ""; notice = ""
        let token = UUID(); generation = token
        let request = Result { try payload(audio: audio) }
        let requestMode = mode
        let capturedTarget = target
        let preferences = sessionPreferences
        let recording = FailedRecording(url: audio, duration: duration, mode: requestMode,
                                        target: capturedTarget, appName: lastApp)
        task = Task {
            do {
                var payload = try request.get()
                var streamingWarning = ""
                if let stream = speechStream {
                    liveStatus = "Finalizing transcript…"
                    do {
                        let transcript = try await stream.finish()
                        payload["op"] = "process"
                        payload["audio_path"] = nil
                        payload["text"] = transcript
                    } catch {
                        guard generation == token, !Task.isCancelled else { throw CancellationError() }
                        streamingWarning = "Live transcription was interrupted; recovered from the complete recording."
                    }
                    stream.cancel(); speechStream = nil
                }
                try Task.checkCancellation()
                liveStatus = payload["op"] as? String == "process" ? "Processing text…" : "Transcribing recording…"
                let response = try await worker.request(payload, python: preferences.pythonExecutable)
                guard generation == token, !Task.isCancelled else { try? FileManager.default.removeItem(at: audio); return }
                let text = response["text"] as? String ?? ""
                let raw = response["raw_text"] as? String ?? text
                let warning = [streamingWarning, response["warning"] as? String ?? ""].filter { !$0.isEmpty }.joined(separator: " ")
                resultText = text; rawText = raw
                if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    notice = "No speech detected. Try speaking closer to the microphone."
                    try? FileManager.default.removeItem(at: audio)
                } else {
                    let entry = HistoryEntry(mode: requestMode, appName: lastApp, rawText: raw, text: text,
                                             duration: duration, warning: warning.isEmpty ? nil : warning)
                    if let retentionError = store.add(entry, recording: audio) {
                        retryRecording = recording
                        self.error = retentionError
                    }
                    notice = warning
                    if allowInsertion, preferences.autoPaste, requestMode != .ask, let capturedTarget {
                        do {
                            try await TextInsertion.insert(text, into: capturedTarget)
                            guard generation == token else { return }
                            if notice.isEmpty { notice = "Inserted into \(capturedTarget.applicationName)." }
                        } catch {
                            notice = "Ready to copy. \(error.localizedDescription)"
                            showMainWindow?()
                        }
                    } else {
                        if notice.isEmpty { notice = requestMode == .ask ? "Your answer is ready." : "Your text is ready to copy." }
                        showMainWindow?()
                    }
                }
                guard generation == token else { return }
                target = nil; phase = .idle; hideVoicePanel?()
                if !self.error.isEmpty { showMainWindow?() }
            } catch {
                guard generation == token else { try? FileManager.default.removeItem(at: audio); return }
                speechStream?.cancel(); speechStream = nil
                retryRecording = recording
                target = nil; phase = .idle; hideVoicePanel?()
                self.error = error.localizedDescription
                if let raw = (error as? WorkerFailure)?.rawText, !raw.isEmpty {
                    rawText = raw; resultText = raw
                    notice = "Text processing failed. The original transcript is ready to copy."
                }
                showMainWindow?()
            }
        }
    }

    var canRetry: Bool { retryRecording != nil && phase == .idle }
    func retryLast() {
        guard phase == .idle, let recording = retryRecording else { return }
        retryRecording = nil
        target = recording.target; mode = recording.mode; lastApp = recording.appName
        sessionPreferences = store.preferences
        sessionAPIKey = textAPIKey
        run(audio: recording.url, duration: recording.duration, allowInsertion: false)
    }

    func retry(_ entry: HistoryEntry) {
        guard phase == .idle, let audio = store.audioURL(for: entry) else { return }
        if entry.mode == .edit || entry.mode == .ask {
            error = "Record a new request for this mode. Selected text is not retained in history."
            return
        }
        do {
            let copy = FileManager.default.temporaryDirectory.appendingPathComponent("OmniTyper-\(UUID()).wav")
            try FileManager.default.copyItem(at: audio, to: copy)
            discardRetryRecording()
            target = nil; mode = entry.mode; lastApp = entry.appName; sessionPreferences = store.preferences
            sessionAPIKey = textAPIKey
            run(audio: copy, duration: entry.duration, allowInsertion: false)
        } catch { self.error = error.localizedDescription }
    }

    func prepareModels() {
        guard phase == .idle else { return }
        phase = .preparing; error = ""; notice = ""
        let preferences = store.preferences
        let token = UUID(); generation = token
        task = Task {
            do {
                _ = try await worker.request(["op": "prepare", "asr_model": preferences.asrModel], python: preferences.pythonExecutable)
                guard generation == token else { return }
                notice = "Speech model ready. Text processing uses your configured API."
            } catch {
                guard generation == token else { return }
                self.error = error.localizedDescription
            }
            phase = .idle
        }
    }

    func loadTextModels() {
        guard phase == .idle else { return }
        error = ""; notice = ""
        do {
            var request = try store.preferences.textSettings.payload(apiKey: textAPIKey, requireModel: false)
            request["op"] = "models"
            let python = store.preferences.pythonExecutable
            phase = .preparing
            let token = UUID(); generation = token
            task = Task {
                do {
                    let response = try await worker.request(request, python: python)
                    guard generation == token else { return }
                    textModels = response["models"] as? [String] ?? []
                    notice = textModels.isEmpty ? "Connected, but no models were listed. Add a model in your server, or enter its name manually." : "Connected. Choose a model or enter a custom model name."
                } catch {
                    guard generation == token else { return }
                    self.error = error.localizedDescription
                }
                phase = .idle
            }
        } catch { self.error = error.localizedDescription }
    }

    func releaseModels() { if phase == .idle { worker.stop(); notice = "Speech model unloaded. Your text API server is managed separately." } }

    func cancel() {
        generation = UUID(); task?.cancel(); task = nil
        speechStream?.cancel(); speechStream = nil; liveText = ""; liveStatus = ""
        recorder.cancel(); worker.stop(); target = nil; phase = .idle; hideVoicePanel?()
        notice = "Cancelled."
    }

    func shutdown() {
        cancel(); shortcut.stop(); timer?.invalidate()
        textAPIKey = ""; sessionAPIKey = ""
        discardRetryRecording()
    }

    private func discardRetryRecording() {
        if let retryRecording { try? FileManager.default.removeItem(at: retryRecording.url) }
        retryRecording = nil
    }

    func copyResult() { TextInsertion.copy(resultText); notice = "Copied to clipboard." }
}
