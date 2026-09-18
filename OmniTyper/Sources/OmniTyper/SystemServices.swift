import AppKit
import AVFoundation
import ApplicationServices
import AudioToolbox
import Combine
import CoreAudio

struct MicrophoneDevice: Identifiable, Hashable {
    let id: String
    let name: String
}

enum SystemServiceError: LocalizedError {
    case unavailable(String)

    var errorDescription: String? {
        switch self {
        case .unavailable(let message): return message
        }
    }
}

enum TextInsertionError: LocalizedError {
    case secureField

    var errorDescription: String? {
        "Password and secure fields cannot be used for voice typing."
    }
}

// AVAudioEngine calls this on its audio thread. All file/converter access stays
// behind the same lock, including closing the file after removing the tap.
final class AudioCaptureSink: @unchecked Sendable {
    private let lock = NSLock()
    private let converter: AVAudioConverter
    private let format: AVAudioFormat
    private let onPCM: (@Sendable (Data) -> Void)?
    private var file: AVAudioFile?
    private var failure: Error?
    private var meter = 0.0
    private var framesWritten: AVAudioFrameCount = 0
    private let maximumFrames: AVAudioFrameCount = 300 * 16_000

    init(input: AVAudioFormat, url: URL, onPCM: (@Sendable (Data) -> Void)? = nil) throws {
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: 16_000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: input, to: format) else {
            throw SystemServiceError.unavailable("The microphone's audio format is not supported.")
        }
        self.format = format
        self.converter = converter
        self.onPCM = onPCM
        file = try AVAudioFile(forWriting: url, settings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ], commonFormat: .pcmFormatFloat32, interleaved: false)
    }

    func consume(_ input: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard let file, failure == nil, framesWritten < maximumFrames else { return }
        let capacity = AVAudioFrameCount(ceil(Double(input.frameLength) * format.sampleRate / input.format.sampleRate)) + 32
        guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return }
        var supplied = false
        var conversionError: NSError?
        converter.convert(to: output, error: &conversionError) { _, status in
            if supplied {
                status.pointee = .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return input
        }
        if let conversionError {
            failure = conversionError
            return
        }
        output.frameLength = min(output.frameLength, maximumFrames - framesWritten)
        guard output.frameLength > 0 else { return }
        if let samples = output.floatChannelData?[0] {
            var squares: Double = 0
            for index in 0..<Int(output.frameLength) {
                squares += Double(samples[index]) * Double(samples[index])
            }
            meter = min(1, sqrt(squares / Double(output.frameLength)) * 5)
        }
        do {
            try file.write(from: output)
            framesWritten += output.frameLength
            if let onPCM, let samples = output.floatChannelData?[0] {
                let pcm = (0..<Int(output.frameLength)).map { index -> Int16 in
                    let sample = samples[index].isFinite ? samples[index] : 0
                    return Int16(min(32767, max(-32768, sample * 32768))).littleEndian
                }
                pcm.withUnsafeBytes { onPCM(Data($0)) }
            }
        }
        catch { failure = error }
    }

    func level() -> Double {
        lock.lock()
        defer { lock.unlock() }
        return meter
    }

    func fail(_ error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if failure == nil { failure = error }
    }

    func close() throws {
        lock.lock()
        defer { lock.unlock() }
        file = nil
        if let failure { throw failure }
    }
}

@MainActor
final class AudioRecorder: ObservableObject {
    @Published private(set) var level = 0.0
    @Published private(set) var elapsed = 0.0

    private var engine: AVAudioEngine?
    private var sink: AudioCaptureSink?
    private var outputURL: URL?
    private var meterTask: Task<Void, Never>?
    private var configurationObserver: NSObjectProtocol?
    private var generation = UUID()
    private var isStarting = false

    deinit {
        meterTask?.cancel()
        if let configurationObserver { NotificationCenter.default.removeObserver(configurationObserver) }
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        try? sink?.close()
        if let outputURL { try? FileManager.default.removeItem(at: outputURL) }
    }

    static func devices() -> [MicrophoneDevice] {
        inputDevices().compactMap { device in
            guard let uid = stringProperty(device, selector: kAudioDevicePropertyDeviceUID),
                  let name = stringProperty(device, selector: kAudioObjectPropertyName) else { return nil }
            return MicrophoneDevice(id: uid, name: name)
        }.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    func start(deviceUID: String, onPCM: (@Sendable (Data) -> Void)? = nil) async throws {
        guard engine == nil, !isStarting else {
            throw SystemServiceError.unavailable("Recording is already in progress.")
        }
        isStarting = true
        let currentGeneration = UUID()
        generation = currentGeneration
        defer { isStarting = false }
        let authorized: Bool
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: authorized = true
        case .notDetermined: authorized = await AVCaptureDevice.requestAccess(for: .audio)
        default: authorized = false
        }
        try Task.checkCancellation()
        guard generation == currentGeneration else { throw CancellationError() }
        guard authorized else {
            throw SystemServiceError.unavailable("Enable Microphone access for OmniTyper in System Settings → Privacy & Security.")
        }

        let engine = AVAudioEngine()
        let input = engine.inputNode
        if !deviceUID.isEmpty {
            guard var device = Self.inputDevices().first(where: {
                Self.stringProperty($0, selector: kAudioDevicePropertyDeviceUID) == deviceUID
            }), let unit = input.audioUnit else {
                throw SystemServiceError.unavailable("The selected microphone is no longer connected. Choose another microphone.")
            }
            let result = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice,
                                             kAudioUnitScope_Global, 0, &device,
                                             UInt32(MemoryLayout<AudioDeviceID>.size))
            guard result == noErr else {
                throw SystemServiceError.unavailable("Could not select this microphone (audio error \(result)).")
            }
        }
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw SystemServiceError.unavailable("The microphone has no available audio input.")
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("OmniTyper-\(UUID().uuidString).wav")
        let sink = try AudioCaptureSink(input: format, url: url, onPCM: onPCM)
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
            sink.consume(buffer)
        }
        do {
            engine.prepare()
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            engine.stop()
            try? sink.close()
            try? FileManager.default.removeItem(at: url)
            throw error
        }
        self.engine = engine
        self.sink = sink
        outputURL = url
        level = 0
        elapsed = 0
        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { _ in
            sink.fail(SystemServiceError.unavailable("The microphone configuration changed. Please record again."))
        }
        let started = ProcessInfo.processInfo.systemUptime
        meterTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 50_000_000)
                guard !Task.isCancelled, let self else { break }
                self.level = sink.level()
                self.elapsed = ProcessInfo.processInfo.systemUptime - started
            }
        }
    }

    func stop() throws -> URL {
        guard let url = outputURL else {
            throw SystemServiceError.unavailable("No recording is in progress.")
        }
        let sink = self.sink
        releaseAudio()
        do { try sink?.close() }
        catch {
            try? FileManager.default.removeItem(at: url)
            throw error
        }
        return url
    }

    func cancel() {
        generation = UUID()
        let url = outputURL
        let sink = self.sink
        releaseAudio()
        try? sink?.close()
        if let url { try? FileManager.default.removeItem(at: url) }
        elapsed = 0
    }

    private func releaseAudio() {
        meterTask?.cancel()
        meterTask = nil
        if let configurationObserver { NotificationCenter.default.removeObserver(configurationObserver) }
        configurationObserver = nil
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        sink = nil
        outputURL = nil
        level = 0
    }

    private static func inputDevices() -> [AudioDeviceID] {
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices,
                                                mScope: kAudioObjectPropertyScopeGlobal,
                                                mElement: kAudioObjectPropertyElementMain)
        var bytes: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &bytes) == noErr else { return [] }
        var devices = [AudioDeviceID](repeating: 0, count: Int(bytes) / MemoryLayout<AudioDeviceID>.size)
        guard !devices.isEmpty,
              AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &bytes, &devices) == noErr else { return [] }
        return devices.filter { device in
            var inputAddress = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreams,
                                                         mScope: kAudioDevicePropertyScopeInput,
                                                         mElement: kAudioObjectPropertyElementMain)
            var size: UInt32 = 0
            return AudioObjectGetPropertyDataSize(device, &inputAddress, 0, nil, &size) == noErr && size > 0
        }
    }

    private static func stringProperty(_ device: AudioDeviceID, selector: AudioObjectPropertySelector) -> String? {
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                                mElement: kAudioObjectPropertyElementMain)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &value) == noErr else { return nil }
        return value?.takeRetainedValue() as String?
    }
}

@MainActor
final class GlobalShortcut {
    private var tap: CFMachPort?
    private var source: CFRunLoopSource?
    private var monitorTask: Task<Void, Never>?
    private var keyCode: UInt16 = 49
    private var modifiers: UInt64 = 0
    private var hold = false
    private var pressed = false
    private var capturedKey = false
    private var onStart: (() -> Void)?
    private var onStop: (() -> Void)?
    private var onCancel: (() -> Void)?
    private let modifierMask: UInt64 = CGEventFlags.maskCommand.rawValue | CGEventFlags.maskAlternate.rawValue
        | CGEventFlags.maskControl.rawValue | CGEventFlags.maskShift.rawValue | CGEventFlags.maskSecondaryFn.rawValue

    private var triggerModifier: UInt64 {
        switch keyCode {
        case 63: return CGEventFlags.maskSecondaryFn.rawValue
        case 54, 55: return CGEventFlags.maskCommand.rawValue
        case 56, 60: return CGEventFlags.maskShift.rawValue
        case 58, 61: return CGEventFlags.maskAlternate.rawValue
        case 59, 62: return CGEventFlags.maskControl.rawValue
        default: return 0
        }
    }

    private func matchingFlags(_ flags: CGEventFlags) -> UInt64 {
        // Function keys carry secondaryFn even on keyboards with no Fn held.
        let fn = CGEventFlags.maskSecondaryFn.rawValue
        let mask = modifiers & fn != 0 || keyCode == 63 ? modifierMask : modifierMask & ~fn
        return flags.rawValue & mask
    }

    deinit {
        monitorTask?.cancel()
        if let source { CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes) }
        if let tap { CFMachPortInvalidate(tap) }
    }

    func start(keyCode: UInt16, modifiers: UInt64, hold: Bool,
               onStart: @escaping () -> Void, onStop: @escaping () -> Void,
               onCancel: @escaping () -> Void) {
        stop()
        self.keyCode = keyCode
        self.modifiers = modifiers & modifierMask
        self.hold = hold
        self.onStart = onStart
        self.onStop = onStop
        self.onCancel = onCancel
        installTap()
        monitorTask = Task { [weak self] in
            var ticks = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 100_000_000)
                guard !Task.isCancelled, let self else { break }
                ticks += 1
                if self.tap == nil && ticks % 10 == 0 { self.installTap() }
                if self.pressed {
                    let flags = self.matchingFlags(CGEventSource.flagsState(.combinedSessionState))
                    let down = self.triggerModifier == 0
                        ? CGEventSource.keyState(.combinedSessionState, key: self.keyCode)
                        : flags & self.triggerModifier != 0
                    if !down || flags != self.modifiers | self.triggerModifier { self.release() }
                }
            }
        }
    }

    func stop() {
        monitorTask?.cancel()
        monitorTask = nil
        release()
        if let tap { CGEvent.tapEnable(tap: tap, enable: false) }
        if let source { CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes) }
        if let tap { CFMachPortInvalidate(tap) }
        source = nil
        tap = nil
        onStart = nil
        onStop = nil
        onCancel = nil
        capturedKey = false
    }

    private func installTap() {
        guard tap == nil else { return }
        let mask = (CGEventMask(1) << CGEventType.keyDown.rawValue)
            | (CGEventMask(1) << CGEventType.keyUp.rawValue)
            | (CGEventMask(1) << CGEventType.flagsChanged.rawValue)
        guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap,
                                         options: .defaultTap, eventsOfInterest: mask,
                                         callback: { _, type, event, context in
            guard let context else { return Unmanaged.passUnretained(event) }
            let owner = Unmanaged<GlobalShortcut>.fromOpaque(context).takeUnretainedValue()
            let consumed = MainActor.assumeIsolated { owner.receive(type, event: event) }
            return consumed ? nil : Unmanaged.passUnretained(event)
        }, userInfo: Unmanaged.passUnretained(self).toOpaque()) else { return }
        self.tap = tap
        source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    private func receive(_ type: CGEventType, event: CGEvent) -> Bool {
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            release()
            if let tap { CGEvent.tapEnable(tap: tap, enable: true) }
            return false
        }
        let code = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
        if type == .keyDown && code == 53 {
            // Escape cancels rather than completing a held recording.
            pressed = false
            deliver(onCancel)
            return false
        }
        if type == .keyUp && code == keyCode {
            let consumed = capturedKey
            capturedKey = false
            release()
            return consumed
        }
        let flags = matchingFlags(event.flags)
        if type == .flagsChanged && code == keyCode && triggerModifier != 0 {
            if flags == modifiers | triggerModifier, flags & triggerModifier != 0 {
                if !pressed { pressed = true; capturedKey = true; deliver(onStart) }
                return true
            }
            let consumed = capturedKey
            capturedKey = false
            release()
            return consumed
        }
        if type == .flagsChanged && pressed && flags != modifiers | triggerModifier {
            release()
        } else if type == .keyDown && code == keyCode && flags == modifiers {
            if event.getIntegerValueField(.keyboardEventAutorepeat) == 0 && !pressed {
                pressed = true
                capturedKey = true
                deliver(onStart)
            }
            return true
        }
        return false
    }

    private func release() {
        guard pressed else { return }
        pressed = false
        if hold { deliver(onStop) }
    }

    private func deliver(_ callback: (() -> Void)?) {
        guard let callback else { return }
        // Permission/UI/AX work must run after returning from the event tap.
        DispatchQueue.main.async { callback() }
    }
}

struct InsertionTarget {
    let applicationName: String
    let bundleID: String
    let selectedText: String
    fileprivate let application: NSRunningApplication
    fileprivate let element: AXUIElement
    fileprivate let range: CFRange
    fileprivate let value: String?
    fileprivate let window: AXUIElement?
    fileprivate let windowTitle: String?
    fileprivate let document: String?
}

@MainActor
enum TextInsertion {
    static var isTrusted: Bool { AXIsProcessTrusted() }

    static func requestPermission() {
        AXIsProcessTrustedWithOptions([kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary)
    }

    static func capture() throws -> InsertionTarget {
        guard isTrusted else {
            throw SystemServiceError.unavailable("Enable Accessibility access to insert text into other applications.")
        }
        guard let app = NSWorkspace.shared.frontmostApplication, !app.isTerminated,
              app.processIdentifier != ProcessInfo.processInfo.processIdentifier else {
            throw SystemServiceError.unavailable("Focus a text field in another application before recording.")
        }
        let element = try focusedElement()
        var pid: pid_t = 0
        guard AXUIElementGetPid(element, &pid) == .success, pid == app.processIdentifier else {
            throw SystemServiceError.unavailable("The focused application changed. Please try again.")
        }
        try rejectSecure(element)
        let role = attribute(element, kAXRoleAttribute) as? String ?? ""
        guard [kAXTextFieldRole, kAXTextAreaRole, kAXComboBoxRole].contains(role)
                || (attribute(element, "AXEditable") as? NSNumber)?.boolValue == true else {
            throw SystemServiceError.unavailable("Focus an editable text field. Your result can still be copied.")
        }
        let range = try selectionRange(element)
        let selectedText = try selectedText(element, range: range)
        let window = elementAttribute(element, kAXWindowAttribute)
        return InsertionTarget(applicationName: app.localizedName ?? "Application", bundleID: app.bundleIdentifier ?? "",
                               selectedText: selectedText, application: app, element: element, range: range,
                               value: attribute(element, kAXValueAttribute) as? String, window: window,
                               windowTitle: window.flatMap { attribute($0, kAXTitleAttribute) as? String },
                               document: window.flatMap { attribute($0, kAXDocumentAttribute) as? String })
    }

    static func insert(_ text: String, into target: InsertionTarget) async throws {
        guard !text.isEmpty else { return }
        try Task.checkCancellation()
        try validate(target)
        var settable = DarwinBoolean(false)
        if AXUIElementIsAttributeSettable(target.element, kAXSelectedTextAttribute as CFString, &settable) == .success,
           settable.boolValue {
            let result = AXUIElementSetAttributeValue(target.element, kAXSelectedTextAttribute as CFString, text as CFString)
            guard result == .success else {
                throw SystemServiceError.unavailable("This application could not replace the selected text. Copy the result instead.")
            }
            return
        }

        let clipboard = NSPasteboard.general
        let originalCount = clipboard.changeCount
        let snapshot = try (clipboard.pasteboardItems ?? []).map { item -> NSPasteboardItem in
            let copy = NSPasteboardItem()
            for type in item.types {
                guard let data = item.data(forType: type) else {
                    throw SystemServiceError.unavailable("The current clipboard cannot be preserved. Copy the result to paste manually.")
                }
                copy.setData(data, forType: type)
            }
            return copy
        }
        guard clipboard.changeCount == originalCount else {
            throw SystemServiceError.unavailable("The clipboard changed. Copy the result to paste it manually.")
        }
        guard let source = CGEventSource(stateID: .privateState),
              let down = CGEvent(keyboardEventSource: source, virtualKey: 9, keyDown: true),
              let up = CGEvent(keyboardEventSource: source, virtualKey: 9, keyDown: false) else {
            throw SystemServiceError.unavailable("Could not create the paste keyboard event.")
        }
        try validate(target)
        try Task.checkCancellation()
        clipboard.clearContents()
        let clearedCount = clipboard.changeCount
        guard clipboard.setString(text, forType: .string) else {
            if clipboard.changeCount == clearedCount {
                clipboard.clearContents()
                if !snapshot.isEmpty { clipboard.writeObjects(snapshot) }
            }
            throw SystemServiceError.unavailable("Could not place the result on the clipboard.")
        }
        let ownedCount = clipboard.changeCount
        defer {
            // Do not overwrite a clipboard copy made while the destination pastes.
            if clipboard.changeCount == ownedCount {
                clipboard.clearContents()
                if !snapshot.isEmpty { clipboard.writeObjects(snapshot) }
            }
        }
        try validate(target)
        down.flags = .maskCommand
        up.flags = .maskCommand
        down.postToPid(target.application.processIdentifier)
        up.postToPid(target.application.processIdentifier)
        // Once sent, paste cannot be revoked. Do not let cancellation restore the
        // old clipboard while the queued paste is still being consumed.
        await Task.detached { try? await Task.sleep(nanoseconds: 300_000_000) }.value
    }

    static func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private static func validate(_ target: InsertionTarget) throws {
        guard isTrusted, !target.application.isTerminated,
              NSWorkspace.shared.frontmostApplication?.processIdentifier == target.application.processIdentifier else {
            throw SystemServiceError.unavailable("The destination application changed. Your result is ready to copy.")
        }
        let focused = try focusedElement()
        guard CFEqual(focused, target.element) else {
            throw SystemServiceError.unavailable("The focused field changed. Your result is ready to copy.")
        }
        try rejectSecure(focused)
        let window = elementAttribute(focused, kAXWindowAttribute)
        guard attribute(focused, kAXValueAttribute) as? String == target.value,
              (window == nil && target.window == nil) || (window != nil && target.window != nil && CFEqual(window!, target.window!)),
              window.flatMap({ attribute($0, kAXTitleAttribute) as? String }) == target.windowTitle,
              window.flatMap({ attribute($0, kAXDocumentAttribute) as? String }) == target.document else {
            throw SystemServiceError.unavailable("The destination content changed. Your result is ready to copy.")
        }
        let range = try selectionRange(focused)
        guard range.location == target.range.location, range.length == target.range.length,
              try selectedText(focused, range: range) == target.selectedText else {
            throw SystemServiceError.unavailable("The cursor or selection changed. Your result is ready to copy.")
        }
    }

    private static func focusedElement() throws -> AXUIElement {
        let system = AXUIElementCreateSystemWide()
        AXUIElementSetMessagingTimeout(system, 1)
        guard let value = attribute(system, kAXFocusedUIElementAttribute),
              CFGetTypeID(value) == AXUIElementGetTypeID() else {
            throw SystemServiceError.unavailable("The focused field is not accessible. Your result can still be copied.")
        }
        let element = value as! AXUIElement
        AXUIElementSetMessagingTimeout(element, 1)
        return element
    }

    private static func rejectSecure(_ element: AXUIElement) throws {
        var current: AXUIElement? = element
        for _ in 0..<8 {
            guard let node = current else { break }
            let role = attribute(node, kAXRoleAttribute) as? String
            let subrole = attribute(node, kAXSubroleAttribute) as? String
            if role == "AXSecureTextField" || subrole == kAXSecureTextFieldSubrole
                || (attribute(node, "AXProtectedContent") as? NSNumber)?.boolValue == true {
                throw TextInsertionError.secureField
            }
            guard let parent = attribute(node, kAXParentAttribute), CFGetTypeID(parent) == AXUIElementGetTypeID() else { break }
            current = (parent as! AXUIElement)
        }
    }

    private static func selectionRange(_ element: AXUIElement) throws -> CFRange {
        guard let value = attribute(element, kAXSelectedTextRangeAttribute),
              CFGetTypeID(value) == AXValueGetTypeID() else {
            throw SystemServiceError.unavailable("This field does not expose its cursor position. Your result can still be copied.")
        }
        let axValue = value as! AXValue
        var range = CFRange()
        guard AXValueGetType(axValue) == .cfRange, AXValueGetValue(axValue, .cfRange, &range),
              range.location >= 0, range.length >= 0 else {
            throw SystemServiceError.unavailable("This field has an unsupported selection. Your result can still be copied.")
        }
        return range
    }

    private static func selectedText(_ element: AXUIElement, range: CFRange) throws -> String {
        if range.length == 0 { return "" }
        if let text = attribute(element, kAXSelectedTextAttribute) as? String { return text }
        if let value = attribute(element, kAXValueAttribute) as? String,
           range.location <= (value as NSString).length,
           range.length <= (value as NSString).length - range.location {
            return (value as NSString).substring(with: NSRange(location: range.location, length: range.length))
        }
        throw SystemServiceError.unavailable("The selected text cannot be read. Your result can still be copied.")
    }

    private static func attribute(_ element: AXUIElement, _ key: String) -> CFTypeRef? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, key as CFString, &value) == .success else { return nil }
        return value
    }

    private static func elementAttribute(_ element: AXUIElement, _ key: String) -> AXUIElement? {
        guard let value = attribute(element, key), CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
        return (value as! AXUIElement)
    }
}
