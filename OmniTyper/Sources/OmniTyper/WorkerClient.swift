import Combine
import Foundation
import Darwin

enum WorkerError: LocalizedError {
    case unavailable(String)

    var errorDescription: String? {
        switch self {
        case .unavailable(let message): return message
        }
    }
}

struct WorkerFailure: LocalizedError {
    let message: String
    let rawText: String?
    var errorDescription: String? { message }
}

@MainActor
final class WorkerClient: ObservableObject {
    @Published private(set) var status = "Models are not loaded"
    @Published private(set) var isRunning = false

    private struct Pending {
        let id: String
        let continuation: CheckedContinuation<[String: Any], Error>
    }

    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var errors: FileHandle?
    private var pythonPath: String?
    private var generation = UUID()
    private var pending: Pending?
    private var stdoutBuffer = Data()
    private var stdoutEnded = false
    private var exitStatus: Int32?
    private var timeoutTask: Task<Void, Never>?
    private var exitTask: Task<Void, Never>?
    private var diagnostics = ""
    private let maximumLineBytes = 1_048_576

    func request(_ payload: [String: Any], python: String) async throws -> [String: Any] {
        let requestID = UUID().uuidString
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                do {
                    guard pending == nil else { throw WorkerError.unavailable("The local model is already busy.") }
                    var message = payload
                    message["id"] = requestID
                    guard JSONSerialization.isValidJSONObject(message) else {
                        throw WorkerError.unavailable("The model request contains unsupported data.")
                    }
                    var data = try JSONSerialization.data(withJSONObject: message)
                    guard data.count < 256 * 1_024 else {
                        throw WorkerError.unavailable("The model request exceeds 256 KiB. Shorten the selected text or dictionary.")
                    }
                    data.append(0x0A)
                    try ensureProcess(python: python)
                    guard let input else { throw WorkerError.unavailable("The local worker has no input connection.") }
                    pending = Pending(id: requestID, continuation: continuation)
                    status = payload["op"] as? String == "prepare" ? "Preparing speech model…" : "Processing…"
                    let workerGeneration = generation
                    let seconds: UInt64 = payload["op"] as? String == "prepare" ? 1_800 : 600
                    timeoutTask = Task { [weak self] in
                        do { try await Task.sleep(nanoseconds: seconds * 1_000_000_000) }
                        catch { return }
                        guard let self, self.pending?.id == requestID else { return }
                        self.failAndStop(WorkerError.unavailable("The local model timed out. Try again or choose a smaller model."))
                    }
                    // A full pipe must never block the UI, including a stuck worker.
                    DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                        do { try input.write(contentsOf: data) }
                        catch {
                            DispatchQueue.main.async {
                                guard let self, self.generation == workerGeneration, self.pending?.id == requestID else { return }
                                self.failAndStop(WorkerError.unavailable("The local worker stopped accepting requests."))
                            }
                        }
                    }
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        } onCancel: {
            Task { @MainActor [weak self] in
                guard let self, self.pending?.id == requestID else { return }
                self.finish(.failure(CancellationError()))
                self.shutdown()
                self.status = "Cancelled"
            }
        }
    }

    func stop() {
        finish(.failure(CancellationError()))
        shutdown()
        status = "Models are not loaded"
    }

    private func ensureProcess(python: String) throws {
        let executable = try resolvePython(python)
        if let process, process.isRunning, pythonPath == executable.path { return }
        shutdown()
        let bundled = Bundle.main.resourceURL?.appendingPathComponent("backend/worker.py")
        let overridden = ProcessInfo.processInfo.environment["OMNITYPER_WORKER"].map {
            URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)
        }
        guard let worker = [bundled, overridden].compactMap({ $0 }).first(where: {
            FileManager.default.isReadableFile(atPath: $0.path)
        }) else {
            throw WorkerError.unavailable("The local worker is missing. Rebuild the app, or set OMNITYPER_WORKER to backend/worker.py.")
        }
        let child = Process()
        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        child.executableURL = executable
        child.arguments = ["-u", worker.path]
        child.currentDirectoryURL = worker.deletingLastPathComponent()
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        child.environment = environment
        child.standardInput = stdin
        child.standardOutput = stdout
        child.standardError = stderr
        let workerGeneration = UUID()
        generation = workerGeneration
        stdoutBuffer.removeAll(keepingCapacity: true)
        stdoutEnded = false
        exitStatus = nil
        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { handle.readabilityHandler = nil }
            DispatchQueue.main.async {
                guard let self, self.generation == workerGeneration else { return }
                self.receive(data)
            }
        }
        stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { handle.readabilityHandler = nil; return }
            DispatchQueue.main.async {
                guard let self, self.generation == workerGeneration else { return }
                // Third-party model logs can echo prompts or audio paths. Keep
                // diagnostics bounded without ever storing their raw contents.
                self.log("Worker stderr: \(data.count) bytes (content omitted for privacy)")
            }
        }
        child.terminationHandler = { [weak self] child in
            let code = child.terminationStatus
            DispatchQueue.main.async {
                guard let self, self.generation == workerGeneration else { return }
                self.exitStatus = code
                self.isRunning = false
                self.log("Worker exited with status \(code)")
                if self.stdoutEnded { self.handleExit() }
                else {
                    // Allow the pipe's final response/EOF to reach the main
                    // queue before treating a normal exit as a lost response.
                    self.exitTask = Task { [weak self] in
                        try? await Task.sleep(nanoseconds: 1_000_000_000)
                        guard !Task.isCancelled, let self, self.generation == workerGeneration else { return }
                        self.handleExit()
                    }
                }
            }
        }
        do { try child.run() }
        catch {
            stdout.fileHandleForReading.readabilityHandler = nil
            stderr.fileHandleForReading.readabilityHandler = nil
            throw WorkerError.unavailable("Could not start Python at \(executable.path). Run the dependency setup script first.")
        }
        process = child
        input = stdin.fileHandleForWriting
        output = stdout.fileHandleForReading
        errors = stderr.fileHandleForReading
        pythonPath = executable.path
        isRunning = true
        log("Worker started")
    }

    private func receive(_ data: Data) {
        if data.isEmpty {
            output?.readabilityHandler = nil
            stdoutEnded = true
            if exitStatus != nil { handleExit() }
            else if pending != nil {
                failAndStop(WorkerError.unavailable("The local worker closed its response connection."))
            }
            return
        }
        stdoutBuffer.append(data)
        while let newline = stdoutBuffer.firstIndex(of: 0x0A) {
            let line = stdoutBuffer.prefix(upTo: newline)
            guard line.count <= maximumLineBytes else {
                failAndStop(WorkerError.unavailable("The local worker sent an oversized response."))
                return
            }
            let complete = Data(line)
            stdoutBuffer.removeSubrange(...newline)
            if complete.isEmpty { continue }
            guard let message = (try? JSONSerialization.jsonObject(with: complete)) as? [String: Any],
                  let responseID = message["id"] as? String else {
                failAndStop(WorkerError.unavailable("The local worker sent an invalid response. Check the installed dependencies."))
                return
            }
            guard responseID == pending?.id else { continue }
            if message["event"] as? String == "progress" {
                if let progress = message["message"] as? String { status = String(progress.prefix(300)) }
            } else if let ok = message["ok"] as? Bool {
                if ok {
                    status = "Ready"
                    finish(.success(message))
                } else {
                    let description = message["error"] as? String ?? "Local model processing failed."
                    status = "Processing failed"
                    finish(.failure(WorkerFailure(message: String(description.prefix(2_000)),
                                                  rawText: message["raw_text"] as? String)))
                }
            } else {
                failAndStop(WorkerError.unavailable("The local worker sent an incomplete response."))
                return
            }
        }
        if stdoutBuffer.count > maximumLineBytes {
            failAndStop(WorkerError.unavailable("The local worker sent an oversized response."))
        }
    }

    private func finish(_ result: Result<[String: Any], Error>) {
        guard let pending else { return }
        self.pending = nil
        timeoutTask?.cancel()
        timeoutTask = nil
        pending.continuation.resume(with: result)
    }

    private func handleExit() {
        let code = exitStatus ?? -1
        if pending != nil {
            finish(.failure(WorkerError.unavailable("The local worker exited (status \(code)). Check Python dependencies and available memory, then retry.")))
            status = "Worker stopped"
        } else if status == "Ready" { status = "Models are not loaded" }
        shutdown()
    }

    private func failAndStop(_ error: Error) {
        finish(.failure(error))
        shutdown()
        status = "Worker stopped"
    }

    private func shutdown() {
        generation = UUID()
        timeoutTask?.cancel()
        timeoutTask = nil
        exitTask?.cancel()
        exitTask = nil
        output?.readabilityHandler = nil
        errors?.readabilityHandler = nil
        try? input?.close()
        input = nil
        output = nil
        errors = nil
        stdoutBuffer.removeAll(keepingCapacity: true)
        stdoutEnded = false
        exitStatus = nil
        isRunning = false
        pythonPath = nil
        guard let child = process else { return }
        process = nil
        child.terminationHandler = nil
        if child.isRunning {
            child.terminate()
            // The Python worker gets time to stop its owned native model server.
            // Escalate only this still-running child, never other Python processes.
            Task {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                if child.isRunning { Darwin.kill(child.processIdentifier, SIGKILL) }
            }
        }
        log("Worker stopped")
    }

    private func resolvePython(_ supplied: String) throws -> URL {
        let expanded = (supplied.trimmingCharacters(in: .whitespacesAndNewlines) as NSString).expandingTildeInPath
        guard !expanded.isEmpty else { throw WorkerError.unavailable("Choose the Python executable created by the setup script.") }
        if expanded.contains("/") {
            let url = URL(fileURLWithPath: expanded)
            guard FileManager.default.isExecutableFile(atPath: url.path) else {
                throw WorkerError.unavailable("Python is not executable at \(url.path). Run the dependency setup script first.")
            }
            return url
        }
        let path = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"
        for directory in path.split(separator: ":") {
            let url = URL(fileURLWithPath: String(directory)).appendingPathComponent(expanded)
            if FileManager.default.isExecutableFile(atPath: url.path) { return url }
        }
        throw WorkerError.unavailable("Could not find \(expanded). Choose the Python executable created by the setup script.")
    }

    private func log(_ message: String) {
        diagnostics += "\(Date().ISO8601Format()) \(message)\n"
        if diagnostics.utf8.count > 8_192 { diagnostics = String(diagnostics.suffix(4_096)) }
        guard let library = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first else { return }
        let directory = library.appendingPathComponent("Logs/OmniTyper", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data(diagnostics.utf8).write(to: directory.appendingPathComponent("worker.log"), options: .atomic)
        } catch { /* Diagnostics must not interrupt dictation. */ }
    }
}
