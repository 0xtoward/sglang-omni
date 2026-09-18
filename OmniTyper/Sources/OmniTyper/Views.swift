// SPDX-License-Identifier: Apache-2.0
import SwiftUI
import AppKit
import ServiceManagement
import UniformTypeIdentifiers

// Use the property wrapper explicitly; some Command Line Tools SDKs expose an unavailable State macro.
private typealias ViewState<Value> = SwiftUI.State<Value>

private let brandGreen = Color(red: 0.13, green: 0.46, blue: 0.36)
private let accent = Color(nsColor: NSColor(name: nil) { appearance in
    appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        ? NSColor(red: 0.40, green: 0.78, blue: 0.64, alpha: 1)
        : NSColor(red: 0.13, green: 0.46, blue: 0.36, alpha: 1)
})
private let cardBackground = Color(nsColor: .controlBackgroundColor)
private let styles = ["clean", "verbatim", "casual", "formal", "concise"]
private let languages = ["English", "Chinese", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Italian", "Russian", "Arabic", "Hindi", "Cantonese"]

enum Page: String, CaseIterable {
    case home = "Home", history = "History", dictionary = "Dictionary", rules = "Writing style", settings = "Settings"
    var icon: String {
        switch self { case .home: return "square.grid.2x2"; case .history: return "clock.arrow.circlepath"
        case .dictionary: return "book.closed"; case .rules: return "slider.horizontal.3"; case .settings: return "gearshape" }
    }
}

struct RootView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: AppStore
    @ViewState private var page: Page = .home
    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            VStack(spacing: 0) {
                HStack {
                    Text(page.rawValue).font(.system(size: 14, weight: .semibold))
                    Spacer()
                    Label("ASR ON YOUR MAC", systemImage: "lock.shield")
                        .font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundStyle(accent)
                    Circle().fill(accent).frame(width: 6, height: 6)
                }.padding(.horizontal, 32).frame(height: 60)
                Divider().opacity(0.5)
                ScrollView {
                    VStack(alignment: .leading, spacing: 22) {
                        if !store.storageError.isEmpty { message(store.storageError, error: true) }
                        if !model.error.isEmpty {
                            message(model.error, error: true)
                            if model.canRetry { Button("Retry last recording") { model.retryLast() } }
                        }
                        if !model.notice.isEmpty { message(model.notice, error: false) }
                        if model.phase == .preparing { PreparationCard(model: model, worker: model.worker) }
                        switch page {
                        case .home: HomeView(model: model, store: store)
                        case .history: HistoryView(model: model, store: store)
                        case .dictionary: DictionaryView(store: store)
                        case .rules: RulesView(store: store)
                        case .settings: PreferencesView(model: model, store: store)
                        }
                    }.padding(32).frame(maxWidth: 940, alignment: .leading).frame(maxWidth: .infinity)
                }
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .tint(accent)
        .preferredColorScheme(store.preferences.appearance == "light" ? .light : store.preferences.appearance == "dark" ? .dark : nil)
        .onChange(of: model.resultText) { _, _ in page = .home }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 30) {
            HStack(spacing: 10) {
                Image(systemName: "waveform").font(.system(size: 23, weight: .medium))
                    .foregroundStyle(.white).frame(width: 40, height: 40)
                    .background(brandGreen, in: RoundedRectangle(cornerRadius: 13))
                VStack(alignment: .leading, spacing: 2) {
                    Text("OmniTyper").font(.system(size: 16, weight: .semibold))
                    Text("Local voice typing.").font(.system(size: 10)).foregroundStyle(.secondary)
                }
            }.padding(.top, 35).padding(.horizontal, 18)
            VStack(spacing: 6) {
                ForEach(Page.allCases, id: \.self) { item in
                    Button { page = item } label: {
                        HStack(spacing: 12) {
                            Image(systemName: item.icon).frame(width: 20)
                            Text(item.rawValue).font(.system(size: 13, weight: page == item ? .semibold : .regular))
                            Spacer()
                            if item == .history && !store.history.isEmpty {
                                Text("\(store.history.count)").font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
                            }
                        }.padding(.horizontal, 14).padding(.vertical, 12)
                            .background(page == item ? accent.opacity(0.1) : .clear, in: RoundedRectangle(cornerRadius: 10))
                            .foregroundStyle(page == item ? accent : .primary)
                    }.buttonStyle(.plain)
                }
            }.padding(.horizontal, 12)
            Spacer()
            VStack(alignment: .leading, spacing: 12) {
                Label("Your voice. Your device.", systemImage: "desktopcomputer").font(.system(size: 11, weight: .medium))
                Text("Local speech recognition.\nYour choice of text API.")
                    .font(.system(size: 11)).foregroundStyle(.secondary).lineSpacing(4)
                HStack {
                    Text("SGLang-Omni + MLX").font(.system(size: 9, weight: .medium, design: .monospaced))
                    Spacer()
                    Text("0.1").font(.system(size: 9, design: .monospaced))
                }.foregroundStyle(.tertiary)
            }.padding(18)
        }.frame(width: 218).background(cardBackground.opacity(0.48))
    }

    private func message(_ text: String, error: Bool) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: error ? "exclamationmark.circle" : "checkmark.circle")
                .foregroundStyle(error ? .orange : accent)
            Text(text).font(.system(size: 12)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
            Button { if error { model.error = "" } else { model.notice = "" } } label: { Image(systemName: "xmark") }
                .buttonStyle(.plain).accessibilityLabel("Dismiss message")
        }.padding(14).background((error ? Color.orange : accent).opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct Card<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        content.padding(22).frame(maxWidth: .infinity, alignment: .leading)
            .background(cardBackground, in: RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).stroke(.primary.opacity(0.055), lineWidth: 1))
    }
}

struct HomeView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: AppStore
    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            VStack(alignment: .leading, spacing: 9) {
                Text("MAKE ROOM FOR YOUR THOUGHTS").font(.system(size: 10, weight: .semibold, design: .monospaced)).tracking(2).foregroundStyle(accent)
                Text("Say it. Make it yours.").font(.system(size: 34, weight: .semibold, design: .rounded))
                Text("Speak in any app. Let your Mac take care of the words.").font(.system(size: 14)).foregroundStyle(.secondary)
            }.padding(.bottom, 4)
            if !model.microphoneAllowed || !model.accessibilityAllowed {
                Card {
                    VStack(alignment: .leading, spacing: 16) {
                        Label("Make yourself at home", systemImage: "hand.wave").font(.system(size: 16, weight: .semibold))
                        Text("Two macOS permissions connect your voice to wherever you write.").font(.system(size: 12)).foregroundStyle(.secondary)
                        permission("Microphone", subtitle: "Hear you only while recording", ready: model.microphoneAllowed, action: model.requestMicrophone)
                        permission("Accessibility", subtitle: "Use the shortcut and insert at your cursor", ready: model.accessibilityAllowed, action: model.requestAccessibility)
                    }
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 20) {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 7) {
                            Text(model.mode.title).font(.system(size: 23, weight: .semibold))
                            Text(model.mode.detail).font(.system(size: 13)).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Image(systemName: model.mode.icon).font(.system(size: 30)).foregroundStyle(accent)
                            .frame(width: 64, height: 64).background(accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 18))
                    }
                    Picker("Voice mode", selection: $model.mode) {
                        ForEach(VoiceMode.allCases) { Label($0.title, systemImage: $0.icon).tag($0) }
                    }.pickerStyle(.segmented).labelsHidden().disabled(model.isBusy)
                    HStack(spacing: 12) {
                        Button { model.toggle() } label: {
                            Label(model.phase == .recording ? "Finish recording" : model.phase == .processing ? "Working…" : model.phase == .starting ? "Loading speech…" : "Start speaking",
                                  systemImage: model.phase == .recording ? "stop.fill" : "mic.fill")
                                .frame(minWidth: 142).padding(.vertical, 6)
                        }.buttonStyle(.borderedProminent).controlSize(.large)
                            .disabled(model.isBusy && model.phase != .recording)
                        if model.isBusy { Button("Cancel", role: .cancel) { model.cancel() } }
                        Spacer()
                        VStack(alignment: .trailing, spacing: 5) {
                            Text(model.shortcutLabel).font(.system(size: 12, weight: .medium, design: .monospaced))
                                .padding(.horizontal, 10).padding(.vertical, 6).background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 7))
                            Text(store.preferences.holdToTalk ? "Hold to speak" : "Press to start · press to finish").font(.system(size: 10)).foregroundStyle(.secondary)
                        }
                    }
                    if model.mode == .translate {
                        HStack {
                            Text("Write in").foregroundStyle(.secondary)
                            Picker("Translation language", selection: $store.preferences.targetLanguage) { ForEach(languages, id: \.self) { Text($0).tag($0) } }
                                .labelsHidden().frame(width: 160)
                        }.font(.system(size: 12))
                    }
                    if model.mode == .ask {
                        Text("Answers come from your configured text API. OmniTyper does not provide web browsing tools.").font(.system(size: 11)).foregroundStyle(.secondary)
                    }
                }
            }
            if model.phase == .starting || model.phase == .recording || model.phase == .processing {
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        Label("Live transcript", systemImage: "waveform").font(.system(size: 13, weight: .semibold))
                        Text(model.liveStatus).font(.system(size: 11)).foregroundStyle(.secondary)
                        Text(model.liveText.isEmpty ? "Your words will appear here while you speak." : model.liveText)
                            .font(.system(size: 15)).lineSpacing(4).textSelection(.enabled)
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            if !model.resultText.isEmpty {
                Card {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            Label("Your words, ready", systemImage: "text.alignleft").font(.system(size: 13, weight: .semibold))
                            Spacer()
                            Button { model.copyResult() } label: { Label("Copy", systemImage: "doc.on.doc") }
                        }
                        Text(model.resultText).font(.system(size: 15)).lineSpacing(5).textSelection(.enabled)
                        if model.rawText != model.resultText && !model.rawText.isEmpty {
                            DisclosureGroup("Original transcript") { Text(model.rawText).font(.system(size: 12)).foregroundStyle(.secondary).textSelection(.enabled).padding(.top, 6) }
                                .font(.system(size: 11))
                        }
                    }
                }
            }
            HStack(spacing: 14) {
                stat("Words captured", value: "\(store.history.reduce(0) { $0 + $1.units })", icon: "text.word.spacing")
                stat("Time speaking", value: "\(Int(store.history.reduce(0) { $0 + $1.duration } / 60)) min", icon: "waveform")
                stat("Your vocabulary", value: "\(store.dictionary.count) words", icon: "book.closed")
            }
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "lightbulb").foregroundStyle(accent)
                Text("Keep your cursor in your writing app and use the shortcut. Voice edit replaces selected text; Ask keeps it unchanged.")
                    .font(.system(size: 12)).foregroundStyle(.secondary).lineSpacing(4)
            }.padding(.horizontal, 4)
        }
    }
    private func permission(_ title: String, subtitle: String, ready: Bool, action: @escaping () -> Void) -> some View {
        HStack {
            Image(systemName: ready ? "checkmark.circle.fill" : "circle").foregroundStyle(ready ? accent : .secondary)
            VStack(alignment: .leading, spacing: 3) { Text(title).font(.system(size: 12, weight: .medium)); Text(subtitle).font(.system(size: 11)).foregroundStyle(.secondary) }
            Spacer()
            if ready { Text("Ready").font(.system(size: 11)).foregroundStyle(accent) }
            else { Button("Allow", action: action).controlSize(.small) }
        }
    }
    private func stat(_ title: String, value: String, icon: String) -> some View {
        Card {
            VStack(alignment: .leading, spacing: 12) {
                Image(systemName: icon).foregroundStyle(accent)
                Text(value).font(.system(size: 21, weight: .semibold, design: .rounded))
                Text(title).font(.system(size: 10)).foregroundStyle(.secondary)
            }
        }
    }
}

private struct PreparationCard: View {
    @ObservedObject var model: AppModel
    @ObservedObject var worker: WorkerClient
    var body: some View {
        HStack(spacing: 14) {
            ProgressView().controlSize(.small)
            VStack(alignment: .leading, spacing: 5) {
                Text("Connecting your models").font(.system(size: 13, weight: .semibold))
                Text(worker.status.isEmpty ? "The first download can take several minutes." : worker.status).font(.system(size: 11)).foregroundStyle(.secondary)
            }
            Spacer(); Button("Cancel") { model.cancel() }
        }.padding(20).background(accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
    }
}

struct VoicePanel: View {
    @ObservedObject var model: AppModel
    @ObservedObject var recorder: AudioRecorder
    @ObservedObject var worker: WorkerClient
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 14) {
                if model.phase == .recording {
                    HStack(alignment: .center, spacing: 3) {
                        ForEach(0..<9) { index in
                            Capsule().fill(accent).frame(width: 3, height: 5 + 30 * recorder.level * (index % 2 == 0 ? 1 : 0.55))
                        }
                    }.frame(width: 45, height: 38).animation(.easeOut(duration: 0.1), value: recorder.level)
                } else { ProgressView().controlSize(.small).frame(width: 45) }
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.phase == .recording ? "Listening · \(model.mode.title)" : model.phase == .starting ? "Loading speech model…" : model.liveStatus)
                        .font(.system(size: 12, weight: .semibold)).lineLimit(1)
                    Text(model.phase == .recording ? String(format: "%d:%02d · Esc to cancel", Int(recorder.elapsed) / 60, Int(recorder.elapsed) % 60) : worker.status)
                        .font(.system(size: 10)).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer(minLength: 0)
                if model.phase == .recording {
                    Button { model.finish() } label: { Image(systemName: "stop.fill").foregroundStyle(accent) }.buttonStyle(.plain).accessibilityLabel("Finish recording")
                }
                Button { model.cancel() } label: { Image(systemName: "xmark").foregroundStyle(.secondary) }.buttonStyle(.plain).accessibilityLabel("Cancel recording")
            }
            Divider()
            Text(model.liveText.isEmpty ? (model.phase == .starting ? "Wait for Listening before speaking." : "Your words will appear here…") : String(model.liveText.suffix(600)))
                .font(.system(size: 14)).lineSpacing(3).lineLimit(3).truncationMode(.head)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            Text(model.phase == .recording ? model.liveStatus : "Text is inserted only after you finish.")
                .font(.system(size: 10)).foregroundStyle(.secondary).lineLimit(1)
        }.padding(18).frame(width: 460, height: 190).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 22))
            .overlay(RoundedRectangle(cornerRadius: 22).stroke(.white.opacity(0.2), lineWidth: 1))
    }
}

struct HistoryView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: AppStore
    @ViewState private var query = ""
    @ViewState private var filter = "all"
    @ViewState private var editing: HistoryEntry?
    @ViewState private var confirmDelete = false
    private var entries: [HistoryEntry] {
        store.history.filter { (filter == "all" || $0.mode.rawValue == filter) && (query.isEmpty || $0.text.localizedCaseInsensitiveContains(query) || $0.appName.localizedCaseInsensitiveContains(query)) }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            pageTitle("A thought worth keeping.", detail: "Search, copy, and revisit the words you have captured.")
            HStack {
                TextField("Search your history", text: $query).textFieldStyle(.roundedBorder)
                Picker("Filter", selection: $filter) {
                    Text("All modes").tag("all")
                    ForEach(VoiceMode.allCases) { Text($0.title).tag($0.rawValue) }
                }.labelsHidden().frame(width: 135)
                Menu {
                    Button("Export history…") { FileActions.exportHistory(store.history) }
                    Button("Delete all history", role: .destructive) { confirmDelete = true }
                } label: { Image(systemName: "ellipsis.circle") }.menuStyle(.borderlessButton).frame(width: 28)
            }
            if entries.isEmpty {
                ContentUnavailableView(query.isEmpty ? "Your words will live here" : "No matching transcripts", systemImage: "text.bubble", description: Text(query.isEmpty ? "Start a dictation to build your local history." : "Try another search."))
                    .frame(maxWidth: .infinity, minHeight: 230)
            }
            LazyVStack(spacing: 14) {
                ForEach(entries) { entry in
                    Card {
                        VStack(alignment: .leading, spacing: 13) {
                            HStack {
                                Label(entry.mode.title, systemImage: entry.mode.icon).font(.system(size: 11, weight: .semibold)).foregroundStyle(accent)
                                Text("· \(entry.appName)").font(.system(size: 11)).foregroundStyle(.secondary)
                                Spacer()
                                Text(entry.date, format: .dateTime.month(.abbreviated).day().hour().minute()).font(.system(size: 10)).foregroundStyle(.secondary)
                            }
                            Text(entry.text).font(.system(size: 14)).lineSpacing(4).textSelection(.enabled)
                            if let warning = entry.warning, !warning.isEmpty { Text(warning).font(.system(size: 11)).foregroundStyle(.orange) }
                            HStack {
                                Text("\(entry.units) words · \(Int(entry.duration))s").font(.system(size: 10)).foregroundStyle(.tertiary)
                                Spacer()
                                Button { TextInsertion.copy(entry.text) } label: { Label("Copy", systemImage: "doc.on.doc") }
                                Menu {
                                    Button("Correct transcript…") { editing = entry }
                                    if store.audioURL(for: entry) != nil {
                                        Button("Retry audio") { model.retry(entry) }.disabled(model.isBusy)
                                        Button("Export audio…") { if let url = store.audioURL(for: entry) { FileActions.exportAudio(url) } }
                                    }
                                    Button("Delete", role: .destructive) { store.delete([entry.id]) }
                                } label: { Image(systemName: "ellipsis") }.menuStyle(.borderlessButton).frame(width: 20)
                            }.controlSize(.small)
                            if entry.rawText != entry.text {
                                DisclosureGroup("Original transcript") { Text(entry.rawText).textSelection(.enabled).padding(.top, 6) }.font(.system(size: 11)).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
        }.sheet(item: $editing) { entry in CorrectionView(store: store, entry: entry) }
            .confirmationDialog("Delete all transcripts and retained audio?", isPresented: $confirmDelete) {
                Button("Delete all", role: .destructive) { store.delete(Set(store.history.map(\.id))) }
            }
    }
}

private struct CorrectionView: View {
    @ObservedObject var store: AppStore
    let entry: HistoryEntry
    @Environment(\.dismiss) var dismiss
    @ViewState private var corrected = ""
    @ViewState private var spoken = ""
    @ViewState private var written = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Make it sound like you").font(.title2.bold())
            TextEditor(text: $corrected).font(.body).frame(height: 150).border(.secondary.opacity(0.2))
            Text("Remember a corrected word (optional)").font(.subheadline)
            HStack { TextField("Heard as", text: $spoken); Image(systemName: "arrow.right"); TextField("Write as", text: $written) }.textFieldStyle(.roundedBorder)
            Text("Corrections stay on this Mac. Only the word pair is added to your dictionary.").font(.caption).foregroundStyle(.secondary)
            HStack {
                Spacer(); Button("Cancel") { dismiss() }
                Button("Save correction") {
                    if let index = store.history.firstIndex(where: { $0.id == entry.id }) { store.history[index].text = corrected }
                    store.addWord(spoken: spoken, written: written, learned: true); dismiss()
                }.buttonStyle(.borderedProminent).disabled(corrected.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }.padding(28).frame(width: 520).onAppear { corrected = entry.text }
    }
}

struct DictionaryView: View {
    @ObservedObject var store: AppStore
    @ViewState private var spoken = ""
    @ViewState private var written = ""
    @ViewState private var query = ""
    @ViewState private var importMessage = ""
    @ViewState private var editingID: UUID?
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            pageTitle("Your words belong here.", detail: "Names, technical terms, and the spellings that make your writing yours.")
            Card {
                VStack(alignment: .leading, spacing: 14) {
                    Text(editingID == nil ? "Add to your vocabulary" : "Edit your vocabulary").font(.headline)
                    HStack {
                        TextField("Spoken or misheard phrase", text: $spoken)
                        Image(systemName: "arrow.right").foregroundStyle(.secondary)
                        TextField("Preferred spelling", text: $written)
                        Button(editingID == nil ? "Add word" : "Save word") {
                            guard store.addWord(spoken: spoken, written: written.isEmpty ? spoken : written, replacing: editingID) else {
                                importMessage = "Could not save. Use a unique spoken phrase and at most 200 entries."
                                return
                            }
                            spoken = ""; written = ""; editingID = nil; importMessage = ""
                        }.buttonStyle(.borderedProminent)
                            .disabled(!DictionaryEntry.isValidPhrase(spoken) || (!written.isEmpty && !DictionaryEntry.isValidPhrase(written)))
                        if editingID != nil { Button("Cancel") { spoken = ""; written = ""; editingID = nil } }
                    }.textFieldStyle(.roundedBorder)
                    Text("Leave preferred spelling blank to add a recognition hint. Up to 200 entries, 120 characters each.").font(.system(size: 11)).foregroundStyle(.secondary)
                }
            }
            HStack {
                TextField("Find a word", text: $query).textFieldStyle(.roundedBorder)
                Button("Import CSV…") {
                    do { if let text = try FileActions.importText() { importMessage = "Imported \(try store.importWords(text)) entries." } }
                    catch { importMessage = error.localizedDescription }
                }
                Button("Export…") { FileActions.exportDictionary(store.dictionary) }
            }
            if !importMessage.isEmpty { Text(importMessage).font(.caption).foregroundStyle(.secondary) }
            if store.dictionary.isEmpty {
                ContentUnavailableView("Teach it your vocabulary", systemImage: "book.closed", description: Text("For example: “S G Lang” → “SGLang”."))
                    .frame(maxWidth: .infinity, minHeight: 180)
            }
            LazyVStack(spacing: 8) {
                ForEach(store.dictionary.filter { query.isEmpty || $0.spoken.localizedCaseInsensitiveContains(query) || $0.written.localizedCaseInsensitiveContains(query) }) { entry in
                    HStack(spacing: 12) {
                        Text(entry.spoken).frame(maxWidth: .infinity, alignment: .leading)
                        Image(systemName: "arrow.right").foregroundStyle(.tertiary)
                        Text(entry.written).frame(maxWidth: .infinity, alignment: .leading)
                        if !entry.isValid { Image(systemName: "exclamationmark.circle").foregroundStyle(.orange).help("Edit or delete this invalid entry before recording.") }
                        if entry.learned { Image(systemName: "sparkle").foregroundStyle(accent).help("Added from your correction") }
                        Button("Edit") { editingID = entry.id; spoken = entry.spoken; written = entry.written }
                        Button {
                            store.dictionary.removeAll { $0.id == entry.id }
                            if editingID == entry.id { editingID = nil; spoken = ""; written = "" }
                        } label: { Image(systemName: "trash").foregroundStyle(.secondary) }.buttonStyle(.plain).accessibilityLabel("Delete word")
                    }.textFieldStyle(.plain).padding(15).background(cardBackground, in: RoundedRectangle(cornerRadius: 10))
                }
            }
        }
    }
}

struct RulesView: View {
    @ObservedObject var store: AppStore
    @ViewState private var bundleID = ""
    @ViewState private var appName = ""
    @ViewState private var selectedStyle = "clean"
    @ViewState private var instructions = ""
    @ViewState private var apps: [NSRunningApplication] = []
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            pageTitle("A voice for every context.", detail: "Write a friendly message or a polished email. Choose what fits each app.")
            Card {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Everywhere").font(.headline)
                    Picker("Default style", selection: $store.preferences.style) { ForEach(styles, id: \.self) { Text($0.capitalized).tag($0) } }
                    TextField("Writing preferences, e.g. Use British English. Keep technical terms in English.", text: $store.preferences.instructions, axis: .vertical)
                        .textFieldStyle(.roundedBorder).lineLimit(3...5)
                    Text("\(store.preferences.instructions.unicodeScalars.count)/1,000 characters")
                        .font(.caption).foregroundStyle(store.preferences.instructions.unicodeScalars.count > 1000 ? .orange : .secondary)
                    if let error = instructionError(store.preferences.instructions, "") { Text(error).font(.caption).foregroundStyle(.orange) }
                    Text("Verbatim skips the text model for dictation. Other styles remove filler words, preserve your meaning, and add punctuation.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Add an app preference").font(.headline)
                    Picker("Running app", selection: $bundleID) {
                        Text("Choose an app").tag("")
                        ForEach(apps, id: \.processIdentifier) { app in Text(app.localizedName ?? app.bundleIdentifier ?? "App").tag(app.bundleIdentifier ?? "") }
                    }.onChange(of: bundleID) { _, value in appName = apps.first { $0.bundleIdentifier == value }?.localizedName ?? value }
                    HStack { TextField("Bundle ID", text: $bundleID); TextField("Display name", text: $appName) }.textFieldStyle(.roundedBorder)
                    Picker("Style", selection: $selectedStyle) { ForEach(styles, id: \.self) { Text($0.capitalized).tag($0) } }
                    TextField("Instructions for this app", text: $instructions, axis: .vertical).textFieldStyle(.roundedBorder)
                    Text("\(instructions.unicodeScalars.count)/1,000 characters · 2,000 combined with defaults")
                        .font(.caption).foregroundStyle(instructions.unicodeScalars.count > 1000 ? .orange : .secondary)
                    if let error = instructionError(store.preferences.instructions, instructions) { Text(error).font(.caption).foregroundStyle(.orange) }
                    Button("Save app preference") {
                        store.rules.removeAll { $0.bundleID == bundleID }
                        store.rules.append(AppRule(bundleID: bundleID, name: appName.isEmpty ? bundleID : appName, style: selectedStyle, instructions: instructions))
                        bundleID = ""; appName = ""; instructions = ""
                    }.disabled(bundleID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || instructionError(store.preferences.instructions, instructions) != nil).buttonStyle(.borderedProminent)
                }
            }
            ForEach(store.rules) { rule in
                Card {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(rule.name).font(.headline)
                            Text("\(rule.style.capitalized) · \(rule.bundleID)").font(.caption).foregroundStyle(.secondary)
                            if !rule.instructions.isEmpty { Text(rule.instructions).font(.subheadline) }
                        }
                        Spacer()
                        Button("Edit") { bundleID = rule.bundleID; appName = rule.name; selectedStyle = rule.style; instructions = rule.instructions }
                        Button { store.rules.removeAll { $0.id == rule.id } } label: { Image(systemName: "trash") }.accessibilityLabel("Delete app preference")
                    }
                }
            }
        }.onAppear { apps = NSWorkspace.shared.runningApplications.filter { $0.activationPolicy == .regular && $0.bundleIdentifier != nil }.sorted { ($0.localizedName ?? "") < ($1.localizedName ?? "") } }
    }
    private func instructionError(_ defaults: String, _ app: String) -> String? {
        do { _ = try Preferences.combinedInstructions(defaults, app); return nil }
        catch { return error.localizedDescription }
    }
}

struct PreferencesView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: AppStore
    @ViewState private var microphones: [MicrophoneDevice] = []
    @ViewState private var captureMonitor: Any?
    @ViewState private var capturing = false
    @ViewState private var login = SMAppService.mainApp.status == .enabled
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            pageTitle("Settle into your flow.", detail: "Your shortcuts, your languages, your privacy.")
            Card {
                VStack(alignment: .leading, spacing: 15) {
                    Label("Keyboard & audio", systemImage: "keyboard").font(.headline)
                    HStack {
                        Text("Global shortcut"); Spacer()
                        Button(capturing ? "Press a key combination…" : model.shortcutLabel) { captureShortcut() }
                            .font(.system(.body, design: .monospaced))
                        Button("Reset") { store.preferences.shortcutKeyCode = 49; store.preferences.shortcutModifiers = 786432 }
                    }
                    Toggle("Hold shortcut to talk", isOn: $store.preferences.holdToTalk)
                    Text("Turn off for press-to-start, press-to-finish. Escape cancels. Use a modifier with ordinary keys to avoid interrupting typing.").font(.caption).foregroundStyle(.secondary)
                    Divider()
                    Picker("Microphone", selection: $store.preferences.microphoneUID) {
                        Text("System default").tag("")
                        ForEach(microphones) { Text($0.name).tag($0.id) }
                    }
                    Toggle("Play start and finish sounds", isOn: $store.preferences.sounds)
                    Toggle("Insert text automatically at the original cursor", isOn: $store.preferences.autoPaste)
                    Text("If the target or cursor changes, your transcript stays available to copy.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 15) {
                    Label("Languages", systemImage: "globe").font(.headline)
                    Picker("Speech language", selection: $store.preferences.language) {
                        Text("Detect automatically").tag("")
                        ForEach(languages, id: \.self) { Text($0).tag($0) }
                    }
                    Picker("Translate into", selection: $store.preferences.targetLanguage) { ForEach(languages, id: \.self) { Text($0).tag($0) } }
                    Text("Recognition coverage follows Qwen3-ASR. Translation quality depends on your text API model.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 15) {
                    Label("Local speech model", systemImage: "cpu").font(.headline)
                    Text("Qwen3-ASR · 0.6B · MLX 4-bit").font(.subheadline)
                    Text("SGLang-Omni performs speech recognition on Apple Silicon. Text processing is configured separately below.").font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Button("Download & prepare ASR") { model.prepareModels() }.buttonStyle(.borderedProminent).disabled(model.isBusy)
                        Button("Unload ASR") { model.releaseModels() }.disabled(model.isBusy)
                    }
                    DisclosureGroup("Runtime location") {
                        TextField("Python executable", text: $store.preferences.pythonExecutable).textFieldStyle(.roundedBorder).padding(.top, 8)
                        Text("Set up the runtime with OmniTyper/scripts/setup.sh before preparing models.").font(.caption).foregroundStyle(.secondary)
                    }
                    Text("First use downloads ASR weights from Hugging Face. Audio stays on this Mac; no telemetry is collected.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 15) {
                    Label("Text API", systemImage: "network").font(.headline)
                    Text("Connect Ollama or another OpenAI-compatible server for cleanup, translation, editing, and answers.").font(.caption).foregroundStyle(.secondary)
                    TextField("Base URL (including /v1)", text: $store.preferences.textSettings.baseURL)
                        .textFieldStyle(.roundedBorder).accessibilityLabel("Text API base URL")
                    HStack {
                        TextField("Model name from your server", text: $store.preferences.textSettings.model)
                            .textFieldStyle(.roundedBorder).accessibilityLabel("Text API model")
                        if !model.textModels.isEmpty {
                            Menu("Choose model") {
                                ForEach(model.textModels, id: \.self) { name in
                                    Button(name) { store.preferences.textSettings.model = name }
                                }
                            }
                        }
                    }
                    SecureField("API key (optional, this session only)", text: $model.textAPIKey).textFieldStyle(.roundedBorder)
                    Text("Local Ollama normally needs no key. Keys are cleared when the address changes or the app quits.").font(.caption).foregroundStyle(.secondary)
                    Button("Connect & load models") { model.loadTextModels() }.disabled(model.isBusy)
                    DisclosureGroup("Request options (JSON)") {
                        TextEditor(text: $store.preferences.textSettings.optionsJSON)
                            .font(.system(.caption, design: .monospaced)).frame(height: 90)
                            .accessibilityLabel("Text API request options JSON")
                        Text("Leave {} to use server defaults. Optional fields such as temperature and max_tokens are passed through; model, messages, and stream are managed by the app. Configure context size and custom models in Ollama.").font(.caption).foregroundStyle(.secondary)
                    }
                    Text("Transcripts, writing preferences, and selected text for Edit/Ask are sent to this endpoint. Choose a local model/server to keep text on your Mac; cloud routing is controlled by your provider. Verbatim dictation skips this API.").font(.caption).foregroundStyle(.secondary)
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 15) {
                    Label("Privacy & history", systemImage: "lock.shield").font(.headline)
                    Toggle("Keep dictation history on this Mac", isOn: $store.preferences.saveHistory)
                    Picker("Keep history for", selection: $store.preferences.historyDays) {
                        Text("24 hours").tag(1); Text("7 days").tag(7); Text("30 days").tag(30)
                        Text("1 year").tag(365); Text("Forever").tag(0)
                    }.disabled(!store.preferences.saveHistory)
                    Toggle("Keep audio for retry and export", isOn: $store.preferences.keepAudio).disabled(!store.preferences.saveHistory)
                    Text("Turning history off deletes saved transcripts and audio. Turning audio retention off deletes retained recordings. At most 1,000 transcripts are kept.").font(.caption).foregroundStyle(.secondary)
                    Button("Open local data folder") { NSWorkspace.shared.open(store.directory) }
                }
            }
            Card {
                VStack(alignment: .leading, spacing: 15) {
                    Label("General", systemImage: "gearshape").font(.headline)
                    Picker("Appearance", selection: $store.preferences.appearance) { Text("System").tag("system"); Text("Light").tag("light"); Text("Dark").tag("dark") }
                    Toggle("Open at login", isOn: Binding(get: { login }, set: { value in
                        do {
                            if value { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
                            login = SMAppService.mainApp.status == .enabled
                        } catch { model.error = "Could not change login settings: \(error.localizedDescription)" }
                    }))
                    HStack {
                        Button("Microphone settings") { NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!) }
                        Button("Accessibility settings") { model.requestAccessibility() }
                    }
                    Text("OmniTyper 0.1 · Apache-2.0 · Independent of Typeless").font(.caption).foregroundStyle(.secondary)
                }
            }
        }.onAppear { microphones = AudioRecorder.devices() }
            .onDisappear { endCapture() }
    }
    private func captureShortcut() {
        endCapture(); capturing = true
        captureMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            if event.keyCode == 53 { endCapture(); return nil }
            let mask: NSEvent.ModifierFlags = [.command, .control, .option, .shift]
            let flags = event.modifierFlags.intersection(mask)
            guard !flags.isEmpty || [96, 97, 98, 99, 100, 101, 109, 111].contains(event.keyCode) else { NSSound.beep(); return nil }
            store.preferences.shortcutKeyCode = event.keyCode
            store.preferences.shortcutModifiers = UInt64(flags.rawValue)
            endCapture(); return nil
        }
    }
    private func endCapture() { if let captureMonitor { NSEvent.removeMonitor(captureMonitor) }; captureMonitor = nil; capturing = false }
}

private func pageTitle(_ title: String, detail: String) -> some View {
    VStack(alignment: .leading, spacing: 9) { Text(title).font(.system(size: 27, weight: .semibold, design: .rounded)); Text(detail).font(.system(size: 13)).foregroundStyle(.secondary) }
}

@MainActor
private enum FileActions {
    static func importText() throws -> String? {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.commaSeparatedText, .plainText]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard size <= 1_000_000 else { throw AppError.message("Dictionary file is larger than 1 MB.") }
        return try String(contentsOf: url, encoding: .utf8)
    }
    static func exportHistory(_ entries: [HistoryEntry]) {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
        save(name: "OmniTyper-history.json", type: .json) { try encoder.encode(entries) }
    }
    static func exportDictionary(_ entries: [DictionaryEntry]) {
        func escape(_ text: String) -> String { "\"" + text.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }
        let text = "spoken,written\n" + entries.map { "\(escape($0.spoken)),\(escape($0.written))" }.joined(separator: "\n")
        save(name: "OmniTyper-dictionary.csv", type: .commaSeparatedText) { Data(text.utf8) }
    }
    static func exportAudio(_ url: URL) { save(name: "OmniTyper-recording.wav", type: .wav) { try Data(contentsOf: url) } }
    private static func save(name: String, type: UTType, data: () throws -> Data) {
        let panel = NSSavePanel(); panel.nameFieldStringValue = name; panel.allowedContentTypes = [type]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try data().write(to: url, options: .atomic) }
        catch { let alert = NSAlert(error: error); alert.runModal() }
    }
}
