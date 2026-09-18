// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

@main
struct OmniTyperApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    var body: some Scene { Settings { EmptyView() } }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private var model: AppModel!
    private var window: NSWindow!
    private var panel: NSPanel!
    private var statusItem: NSStatusItem!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        model = AppModel()
        let content = RootView(model: model, store: model.store)
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1060, height: 750),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                          backing: .buffered, defer: false)
        window.title = "OmniTyper"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.contentView = NSHostingView(rootView: content)
        window.minSize = NSSize(width: 920, height: 660)
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.center()
        window.setFrameAutosaveName("OmniTyperMainWindow")
        model.showMainWindow = { [weak self] in self?.openWindow() }
        model.showVoicePanel = { [weak self] in self?.showPanel() }
        model.hideVoicePanel = { [weak self] in self?.panel?.orderOut(nil) }
        setupMenu()
        if !ProcessInfo.processInfo.arguments.contains("--background") { openWindow() }
        if let index = ProcessInfo.processInfo.arguments.firstIndex(of: "--snapshot"),
           ProcessInfo.processInfo.arguments.count > index + 1 {
            let path = ProcessInfo.processInfo.arguments[index + 1]
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in self?.saveSnapshot(path) }
        }
    }

    private func setupMenu() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        statusItem.button?.image = NSImage(systemSymbolName: "waveform", accessibilityDescription: "OmniTyper")
        statusItem.button?.toolTip = "OmniTyper — local voice input"
        let menu = NSMenu()
        menu.addItem(withTitle: "Open OmniTyper", action: #selector(openWindow), keyEquivalent: "")
        menu.addItem(.separator())
        for mode in VoiceMode.allCases {
            let item = NSMenuItem(title: mode.title, action: #selector(startFromMenu(_:)), keyEquivalent: "")
            item.representedObject = mode.rawValue; item.target = self; menu.addItem(item)
        }
        menu.addItem(withTitle: "Stop recording", action: #selector(stopRecording), keyEquivalent: "")
        menu.addItem(withTitle: "Cancel", action: #selector(cancelRecording), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit OmniTyper", action: #selector(quit), keyEquivalent: "q")
        for item in menu.items { item.target = self }
        statusItem.menu = menu
    }

    @objc func openWindow() {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
    }

    @objc private func startFromMenu(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String, let mode = VoiceMode(rawValue: raw) else { return }
        model.toggle(mode)
    }
    @objc private func stopRecording() { model.finish() }
    @objc private func cancelRecording() { model.cancel() }
    @objc private func quit() { NSApp.terminate(nil) }

    func windowWillClose(_ notification: Notification) { NSApp.setActivationPolicy(.accessory) }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool { openWindow(); return true }
    func applicationWillTerminate(_ notification: Notification) { model.shutdown() }

    private func showPanel() {
        if panel == nil {
            panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 460, height: 190),
                            styleMask: [.nonactivatingPanel, .borderless], backing: .buffered, defer: false)
            panel.level = .floating; panel.isOpaque = false; panel.backgroundColor = .clear
            panel.hasShadow = true; panel.hidesOnDeactivate = false
            panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            panel.contentView = NSHostingView(rootView: VoicePanel(model: model, recorder: model.recorder, worker: model.worker))
        }
        let pointer = NSEvent.mouseLocation
        let screen = NSScreen.screens.first(where: { NSMouseInRect(pointer, $0.frame, false) }) ?? NSScreen.main
        if let visible = screen?.visibleFrame { panel.setFrameOrigin(NSPoint(x: visible.midX - 230, y: visible.minY + 28)) }
        panel.orderFrontRegardless()
    }

    private func saveSnapshot(_ path: String) {
        guard let view = window.contentView,
              let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: bitmap)
        if let png = bitmap.representation(using: .png, properties: [:]) {
            try? png.write(to: URL(fileURLWithPath: path))
        }
        NSApp.terminate(nil)
    }
}
