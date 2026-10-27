import AppKit
import SwiftUI
import Sparkle

/// One signed update channel for the installed application. No model data is updated.
@MainActor final class AppUpdater: NSObject, ObservableObject, SPUUpdaterDelegate {
    @Published private(set) var availableVersion: String?
    @Published private(set) var checking = false
    @Published private(set) var message = "Updates are not configured for this build."
    @Published private(set) var configured = false
    private var controller: SPUStandardUpdaterController?
    private var polling: Task<Void, Never>?
    var mayInstall: () -> Bool = { true }
    var prepareForRelaunch: (() async -> Void)?
    var currentVersion: String { Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Development" }

    func start(disabled: Bool) {
        guard controller == nil else { return }
        if disabled && !CommandLine.arguments.contains("--update-probe") {
            if CommandLine.arguments.contains("--preview-update-available") { availableVersion = "Next release" }
            return
        }
        guard let feed = Bundle.main.object(forInfoDictionaryKey: "SUFeedURL") as? String,
              let url = URL(string: feed), url.scheme == "https" || (url.scheme == "http" && url.host == "127.0.0.1"),
              Bundle.main.object(forInfoDictionaryKey: "SUPublicEDKey") is String else { return }
        let value = SPUStandardUpdaterController(startingUpdater: false, updaterDelegate: self, userDriverDelegate: nil)
        controller = value
        do {
            try value.updater.start()
            configured = true
            check()
            polling = Task { [weak self] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(url.host == "127.0.0.1" ? 60 : 6 * 60 * 60))
                    guard !Task.isCancelled else { return }
                    self?.check()
                }
            }
        } catch { message = "Updates could not start: \(error.localizedDescription)" }
    }
    func check() {
        guard configured, let updater = controller?.updater, updater.canCheckForUpdates else { return }
        checking = true; message = "Checking for updates…"
        // Information checks never download, quit the app or open an alert.
        updater.checkForUpdateInformation()
    }
    func install() {
        guard mayInstall() else { message = "Finish the current response before updating."; return }
        guard let controller, controller.updater.canCheckForUpdates else { return }
        controller.checkForUpdates(nil)
    }
    func updater(_ updater: SPUUpdater, didFindValidUpdate item: SUAppcastItem) {
        availableVersion = item.displayVersionString
        message = "Particle \(item.displayVersionString) is available."
    }
    func updaterDidNotFindUpdate(_ updater: SPUUpdater) {
        availableVersion = nil; message = "You’re up to date."
    }
    func updater(_ updater: SPUUpdater, didAbortWithError error: Error) {
        checking = false
        let failure = error as NSError
        if failure.domain == SUSparkleErrorDomain && failure.code == Int(SUError.noUpdateError.rawValue) {
            availableVersion = nil; message = "You’re up to date."
        } else { message = "Couldn’t check for updates. Try again later." }
    }
    func updater(_ updater: SPUUpdater, didFinishUpdateCycleFor updateCheck: SPUUpdateCheck, error: Error?) {
        checking = false
    }
    func updater(_ updater: SPUUpdater, shouldPostponeRelaunchForUpdate item: SUAppcastItem, untilInvokingBlock installHandler: @escaping () -> Void) -> Bool {
        guard let prepareForRelaunch else { return false }
        Task { await prepareForRelaunch(); installHandler() }
        return true
    }
    func updater(_ updater: SPUUpdater, mayPerform updateCheck: SPUUpdateCheck) throws {
        if updateCheck != .updateInformation && !mayInstall() {
            throw NSError(domain: "ParticleOne.Updates", code: 1, userInfo: [NSLocalizedDescriptionKey: "Finish the current response before updating."])
        }
    }
}

struct UpdateIndicator: View {
    @ObservedObject var updater: AppUpdater
    let busy: Bool
    var body: some View {
        if let version = updater.availableVersion {
            Button { updater.install() } label: {
                Image(systemName: "arrow.down.circle.fill").font(.system(size: 18)).foregroundStyle(Design.green)
                    .frame(width: 40, height: 40).contentShape(Rectangle())
            }.buttonStyle(.plain).disabled(busy)
                .help(busy ? "Finish the current response before updating" : "Update to Particle \(version)")
                .accessibilityLabel("Update available: Particle \(version)")
        }
    }
}

struct UpdateSettings: View {
    @ObservedObject var updater: AppUpdater
    let busy: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("App updates").font(Design.heading(20))
            Text("Particle \(updater.currentVersion)").font(Design.body(11))
            Text(updater.message).font(Design.body(10)).foregroundStyle(Design.muted)
            HStack {
                Button(updater.checking ? "Checking…" : "Check for updates") { updater.check() }
                    .buttonStyle(HitButtonStyle()).disabled(!updater.configured || updater.checking)
                if updater.availableVersion != nil {
                    Spacer()
                    Button("Update & restart…") { updater.install() }.buttonStyle(WashedButtonStyle(compact: true)).disabled(busy)
                }
            }
            Text("Checks run quietly at launch and periodically. Installation asks before restarting. Your model, conversations and memories stay on this Mac.")
                .font(Design.body(10)).foregroundStyle(Design.muted)
        }
    }
}

struct UpdateMenuCommand: View {
    @ObservedObject var updater: AppUpdater
    let busy: Bool
    var body: some View {
        Button("Check for updates…") { updater.install() }.disabled(busy || !updater.configured)
    }
}
