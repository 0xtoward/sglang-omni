// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OmniTyper",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "OmniTyper", targets: ["OmniTyper"])],
    targets: [
        .executableTarget(name: "OmniTyper"),
        .testTarget(name: "OmniTyperTests", dependencies: ["OmniTyper"])
    ]
)
