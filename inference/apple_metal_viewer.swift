import Cocoa
import SceneKit

// Native macOS Metal 3D Viewer for Neural Meshes
class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let screenRect = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1000, height: 900)
        let windowWidth: CGFloat = 900
        let windowHeight: CGFloat = 850
        let windowX = screenRect.origin.x + (screenRect.width - windowWidth) / 2
        let windowY = screenRect.origin.y + (screenRect.height - windowHeight) / 2

        window = NSWindow(
            contentRect: NSRect(x: windowX, y: windowY, width: windowWidth, height: windowHeight),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Apple Metal 3D Viewer - Neural Ficus Tree (Metal 120 FPS)"
        window.isReleasedWhenClosed = false

        let scnView = SCNView(frame: window.contentView!.bounds)
        scnView.autoresizingMask = [.width, .height]
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = false
        scnView.showsStatistics = true
        scnView.backgroundColor = NSColor(red: 0.05, green: 0.07, blue: 0.10, alpha: 1.0)

        // Try loading 1024 mesh, fallback to standard mesh
        let scriptPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "sample_ficus.obj"
        let currentDir = FileManager.default.currentDirectoryPath
        var modelUrl = URL(fileURLWithPath: scriptPath)
        if !FileManager.default.fileExists(atPath: modelUrl.path) {
            modelUrl = URL(fileURLWithPath: currentDir).appendingPathComponent("inference").appendingPathComponent(scriptPath)
        }

        print("Loading 3D model into Apple Metal pipeline: \(modelUrl.path)")
        guard let scene = try? SCNScene(url: modelUrl) else {
            print("Error: Could not load 3D model from \(modelUrl.path)")
            NSApplication.shared.terminate(nil)
            return
        }

        // Apply botanical emerald PBR material to leaves and branches
        let plantColor = NSColor(red: 0.18, green: 0.72, blue: 0.38, alpha: 1.0)
        scene.rootNode.enumerateChildNodes { (node, _) in
            if let geometry = node.geometry {
                for material in geometry.materials {
                    material.diffuse.contents = plantColor
                    material.lightingModel = .physicallyBased
                    material.roughness.contents = 0.4
                    material.metalness.contents = 0.1
                }
            }
        }

        // Add 3-point studio lighting
        // 1. Key Light (Main soft white sun)
        let keyLight = SCNLight()
        keyLight.type = .directional
        keyLight.intensity = 1000
        keyLight.color = NSColor.white
        let keyLightNode = SCNNode()
        keyLightNode.light = keyLight
        keyLightNode.eulerAngles = SCNVector3(x: -CGFloat.pi / 4, y: CGFloat.pi / 4, z: 0)
        scene.rootNode.addChildNode(keyLightNode)

        // 2. Fill Light (Soft cool ambient)
        let fillLight = SCNLight()
        fillLight.type = .ambient
        fillLight.intensity = 350
        fillLight.color = NSColor(red: 0.7, green: 0.8, blue: 0.9, alpha: 1.0)
        let fillLightNode = SCNNode()
        fillLightNode.light = fillLight
        scene.rootNode.addChildNode(fillLightNode)

        // 3. Rim Light (Back warm highlight)
        let rimLight = SCNLight()
        rimLight.type = .directional
        rimLight.intensity = 600
        rimLight.color = NSColor(red: 0.9, green: 0.95, blue: 1.0, alpha: 1.0)
        let rimLightNode = SCNNode()
        rimLightNode.light = rimLight
        rimLightNode.eulerAngles = SCNVector3(x: CGFloat.pi / 6, y: -CGFloat.pi / 3, z: 0)
        scene.rootNode.addChildNode(rimLightNode)

        scnView.scene = scene
        window.contentView!.addSubview(scnView)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        print("Apple Metal 3D Viewer launched successfully! Interact with trackpad/mouse.")
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
