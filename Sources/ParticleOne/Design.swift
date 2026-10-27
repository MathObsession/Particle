import SwiftUI
import AppKit
import CoreText
import ParticleOneCore

@MainActor enum Design {
    static func adaptive(_ light:UInt32,_ dark:UInt32)->Color {
        Color(nsColor:NSColor(name:nil){appearance in NSColor(Color(hex:appearance.bestMatch(from:[.darkAqua,.aqua]) == .darkAqua ? dark : light))})
    }
    static let ink=adaptive(0x294336,0xE1E9DA),muted=adaptive(0x758276,0xA5B5A5),paper=adaptive(0xF7F8F2,0x20241F),sidebar=adaptive(0xECF0E6,0x20241F),line=adaptive(0xDDE4D7,0x383E34),green=adaptive(0x58794A,0xB2CF8B),code=Color(hex:0x202C27),surface=adaptive(0xFFFFFF,0x292F27)
    static func heading(_ size: CGFloat) -> Font {.custom("PTSerif-Regular",size:size)}
    static func body(_ size: CGFloat = 13) -> Font {.custom("Poppins-Regular",size:size)}
    static func medium(_ size: CGFloat = 13) -> Font {.custom("Poppins-Medium",size:size)}
    static func mono(_ size: CGFloat = 12) -> Font {.custom("SpaceMono-Regular",size:size)}
    static func registerFonts() {
        guard let folder=Bundle.module.url(forResource:"Fonts",withExtension:nil,subdirectory:"Resources"),let paths=try? FileManager.default.contentsOfDirectory(at:folder,includingPropertiesForKeys:nil) else{return}
        for file in paths where file.pathExtension=="ttf" {CTFontManagerRegisterFontsForURL(file as CFURL,.process,nil)}
    }
    static let artwork: [String:NSImage] = {
        var images=[String:NSImage]()
        for name in ["hero","wash","icon","icon-alpha","hero-blended-light","hero-blended-dark","code-scene-light","code-scene-dark","code-panel-light","code-panel-dark","interface-grain"] {if let url=Bundle.module.url(forResource:name,withExtension:"png",subdirectory:"Resources/Art"),let image=NSImage(contentsOf:url){images[name]=image}}
        return images
    }()
    static func image(_ name: String) -> Image {Image(nsImage:artwork[name] ?? NSImage())}
}
extension Color {
    init(hex: UInt32) {self.init(.sRGB,red:Double((hex>>16)&255)/255,green:Double((hex>>8)&255)/255,blue:Double(hex&255)/255,opacity:1)}
}
struct WashSurface: View {
    var radius: CGFloat=16
    @Environment(\.colorScheme) private var scheme
    var body: some View {
        GeometryReader { geometry in
            Design.image("wash").resizable().scaledToFill().frame(width:geometry.size.width,height:geometry.size.height).clipped()
                .overlay((scheme == .dark ? Color.black : .white).opacity(scheme == .dark ? 0.68 : 0.13))
        }
        .clipShape(RoundedRectangle(cornerRadius:radius))
        .overlay(RoundedRectangle(cornerRadius:radius).strokeBorder(.white.opacity(scheme == .dark ? 0.12 : 0.65),lineWidth:1))
    }
}
struct WashedButtonStyle: ButtonStyle {
    var compact=false
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.font(Design.medium(compact ? 12 : 13)).foregroundStyle(Design.ink)
            .padding(.horizontal,compact ? 12 : 17).padding(.vertical,compact ? 8 : 12)
            .background(WashSurface(radius:compact ? 10 : 13))
            .shadow(color:Design.green.opacity(configuration.isPressed ? 0.02 : 0.1),radius:8,y:3)
            .contentShape(Rectangle()).opacity(configuration.isPressed ? 0.74 : 1)
    }
}
struct QuietButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.padding(10).frame(minWidth:36,minHeight:36).foregroundStyle(Design.muted).background(configuration.isPressed ? Design.line : .clear,in:RoundedRectangle(cornerRadius:8)).contentShape(Rectangle())
    }
}

/// Finished opaque ParticleDesign PNGs keep their original green background.
/// Their outside colors gently meet the surrounding page; no live effects.
struct FadingHeroBanner: View {
    var height:CGFloat=250
    @Environment(\.colorScheme) private var scheme
    var body:some View {
        Design.image(scheme == .dark ? "hero-blended-dark" : "hero-blended-light")
            .resizable().aspectRatio(contentMode:.fit).frame(maxWidth:.infinity).frame(height:height)
            .accessibilityLabel("Particle")
    }
}
struct ParticleMark:View {
    var size:CGFloat=30
    var body:some View {Design.image("icon-alpha").resizable().renderingMode(.template).foregroundStyle(Design.green).frame(width:size,height:size).accessibilityHidden(true)}
}
struct HitButtonStyle:ButtonStyle {
    func makeBody(configuration:Configuration)->some View {
        configuration.label.padding(.horizontal,7).padding(.vertical,7).frame(minHeight:32).contentShape(Rectangle()).opacity(configuration.isPressed ? 0.6 : 1)
    }
}

extension AppAppearance {
    var colorScheme:ColorScheme? {switch self{case .system:nil;case .light:.light;case .dark:.dark}}
}

/// Actual code-rendered ParticleDesign PNGs, not a live substitute gradient.
struct CodeMaterial:View {
    var panel=false
    @Environment(\.colorScheme) private var scheme
    var body:some View {
        GeometryReader{g in Design.image("code-"+(panel ? "panel-" : "scene-")+(scheme == .dark ? "dark" : "light")).resizable().scaledToFill().frame(width:g.size.width,height:g.size.height).clipped()}.allowsHitTesting(false).accessibilityHidden(true)
    }
}
struct InterfaceGrain:View {
    var body:some View {Design.image("interface-grain").resizable(resizingMode:.tile).allowsHitTesting(false).accessibilityHidden(true)}
}
