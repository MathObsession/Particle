import AppKit
import SwiftUI
import TurboFieldfareMacPresentation

/// Reuse the upstream full Markdown renderer (lists, tables, quotes and links),
/// then apply ParticleOne typography. Remote images remain text; code is inert.
@MainActor enum ParticleMarkdown {
    static func render(_ source: String,compact: Bool = false) -> NSAttributedString {
        let rendered=ResponseMarkdownRenderer(environment:["TURBO_FIELDFARE_DISABLE_MATH":"1"]).render(source,typesetsMath:false).attributedString
        let output=NSMutableAttributedString(attributedString:rendered)
        let full=NSRange(location:0,length:output.length)
        output.addAttribute(.foregroundColor,value:NSColor(compact ? Design.muted : Design.ink),range:full)
        rendered.enumerateAttribute(.font,in:full) { value,range,_ in
            guard let original=value as? NSFont else{return}
            let scale:CGFloat=compact ? 11.0/13.0 : 1
            let size=original.pointSize*scale
            let traits=original.fontDescriptor.symbolicTraits
            let heading=original.pointSize>NSFont.systemFontSize+0.5
            let name=original.isFixedPitch ? "SpaceMono-Regular" : heading ? "PTSerif-Regular" : traits.contains(.bold) ? "Poppins-SemiBold" : "Poppins-Regular"
            var font=NSFont(name:name,size:size) ?? NSFont.systemFont(ofSize:size)
            if traits.contains(.italic) {font=NSFontManager.shared.convert(font,toHaveTrait:.italicFontMask)}
            output.addAttribute(.font,value:font,range:range)
            if original.isFixedPitch {output.addAttribute(.backgroundColor,value:NSColor(Design.sidebar),range:range)}
        }
        return output
    }
}

struct MarkdownContent: NSViewRepresentable {
    let source: String
    var compact=false
    @Environment(\.colorScheme) private var colorScheme
    func makeNSView(context: Context) -> NSTextView {
        let view=NSTextView(frame:.zero)
        view.isEditable=false;view.isSelectable=true;view.drawsBackground=false
        view.isRichText=true;view.isAutomaticLinkDetectionEnabled=false
        view.textContainerInset = .zero;view.textContainer?.lineFragmentPadding=0
        view.isHorizontallyResizable=false;view.isVerticallyResizable=true
        view.textContainer?.widthTracksTextView=true;view.textContainer?.heightTracksTextView=false
        view.linkTextAttributes=[.foregroundColor:NSColor(Design.green),.underlineStyle:NSUnderlineStyle.single.rawValue]
        view.setContentHuggingPriority(.defaultHigh,for:.vertical)
        view.setContentCompressionResistancePriority(.defaultLow,for:.horizontal)
        return view
    }
    func updateNSView(_ view: NSTextView,context: Context) {
        let text=ParticleMarkdown.render(source.trimmingCharacters(in:.whitespacesAndNewlines),compact:compact)
        if view.attributedString() != text {view.textStorage?.setAttributedString(text)}
    }
    func sizeThatFits(_ proposal: ProposedViewSize,nsView view: NSTextView,context: Context) -> CGSize? {
        let width=max(1,proposal.width ?? 720)
        guard let container=view.textContainer,let layout=view.layoutManager else{return nil}
        container.containerSize=NSSize(width:width,height:1_000_000)
        layout.ensureLayout(for:container)
        return CGSize(width:width,height:ceil(layout.usedRect(for:container).height)+2)
    }
}
