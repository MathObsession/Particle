import AppKit
import SwiftUI

/// Return sends; Shift-Return inserts a newline. Marked text is left to the
/// input method so confirming Hindi/Japanese/etc. composition never sends.
struct ComposerInput:NSViewRepresentable {
    @Binding var text:String
    var submit:()->Void
    func makeCoordinator()->Coordinator {Coordinator(self)}
    func makeNSView(context:Context)->NSScrollView {
        let scroll=NSScrollView();scroll.drawsBackground=false;scroll.hasVerticalScroller=true
        let editor=SendTextView();editor.isRichText=false;editor.drawsBackground=false;editor.font=NSFont(name:"Poppins-Regular",size:14)
        editor.isAutomaticQuoteSubstitutionEnabled=false;editor.isAutomaticDashSubstitutionEnabled=false
        editor.isAutomaticTextReplacementEnabled=false;editor.textContainerInset=NSSize(width:4,height:7)
        editor.isVerticallyResizable=true;editor.isHorizontallyResizable=false;editor.autoresizingMask=[.width]
        editor.textContainer?.widthTracksTextView=true;editor.delegate=context.coordinator;editor.onSend={context.coordinator.parent.submit()}
        editor.setAccessibilityLabel("Message to Particle");scroll.documentView=editor
        return scroll
    }
    func updateNSView(_ scroll:NSScrollView,context:Context){
        context.coordinator.parent=self
        guard let editor=scroll.documentView as? SendTextView else{return}
        if editor.string != text {editor.string=text;editor.setSelectedRange(NSRange(location:(text as NSString).length,length:0))}
        editor.textColor=NSColor(Design.ink);editor.insertionPointColor=NSColor(Design.green)
    }
    final class Coordinator:NSObject,NSTextViewDelegate {
        var parent:ComposerInput
        init(_ parent:ComposerInput){self.parent=parent}
        func textDidChange(_ notification:Notification){if let editor=notification.object as? NSTextView {parent.text=editor.string}}
    }
}
final class SendTextView:NSTextView {
    var onSend:(()->Void)?
    override func keyDown(with event:NSEvent){
        if (event.keyCode==36 || event.keyCode==76),!event.modifierFlags.contains(.shift),!hasMarkedText(){onSend?();return}
        super.keyDown(with:event)
    }
}
