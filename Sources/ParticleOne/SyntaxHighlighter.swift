import AppKit
import SwiftUI

/// A bounded lexical highlighter. The original string remains the copy source;
/// code is never executed. Unknown languages use the same readable plain style.
@MainActor enum SyntaxHighlighter {
    static func render(_ code:String,language:String)->NSAttributedString {
        let output=NSMutableAttributedString(string:code,attributes:[.font:NSFont(name:"SpaceMono-Regular",size:12) ?? NSFont.monospacedSystemFont(ofSize:12,weight:.regular),.foregroundColor:NSColor(Color(hex:0xD8E8CE))])
        guard code.utf8.count<=200_000 else{return output}
        let lang=language.lowercased().trimmingCharacters(in:.whitespaces)
        let cFamily=["swift","javascript","js","typescript","ts","tsx","jsx","c","cpp","c++","csharp","cs","java","rust","rs","go","kotlin"]
        let hashFamily=["python","py","bash","sh","shell","zsh","ruby","rb","yaml","yml","toml"]
        guard cFamily.contains(lang) || hashFamily.contains(lang) || ["json","sql","html","xml","css"].contains(lang) else{return output}
        let comments=hashFamily.contains(lang) ? #"#[^\n]*"# : lang=="sql" ? #"--[^\n]*|/\*[\s\S]*?(?:\*/|$)"# : lang=="html" || lang=="xml" ? #"<!--[\s\S]*?(?:-->|$)"# : #"//[^\n]*|/\*[\s\S]*?(?:\*/|$)"#
        let strings = #"\"\"\"[\s\S]*?(?:\"\"\"|$)|\"(?:\\[\s\S]|[^\"\\])*(?:\"|$)|'(?:\\[\s\S]|[^'\\])*(?:'|$)|`(?:\\[\s\S]|[^`\\])*(?:`|$)"#
        let numbers = #"\b(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b"#
        let words = #"\b(?:func|let|var|if|else|return|import|struct|class|enum|protocol|extension|guard|throw|throws|try|await|async|public|private|static|init|self|nil|true|false|def|from|as|in|for|while|with|lambda|None|True|False|and|or|not|pass|yield|except|raise|const|function|export|default|new|switch|case|break|continue|interface|type|fn|mut|use|impl|pub|match|package|SELECT|FROM|WHERE|JOIN|ON|AS|INSERT|INTO|UPDATE|DELETE|CREATE|TABLE|NULL|AND|OR|ORDER|BY|LIMIT)\b"#
        let pattern="("+comments+")|("+strings+")|("+numbers+")|("+words+")"
        guard let regex=try? NSRegularExpression(pattern:pattern) else{return output}
        let colors=[0x82917E,0xC8D98D,0xE7BE8A,0xAFBDD9].map{NSColor(Color(hex:UInt32($0)))}
        for match in regex.matches(in:code,range:NSRange(location:0,length:(code as NSString).length)) {
            for group in 1...4 where match.range(at:group).location != NSNotFound {output.addAttribute(.foregroundColor,value:colors[group-1],range:match.range);break}
        }
        return output
    }
}
