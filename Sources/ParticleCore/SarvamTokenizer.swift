import Foundation
import CryptoKit
import Tokenizers
import Hub
import ParticleGGUF
import ParticleDesktopProtocol

public final class SarvamTokenizer: @unchecked Sendable {
    private let tokenizer: any Tokenizer
    private let template: String
    public let supportsIncrementalChat: Bool
    public let stopIDs: Set<Int> = [1,26]
    public static func load(folder: URL, vocabulary: [String]? = nil) async throws -> SarvamTokenizer {
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
            guard FileManager.default.fileExists(atPath: folder.appendingPathComponent(name).path) else {
                throw ParticleError.invalid("Missing \(name) in \(folder.path). Run Scripts/fetch_sarvam_tokenizer.py or pass --tokenizer.")
            }
        }
        // Hub's binary-distinct JSON parser preserves canonically equivalent
        // token spellings that Foundation JSONDecoder can collapse as keys.
        let local = LanguageModelConfigurationFromHub(modelFolder:folder)
        let source = try await local.tokenizerData
        guard var data = source.dictionary(), source["normalizer"]["type"].string() == "Replace",
              source["normalizer"]["pattern"]["String"].string() == " ",
              source["normalizer"]["content"].string() == "▁" else {
            throw ParticleError.invalid("Unsupported Sarvam tokenizer normalization")
        }
        // The dependency's literal String replacement works on grapheme
        // boundaries. Its regex replacement works on scalar boundaries,
        // matching Rust tokenizers even for a space plus combining mark.
        data["normalizer"] = ["type":"Replace","pattern":["Regex":" "],"content":"▁"]
        guard let config = try await local.tokenizerConfig else { throw ParticleError.invalid("Missing tokenizer configuration") }
        let tokenizer = try AutoTokenizer.from(tokenizerConfig:config,tokenizerData:Config(data))
        let template = try String(contentsOf: folder.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        for (token, id) in [("<pad>",0),("<eos>",1),("[@BOS@]",6),("<|user|>",18),("<|assistant|>",19),
                            ("<think>",23),("</think>",24),("<|start_of_turn|>",25),("<|end_of_turn|>",26),("<|nothink|>",28)] {
            guard tokenizer.convertTokenToId(token) == id, tokenizer.convertIdToToken(id) == token else {
                throw ParticleError.invalid("Sarvam tokenizer special-token mismatch: \(token)")
            }
        }
        if let vocabulary {
            guard vocabulary.count == 262144 else { throw ParticleError.invalid("Unsupported vocabulary size") }
            let encoding = byteEncoding()
            for (id, expected) in vocabulary.enumerated() {
                guard let token = tokenizer.convertIdToToken(id) else { throw ParticleError.invalid("Tokenizer missing ID \(id)") }
                // This GGUF stores Sarvam's metaspace vocabulary in GPT-2's
                // byte-to-Unicode representation. IDs, not spellings, must agree.
                let bytes: [UInt8]
                if let byte = byteFallback(token) { bytes = [byte] }
                else { bytes = Array(decodeMetaspace(token).utf8) }
                let encoded = bytes.map { encoding[Int($0)] }.joined()
                let literal = decodeMetaspace(token)
                // GGUF user-defined whitespace and byte-fallback tokens retain
                // literal spellings, while normal BPE tokens are byte-encoded.
                guard encoded.utf8.elementsEqual(expected.utf8) || token.utf8.elementsEqual(expected.utf8)
                        || literal.utf8.elementsEqual(expected.utf8) else {
                    throw ParticleError.invalid("Tokenizer/GGUF vocabulary mismatch at ID \(id): token=\(token.debugDescription), encoded=\(encoded.debugDescription), GGUF=\(expected.debugDescription)")
                }
            }
        }
        let pinned = [
            "tokenizer.json":"a574ceaaff7c7a8f091179c53fd17ae33567089c099d4ff37d4cb3bc1a87e80e",
            "tokenizer_config.json":"3cdb41a3bcbc8a7e14dbcf8987ccad0cbfe8f82091f2a2c82020f1d0e219cdd5",
            "chat_template.jinja":"07a8ca6f96384a6878c9aa6d4e0616ae57f5937fa8d0ff014855704fa26a0ef4",
        ]
        let verified = try pinned.allSatisfy { try Self.digest(folder.appendingPathComponent($0.key)) == $0.value }
        return SarvamTokenizer(tokenizer:tokenizer,template:template,supportsIncrementalChat:verified)
    }
    private init(tokenizer: any Tokenizer, template: String, supportsIncrementalChat: Bool) {
        self.tokenizer=tokenizer; self.template=template; self.supportsIncrementalChat=supportsIncrementalChat
    }
    private static func digest(_ url: URL) throws -> String {
        let file=try FileHandle(forReadingFrom:url); defer { try? file.close() }
        var digest=SHA256()
        while true {
            let count=try autoreleasepool {
                let bytes=try file.read(upToCount:1024*1024) ?? Data()
                digest.update(data:bytes); return bytes.count
            }
            if count==0 { break }
        }
        return digest.finalize().map { String(format:"%02x",$0) }.joined()
    }
    /// The caller supplies the canonical historical prefix for this exact
    /// conversation. Only pinned assets permit token concatenation: added-token
    /// boundaries prevent BPE from merging across the new turn marker.
    /// Unknown template/tokenizer revisions return nil for full-render fallback.
    public func continuingChat(historicalPrefix: [Int], prompt: String, thinking: Bool = false) throws -> [Int]? {
        guard supportsIncrementalChat, historicalPrefix.starts(with:[6,107]),
              historicalPrefix.suffix(2).elementsEqual([26,107]) else { return nil }
        let turn=try chat(prompt:prompt,thinking:thinking)
        guard turn.starts(with:[6,107,25,18]) else { return nil }
        return historicalPrefix+turn.dropFirst(2)
    }
    public func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }
    public func chat(prompt: String, system: String? = nil, thinking: Bool = false) throws -> [Int] {
        var messages: [SarvamMessage] = []
        if let system { messages.append(SarvamMessage(role:.system,content:system)) }
        messages.append(SarvamMessage(role:.user,content:prompt))
        return try chat(messages:messages,thinking:thinking)
    }
    public func chat(messages: [SarvamMessage], thinking: Bool = false) throws -> [Int] {
        let rendered: [Message] = messages.map { ["role":$0.role.rawValue,"content":$0.content] }
        return try tokenizer.applyChatTemplate(messages:rendered,chatTemplate:.literal(template),addGenerationPrompt:true,
                                               truncation:false,maxLength:nil,tools:nil,additionalContext:["enable_thinking":thinking])
    }
    public func agentChat(messages:[SarvamMessage],tools:[DesktopTool],thinking:Bool=false)throws->[Int]{
        let rendered:[Message]=messages.map{["role":$0.role.rawValue,"content":$0.content]}
        let specs:[ToolSpec]=tools.map { tool in
            let properties:[String:[String:String]]=tool.properties.mapValues{["type":"string","description":$0]}
            let parameters:[String:any Sendable]=["type":"object","properties":properties,"required":tool.required,"additionalProperties":false]
            let function:[String:any Sendable]=["name":tool.name,"description":tool.description,"parameters":parameters]
            return ["type":"function","function":function]
        }
        return try tokenizer.applyChatTemplate(messages:rendered,chatTemplate:.literal(template),addGenerationPrompt:true,truncation:false,maxLength:nil,tools:specs,additionalContext:["enable_thinking":thinking])
    }
    public func historicalPrefix(messages: [SarvamMessage], thinking: Bool = false) throws -> [Int] {
        let boundary = "{% for m in messages %}\n{%- if m.role == 'user' -%}"
        guard template.components(separatedBy:boundary).count == 2 else {
            throw ParticleError.invalid("Unsupported official template boundary; idle preparation skipped")
        }
        let historical = template.replacingOccurrences(of:boundary,with:
            "{%- set ns.last_user_index = messages|length %}\n" + boundary)
        let rendered: [Message] = messages.map { ["role":$0.role.rawValue,"content":$0.content] }
        return try tokenizer.applyChatTemplate(messages:rendered,chatTemplate:.literal(historical),addGenerationPrompt:false,
            truncation:false,maxLength:nil,tools:nil,additionalContext:["enable_thinking":thinking])
    }
    /// Canonicalize only the newly generated assistant. The supplied prompt
    /// already contains canonical older turns and the final assistant header.
    /// Pinned special-token boundaries make this identical to rendering the
    /// entire history with a future user, without re-tokenizing earlier turns.
    public func completedChatPrefix(promptTokens: [Int], assistant: String, thinking: Bool = false) throws -> [Int]? {
        guard supportsIncrementalChat, promptTokens.starts(with:[6,107]),
              promptTokens.suffix(3).elementsEqual([25,19,107]) else { return nil }
        let reply=try historicalPrefix(messages:[SarvamMessage(role:.assistant,content:assistant)],thinking:thinking)
        guard reply.starts(with:[6,107,25,19,107]) else { return nil }
        return Array(promptTokens.dropLast(3))+reply.dropFirst(2)
    }
    public func token(_ id: Int) throws -> String {
        guard let token = tokenizer.convertIdToToken(id) else { throw ParticleError.invalid("Unknown output token ID \(id)") }
        return token
    }
    public func decode(_ ids: [Int]) throws -> String {
        var decoder = SarvamTextDecoder(), text = ""
        for id in ids { text += decoder.append(try token(id)) }
        return text + decoder.finish()
    }
}

private func byteEncoding() -> [String] {
    var table = [String](repeating: "", count: 256), extra = 0
    for b in 0..<256 {
        if (33...126).contains(b) || (161...172).contains(b) || (174...255).contains(b) {
            table[b] = String(UnicodeScalar(b)!)
        } else { table[b] = String(UnicodeScalar(256 + extra)!); extra += 1 }
    }
    return table
}
private func byteFallback(_ token: String) -> UInt8? {
    guard token.utf8.count == 6, token.hasPrefix("<0x"), token.hasSuffix(">") else { return nil }
    return UInt8(token.dropFirst(3).dropLast(), radix: 16)
}

// Replace Unicode scalars rather than Characters: a following combining mark
// can join the metaspace into one grapheme and defeat Foundation replacement.
private func decodeMetaspace(_ text: String) -> String {
    String(String.UnicodeScalarView(text.unicodeScalars.map { $0.value == 0x2581 ? UnicodeScalar(32)! : $0 }))
}

/// Commit byte-fallback runs as units. A later byte may invalidate a complete
/// run, so delaying its emission prevents rewriting already streamed output.
public struct SarvamTextDecoder {
    private var bytes: [UInt8] = []
    public init() {}
    public mutating func append(_ token: String) -> String {
        if let byte = byteFallback(token) { bytes.append(byte); return "" }
        return finish() + decodeMetaspace(token)
    }
    public mutating func finish() -> String {
        defer { bytes.removeAll(keepingCapacity: true) }
        return String(bytes: bytes, encoding: .utf8) ?? String(repeating: "\u{FFFD}", count: bytes.count)
    }
}

public struct ParticleSampler {
    private var state: UInt64
    public init(seed: UInt64 = 42) { state = seed }
    static func topCandidates(_ logits: [Float], count: Int) -> [Int] {
        func better(_ a: Int, _ b: Int) -> Bool { logits[a] == logits[b] ? a < b : logits[a] > logits[b] }
        var heap: [Int] = []; heap.reserveCapacity(count)
        for id in logits.indices {
            if heap.count < count {
                heap.append(id); var i = heap.count-1
                while i > 0 {
                    let parent = (i-1)/2
                    if !better(heap[parent],heap[i]) { break }
                    heap.swapAt(parent,i); i = parent
                }
            } else if better(id,heap[0]) {
                heap[0] = id; var i = 0
                while 2*i+1 < heap.count {
                    var child = 2*i+1
                    if child+1 < heap.count && better(heap[child],heap[child+1]) { child += 1 }
                    if !better(heap[i],heap[child]) { break }
                    heap.swapAt(i,child); i = child
                }
            }
        }
        return heap.sorted(by:better)
    }
    public mutating func sample(_ logits: [Float], temperature: Float, topK: Int, topP: Float) throws -> Int {
        guard !logits.isEmpty, logits.allSatisfy(\.isFinite), temperature.isFinite, temperature >= 0,
              topK > 0, topK <= logits.count, topP.isFinite, topP > 0, topP <= 1 else {
            throw ParticleError.invalid("Invalid sampling configuration or logits")
        }
        if temperature == 0 {
            return logits.indices.max { logits[$0] == logits[$1] ? $0 > $1 : logits[$0] < logits[$1] }!
        }
        let ranked = Self.topCandidates(logits,count:topK)
        let maximum = logits[ranked.first!]
        let candidates = ranked.map { ($0, exp(Double((logits[$0] - maximum) / temperature))) }
        let total = candidates.reduce(0.0) { $0 + $1.1 }
        var kept: [(Int, Double)] = [], cumulative = 0.0
        for candidate in candidates {
            kept.append(candidate); cumulative += candidate.1
            if cumulative / total >= Double(topP) { break }
        }
        state &+= 0x9E3779B97F4A7C15
        var z = state; z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) &* 0x94D049BB133111EB; z ^= z >> 31
        let threshold = Double(z >> 11) / 9007199254740992.0 * cumulative
        var partial = 0.0
        for (id, probability) in kept { partial += probability; if threshold < partial { return id } }
        return kept.last!.0
    }
}
