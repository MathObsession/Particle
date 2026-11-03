import Foundation
import ParticleModelLibrary

/// Qwen's qualified tokenizer uses GPT-2 byte-level spellings. Decode complete
/// UTF-8 bytes, rather than treating a replacement character as an incomplete token.
struct ChronosByteDecoder {
    private(set) var bytes = Data()
    static let alphabet: [UnicodeScalar:UInt8] = {
        let visible = Array(33...126)+Array(161...172)+Array(174...255)
        let existing = Set(visible);var result = [UnicodeScalar:UInt8]();var extra = 256
        for byte in 0...255 {
            let scalar = existing.contains(byte) ? byte : extra
            result[UnicodeScalar(scalar)!] = UInt8(byte)
            if !existing.contains(byte) {extra += 1}
        }
        return result
    }()
    mutating func append(_ spelling: String) throws -> String? {
        for scalar in spelling.unicodeScalars {
            guard let byte = Self.alphabet[scalar] else {throw ModelLibraryError.incompatible("The tokenizer uses an unqualified byte-decoder alphabet.")}
            bytes.append(byte)
        }
        return String(data:bytes,encoding:.utf8)
    }
    static func suffix(_ text: String, emittedBytes: Int) throws -> String {
        guard emittedBytes <= text.utf8.count else {throw ModelLibraryError.invalid("The decoded stream moved backward.")}
        return String(decoding:text.utf8.dropFirst(emittedBytes),as:UTF8.self)
    }
}
