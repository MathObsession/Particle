import Foundation
import Testing
@testable import ParticleMLXRuntime

@Suite("Chronos UTF-8 streaming") struct ChronosTextTests {
    private func spelling(_ bytes:[UInt8]) -> String {
        let inverse = Dictionary(uniqueKeysWithValues:ChronosByteDecoder.alphabet.map{($0.value,$0.key)})
        return bytes.map{String(inverse[$0]!)}.joined()
    }
    @Test func incompleteEmojiWaitsForWholeUTF8Sequence() throws {
        var decoder = ChronosByteDecoder()
        let bytes = Array("🙂".utf8)
        #expect(try decoder.append(spelling(Array(bytes.prefix(2)))) == nil)
        #expect(try decoder.append(spelling(Array(bytes.dropFirst(2)))) == "🙂")
    }
    @Test func combiningMarkIsAnAppendRatherThanAGraphemeReplacement() throws {
        var decoder = ChronosByteDecoder()
        #expect(try decoder.append("a") == "a")
        let decoded = try decoder.append(spelling(Array("\u{301}".utf8)))
        let text = try #require(decoded)
        #expect(try ChronosByteDecoder.suffix(text,emittedBytes:1).utf8.elementsEqual("\u{301}".utf8))
    }
    @Test func literalReplacementCharacterIsValidText() throws {
        var decoder = ChronosByteDecoder()
        #expect(try decoder.append(spelling(Array("�".utf8))) == "�")
    }
    @Test func newlinesSpacesAndThinkingMarkersAreByteExact() throws {
        var decoder = ChronosByteDecoder()
        let expected = "<think>\n\n</think>\n\nHello! नमस्ते"
        #expect(try decoder.append(spelling(Array(expected.utf8)))?.utf8.elementsEqual(expected.utf8) == true)
    }
}
