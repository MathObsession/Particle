import Foundation
import CryptoKit

/// Newline-delimited JSON only on the opt-in desktop transport. Never carries secrets.
public struct DesktopMessage: Codable, Sendable, Equatable {
    public let role: String
    public let content: String
    public init(role: String, content: String) { self.role=role; self.content=content }
}
public struct DesktopCommand: Codable, Sendable {
    public var tools: [DesktopTool]?
    public var type: String
    public var text: String?
    public var messages: [DesktopMessage]?
    public var system: String?
    public var reasoningBudget: Int?
    public init(_ type: String, text: String? = nil, messages: [DesktopMessage]? = nil, system: String? = nil, reasoningBudget: Int? = nil, tools: [DesktopTool]? = nil) {
        self.tools=tools; self.type=type; self.text=text; self.messages=messages; self.system=system; self.reasoningBudget=reasoningBudget
    }
    public func validatedHistory() throws -> [DesktopMessage] {
        let history=messages ?? []
        guard reasoningBudget.map({(-1...4096).contains($0)}) ?? true, (system?.utf8.count ?? 0)<=65_536, history.count <= 2048, history.count.isMultiple(of:2),
              history.enumerated().allSatisfy({ $0.element.role == ($0.offset.isMultiple(of:2) ? "user" : "assistant") }),
              history.reduce(0, { $0+$1.content.utf8.count }) <= 1_048_576 else {
            throw CocoaError(.coderReadCorrupt)
        }
        return history
    }
    public func validatedAgentHistory() throws -> [DesktopMessage] {
        let history=messages ?? []
        guard !history.isEmpty,history.count<=256,history.first?.role=="user",["user","tool"].contains(history.last?.role ?? ""),
              history.allSatisfy({["user","assistant","tool"].contains($0.role)}),
              history.reduce(0,{$0+$1.content.utf8.count})<=900_000,(system?.utf8.count ?? 0)<=65_536,
              let tools,!tools.isEmpty,tools.count<=16,Set(tools.map(\.name)).count==tools.count,
              tools.allSatisfy({$0.valid}) else{throw CocoaError(.coderReadCorrupt)}
        for (index,message) in history.enumerated() where message.role=="tool" {
            guard index>0,["assistant","tool"].contains(history[index-1].role) else{throw CocoaError(.coderReadCorrupt)}
        }
        return history
    }
    public func line() throws -> Data { try JSONEncoder().encode(self)+Data([10]) }
}
/// The stream decoder retains incomplete UTF-8 and JSON until a complete line arrives.
public struct DesktopLineDecoder: Sendable {
    private var pending=Data()
    public init() {}
    public mutating func append(_ data: Data) throws -> [Data] {
        pending.append(data)
        var lines=[Data]()
        while let end=pending.firstIndex(of:10) {
            guard end <= 2_097_152 else { throw CocoaError(.coderReadCorrupt) }
            let line=Data(pending[..<end]); pending.removeSubrange(...end)
            if !line.isEmpty { lines.append(line) }
        }
        guard pending.count <= 2_097_152 else { throw CocoaError(.coderReadCorrupt) }
        return lines
    }
}

public enum DesktopTextIdentity {
    /// Only a digest and byte count cross the acknowledgment channel. User
    /// content is never added to the diagnostic log.
    public static func fingerprint(_ text: String) -> String {
        SHA256.hash(data:Data(text.utf8)).map {String(format:"%02x",$0)}.joined()
    }
}

/// A real decode constraint: once a thinking span reaches the allowed token
/// budget, the caller feeds the known closing token through the model itself.
/// Every injected token must still be forwarded and included in KV lineage.
public struct DesktopReasoningLimiter: Sendable {
    public let budget: Int
    public private(set) var thinking=false
    public private(set) var used=0
    public private(set) var closures=0
    public init(budget: Int){self.budget=budget}
    public var mustClose: Bool {budget>=0 && thinking && used>=budget}
    public var exhausted: Bool {closures>=3 && mustClose}
    public mutating func accept(_ token: Int) {
        if token==23 {thinking=true}
        else if token==24 {if thinking {closures+=1};thinking=false}
        else if thinking {used+=1}
    }
}

public struct DesktopTool:Codable,Sendable,Equatable {
    public let name:String
    public let description:String
    public let properties:[String:String]
    public let required:[String]
    public init(_ name:String,_ description:String,_ properties:[String:String],required:[String]){self.name=name;self.description=description;self.properties=properties;self.required=required}
    public var valid:Bool {name.range(of:"^[a-z_]{1,40}$",options:.regularExpression) != nil && description.utf8.count<=600 && properties.count<=8 && properties.allSatisfy({$0.key.range(of:"^[a-z_]{1,40}$",options:.regularExpression) != nil && $0.value.utf8.count<=500}) && Set(required).isSubset(of:Set(properties.keys))}
}
