import Foundation
import CryptoKit

/// A generated execution/storage contract. Backend graphs are trusted bundled
/// implementations; model repositories supply data, never executable code.
public struct ParticleGraphPlan:Codable,Equatable,Sendable {
    public struct Assembly:Codable,Equatable,Sendable {
        public let target:SafetensorRecord
        public let sources:[SafetensorRecord]
        public init(target:SafetensorRecord,sources:[SafetensorRecord]) {self.target=target;self.sources=sources}
    }
    public struct Projection:Codable,Equatable,Sendable {
        public let path:String
        public let bank:Int
        public let input:Int
        public let output:Int
        public let experts:Int
        public let bits:Int?
        public let groupSize:Int?
        public let bias:Bool
        public let mode:String
        public var tensors:[String] {[path+".weight"]+(bits == nil ? [] : [path+".scales"]+(mode == "affine" ? [path+".biases"] : []))+(bias ? [path+".bias"] : [])}
        public init(path:String,bank:Int,input:Int,output:Int,experts:Int,bits:Int?,groupSize:Int?,bias:Bool,mode:String="affine") {self.path=path;self.bank=bank;self.input=input;self.output=output;self.experts=experts;self.bits=bits;self.groupSize=groupSize;self.bias=bias;self.mode=mode}
    }
    public let version:Int
    public let backend:String
    public let modelType:String
    public let configurationSHA256:String
    public let tensorsSHA256:String
    public let sourceTensorsSHA256:String
    public let assemblies:[Assembly]
    public let projections:[Projection]
    public let routingFields:[String]
    public let nativeExperts:Int
    public let hiddenLayers:Int
    public let contextLimit:Int
    public let kvBytesPerToken:Int64
    public let cacheTypes:[String]
    public let expertBytesPerSlot:Int64
    public let residentBytes:Int64
    public init(modelType:String,configurationSHA256:String,tensorsSHA256:String,projections:[Projection],routingFields:[String],nativeExperts:Int,hiddenLayers:Int,contextLimit:Int,kvBytesPerToken:Int64,cacheTypes:[String],expertBytesPerSlot:Int64,residentBytes:Int64,sourceTensorsSHA256:String,assemblies:[Assembly]) {
        version=1;backend="mlx-swift-lm/3.32.3:particle-graph-1";self.modelType=modelType;self.configurationSHA256=configurationSHA256;self.tensorsSHA256=tensorsSHA256;self.projections=projections;self.routingFields=routingFields;self.nativeExperts=nativeExperts;self.hiddenLayers=hiddenLayers;self.contextLimit=contextLimit;self.kvBytesPerToken=kvBytesPerToken;self.cacheTypes=cacheTypes;self.expertBytesPerSlot=expertBytesPerSlot;self.residentBytes=residentBytes;self.sourceTensorsSHA256=sourceTensorsSHA256;self.assemblies=assemblies
    }
    public var banks:[Int:Int] {Dictionary(projections.map{($0.bank,$0.experts)},uniquingKeysWith:{old,_ in old})}
    public func bank(for tensor:String)->Int? {projections.first{$0.tensors.contains(tensor)}?.bank}
    public func validate() throws {
        guard version == 1,backend == "mlx-swift-lm/3.32.3:particle-graph-1",!modelType.isEmpty,modelType.utf8.count <= 128,
              configurationSHA256.count == 64,tensorsSHA256.count == 64,configurationSHA256.allSatisfy(\.isHexDigit),tensorsSHA256.allSatisfy(\.isHexDigit),
              !projections.isEmpty,projections.count <= 4096,Set(projections.map(\.path)).count == projections.count,
              banks.count <= 512,Set(banks.keys) == Set(0..<banks.count),(1...64).contains(nativeExperts),
              (1...512).contains(hiddenLayers),(4096...262144).contains(contextLimit),kvBytesPerToken > 0,
              expertBytesPerSlot > 0,residentBytes >= 0,expertBytesPerSlot <= 256*1024*1024*512,residentBytes <= 1_099_511_627_776,
              assemblies.count <= 100_000,sourceTensorsSHA256.count == 64,
              routingFields.count <= 16,routingFields.allSatisfy({$0.range(of:"^[A-Za-z0-9_.]+$",options:.regularExpression) != nil}) else {throw ModelLibraryError.invalid("Invalid generated graph plan.")}
        for p in projections {
            guard p.path.range(of:"^[A-Za-z0-9_]+(?:\\.[A-Za-z0-9_]+)+$",options:.regularExpression) != nil,
                  (1...65536).contains(p.input),(1...65536).contains(p.output),(2...4096).contains(p.experts),p.experts >= nativeExperts,
                  p.experts == banks[p.bank],p.bits == nil ? p.groupSize == nil : ([2,3,4,5,6,8].contains(p.bits!) && [16,32,64,128].contains(p.groupSize ?? 0) && ["affine","mxfp4","mxfp8","nvfp4"].contains(p.mode)) else {throw ModelLibraryError.invalid("Invalid streamed projection binding.")}
        }
    }
    public func canonicalTensors(_ original:[SafetensorRecord])->[SafetensorRecord] {
        var values=Dictionary(uniqueKeysWithValues:original.map{($0.name,$0)})
        for assembly in assemblies {for source in assembly.sources {values[source.name]=nil}}
        for assembly in assemblies {values[assembly.target.name]=assembly.target}
        return Array(values.values)
    }
    public func originalTensors(_ canonical:[SafetensorRecord])->[SafetensorRecord] {
        var values=Dictionary(uniqueKeysWithValues:canonical.map{($0.name,$0)})
        for assembly in assemblies {values[assembly.target.name]=nil}
        for assembly in assemblies {for source in assembly.sources {values[source.name]=source}}
        return Array(values.values)
    }
    public static func digest(_ data:Data)->String {SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()}
    public static func tensorDigest(_ tensors:[SafetensorRecord]) throws -> String {
        struct Header:Codable {let name:String;let dtype:String;let shape:[Int];let bytes:Int64}
        let headers=tensors.sorted{$0.name < $1.name}.map{Header(name:$0.name,dtype:$0.dtype,shape:$0.shape,bytes:$0.bytes)}
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys];return digest(try encoder.encode(headers))
    }
}
public struct ParticleGraphRequest:Codable,Sendable {
    public let configuration:Data
    public let tensors:[SafetensorRecord]
    public init(configuration:Data,tensors:[SafetensorRecord]) {self.configuration=configuration;self.tensors=tensors}
}
