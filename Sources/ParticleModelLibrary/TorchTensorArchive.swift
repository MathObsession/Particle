import Foundation
import CryptoKit

/// Data-only reader for torch.save's uncompressed ZIP64 state dictionaries.
/// GLOBAL/REDUCE are recognized as inert metadata symbols, never invoked. No
/// Python, pickle execution, decompression or arbitrary object construction.
public enum TorchTensorArchive {
    private struct Member:Sendable {let name:String,offset:Int64,bytes:Int64}
    public static func records(source:any ModelByteSource,file:HubFile) async throws -> [String:SafetensorRecord] {
        let cacheRoot=FileManager.default.temporaryDirectory.appendingPathComponent("particle-torch-metadata-v1",isDirectory:true)
        let key=SHA256.hash(data:Data((source.identity+"|"+file.path+"|"+String(file.bytes)+"|"+(file.sha256 ?? "")).utf8)).map{String(format:"%02x",$0)}.joined()
        let cached=cacheRoot.appendingPathComponent(key+".json")
        if let data=try? ArchitectureRegistry.boundedData(cached,limit:8*1024*1024),let records=try? JSONDecoder().decode([String:SafetensorRecord].self,from:data),!records.isEmpty,records.count<=100_000,
           records.allSatisfy({name,record in name == record.name && name.utf8.count<=1024 && record.offset>=0 && record.bytes>0 && record.offset<=file.bytes && record.bytes<=file.bytes-record.offset && (1...8).contains(record.shape.count)}) {return records}
        let read:@Sendable (Int64,Int) async throws -> Data = {offset,count in
            guard offset>=0,count>0,offset<=file.bytes,Int64(count)<=file.bytes-offset,count<=16*1024*1024 else {throw invalid()}
            return try await source.read(file.path,range:offset..<offset+Int64(count))
        }
        let u:@Sendable (Data,Int,Int) throws -> Int64 = {data,offset,count in
            guard offset>=0,offset+count<=data.count,count<=8 else {throw invalid()}
            let value=(0..<count).reduce(UInt64(0)){$0 | UInt64(data[offset+$1]) << (8*$1)}
            guard value<=UInt64(Int64.max) else {throw invalid()};return Int64(value)
        }
        let tailStart=max(0,file.bytes-65557),tail=try await read(tailStart,Int(file.bytes-tailStart))
        guard let end=(0..<max(0,tail.count-21)).reversed().first(where:{i in (try? u(tail,i,4)) == 0x06054b50 && (try? u(tail,i+20,2)) == Int64(tail.count-i-22)}) else {throw invalid()}
        var centralSize=try u(tail,end+12,4),centralOffset=try u(tail,end+16,4)
        if centralSize == 0xffffffff || centralOffset == 0xffffffff {
            let locator=try await read(tailStart+Int64(end)-20,20)
            guard try u(locator,0,4) == 0x07064b50 else {throw invalid()}
            let zip64=try await read(u(locator,8,8),56)
            guard try u(zip64,0,4) == 0x06064b50 else {throw invalid()}
            centralSize=try u(zip64,40,8);centralOffset=try u(zip64,48,8)
        }
        guard centralSize>0,centralSize<=16*1024*1024 else {throw invalid()}
        let central=try await read(centralOffset,Int(centralSize));var members=[String:Member](),position=0
        while position<central.count {
            guard members.count<100_000,try u(central,position,4) == 0x02014b50,try u(central,position+10,2) == 0,try u(central,position+8,2)&1 == 0 else {throw invalid()}
            let nameLength=Int(try u(central,position+28,2)),extraLength=Int(try u(central,position+30,2)),commentLength=Int(try u(central,position+32,2)),start=position+46
            guard start+nameLength+extraLength+commentLength<=central.count,let name=String(data:central.subdata(in:start..<start+nameLength),encoding:.utf8),!name.contains(".."),!name.hasPrefix("/"),members[name] == nil else {throw invalid()}
            var size=try u(central,position+24,4),compressed=try u(central,position+20,4),offset=try u(central,position+42,4)
            var extra=start+nameLength
            while extra<start+nameLength+extraLength {
                let kind=try u(central,extra,2),length=Int(try u(central,extra+2,2));var cursor=extra+4
                guard cursor+length<=start+nameLength+extraLength else {throw invalid()}
                if kind == 1 {
                    if size == 0xffffffff {size=try u(central,cursor,8);cursor+=8}
                    if compressed == 0xffffffff {compressed=try u(central,cursor,8);cursor+=8}
                    if offset == 0xffffffff {offset=try u(central,cursor,8)}
                }
                extra+=4+length
            }
            guard compressed == size,offset>=0,size>=0,offset<file.bytes else {throw invalid()}
            members[name]=Member(name:name,offset:offset,bytes:size);position=start+nameLength+extraLength+commentLength
        }
        guard let pickle=members.values.first(where:{$0.name.hasSuffix("/data.pkl")}),pickle.bytes<=16*1024*1024 else {throw invalid()}
        let prefix=String(pickle.name.dropLast("data.pkl".count))
        let payload:@Sendable (Member) async throws -> Int64 = {member in
            let local=try await read(member.offset,30)
            guard try u(local,0,4) == 0x04034b50,try u(local,8,2) == 0 else {throw invalid()}
            let offset=member.offset+30+(try u(local,26,2))+(try u(local,28,2))
            guard offset<=file.bytes,member.bytes<=file.bytes-offset else {throw invalid()};return offset
        }
        if let byteOrder=members[prefix+"byteorder"] {
            let data=try await read(payload(byteOrder),Int(byteOrder.bytes))
            guard String(data:data,encoding:.utf8) == "little" else {throw ModelLibraryError.incompatible("Only little-endian PyTorch tensor archives are supported.")}
        }
        let data=try await read(payload(pickle),Int(pickle.bytes)),tensors=try PickleMetadata(data).decode()
        var result=[String:SafetensorRecord](),offsets=[String:Int64]()
        let keys=Array(Set(tensors.values.map(\.key))).sorted()
        let storageMembers=try keys.map{key -> (String,Member) in guard let member=members[prefix+"data/"+key] else {throw invalid()};return (key,member)}
        try await withThrowingTaskGroup(of:(String,Int64).self) {group in
            var next=0
            for _ in 0..<min(8,storageMembers.count) {
                let (key,member)=storageMembers[next];next+=1;group.addTask{(key,try await payload(member))}
            }
            while let (key,offset)=try await group.next() {
                offsets[key]=offset
                if next<storageMembers.count {let (key,member)=storageMembers[next];next+=1;group.addTask{(key,try await payload(member))}}
            }
        }
        for (name,tensor) in tensors {
            guard let storage=members[prefix+"data/"+tensor.key],tensor.shape.count>0,tensor.shape.count<=8 else {throw invalid()}
            if offsets[tensor.key] == nil {offsets[tensor.key]=try await payload(storage)}
            var elements:Int64=1
            for dimension in tensor.shape.reversed() {
                guard dimension>0,dimension<=Int(Int32.max),elements<=Int64.max/Int64(dimension) else {throw invalid()};elements*=Int64(dimension)
            }
            var contiguous:Int64=1
            for (dim,stride) in zip(tensor.shape,tensor.strides).reversed() {guard stride == contiguous else {throw ModelLibraryError.incompatible("A PyTorch tensor is not contiguous: \(name). Export safetensors first.")};contiguous*=Int64(dim)}
            let width=Int64(tensor.width)
            guard tensor.offset>=0,tensor.offset<=storage.bytes/width,elements<=storage.bytes/width-tensor.offset else {throw invalid()}
            result[name] = .init(name:name,dtype:tensor.dtype,shape:tensor.shape,offset:offsets[tensor.key]!+tensor.offset*width,bytes:elements*width)
        }
        guard !result.isEmpty else {throw invalid()}
        // This cache contains headers only, never weight bytes or credentials.
        // Its immutable source identity prevents cross-revision reuse.
        if let data=try? JSONEncoder().encode(result),data.count<=8*1024*1024 {
            try? FileManager.default.createDirectory(at:cacheRoot,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
            try? data.write(to:cached,options:.atomic)
            let files=(try? FileManager.default.contentsOfDirectory(at:cacheRoot,includingPropertiesForKeys:[.contentModificationDateKey,.fileSizeKey])) ?? []
            var kept:Int64=0
            for file in files.sorted(by:{((try? $0.resourceValues(forKeys:[.contentModificationDateKey]).contentModificationDate) ?? .distantPast)>((try? $1.resourceValues(forKeys:[.contentModificationDateKey]).contentModificationDate) ?? .distantPast)}) {
                let bytes=Int64((try? file.resourceValues(forKeys:[.fileSizeKey]).fileSize) ?? 0)
                if bytes>32*1024*1024-kept {try? FileManager.default.removeItem(at:file)} else {kept+=bytes}
            }
        }
        return result
    }
    private final class LocalResult:@unchecked Sendable {
        let condition=NSCondition();var result:Result<[String:SafetensorRecord],Error>?
        func complete(_ value:Result<[String:SafetensorRecord],Error>) {condition.lock();result=value;condition.broadcast();condition.unlock()}
        func wait()throws->[String:SafetensorRecord] {condition.lock();defer{condition.unlock()};while result == nil {condition.wait()};return try result!.get()}
    }
    public static func localRecords(_ source:LocalModelByteSource,file:HubFile)throws->[String:SafetensorRecord] {
        let box=LocalResult()
        Task.detached {do {box.complete(.success(try await records(source:source,file:file)))} catch {box.complete(.failure(error))}}
        return try box.wait()
    }
    private static func invalid()->ModelLibraryError {.invalid("Unsupported or corrupt data-only PyTorch tensor archive. Export a contiguous safetensors checkpoint; no pickle code was executed.")}
    private struct Tensor {let key:String,dtype:String,width:Int,offset:Int64,shape:[Int],strides:[Int64]}
    private final class DictionaryValue {var entries=[String:Value]()}
    private indirect enum Value {case mark,none,integer(Int64),string(String),symbol(String),tuple([Value]),list([Value]),dictionary(DictionaryValue),storage(String,String,Int),tensor(Tensor)}
    private final class PickleMetadata {
        let data:Data;var cursor=0,stack=[Value](),memo=[Int:Value](),steps=0
        init(_ data:Data) {self.data=data}
        func bytes(_ count:Int)throws->Data {guard count>=0,count<=data.count-cursor else {throw invalid()};defer{cursor+=count};return data.subdata(in:cursor..<cursor+count)}
        func integer(_ count:Int)throws->Int64 {let b=try bytes(count);let value=b.enumerated().reduce(UInt64(0)){$0 | UInt64($1.element) << (8*$1.offset)};guard value<=UInt64(Int64.max) else {throw invalid()};return Int64(value)}
        func text(_ count:Int)throws->String {guard let value=String(data:try bytes(count),encoding:.utf8),value.utf8.count<=1_048_576 else {throw invalid()};return value}
        func line()throws->String {guard let end=data[cursor...].firstIndex(of:10) else {throw invalid()};let value=try text(end-cursor);cursor+=1;return value}
        func pop()throws->Value {guard let value=stack.popLast() else {throw invalid()};return value}
        func marked()throws->[Value] {guard let mark=stack.lastIndex(where:{if case .mark=$0{return true};return false}) else {throw invalid()};let values=Array(stack.dropFirst(mark+1));stack.removeSubrange(mark...);return values}
        func numbers(_ value:Value)throws->[Int64] {guard case .tuple(let values)=value else {throw invalid()};return try values.map{guard case .integer(let n)=$0 else {throw invalid()};return n}}
        func decode()throws->[String:Tensor] {
            while cursor<data.count {
                steps+=1;guard steps<=2_000_000,stack.count<200_000,memo.count<200_000 else {throw invalid()}
                let op=try integer(1)
                switch op {
                case 0x80:guard try integer(1)<=5 else {throw invalid()}
                case 0x95:_=try integer(8)
                case 0x28:stack.append(.mark)
                case 0x4e,0x88,0x89:stack.append(.none)
                case 0x4b:stack.append(.integer(try integer(1)))
                case 0x4d:stack.append(.integer(try integer(2)))
                case 0x4a:stack.append(.integer(Int64(Int32(bitPattern:UInt32(try integer(4))))))
                case 0x8a:let count=Int(try integer(1));guard count<=8 else {throw invalid()};stack.append(.integer(try integer(count)))
                case 0x58:stack.append(.string(try text(Int(integer(4)))))
                case 0x8c:stack.append(.string(try text(Int(integer(1)))))
                case 0x8d:stack.append(.string(try text(Int(integer(8)))))
                case 0x63:stack.append(.symbol(try line()+"."+line()))
                case 0x93:guard case .string(let name)=try pop(),case .string(let module)=try pop() else {throw invalid()};stack.append(.symbol(module+"."+name))
                case 0x71,0x72:let index=Int(try integer(op == 0x71 ? 1 : 4));guard let last=stack.last else {throw invalid()};memo[index]=last
                case 0x94:guard let last=stack.last else {throw invalid()};memo[memo.count]=last
                case 0x68,0x6a:let index=Int(try integer(op == 0x68 ? 1 : 4));guard let value=memo[index] else {throw invalid()};stack.append(value)
                case 0x29:stack.append(.tuple([]))
                case 0x74:stack.append(.tuple(try marked()))
                case 0x85,0x86,0x87:let count=Int(op-0x84);guard stack.count>=count else {throw invalid()};let values=Array(stack.suffix(count));stack.removeLast(count);stack.append(.tuple(values))
                case 0x7d:stack.append(.dictionary(DictionaryValue()))
                case 0x5d:stack.append(.list([]))
                case 0x51:
                    guard case .tuple(let args)=try pop(),args.count>=5,case .string("storage")=args[0],case .symbol(let type)=args[1],case .string(let key)=args[2] else {throw invalid()}
                    let types=["torch.BFloat16Storage":("BF16",2),"torch.HalfStorage":("F16",2),"torch.FloatStorage":("F32",4),"torch.IntStorage":("I32",4),"torch.LongStorage":("I64",8),"torch.ByteStorage":("U8",1)]
                    guard let (dtype,width)=types[type],key.range(of:"^[0-9]+$",options:.regularExpression) != nil else {throw invalid()};stack.append(.storage(key,dtype,width))
                case 0x52:
                    guard case .tuple(let args)=try pop(),case .symbol(let function)=try pop() else {throw invalid()}
                    if function == "collections.OrderedDict",args.isEmpty {stack.append(.dictionary(DictionaryValue()))}
                    else if ["torch._utils._rebuild_tensor_v2","torch._utils._rebuild_tensor"].contains(function),args.count>=4,case .storage(let key,let dtype,let width)=args[0],case .integer(let offset)=args[1] {
                        stack.append(.tensor(.init(key:key,dtype:dtype,width:width,offset:offset,shape:try numbers(args[2]).map(Int.init),strides:try numbers(args[3]))))
                    } else {throw invalid()}
                case 0x75:
                    let values=try marked();guard case .dictionary(let dict)=stack.last,values.count.isMultiple(of:2) else {throw invalid()}
                    for i in stride(from:0,to:values.count,by:2) {guard case .string(let key)=values[i] else {throw invalid()};dict.entries[key]=values[i+1]}
                case 0x73:let value=try pop();guard case .string(let key)=try pop(),case .dictionary(let dict)=stack.last else {throw invalid()};dict.entries[key]=value
                case 0x62:_=try pop();guard case .dictionary=stack.last else {throw invalid()}
                case 0x2e:
                    guard case .dictionary(let dict)=try pop(),stack.isEmpty else {throw invalid()}
                    var tensors=[String:Tensor]()
                    for (name,value) in dict.entries {if case .tensor(let tensor)=value {tensors[name]=tensor} else if name != "_metadata" {throw invalid()}}
                    return tensors
                default:throw invalid()
                }
            }
            throw invalid()
        }
    }
}
