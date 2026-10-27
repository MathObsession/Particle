import Foundation
import Testing
@preconcurrency import Metal
import ParticleGGUF
@testable import ParticleCore

@Suite(.serialized) struct AttentionOptimizationTests {
    @Test func groupedAndSplitAttentionKeepAllHistoryAndCausality() throws {
        let gpu=try ParticleGPU(), capacity=8192
        let keys=try gpu.buffer(capacity*256), values=try gpu.buffer(capacity*256)
        let k=keys.contents().assumingMemoryBound(to:Float.self), v=values.contents().assumingMemoryBound(to:Float.self)
        for i in 0..<capacity*256 { k[i]=sin(Float(i)*0.013)*0.3; v[i]=cos(Float(i)*0.017) }
        for (base,count) in [(0,1),(31,3),(255,5),(511,16),(2048,3),(8189,3),(0,128),(1,256)] {
            try autoreleasepool {
                let queries=try (0..<count).map { _ in try gpu.buffer(4096) }
                let reference=try (0..<count).map { _ in try gpu.buffer(4096) }
                let tiled=try (0..<count).map { _ in try gpu.buffer(4096) }
                let split=try (0..<count).map { _ in try gpu.buffer(4096) }
                for (row,q) in queries.enumerated() {
                    let p=q.contents().assumingMemoryBound(to:Float.self)
                    for i in 0..<4096 { p[i]=cos(Float(i+row*19)*0.021) }
                }
                let cb=try gpu.command()
                try gpu.batchAttention(cb,queries:queries,keys:keys,values:values,outputs:reference,base:base,capacity:capacity,mode:.reference)
                try gpu.batchAttention(cb,queries:queries,keys:keys,values:values,outputs:tiled,base:base,capacity:capacity,mode:.tiled)
                try gpu.batchAttention(cb,queries:queries,keys:keys,values:values,outputs:split,base:base,capacity:capacity,mode:.split)
                try gpu.finish(cb)
                var exact=true, splitError: Float=0
                for row in 0..<count {
                    let a=reference[row].contents().assumingMemoryBound(to:Float.self)
                    let b=tiled[row].contents().assumingMemoryBound(to:Float.self)
                    let c=split[row].contents().assumingMemoryBound(to:Float.self)
                    for i in 0..<4096 { exact = exact && a[i].bitPattern == b[i].bitPattern; splitError=max(splitError,abs(a[i]-c[i])) }
                }
                #expect(exact,"Tiled attention must retain the scalar accumulation order, base=\(base), rows=\(count)")
                #expect(splitError<0.00002,"Split stable-softmax merge, base=\(base), rows=\(count), error=\(splitError)")
            }
        }
        // Independent FP64 reference on a complete 513-position query. Future
        // KV entries are populated but must never contribute to the result.
        let query=try gpu.buffer(4096), output=try gpu.buffer(4096)
        let q=query.contents().assumingMemoryBound(to:Float.self)
        for i in 0..<4096 { q[i]=sin(Float(i)*0.01) }
        let cb=try gpu.command()
        try gpu.batchAttention(cb,queries:[query],keys:keys,values:values,outputs:[output],base:512,capacity:capacity,mode:.split)
        try gpu.finish(cb)
        let o=output.contents().assumingMemoryBound(to:Float.self)
        for head in [0,15,16,31,32,47,48,63] {
            let scores=(0...512).map { t in (0..<64).reduce(0.0) { $0+Double(q[head*64+$1])*Double(k[t*256+(head/16)*64+$1])*0.125 } }
            let maximum=scores.max()!, weights=scores.map { exp($0-maximum) }, total=weights.reduce(0,+)
            for d in [0,17,32,63] {
                let expected=(0...512).reduce(0.0) { $0+weights[$1]*Double(v[$1*256+(head/16)*64+d]) }/total
                #expect(abs(Double(o[head*64+d])-expected)<0.00002)
            }
        }
        #expect(throws:ParticleError.self) { try gpu.batchAttention(gpu.command(),queries:[query],keys:keys,values:values,outputs:[output],base:8192,capacity:8192) }
    }
    @Test func split512MatchesReference() throws {
        var options=ParticleExecutionOptions(); options.splitBlock=512
        let gpu=try ParticleGPU(execution:options), keys=try gpu.buffer(1025*256), values=try gpu.buffer(1025*256)
        let q=try gpu.buffer(4096), a=try gpu.buffer(4096), b=try gpu.buffer(4096)
        for (buffer,count) in [(keys,1025*256),(values,1025*256),(q,4096)] {
            let p=buffer.contents().assumingMemoryBound(to:Float.self)
            for i in 0..<count { p[i]=sin(Float(i)*0.01)*0.1 }
        }
        let cb=try gpu.command()
        try gpu.batchAttention(cb,queries:[q],keys:keys,values:values,outputs:[a],base:1024,capacity:1025,mode:.reference)
        try gpu.batchAttention(cb,queries:[q],keys:keys,values:values,outputs:[b],base:1024,capacity:1025,mode:.split)
        try gpu.finish(cb)
        for i in 0..<4096 { #expect(abs(a.contents().assumingMemoryBound(to:Float.self)[i]-b.contents().assumingMemoryBound(to:Float.self)[i])<0.00002) }
    }
    @Test func fp32TensorAttentionMatchesFullCausalReference() throws {
        var options=ParticleExecutionOptions(); options.attention = .tensor
        let gpu=try ParticleGPU(execution:options), capacity=8192
        let keys=try gpu.buffer(capacity*256), values=try gpu.buffer(capacity*256)
        for (buffer,scale) in [(keys,Float(0.013)),(values,Float(0.017))] {
            let p=buffer.contents().assumingMemoryBound(to:Float.self)
            for i in 0..<capacity*256 { p[i]=sin(Float(i)*scale)*0.2 }
        }
        for (base,count) in [(0,1),(31,3),(127,17),(2048,16),(8000,128)] {
            let queries=try (0..<count).map { _ in try gpu.buffer(4096) }
            let reference=try (0..<count).map { _ in try gpu.buffer(4096) }
            let output=try (0..<count).map { _ in try gpu.buffer(4096) }
            for (row,q) in queries.enumerated() { for i in 0..<4096 { q.contents().assumingMemoryBound(to:Float.self)[i]=cos(Float(i+row*19)*0.021) } }
            let cb=try gpu.command()
            try gpu.batchAttention(cb,queries:queries,keys:keys,values:values,outputs:reference,base:base,capacity:capacity,mode:.reference)
            try gpu.batchAttention(cb,queries:queries,keys:keys,values:values,outputs:output,base:base,capacity:capacity,mode:.tensor)
            try gpu.finish(cb)
            var maxError: Float=0
            for row in 0..<count { for i in 0..<4096 {
                let a=reference[row].contents().assumingMemoryBound(to:Float.self)[i], b=output[row].contents().assumingMemoryBound(to:Float.self)[i]
                #expect(b.isFinite); maxError=max(maxError,abs(a-b))
            } }
            #expect(maxError<0.00002,"FP32 TensorOps attention error \(maxError), base \(base), rows \(count)")
        }
    }
    @Test func historicalPreparationMatchesOfficialNextTurn() async throws {
        let folder=URL(fileURLWithPath:"/Volumes/Particle/Models/sarvam-30b/tokenizer")
        let tokenizer=try await SarvamTokenizer.load(folder:folder)
        for thinking in [false,true] {
            for reply in ["Hello."," <think>reasoning</think>\n  answer  ","<think></think>\nनमस्ते दुनिया 👋", "", "\n\t", "a\u{0301} text\n"] {
                let history=[SarvamMessage(role:.system,content:"Keep the whole chat."),.init(role:.user,content:"Hi"),.init(role:.assistant,content:reply)]
                let prefix=try tokenizer.historicalPrefix(messages:history,thinking:thinking)
                let prompt=try tokenizer.chat(messages:Array(history.dropLast()),thinking:thinking)
                #expect(try tokenizer.completedChatPrefix(promptTokens:prompt,assistant:reply,thinking:thinking)==prefix)
                for user in ["Next", "हिन्दी में", "\u{0301} combining", "<think>literal</think>"] {
                    let target=try tokenizer.chat(messages:history+[.init(role:.user,content:user)],thinking:thinking)
                    #expect(Array(target.prefix(prefix.count))==prefix)
                    #expect(try tokenizer.continuingChat(historicalPrefix:prefix,prompt:user,thinking:thinking)==target)
                    let plan=try ParticlePromptReuse(committed:prefix,target:target,context:2048,outputAllowance:64)
                    #expect(plan.reused==prefix.count)
                }
            }
        }
    }
    @Test func incrementalCanonicalizationPreservesMultipleEarlierTurns() async throws {
        let tokenizer=try await SarvamTokenizer.load(folder:URL(fileURLWithPath:"/Volumes/Particle/Models/sarvam-30b/tokenizer"))
        let history=[SarvamMessage(role:.system,content:"Keep every turn."),.init(role:.user,content:"First"),
            .init(role:.assistant,content:"<think>old reasoning</think>First answer."),.init(role:.user,content:"Next"),
            .init(role:.assistant,content:"<think>new reasoning</think>नमस्ते 👋 a\u{0301}")]
        let prompt=try tokenizer.chat(messages:Array(history.dropLast()))
        let reference=try tokenizer.historicalPrefix(messages:history)
        #expect(try tokenizer.completedChatPrefix(promptTokens:prompt,assistant:history.last!.content)==reference)
        #expect(try tokenizer.completedChatPrefix(promptTokens:[6,107],assistant:"Answer")==nil)
    }
    @Test func unknownTokenizerRevisionFallsBackFromIncrementalEncoding() async throws {
        let original=URL(fileURLWithPath:"/Volumes/Particle/Models/sarvam-30b/tokenizer")
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent("particle-tokenizer-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:folder) }
        for name in ["config.json","tokenizer.json","chat_template.jinja"] {
            try FileManager.default.createSymbolicLink(at:folder.appendingPathComponent(name),withDestinationURL:original.appendingPathComponent(name))
        }
        // Semantically unchanged JSON, different fingerprint. Never modifies
        // installed tokenizer assets or needs a second copy of their vocabulary.
        var config=try Data(contentsOf:original.appendingPathComponent("tokenizer_config.json")); config.append(10)
        try config.write(to:folder.appendingPathComponent("tokenizer_config.json"))
        let tokenizer=try await SarvamTokenizer.load(folder:folder)
        #expect(!tokenizer.supportsIncrementalChat)
        let history=[SarvamMessage(role:.user,content:"Hi"),.init(role:.assistant,content:"Hello.")]
        let prefix=try tokenizer.historicalPrefix(messages:history)
        #expect(try tokenizer.continuingChat(historicalPrefix:prefix,prompt:"Next")==nil)
        #expect(try !tokenizer.chat(messages:history+[.init(role:.user,content:"Next")]).isEmpty)
    }
    @MainActor @Test func cancellationHandlerCanExecuteOffMainActor() async {
        let cancellation=ParticleCancellation(), handler=cancellation.cancellationHandler()
        await Task.detached { handler() }.value
        #expect(cancellation.isCancelled)
    }
    @Test func inputCancelsIdleOnlyAndQueuesWithoutLosingMessages() {
        let mailbox=ParticleInputMailbox(), idle=ParticleCancellation(), foreground=ParticleCancellation()
        #expect(mailbox.beginIdle(idle))
        mailbox.submit("first"); #expect(idle.isCancelled); #expect(!foreground.isCancelled)
        mailbox.endIdle(); idle.reset(); #expect(!mailbox.beginIdle(idle))
        #expect(mailbox.next()?.text=="first")
        #expect(mailbox.beginIdle(idle)); mailbox.finish(); #expect(idle.isCancelled); #expect(mailbox.next()==nil)
    }
    @Test func workQueueClaimsEveryRowExactlyOnce() {
        let queue=ParticleTileQueue(count:4097), seen=ClaimedRows()
        DispatchQueue.concurrentPerform(iterations:4) { worker in
            while let range=queue.claim(worker==0 ? 4 : 1) { seen.add(Array(range)) }
        }
        #expect(seen.rows.sorted()==Array(0..<4097))
    }
    @Test func coalescingBoundsBytesAndNeverReadsSpeculativeExperts() {
        let pointer=UnsafeMutableRawPointer.allocate(byteCount:1,alignment:1); defer { pointer.deallocate() }
        let ranges=[(0,1000),(1050,1000),(4000,1000),(5000,1000),(20_000_000,5_000_000),(25_000_000,5_000_000)].map {
            ParticleReadRange(offset:$0.0,count:$0.1,destination:pointer)
        }
        let groups=ParticleReadPlan.groups(ranges)
        #expect(groups.flatMap(\.ranges).map(\.offset)==ranges.map(\.offset))
        #expect(groups.first?.count==2050)
        for group in groups {
            let useful=group.ranges.reduce(0) { $0+$1.count }
            #expect(Double(group.count)<=Double(useful)*1.1)
            #expect(group.ranges.count==1 || group.count<=8*1024*1024)
        }
    }
}
private final class ClaimedRows: @unchecked Sendable {
    let lock=NSLock(); var rows: [Int]=[]
    func add(_ values: [Int]) { lock.lock(); rows.append(contentsOf:values); lock.unlock() }
}
