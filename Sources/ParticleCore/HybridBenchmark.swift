import Foundation
import ParticleGGUF

public struct ParticleHybridTiming: Codable {
    public let rows: Int, cpuRows: Int, gpuRows: Int
    public let referenceSeconds: Double, hybridSeconds: Double, speedup: Double
    public let maxError: Float
}
extension ParticleValidation {
    public static func benchmarkHybrid() throws -> [ParticleHybridTiming] {
        let gpu=try ParticleGPU(), n=4096, width=1024
        func weights(_ columns: Int,_ rows: Int) throws -> GPUWeight {
            let buffer=try gpu.bytes(columns*rows/256*144)
            let p=buffer.contents().assumingMemoryBound(to:UInt8.self)
            for i in 0..<buffer.length { p[i]=UInt8(truncatingIfNeeded:i*17+13) }
            for block in 0..<columns*rows/256 { p[block*144]=0x19; p[block*144+1]=0x14; p[block*144+2]=0x19; p[block*144+3]=0x10 }
            return GPUWeight(buffer:buffer,offset:0,columns:columns,rows:rows,type:.q4K)
        }
        let gate=try weights(n,width), up=try weights(n,width), down=try weights(width,n)
        var reports: [ParticleHybridTiming]=[]
        for count in [8,32,128] {
            let inputs=try (0..<count).map { _ in try gpu.buffer(n) }, scratch=try (0..<count).map { _ in try gpu.buffer(width) }
            let reference=try (0..<count).map { _ in try gpu.buffer(n) }, candidate=try (0..<count).map { _ in try gpu.buffer(n) }
            for (row,b) in inputs.enumerated() { for i in 0..<n { b.contents().assumingMemoryBound(to:Float.self)[i]=sin(Float(i+row)*0.03)*0.001 } }
            var controlTimes: [Double]=[], hybridTimes: [Double]=[], ownership=ParticleHybridWorkMetrics(cpuRows:0,gpuRows:0)
            for round in 0..<4 {
                for hybrid in (round%2==0 ? [false,true] : [true,false]) {
                    let start=ProcessInfo.processInfo.systemUptime
                    if hybrid { ownership=try gpu.hybridMLP(inputs:inputs,gate:gate,up:up,down:down,gateScratch:scratch,outputs:candidate,cancellation:nil) }
                    else {
                        let cb=try gpu.command()
                        try gpu.batchMLP(cb,inputs:inputs,gate:gate,up:up,down:down,gateScratch:scratch,outputs:reference)
                        try gpu.finish(cb)
                    }
                    if round>0 { if hybrid { hybridTimes.append(ProcessInfo.processInfo.systemUptime-start) } else { controlTimes.append(ProcessInfo.processInfo.systemUptime-start) } }
                }
            }
            var error: Float=0
            for row in 0..<count { for i in 0..<n {
                error=max(error,abs(reference[row].contents().assumingMemoryBound(to:Float.self)[i]-candidate[row].contents().assumingMemoryBound(to:Float.self)[i]))
            } }
            let control=controlTimes.sorted()[1], hybrid=hybridTimes.sorted()[1]
            reports.append(ParticleHybridTiming(rows:count,cpuRows:ownership.cpuRows,gpuRows:ownership.gpuRows,
                referenceSeconds:control,hybridSeconds:hybrid,speedup:control/hybrid,maxError:error))
        }
        return reports
    }
}
