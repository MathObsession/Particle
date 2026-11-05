import Foundation

/// Recommendations are exactly the curated catalog; backend coverage cannot add entries.
public enum ModelRecommendations {
    // Kept as a source compatibility constant for older tests/data migrations.
    public static let chronosRepository="Lifeboat78/Chronos-Ornith-1.5-35B-A3B-MLX-3bit"
    public static let additionalRepositories=ModelCatalog.entries.map(\.repository)
    public static func reason(for model:ModelDescriptor)->String? {
        guard let entry=ModelCatalog.entry(for:model) else{return nil}
        return entry.fast ? "Fast collection · Actual speed depends on this Mac, storage and expert profile." : "Curated MoE model · Native quantization and configurable experts."
    }
    public static func eligible(_ models:[ModelDescriptor])->[ModelDescriptor] {
        models.filter{ModelCatalog.allows($0) && ArchitectureRegistry.report($0).installable}.sorted{a,b in
            let order=ModelCatalog.entries.map(\.id)
            return (order.firstIndex(of:ModelCatalog.entry(for:a)!.id) ?? 99)<(order.firstIndex(of:ModelCatalog.entry(for:b)!.id) ?? 99)
        }
    }
}
