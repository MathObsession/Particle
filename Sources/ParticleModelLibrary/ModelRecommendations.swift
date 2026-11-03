import Foundation

/// Editorial recommendations never bypass tensor inspection or runtime qualification.
public enum ModelRecommendations {
    public static let chronosRepository = "Lifeboat78/Chronos-Ornith-1.5-35B-A3B-MLX-3bit"
    public static func reason(for model: ModelDescriptor) -> String? {
        if model.id == ModelDescriptor.sarvam.id {
            return "Start here. Particle’s default assistant for everyday chat and coding."
        }
        if model.provenance.repository == chronosRepository {
            return "An alternative for chat, with native 3-bit weights and streamed experts."
        }
        return nil
    }
    public static func eligible(_ models: [ModelDescriptor]) -> [ModelDescriptor] {
        models.filter { reason(for: $0) != nil && ArchitectureRegistry.report($0).installable }
            .sorted { a,b in
                if a.id == ModelDescriptor.sarvam.id { return b.id != a.id }
                if b.id == ModelDescriptor.sarvam.id { return false }
                return a.name.localizedCaseInsensitiveCompare(b.name) == .orderedAscending
            }
    }
}
