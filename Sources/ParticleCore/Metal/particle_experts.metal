// Included with particle.metal; native quantization and FP32 sums are shared.
struct ParticleExpertBindings {
    array<device const uchar *,6> gate [[id(0)]];
    array<device const uchar *,6> up [[id(6)]];
    array<device const uchar *,6> down [[id(12)]];
    array<device float *,6> scratch [[id(18)]];
    array<device float *,6> output [[id(24)]];
};
struct ParticleExpertParams { uint n, hidden, gateType, upType, downType, gateRow, upRow, downRow, count; };
kernel void particle_expert_gate_up(device const ParticleExpertBindings &b [[buffer(0)]],
                                    device const float *x [[buffer(1)]], constant ParticleExpertParams &p [[buffer(2)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group.x*4+sg, expert=group.y;
    if (row>=p.hidden || expert>=p.count) return;
    device const uchar *g=b.gate[expert]+ulong(row)*p.gateRow, *u=b.up[expert]+ulong(row)*p.upRow;
    float a=0,c=0;
    for (uint i=lane; i<p.n; i+=32) { float v=x[i]; a+=p_weight(g,i,p.gateType)*v; c+=p_weight(u,i,p.upType)*v; }
    a=simd_sum(a); c=simd_sum(c);
    if (lane==0) b.scratch[expert][row]=(a/(1+exp(-a)))*c;
}
kernel void particle_expert_down(device const ParticleExpertBindings &b [[buffer(0)]],
                                 constant ParticleExpertParams &p [[buffer(1)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group.x*4+sg, expert=group.y;
    if (row>=p.n || expert>=p.count) return;
    device const uchar *w=b.down[expert]+ulong(row)*p.downRow;
    float sum=0;
    for (uint i=lane; i<p.hidden; i+=32) sum+=p_weight(w,i,p.downType)*b.scratch[expert][i];
    sum=simd_sum(sum);
    if (lane==0) b.output[expert][row]=sum;
}
