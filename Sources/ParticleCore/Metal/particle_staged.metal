
#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
// Both tiles are bounded threadgroup allocations: 8 KiB total. Native
// K-quants are decoded directly; no full FP16 weight or activation matrices.
kernel void particle_staged_gemv(device const ParticleBatchRows &rows [[buffer(0)]],
    device const uchar *weights [[buffer(1)]], constant LinearParams &p [[buffer(2)]],
    constant uint &count [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr auto descriptor=matmul2d_descriptor(32,32,64,false,true,false);
    matmul2d<descriptor,execution_simdgroups<4>> operation;
    using tg_tensor=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>;
    threadgroup half at[32*64], bt[32*64];
    tg_tensor a(at,dextents<int32_t,2>(64,32),array<int32_t,2>({1,64}));
    tg_tensor b(bt,dextents<int32_t,2>(64,32),array<int32_t,2>({1,64}));
    auto sum=operation.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    auto product=operation.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(int i=0;i<sum.get_capacity();++i) sum[i]=0;
    for(uint k=0;k<p.n;k+=64) {
        for(uint i=tid;i<32*64;i+=128) {
            uint token=group.y*32+i/64, output=group.x*32+i/64, col=k+i%64;
            at[i]=token<count ? half(rows.input[token][col]) : half(0);
            bt[i]=output<p.m ? half(p_weight(weights+ulong(output)*p.rowBytes,col,p.type)) : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int i=0;i<product.get_capacity();++i) product[i]=0;
        operation.run(a,b,product);
        for(int i=0;i<sum.get_capacity();++i) sum[i]+=product[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(int i=0;i<sum.get_capacity();++i) {
        if(!sum.is_valid_element(i)) continue;
        auto index=sum.get_multidimensional_index(i);
        uint output=group.x*32+uint(index[0]), token=group.y*32+uint(index[1]);
        if(output<p.m && token<count) rows.output[token][output]=sum[i];
    }
}
kernel void particle_staged_gate_up(device const ParticleBatchRows &rows [[buffer(0)]],
    device const uchar *gate [[buffer(1)]], device const uchar *up [[buffer(2)]], constant LinearParams &p [[buffer(3)]],
    constant uint &count [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr auto descriptor=matmul2d_descriptor(32,32,64,false,true,false);
    matmul2d<descriptor,execution_simdgroups<4>> operation;
    using tg_tensor=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>;
    threadgroup half at[32*64], bt[32*64];
    tg_tensor a(at,dextents<int32_t,2>(64,32),array<int32_t,2>({1,64}));
    tg_tensor b(bt,dextents<int32_t,2>(64,32),array<int32_t,2>({1,64}));
    auto g=operation.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    auto u=operation.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    auto product=operation.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(int i=0;i<g.get_capacity();++i) { g[i]=0; u[i]=0; }
    for(uint k=0;k<p.n;k+=64) {
        for(uint i=tid;i<32*64;i+=128) {
            uint token=group.y*32+i/64, output=group.x*32+i/64, col=k+i%64;
            at[i]=token<count ? half(rows.input[token][col]) : half(0);
            bt[i]=output<p.m ? half(p_weight(gate+ulong(output)*p.rowBytes,col,p.type)) : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int i=0;i<product.get_capacity();++i) product[i]=0;
        operation.run(a,b,product);
        for(int i=0;i<g.get_capacity();++i) g[i]+=product[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<32*64;i+=128) {
            uint output=group.x*32+i/64, col=k+i%64;
            bt[i]=output<p.m ? half(p_weight(up+ulong(output)*p.rowBytes,col,p.type)) : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int i=0;i<product.get_capacity();++i) product[i]=0;
        operation.run(a,b,product);
        for(int i=0;i<u.get_capacity();++i) u[i]+=product[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(int i=0;i<g.get_capacity();++i) {
        if(!g.is_valid_element(i)) continue;
        auto index=g.get_multidimensional_index(i);
        uint output=group.x*32+uint(index[0]), token=group.y*32+uint(index[1]);
        if(output<p.m && token<count) rows.output[token][output]=(g[i]/(1+exp(-g[i])))*u[i];
    }
}
#endif
