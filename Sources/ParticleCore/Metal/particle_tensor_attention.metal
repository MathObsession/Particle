#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
// Four prompt rows x sixteen Q heads per KV head. The same 32-position
// K/V tile serves 64 queries. FP32 QK/PV, online softmax, full causal history.
kernel void particle_tensor_attention(device const ParticleBatchRows &rows [[buffer(0)]],
    device const float *keys [[buffer(1)]], device const float *values [[buffer(2)]],
    constant uint4 &p [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr auto qk_desc=matmul2d_descriptor(64,32,64,false,true,false);
    constexpr auto pv_desc=matmul2d_descriptor(64,64,32,false,true,false);
    matmul2d<qk_desc,execution_simdgroups<4>> qk;
    matmul2d<pv_desc,execution_simdgroups<4>> pv;
    using tg_tensor=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>;
    // Q is overwritten by scores/probabilities after QK completes; K is
    // overwritten by transposed V. Max, denominator and old scale follow.
    threadgroup float workspace[6336];
    tg_tensor qt(workspace,dextents<int32_t,2>(64,64),array<int32_t,2>({1,64}));
    tg_tensor kt(workspace+4096,dextents<int32_t,2>(64,32),array<int32_t,2>({1,64}));
    tg_tensor prob(workspace,dextents<int32_t,2>(32,64),array<int32_t,2>({1,32}));
    tg_tensor vt(workspace+4096,dextents<int32_t,2>(32,64),array<int32_t,2>({1,32}));
    auto score=qk.get_destination_cooperative_tensor<decltype(qt),decltype(kt),float>();
    auto product=pv.get_destination_cooperative_tensor<decltype(prob),decltype(vt),float>();
    auto output=pv.get_destination_cooperative_tensor<decltype(prob),decltype(vt),float>();
    for(int i=0;i<output.get_capacity();++i) output[i]=0;
    if(tid<64) { workspace[6144+tid]=-INFINITY; workspace[6208+tid]=0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint last=p.x+min(group.y*4+3,p.y-1);
    for(uint start=0;start<=last;start+=32) {
        for(uint i=tid;i<4096;i+=128) {
            uint row=group.y*4+(i/64)/16, head=group.x*16+(i/64)%16;
            workspace[i]=row<p.y ? rows.input[row][head*64+i%64] : 0;
        }
        for(uint i=tid;i<2048;i+=128) {
            uint t=start+i/64;
            workspace[4096+i]=t<=last ? keys[t*256+group.x*64+i%64] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int i=0;i<score.get_capacity();++i) score[i]=0;
        qk.run(qt,kt,score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int i=0;i<score.get_capacity();++i) {
            if(!score.is_valid_element(i)) continue;
            auto index=score.get_multidimensional_index(i);
            uint m=uint(index[1]), t=uint(index[0]), row=group.y*4+m/16;
            workspace[m*32+t]=row<p.y && start+t<=p.x+row ? score[i]*0.125f : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint m=sg;m<64;m+=4) {
            uint row=group.y*4+m/16;
            float scoreValue=workspace[m*32+lane];
            float oldMax=workspace[6144+m], oldSum=workspace[6208+m];
            float maximum=max(oldMax,simd_max(scoreValue));
            float oldScale=isfinite(oldMax) ? exp(oldMax-maximum) : 0;
            float weight=row<p.y && isfinite(scoreValue) ? exp(scoreValue-maximum) : 0;
            float denominator=oldSum*oldScale+simd_sum(weight);
            workspace[m*32+lane]=weight;
            if(lane==0) { workspace[6144+m]=maximum; workspace[6208+m]=denominator; workspace[6272+m]=oldScale; }
        }
        for(uint i=tid;i<2048;i+=128) {
            uint t=start+i%32, d=i/32;
            workspace[4096+i]=t<=last ? values[t*256+group.x*64+d] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int i=0;i<output.get_capacity();++i) {
            if(output.is_valid_element(i)) { auto index=output.get_multidimensional_index(i); output[i]*=workspace[6272+uint(index[1])]; }
        }
        for(int i=0;i<product.get_capacity();++i) product[i]=0;
        pv.run(prob,vt,product);
        for(int i=0;i<output.get_capacity();++i) output[i]+=product[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(int i=0;i<output.get_capacity();++i) {
        if(!output.is_valid_element(i)) continue;
        auto index=output.get_multidimensional_index(i);
        uint m=uint(index[1]), row=group.y*4+m/16, head=group.x*16+m%16;
        if(row<p.y) rows.output[row][head*64+uint(index[0])]=output[i]/workspace[6208+m];
    }
}
#endif
