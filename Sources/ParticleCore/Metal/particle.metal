#include <metal_stdlib>
using namespace metal;

// GGML K-quant block formats. See ggml-org/llama.cpp ggml-quants.c,
// dequantize_row_q{4,5,6}_K (MIT). Decode directly, with no requantization.
inline float p_half(device const uchar *p) {
    ushort bits = ushort(ushort(p[0]) | ushort(ushort(p[1]) << 8));
    return float(as_type<half>(bits));
}
inline float p_weight(device const uchar *row, uint i, uint type) {
    if (type == 0) return ((device const float *)row)[i];
    if (type == 1) return p_half(row + 2 * i);
    uint k = i & 255u;
    if (type == 14) {
        device const uchar *p = row + (i >> 8) * 210;
        uint h = k / 128, j = k % 128, q = j / 32, l = j % 32;
        uint lo = p[h * 64 + (q % 2) * 32 + l];
        lo = q < 2 ? (lo & 15) : (lo >> 4);
        int v = int(lo | (((p[128 + h * 32 + l] >> (2 * q)) & 3) << 4)) - 32;
        int sc = int(((device const char *)(p + 192))[h * 8 + q * 2 + l / 16]);
        return p_half(p + 208) * float(sc) * float(v);
    }
    uint stride = type == 12 ? 144 : 176;
    device const uchar *p = row + (i >> 8) * stride;
    uint g = k / 32;
    device const uchar *s = p + 4;
    uint sc = g < 4 ? (s[g] & 63) : ((s[g + 4] & 15) | ((s[g - 4] >> 6) << 4));
    uint mn = g < 4 ? (s[g + 4] & 63) : ((s[g + 4] >> 4) | ((s[g] >> 6) << 4));
    uint l = k % 32;
    uint packed = p[(type == 12 ? 16 : 48) + (k / 64) * 32 + l];
    uint v = (packed >> ((g % 2) * 4)) & 15;
    if (type == 13) v |= ((p[16 + l] >> g) & 1) << 4;
    return p_half(p) * float(sc) * float(v) - p_half(p + 2) * float(mn);
}

struct LinearParams { uint n, m, type, rowBytes; };
kernel void particle_gemv(device const uchar *weights [[buffer(0)]],
                          device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
                          constant LinearParams &p [[buffer(3)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * 4 + simd;
    if (row >= p.m) return;
    device const uchar *w = weights + ulong(row) * p.rowBytes;
    float sum = 0;
    for (uint i = lane; i < p.n; i += 32) sum += p_weight(w, i, p.type) * x[i];
    sum = simd_sum(sum);
    if (lane == 0) y[row] = sum;
}
// Argument buffers retain explicit row bindings; the host declares residency.
struct ParticleBatchRows {
    array<device const float *,256> input [[id(0)]];
    array<device float *,256> output [[id(256)]];
};
kernel void particle_batch_gemv(device const ParticleBatchRows &rows [[buffer(0)]],
                                device const uchar *weights [[buffer(1)]],
                                constant LinearParams &p [[buffer(2)]], constant uint &count [[buffer(3)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * 4 + sg, first = group.y * 4;
    if (row >= p.m || first >= count) return;
    device const uchar *w = weights + ulong(row) * p.rowBytes;
    float a=0, b=0, c=0, d=0;
    for (uint i=lane; i<p.n; i+=32) {
        float v = p_weight(w,i,p.type);
        a += v*rows.input[first][i];
        if (first+1<count) b += v*rows.input[first+1][i];
        if (first+2<count) c += v*rows.input[first+2][i];
        if (first+3<count) d += v*rows.input[first+3][i];
    }
    a=simd_sum(a); b=simd_sum(b); c=simd_sum(c); d=simd_sum(d);
    if (lane==0) {
        rows.output[first][row]=a;
        if (first+1<count) rows.output[first+1][row]=b;
        if (first+2<count) rows.output[first+2][row]=c;
        if (first+3<count) rows.output[first+3][row]=d;
    }
}
kernel void particle_gate_up(device const uchar *gate [[buffer(0)]], device const uchar *up [[buffer(1)]],
                             device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                             constant LinearParams &p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group*4+sg;
    if (row>=p.m) return;
    device const uchar *g=gate+ulong(row)*p.rowBytes, *u=up+ulong(row)*p.rowBytes;
    float a=0,b=0;
    for (uint i=lane;i<p.n;i+=32) {
        float v=x[i]; a+=p_weight(g,i,p.type)*v; b+=p_weight(u,i,p.type)*v;
    }
    a=simd_sum(a); b=simd_sum(b);
    if (lane==0) y[row]=(a/(1+exp(-a)))*b;
}
kernel void particle_batch_gate_up(device const ParticleBatchRows &rows [[buffer(0)]],
                                   device const uchar *gate [[buffer(1)]], device const uchar *up [[buffer(2)]],
                                   constant LinearParams &p [[buffer(3)]], constant uint &count [[buffer(4)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group.x*4+sg, first=group.y*4;
    if (row>=p.m || first>=count) return;
    device const uchar *g=gate+ulong(row)*p.rowBytes, *u=up+ulong(row)*p.rowBytes;
    float4 a=0,b=0;
    for (uint i=lane;i<p.n;i+=32) {
        float gv=p_weight(g,i,p.type), uv=p_weight(u,i,p.type);
        float v=rows.input[first][i]; a.x+=gv*v; b.x+=uv*v;
        if (first+1<count) { v=rows.input[first+1][i]; a.y+=gv*v; b.y+=uv*v; }
        if (first+2<count) { v=rows.input[first+2][i]; a.z+=gv*v; b.z+=uv*v; }
        if (first+3<count) { v=rows.input[first+3][i]; a.w+=gv*v; b.w+=uv*v; }
    }
    a.x=simd_sum(a.x); a.y=simd_sum(a.y); a.z=simd_sum(a.z); a.w=simd_sum(a.w);
    b.x=simd_sum(b.x); b.y=simd_sum(b.y); b.z=simd_sum(b.z); b.w=simd_sum(b.w);
    if (lane==0) {
        rows.output[first][row]=(a.x/(1+exp(-a.x)))*b.x;
        if (first+1<count) rows.output[first+1][row]=(a.y/(1+exp(-a.y)))*b.y;
        if (first+2<count) rows.output[first+2][row]=(a.z/(1+exp(-a.z)))*b.z;
        if (first+3<count) rows.output[first+3][row]=(a.w/(1+exp(-a.w)))*b.w;
    }
}
kernel void particle_embedding(device const uchar *weights [[buffer(0)]], device float *out [[buffer(1)]],
                               constant LinearParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) out[i] = p_weight(weights + ulong(p.m) * p.rowBytes, i, p.type);
}
kernel void particle_rms(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                         device float *y [[buffer(2)]], constant uint &n [[buffer(3)]],
                         uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    float sum = 0;
    for (uint i = tid; i < n; i += 256) sum += x[i] * x[i];
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float total = simd_sum(lane < 8 ? partial[lane] : 0.0f);
        if (lane == 0) partial[0] = rsqrt(total / float(n) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = partial[0];
    for (uint i = tid; i < n; i += 256) y[i] = x[i] * inv * w[i];
}
struct AttentionParams { uint position; uint capacity; };
kernel void particle_qkv(device const float *qkv [[buffer(0)]], device const float *qw [[buffer(1)]],
                         device const float *kw [[buffer(2)]], device float *q [[buffer(3)]],
                         device float *keys [[buffer(4)]], device float *values [[buffer(5)]],
                         constant AttentionParams &p [[buffer(6)]], uint head [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    // Source projection is Q[64,64], K[4,64], V[4,64]. V is neither normalized nor rotated.
    if (head >= 68) {
        values[p.position * 256 + (head - 68) * 64 + tid] = qkv[head * 64 + tid];
        return;
    }
    threadgroup float sums[2];
    device const float *x = qkv + head * 64;
    device const float *w = head < 64 ? qw : kw;
    float square = simd_sum(x[tid] * x[tid]);
    if (lane == 0) sums[sg] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt((sums[0] + sums[1]) / 64.0f + 1e-6f);
    uint partner = (tid + 32) % 64;
    uint pair = tid % 32;
    float angle = float(p.position) * pow(8000000.0f, -float(pair) / 32.0f);
    float v = x[tid] * inv * w[tid] * cos(angle)
        + (tid < 32 ? -1.0f : 1.0f) * x[partner] * inv * w[partner] * sin(angle);
    if (head < 64) q[head * 64 + tid] = v;
    else keys[p.position * 256 + (head - 64) * 64 + tid] = v;
}
kernel void particle_attention(device const float *q [[buffer(0)]], device const float *keys [[buffer(1)]],
                               device const float *values [[buffer(2)]], device float *out [[buffer(3)]],
                               constant AttentionParams &p [[buffer(4)]], uint head [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    uint kvHead = head / 16;
    float q0 = q[head * 64 + lane], q1 = q[head * 64 + lane + 32];
    float maximum = -INFINITY, denominator = 0, a0 = 0, a1 = 0;
    for (uint t = 0; t <= p.position; ++t) {
        uint base = t * 256 + kvHead * 64 + lane;
        float score = simd_sum(q0 * keys[base] + q1 * keys[base + 32]) * 0.125f;
        float next = max(maximum, score), oldScale = exp(maximum - next), weight = exp(score - next);
        denominator = denominator * oldScale + weight;
        a0 = a0 * oldScale + weight * values[base];
        a1 = a1 * oldScale + weight * values[base + 32];
        maximum = next;
    }
    out[head * 64 + lane] = a0 / denominator;
    out[head * 64 + lane + 32] = a1 / denominator;
}
struct AddParams { uint n; float scale; };
kernel void particle_add(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                         constant AddParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) x[i] += y[i] * p.scale;
}
kernel void particle_swiglu(device float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
                            constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) gate[i] = (gate[i] / (1.0f + exp(-gate[i]))) * up[i];
}


// Four query heads and two query rows share one 32-position K/V tile.
// Each SIMD group retains the reference kernel's per-position FP32 order.
kernel void particle_grouped_attention(device const ParticleBatchRows &rows [[buffer(0)]],
    device const float *keys [[buffer(1)]], device const float *values [[buffer(2)]],
    constant uint4 &p [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.y * 2 + sg / 4, head = group.x * 4 + sg % 4;
    uint kv = head / 16, last = p.x + min(group.y * 2 + 1, p.y - 1);
    bool active = row < p.y;
    float q0 = active ? rows.input[row][head*64+lane] : 0;
    float q1 = active ? rows.input[row][head*64+lane+32] : 0;
    float maximum = -INFINITY, denominator = 0, a0 = 0, a1 = 0;
    threadgroup float kt[32*64], vt[32*64];
    for (uint start=0; start<=last; start+=32) {
        uint count = min(32u,last-start+1);
        for (uint i=tid; i<count*64; i+=256) {
            uint src = (start+i/64)*256 + kv*64 + i%64;
            kt[i]=keys[src]; vt[i]=values[src];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint visible = active && start <= p.x+row ? min(count,p.x+row-start+1) : 0;
        for (uint t=0;t<visible;++t) {
            uint base=t*64+lane;
            float score=simd_sum(q0*kt[base]+q1*kt[base+32])*0.125f;
            float next=max(maximum,score), oldScale=exp(maximum-next), weight=exp(score-next);
            denominator=denominator*oldScale+weight;
            a0=a0*oldScale+weight*vt[base];
            a1=a1*oldScale+weight*vt[base+32];
            maximum=next;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        rows.output[row][head*64+lane]=a0/denominator;
        rows.output[row][head*64+lane+32]=a1/denominator;
    }
}

// Independent KV blocks expose parallelism for decode and small prompt tails.
// Scratch is [row,head,block,{max,sum,weighted value[64]}].
kernel void particle_split_attention(device const ParticleBatchRows &rows [[buffer(0)]],
    device const float *keys [[buffer(1)]], device const float *values [[buffer(2)]],
    device float *scratch [[buffer(3)]], constant uint4 &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint head=group.x, row=group.y, block=group.z, kv=head/16;
    uint first=block*p.z, end=min(first+p.z,p.x+row+1);
    float q0=rows.input[row][head*64+lane], q1=rows.input[row][head*64+lane+32];
    float maximum=-INFINITY, denominator=0, a0=0, a1=0;
    for (uint t=first;t<end;++t) {
        uint base=t*256+kv*64+lane;
        float score=simd_sum(q0*keys[base]+q1*keys[base+32])*0.125f;
        float next=max(maximum,score), oldScale=exp(maximum-next), weight=exp(score-next);
        denominator=denominator*oldScale+weight;
        a0=a0*oldScale+weight*values[base]; a1=a1*oldScale+weight*values[base+32]; maximum=next;
    }
    uint offset=((row*64+head)*p.w+block)*66;
    if (lane==0) { scratch[offset]=maximum; scratch[offset+1]=denominator; }
    scratch[offset+2+lane]=a0; scratch[offset+34+lane]=a1;
}
kernel void particle_merge_attention(device const ParticleBatchRows &rows [[buffer(0)]],
    device const float *scratch [[buffer(1)]], constant uint4 &p [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint head=group.x, row=group.y, count=(p.x+row+p.z)/p.z;
    uint base=(row*64+head)*p.w*66;
    float maximum=-INFINITY;
    for(uint b=0;b<count;++b) maximum=max(maximum,scratch[base+b*66]);
    float denominator=0,a0=0,a1=0;
    for(uint b=0;b<count;++b) {
        uint offset=base+b*66; float scale=exp(scratch[offset]-maximum);
        denominator+=scratch[offset+1]*scale;
        a0+=scratch[offset+2+lane]*scale; a1+=scratch[offset+34+lane]*scale;
    }
    rows.output[row][head*64+lane]=a0/denominator;
    rows.output[row][head*64+lane+32]=a1/denominator;
}
// Decode one whole K-quant block per lane, reusing its scale/minimum metadata.
// Accumulation still visits lane + {0,32,...,224}, exactly as the reference.
inline void p_block8(device const uchar *p, uint type, uint lane, thread float *v) {
    float d=p_half(p+(type==14 ? 208 : 0));
    if(type==14) {
        for(uint h=0;h<2;++h) {
            uint a=p[h*64+lane], b=p[h*64+32+lane], hi=p[128+h*32+lane];
            int q[4]={int((a&15)|((hi&3)<<4))-32,int((b&15)|(((hi>>2)&3)<<4))-32,
                      int((a>>4)|(((hi>>4)&3)<<4))-32,int((b>>4)|((hi>>6)<<4))-32};
            for(uint g=0;g<4;++g) {
                int sc=int(((device const char *)(p+192))[h*8+g*2+lane/16]);
                v[h*4+g]=d*float(sc)*float(q[g]);
            }
        }
    } else {
        float minimum=p_half(p+2); device const uchar *s=p+4;
        for(uint g=0;g<8;++g) {
            uint sc=g<4 ? s[g]&63 : (s[g+4]&15)|((s[g-4]>>6)<<4);
            uint mn=g<4 ? s[g+4]&63 : (s[g+4]>>4)|((s[g]>>6)<<4);
            uint packed=p[(type==12 ? 16 : 48)+(g/2)*32+lane];
            uint q=(packed>>((g%2)*4))&15;
            if(type==13) q|=((p[16+lane]>>g)&1)<<4;
            v[g]=d*float(sc)*float(q)-minimum*float(mn);
        }
    }
}
kernel void particle_vector_gemv(device const uchar *weights [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]], constant LinearParams &p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group*4+sg; if(row>=p.m) return;
    device const uchar *w=weights+ulong(row)*p.rowBytes;
    uint stride=p.type==14 ? 210 : p.type==12 ? 144 : 176;
    float sum=0;
    for(uint block=0;block<p.n/256;++block) {
        float v[8]; p_block8(w+block*stride,p.type,lane,v);
        for(uint g=0;g<8;++g) sum+=v[g]*x[block*256+g*32+lane];
    }
    sum=simd_sum(sum); if(lane==0) y[row]=sum;
}
kernel void particle_batch_vector_gemv(device const ParticleBatchRows &rows [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    constant LinearParams &p [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group.x*4+sg, first=group.y*4; if(row>=p.m || first>=count) return;
    device const uchar *w=weights+ulong(row)*p.rowBytes;
    uint stride=p.type==14 ? 210 : p.type==12 ? 144 : 176;
    float a=0,b=0,c=0,d=0;
    for(uint block=0;block<p.n/256;++block) {
        float v[8]; p_block8(w+block*stride,p.type,lane,v);
        for(uint g=0;g<8;++g) {
            uint i=block*256+g*32+lane;
            a+=v[g]*rows.input[first][i];
            if(first+1<count) b+=v[g]*rows.input[first+1][i];
            if(first+2<count) c+=v[g]*rows.input[first+2][i];
            if(first+3<count) d+=v[g]*rows.input[first+3][i];
        }
    }
    a=simd_sum(a); b=simd_sum(b); c=simd_sum(c); d=simd_sum(d);
    if(lane==0) {
        rows.output[first][row]=a;
        if(first+1<count) rows.output[first+1][row]=b;
        if(first+2<count) rows.output[first+2][row]=c;
        if(first+3<count) rows.output[first+3][row]=d;
    }
}
// Compact greedy sampling preserves finite-logit checks and lowest-ID ties.
// Each group emits {max bits, token ID, invalid flag}; no vocabulary copy.
kernel void particle_argmax(device const float *logits [[buffer(0)]], device uint *partial [[buffer(1)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint id=group*256+tid;
    float value=logits[id]; uint invalid=simd_max(uint(!isfinite(value)));
    value=isfinite(value) ? value : -INFINITY;
    float maximum=simd_max(value); uint winner=simd_min(value==maximum ? id : UINT_MAX);
    threadgroup float maxima[8]; threadgroup uint winners[8], bad[8];
    if(lane==0) { maxima[sg]=maximum; winners[sg]=winner; bad[sg]=invalid; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        float m=simd_max(lane<8 ? maxima[lane] : -INFINITY);
        uint w=simd_min(lane<8 && maxima[lane]==m ? winners[lane] : UINT_MAX);
        uint b=simd_max(lane<8 ? bad[lane] : 0);
        if(lane==0) { partial[group*3]=as_type<uint>(m); partial[group*3+1]=w; partial[group*3+2]=b; }
    }
}
kernel void particle_argmax_merge(device const uint *partial [[buffer(0)]], device uint *out [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float maximum=-INFINITY; uint winner=UINT_MAX, invalid=0;
    for(uint group=tid;group<1024;group+=256) {
        float value=as_type<float>(partial[group*3]); uint id=partial[group*3+1];
        if(value>maximum || (value==maximum && id<winner)) { maximum=value; winner=id; }
        invalid|=partial[group*3+2];
    }
    float m=simd_max(maximum); uint w=simd_min(maximum==m ? winner : UINT_MAX), b=simd_max(invalid);
    threadgroup float maxima[8]; threadgroup uint winners[8], bad[8];
    if(lane==0) { maxima[sg]=m; winners[sg]=w; bad[sg]=b; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        float top=simd_max(lane<8 ? maxima[lane] : -INFINITY);
        uint id=simd_min(lane<8 && maxima[lane]==top ? winners[lane] : UINT_MAX);
        uint invalid=simd_max(lane<8 ? bad[lane] : 0);
        if(lane==0) { out[0]=id; out[1]=invalid; }
    }
}

// One SIMD group owns a 128-expert selection. Bias affects ranking only;
// mixture weights come from the unbiased sigmoid scores in top-six order.
kernel void particle_route_six(device const float *logits [[buffer(0)]],
                               device const float *bias [[buffer(1)]],
                               device uint *ids [[buffer(2)]], device float *weights [[buffer(3)]],
                               uint lane [[thread_index_in_simdgroup]]) {
    float score[4], ranking[4], chosenScores[6];
    bool invalid = false;
    for (uint i=0; i<4; ++i) {
        uint id=lane+32*i;
        float x=logits[id], b=bias[id];
        invalid |= !isfinite(x) || !isfinite(b);
        float e=exp(-abs(x));
        score[i]=x>=0 ? 1.0f/(1.0f+e) : e/(1.0f+e);
        ranking[i]=score[i]+b;
    }
    bool failed=simd_any(invalid);
    if (failed) { if (lane==0) ids[6]=1; return; }
    for (uint rank=0; rank<6; ++rank) {
        float best=-INFINITY; uint id=0xffffffffu;
        for (uint i=0; i<4; ++i) {
            uint candidate=lane+32*i;
            if (ranking[i]>best || (ranking[i]==best && candidate<id)) { best=ranking[i]; id=candidate; }
        }
        float maximum=simd_max(best);
        uint selected=simd_min(best==maximum ? id : 0xffffffffu);
        float unbiased=simd_sum(lane==selected%32 ? score[selected/32] : 0.0f);
        if (lane==0) { ids[rank]=selected; chosenScores[rank]=unbiased; }
        if (lane==selected%32) ranking[selected/32]=-INFINITY;
    }
    if (lane==0) {
        float denominator=1e-20f;
        for (uint i=0; i<6; ++i) denominator+=chosenScores[i];
        for (uint i=0; i<6; ++i) weights[i]=chosenScores[i]/denominator*2.5f;
        ids[6]=0;
    }
}
