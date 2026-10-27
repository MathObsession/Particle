// SIMD lanes share native K-quant scales; FP32 product and accumulation order
// remain identical. Every active SIMD group owns one output row/block.
inline void p_cooperative_block8(device const uchar *p, uint type, uint lane, thread float *v) {
    if(type==14) { p_block8(p,type,lane,v); return; } else {
        uint group=lane&7; device const uchar *s=p+4;
        uint sc=group<4 ? s[group]&63 : (s[group+4]&15)|((s[group-4]>>6)<<4);
        uint mn=group<4 ? s[group+4]&63 : (s[group+4]>>4)|((s[group]>>6)<<4);
        float scaled=p_half(p)*float(sc), minimum=p_half(p+2)*float(mn);
        uint high=type==13 ? p[16+lane] : 0;
        for(uint g=0;g<8;++g) {
            uint packed=p[(type==12 ? 16 : 48)+(g/2)*32+lane];
            uint q=(packed>>((g%2)*4))&15;
            if(type==13) q|=((high>>g)&1)<<4;
            v[g]=simd_shuffle(scaled,ushort(g))*float(q)-simd_shuffle(minimum,ushort(g));
        }
    }
}

kernel void particle_cooperative_vector_gemv(device const uchar *weights [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]], constant LinearParams &p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group*4+sg; if(row>=p.m) return;
    device const uchar *w=weights+ulong(row)*p.rowBytes;
    uint stride=p.type==14 ? 210 : p.type==12 ? 144 : 176;
    float sum=0;
    for(uint block=0;block<p.n/256;++block) {
        float v[8]; p_cooperative_block8(w+block*stride,p.type,lane,v);
        for(uint g=0;g<8;++g) sum+=v[g]*x[block*256+g*32+lane];
    }
    sum=simd_sum(sum); if(lane==0) y[row]=sum;
}

kernel void particle_cooperative_batch_vector_gemv(device const ParticleBatchRows &rows [[buffer(0)]], device const uchar *weights [[buffer(1)]],
    constant LinearParams &p [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group.x*4+sg, first=group.y*4; if(row>=p.m || first>=count) return;
    device const uchar *w=weights+ulong(row)*p.rowBytes;
    uint stride=p.type==14 ? 210 : p.type==12 ? 144 : 176;
    float a=0,b=0,c=0,d=0;
    for(uint block=0;block<p.n/256;++block) {
        float v[8]; p_cooperative_block8(w+block*stride,p.type,lane,v);
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

kernel void particle_cooperative_vector_gate_up(device const uchar *gate [[buffer(0)]], device const uchar *up [[buffer(1)]],
                                     device const float *x [[buffer(2)]], device float *out [[buffer(3)]],
                                     constant LinearParams &p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
                                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group*4+sg; if(row>=p.m) return;
    device const uchar *g=gate+ulong(row)*p.rowBytes, *u=up+ulong(row)*p.rowBytes;
    uint stride=p.type==14 ? 210 : p.type==12 ? 144 : 176;
    float a=0,b=0;
    for(uint block=0;block<p.n/256;++block) {
        float gv[8],uv[8]; p_cooperative_block8(g+block*stride,p.type,lane,gv); p_cooperative_block8(u+block*stride,p.type,lane,uv);
        for(uint j=0;j<8;++j) { float v=x[block*256+j*32+lane]; a+=gv[j]*v; b+=uv[j]*v; }
    }
    a=simd_sum(a); b=simd_sum(b);
    if(lane==0) out[row]=(a/(1+exp(-a)))*b;
}

kernel void particle_cooperative_batch_vector_gate_up(device const ParticleBatchRows &rows [[buffer(0)]],
    device const uchar *gate [[buffer(1)]], device const uchar *up [[buffer(2)]],
    constant LinearParams &p [[buffer(3)]], constant uint &count [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group.x*4+sg, first=group.y*4; if(row>=p.m || first>=count) return;
    device const uchar *g=gate+ulong(row)*p.rowBytes, *u=up+ulong(row)*p.rowBytes;
    uint stride=p.type==14 ? 210 : p.type==12 ? 144 : 176;
    float4 a=0,b=0;
    for(uint block=0;block<p.n/256;++block) {
        float gv[8],uv[8]; p_cooperative_block8(g+block*stride,p.type,lane,gv); p_cooperative_block8(u+block*stride,p.type,lane,uv);
        for(uint j=0;j<8;++j) {
            uint i=block*256+j*32+lane; float v=rows.input[first][i];
            a.x+=gv[j]*v; b.x+=uv[j]*v;
            if(first+1<count) { v=rows.input[first+1][i]; a.y+=gv[j]*v; b.y+=uv[j]*v; }
            if(first+2<count) { v=rows.input[first+2][i]; a.z+=gv[j]*v; b.z+=uv[j]*v; }
            if(first+3<count) { v=rows.input[first+3][i]; a.w+=gv[j]*v; b.w+=uv[j]*v; }
        }
    }
    a.x=simd_sum(a.x); a.y=simd_sum(a.y); a.z=simd_sum(a.z); a.w=simd_sum(a.w);
    b.x=simd_sum(b.x); b.y=simd_sum(b.y); b.z=simd_sum(b.z); b.w=simd_sum(b.w);
    if(lane==0) {
        rows.output[first][row]=(a.x/(1+exp(-a.x)))*b.x;
        if(first+1<count) rows.output[first+1][row]=(a.y/(1+exp(-a.y)))*b.y;
        if(first+2<count) rows.output[first+2][row]=(a.z/(1+exp(-a.z)))*b.z;
        if(first+3<count) rows.output[first+3][row]=(a.w/(1+exp(-a.w)))*b.w;
    }
}

kernel void particle_cooperative_expert_gate_up_vector(device const ParticleExpertBindings &b [[buffer(0)]],
    device const float *x [[buffer(1)]], constant ParticleExpertParams &p [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group.x*4+sg, expert=group.y; if(row>=p.hidden || expert>=p.count) return;
    device const uchar *g=b.gate[expert]+ulong(row)*p.gateRow, *u=b.up[expert]+ulong(row)*p.upRow;
    uint gs=p.gateType==14 ? 210 : p.gateType==12 ? 144 : 176;
    uint us=p.upType==14 ? 210 : p.upType==12 ? 144 : 176;
    float a=0,c=0;
    for(uint block=0;block<p.n/256;++block) {
        float gv[8],uv[8]; p_cooperative_block8(g+block*gs,p.gateType,lane,gv); p_cooperative_block8(u+block*us,p.upType,lane,uv);
        for(uint j=0;j<8;++j) { float v=x[block*256+j*32+lane]; a+=gv[j]*v; c+=uv[j]*v; }
    }
    a=simd_sum(a); c=simd_sum(c);
    if(lane==0) b.scratch[expert][row]=(a/(1+exp(-a)))*c;
}

kernel void particle_cooperative_expert_down_vector(device const ParticleExpertBindings &b [[buffer(0)]],
    constant ParticleExpertParams &p [[buffer(1)]], uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group.x*4+sg, expert=group.y; if(row>=p.n || expert>=p.count) return;
    device const uchar *w=b.down[expert]+ulong(row)*p.downRow;
    uint stride=p.downType==14 ? 210 : p.downType==12 ? 144 : 176;
    float sum=0;
    for(uint block=0;block<p.hidden/256;++block) {
        float v[8]; p_cooperative_block8(w+block*stride,p.downType,lane,v);
        for(uint j=0;j<8;++j) sum+=v[j]*b.scratch[expert][block*256+j*32+lane];
    }
    sum=simd_sum(sum);
    if(lane==0) b.output[expert][row]=sum;
}
