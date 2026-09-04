// overhead_microbench.cu
// Isolate the three overhead sources in a batch=1 speculative-decode loop:
//   (1) kernel-launch overhead, (2) forward-pass compute (here: memory-bound
//   weight read), (3) CPU<->GPU D2H transition (the .item() sync).
//
// We MODEL a "draft forward" as one memory-bound kernel that reads a large
// buffer (~ model weight size) — because batch=1 decoding is bandwidth-bound by
// reading all weights to produce one token. A "step" = K such kernels in
// sequence (the draft proposing K tokens). We compare:
//   A  pure launch floor      : K*S trivial kernels (no memory work)
//   B  sequential (no graph)   : K*S mem-bound kernels, normal launches
//   C  CUDA graph             : capture K mem-bound kernels, replay S times
//   D  B + D2H sync per step  : adds a cudaMemcpy(int) host<-device each step
// Derived:
//   launch overhead total  ~= B - C        (graph removes per-launch cost)
//   transition overhead    ~= D - B        (the per-step .item() sync)
//   compute/memory floor   ~= C            (launch removed, pure work left)
//
// Build:  nvcc -O3 -arch=native overhead_microbench.cu -o overhead_microbench
//         (or -arch=sm_80 for A100, sm_89 for RTX 40xx/50xx)
// Run:    ./overhead_microbench [weight_MB=1024] [K=4] [S=22]

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// Memory-bound kernel: grid-stride read of `n` floats, reduce into out[0].
// Mimics a forward pass that must read all weights from HBM.
__global__ void memBound(const float* __restrict__ w, size_t n, float* out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n; i += stride) acc += w[i];
    atomicAdd(out, acc);
}
// Trivial kernel: pure launch-overhead probe (negligible work).
__global__ void tiny(float* out) { if (threadIdx.x==0 && blockIdx.x==0) out[0]+=1.f; }

static float timeMs(cudaEvent_t a, cudaEvent_t b){ float ms; CK(cudaEventElapsedTime(&ms,a,b)); return ms; }

int main(int argc, char** argv){
    size_t weightMB = argc>1 ? atoll(argv[1]) : 1024;   // ~ model weight size
    int K = argc>2 ? atoi(argv[2]) : 4;                 // draft tokens per step
    int S = argc>3 ? atoi(argv[3]) : 22;                // steps (e.g. 96 toks / ~4.3)
    size_t n = weightMB*1024*1024/sizeof(float);
    int dev; CK(cudaGetDevice(&dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    double peakBW = 2.0*p.memoryClockRate*1e3*(p.memoryBusWidth/8)/1e9; // GB/s
    printf("GPU=%s  weight=%zu MB (%zu floats)  K=%d  S=%d  peakHBM~%.0f GB/s\n",
           p.name, weightMB, n, K, S, peakBW);

    float *w, *out; CK(cudaMalloc(&w, n*sizeof(float))); CK(cudaMalloc(&out, sizeof(float)));
    CK(cudaMemset(w, 0, n*sizeof(float)));
    int threads=256, blocks=std::min((size_t)1024,(n+threads-1)/threads);
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // warmup
    for(int i=0;i<5;i++) memBound<<<blocks,threads,0,st>>>(w,n,out);
    CK(cudaStreamSynchronize(st));

    // A: pure launch floor
    CK(cudaEventRecord(e0,st));
    for(int s=0;s<S;s++) for(int k=0;k<K;k++) tiny<<<1,1,0,st>>>(out);
    CK(cudaEventRecord(e1,st)); CK(cudaEventSynchronize(e1));
    float tA=timeMs(e0,e1);

    // B: sequential mem-bound kernels, normal launches
    CK(cudaEventRecord(e0,st));
    for(int s=0;s<S;s++) for(int k=0;k<K;k++) memBound<<<blocks,threads,0,st>>>(w,n,out);
    CK(cudaEventRecord(e1,st)); CK(cudaEventSynchronize(e1));
    float tB=timeMs(e0,e1);

    // C: CUDA graph — capture K mem-bound kernels once, replay S times
    cudaGraph_t graph; cudaGraphExec_t gexec;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for(int k=0;k<K;k++) memBound<<<blocks,threads,0,st>>>(w,n,out);
    CK(cudaStreamEndCapture(st,&graph));
    CK(cudaGraphInstantiate(&gexec,graph,0));
    CK(cudaEventRecord(e0,st));
    for(int s=0;s<S;s++) CK(cudaGraphLaunch(gexec,st));
    CK(cudaEventRecord(e1,st)); CK(cudaEventSynchronize(e1));
    float tC=timeMs(e0,e1);

    // D: B + per-step D2H sync (mimics .item())
    float hostScalar;
    CK(cudaEventRecord(e0,st));
    for(int s=0;s<S;s++){
        for(int k=0;k<K;k++) memBound<<<blocks,threads,0,st>>>(w,n,out);
        CK(cudaMemcpyAsync(&hostScalar,out,sizeof(float),cudaMemcpyDeviceToHost,st));
        CK(cudaStreamSynchronize(st));   // the host<-device stall
    }
    CK(cudaEventRecord(e1,st)); CK(cudaEventSynchronize(e1));
    float tD=timeMs(e0,e1);

    int totalK = K*S;
    printf("\n--- totals over %d kernels (%d steps x K=%d) ---\n", totalK, S, K);
    printf("A pure-launch floor      : %8.3f ms  (%.1f us/launch)\n", tA, tA*1000/totalK);
    printf("B sequential mem-bound   : %8.3f ms\n", tB);
    printf("C CUDA graph (no launch) : %8.3f ms\n", tC);
    printf("D B + D2H sync per step  : %8.3f ms\n", tD);
    printf("\n--- decomposition ---\n");
    printf("forward/compute floor (C)        : %8.3f ms  (%.1f%%)\n", tC, 100.0*tC/tD);
    printf("kernel-launch overhead (B - C)   : %8.3f ms  (%.1f%%)\n", tB-tC, 100.0*(tB-tC)/tD);
    printf("CPU<->GPU transitions  (D - B)   : %8.3f ms  (%.1f%%)\n", tD-tB, 100.0*(tD-tB)/tD);
    double achBW = (double)n*sizeof(float)*totalK / (tC/1e3) / 1e9;
    printf("\nC achieved HBM bandwidth         : %.0f GB/s of ~%.0f peak => %s\n",
           achBW, peakBW, achBW>0.6*peakBW ? "MEMORY-BOUND (compute floor is HBM bandwidth)" : "not bandwidth-saturated");
    CK(cudaFree(w)); CK(cudaFree(out));
    return 0;
}
