#include "common.cuh"
#include <cuda_profiler_api.h>

using namespace std;

struct GpuData
{
    int dev = -1;
    cudaStream_t stream{};
    cublasHandle_t blas{};

    float *dX = nullptr;      // Xt (KxN row-major) interpreted as X (NxK col-major)
    float *dW = nullptr;      // Wt_shard (MshxK row-major) interpreted as Wsh (KxMsh col-major)
    float *dY = nullptr;      // Y_shard (NxMsh) col-major
    float *dY_full = nullptr; // Y_full (NxM) col-major
};

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) { std::cout << "No CUDA devices found\n"; return 1; }

    int ngpus = std::min(args.gpus, device_count);
    const int N = args.N, K = args.K, M = args.M;

    if (M % ngpus != 0)
    {
        std::cerr << "For column-parallel TP, M must be divisible by gpus. "
                  << "M=" << M << " gpus=" << ngpus << "\n";
        return 1;
    }
    const int Msh = M / ngpus;

    std::cout << "[config] gpus=" << ngpus
              << " N=" << N << " K=" << K << " M=" << M
              << " shard(M)=" << Msh
              << " warmup=" << args.warmup
              << " iters=" << args.iters
              << " check=" << args.check
              << " samples=" << args.samples << "\n";

    // Host storage (no per-GPU packing):
    // Xt = X^T row-major KxN  -> used as X col-major NxK
    // Wt = W^T row-major MxK  -> shard rows to get columns of W
    std::vector<float> hXt; // K x N
    std::vector<float> hWt; // M x K
    init_rowmajor(hXt, K, N);
    init_rowmajor(hWt, M, K);

    // NCCL init
    std::vector<int> devs(ngpus);
    for (int i = 0; i < ngpus; ++i) 
        devs[i] = i;

    std::cout << "ngpu " << ngpus << std::endl;
    std::vector<ncclComm_t> comms(ngpus);
    CHECK_NCCL(ncclCommInitAll(comms.data(), ngpus, devs.data()));

    // Per-GPU init/alloc
    std::vector<GpuData> g(ngpus);
    for (int r = 0; r < ngpus; ++r)
    {
        g[r].dev = devs[r];
        CHECK_CUDA(cudaSetDevice(g[r].dev));
        CHECK_CUDA(cudaStreamCreate(&g[r].stream));
        CHECK_CUBLAS(cublasCreate(&g[r].blas));
        CHECK_CUBLAS(cublasSetStream(g[r].blas, g[r].stream));

        CHECK_CUDA(cudaMalloc(&g[r].dX, (size_t)K * (size_t)N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&g[r].dW, (size_t)Msh * (size_t)K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&g[r].dY, (size_t)N * (size_t)Msh * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&g[r].dY_full, (size_t)N * (size_t)M * sizeof(float)));
    }

    // H2D one-time
    for (int r = 0; r < ngpus; ++r)
    {
        CHECK_CUDA(cudaSetDevice(g[r].dev));
        CHECK_CUDA(cudaMemcpyAsync(g[r].dX, hXt.data(),
                                   (size_t)K * (size_t)N * sizeof(float),
                                   cudaMemcpyHostToDevice, g[r].stream));

        const int m0 = r * Msh; // rows range [m0, m0+Msh) in Wt
        const float *src_wt_shard = hWt.data() + (size_t)m0 * (size_t)K;
        CHECK_CUDA(cudaMemcpyAsync(g[r].dW, src_wt_shard,
                                   (size_t)Msh * (size_t)K * sizeof(float),
                                   cudaMemcpyHostToDevice, g[r].stream));
    }
    for (int r = 0; r < ngpus; ++r)
    {
        CHECK_CUDA(cudaSetDevice(g[r].dev));
        CHECK_CUDA(cudaStreamSynchronize(g[r].stream));
    }

    const float alpha = 1.0f, beta = 0.0f;

    auto one_iteration = [&]()
    {
        // GEMMs (async)
        for (int r = 0; r < ngpus; ++r)
        {
            CHECK_CUDA(cudaSetDevice(g[r].dev));
            CHECK_CUBLAS(cublasSgemm(
                g[r].blas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, Msh, K,
                &alpha,
                g[r].dX, N,
                g[r].dW, K,
                &beta,
                g[r].dY, N));
        }

        // AllGather shards -> full Y on each GPU
        CHECK_NCCL(ncclGroupStart());
        for (int r = 0; r < ngpus; ++r)
        {
            CHECK_CUDA(cudaSetDevice(g[r].dev));
            CHECK_NCCL(ncclAllGather(
                g[r].dY,
                g[r].dY_full,
                (size_t)N * (size_t)Msh,
                ncclFloat,
                comms[r],
                g[r].stream));
        }
        CHECK_NCCL(ncclGroupEnd());

        // Sync
        for (int r = 0; r < ngpus; ++r)
        {
            CHECK_CUDA(cudaSetDevice(g[r].dev));
            CHECK_CUDA(cudaStreamSynchronize(g[r].stream));
        }
    };

    // Warmup
    for (int i = 0; i < args.warmup; ++i) one_iteration();

    // Compute
    cudaProfilerStart();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < args.iters; ++i) one_iteration();
    auto t1 = std::chrono::high_resolution_clock::now();
    cudaProfilerStop();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double ms_per_iter = (sec * 1000.0) / (double)args.iters;

    long double flops = 2.0L * (long double)N * (long double)K * (long double)M;
    long double tflops = (flops / 1e12L) / ((long double)ms_per_iter / 1000.0L);

    std::cout << "[time] avg " << ms_per_iter << " ms/iter"
              << " | effective " << (double)tflops << " TFLOP/s (GEMM-equivalent)\n";

    if (args.check)
    {
        auto W_at = [&](int k, int m) -> float {
            // W[k,m] = Wt[m,k], since Wt is MxK row-major
            return hWt[(size_t)m * (size_t)K + (size_t)k];
        };
        verify_samples("col-tp", g[0].dev, g[0].dY_full, N, K, M, hXt, args.samples, W_at);
    }

    // Cleanup
    for (int r = 0; r < ngpus; ++r)
    {
        CHECK_CUDA(cudaSetDevice(g[r].dev));
        if (g[r].dX) CHECK_CUDA(cudaFree(g[r].dX));
        if (g[r].dW) CHECK_CUDA(cudaFree(g[r].dW));
        if (g[r].dY) CHECK_CUDA(cudaFree(g[r].dY));
        if (g[r].dY_full) CHECK_CUDA(cudaFree(g[r].dY_full));
        CHECK_CUBLAS(cublasDestroy(g[r].blas));
        CHECK_CUDA(cudaStreamDestroy(g[r].stream));
    }
    for (int r = 0; r < ngpus; ++r) 
        CHECK_NCCL(ncclCommDestroy(comms[r]));

    return 0;
}
