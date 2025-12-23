#include "common.cuh"
#include <cuda_profiler_api.h>

using namespace std;

// Build:
// nvcc -O3 -std=c++17 -I /home/i.afanasyev/nccl/build/include \
//   tensor_parallel_row.cu -L /home/i.afanasyev/nccl/build/lib -lnccl -lcublas -o tp_row_nccl.bin
//
// Runtime:
// export LD_LIBRARY_PATH=/home/i.afanasyev/nccl/build/lib:$LD_LIBRARY_PATH

struct GpuData
{
    int dev = -1;
    cudaStream_t stream{};
    cublasHandle_t blas{};

    float *dXsh = nullptr; // Xt_shard (KshxN row-major) == X_shard (NxKsh col-major)
    float *dWsh = nullptr; // W_shard  (KshxM row-major) == (W_shard^T) (MxKsh col-major)
    float *dY = nullptr;   // Y (NxM col-major) partial then AllReduce(sum) -> full
};

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) { std::cout << "No CUDA devices found\n"; return 1; }

    int ngpus = std::min(args.gpus, device_count);
    const int N = args.N, K = args.K, M = args.M;

    if (K % ngpus != 0)
    {
        std::cerr << "For row-parallel TP, K must be divisible by gpus. "
                  << "K=" << K << " gpus=" << ngpus << "\n";
        return 1;
    }
    const int Ksh = K / ngpus;

    std::cout << "[config] gpus=" << ngpus
              << " N=" << N << " K=" << K << " M=" << M
              << " shard(K)=" << Ksh
              << " warmup=" << args.warmup
              << " iters=" << args.iters
              << " check=" << args.check
              << " samples=" << args.samples << "\n";

    // Host storage:
    // Xt = X^T row-major KxN (contiguous rows give X shards)
    // W  = W   row-major KxM (contiguous rows give W shards)
    std::vector<float> hXt; // K x N row-major
    std::vector<float> hW;  // K x M row-major
    init_rowmajor(hXt, K, N);
    init_rowmajor(hW, K, M);

    // NCCL init
    std::vector<int> devs(ngpus);
    for (int i = 0; i < ngpus; ++i) devs[i] = i;

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

        CHECK_CUDA(cudaMalloc(&g[r].dXsh, (size_t)Ksh * (size_t)N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&g[r].dWsh, (size_t)Ksh * (size_t)M * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&g[r].dY,   (size_t)N   * (size_t)M * sizeof(float)));
    }

    // H2D one-time: copy shards
    for (int r = 0; r < ngpus; ++r)
    {
        CHECK_CUDA(cudaSetDevice(g[r].dev));
        const int k0 = r * Ksh;

        // Xt_shard: rows [k0, k0+Ksh), contiguous in row-major
        const float* src_xt = hXt.data() + (size_t)k0 * (size_t)N;
        CHECK_CUDA(cudaMemcpyAsync(g[r].dXsh, src_xt,
                                   (size_t)Ksh * (size_t)N * sizeof(float),
                                   cudaMemcpyHostToDevice, g[r].stream));

        // W_shard: rows [k0, k0+Ksh), contiguous in row-major
        const float* src_w = hW.data() + (size_t)k0 * (size_t)M;
        CHECK_CUDA(cudaMemcpyAsync(g[r].dWsh, src_w,
                                   (size_t)Ksh * (size_t)M * sizeof(float),
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
        // Each GPU computes partial Y_part = X_sh (NxKsh) * W_sh (KshxM) => NxM
        // Memory tricks:
        // - dXsh stores Xt_sh (KshxN row-major) == X_sh (NxKsh col-major), lda=N
        // - dWsh stores W_sh  (KshxM row-major) == W_sh^T (MxKsh col-major)
        //   so use op(B)=T with ldb=M.
        for (int r = 0; r < ngpus; ++r)
        {
            CHECK_CUDA(cudaSetDevice(g[r].dev));
            CHECK_CUBLAS(cublasSgemm(
                g[r].blas,
                CUBLAS_OP_N, CUBLAS_OP_T,
                N, M, Ksh,
                &alpha,
                g[r].dXsh, N,  // A: X_sh (NxKsh) col-major
                g[r].dWsh, M,  // Bcol: W_sh^T (MxKsh) col-major, so B = (Bcol)^T
                &beta,
                g[r].dY, N));  // C: Y_part (NxM) col-major
        }

        // Sum partials across GPUs => full Y on every GPU
        CHECK_NCCL(ncclGroupStart());
        for (int r = 0; r < ngpus; ++r)
        {
            CHECK_CUDA(cudaSetDevice(g[r].dev));
            CHECK_NCCL(ncclAllReduce(
                g[r].dY, g[r].dY,
                (size_t)N * (size_t)M,
                ncclFloat,
                ncclSum,
                comms[r],
                g[r].stream));
        }
        CHECK_NCCL(ncclGroupEnd());

        for (int r = 0; r < ngpus; ++r)
        {
            CHECK_CUDA(cudaSetDevice(g[r].dev));
            CHECK_CUDA(cudaStreamSynchronize(g[r].stream));
        }
    };

    // Warmup
    for (int i = 0; i < args.warmup; ++i) one_iteration();

    cudaProfilerStart();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < args.iters; ++i) one_iteration();
    auto t1 = std::chrono::high_resolution_clock::now();
    cudaProfilerStop();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double ms_per_iter = (sec * 1000.0) / (double)args.iters;

    // Full GEMM-equivalent flops: 2*N*K*M (even though each GPU does 1/ngpus of K)
    long double flops = 2.0L * (long double)N * (long double)K * (long double)M;
    long double tflops = (flops / 1e12L) / ((long double)ms_per_iter / 1000.0L);

    std::cout << "[time] avg " << ms_per_iter << " ms/iter"
              << " | effective " << (double)tflops << " TFLOP/s (GEMM-equivalent)\n";

    if (args.check)
    {
        auto W_at = [&](int k, int m) -> float {
            // W is KxM row-major
            return hW[(size_t)k * (size_t)M + (size_t)m];
        };
        verify_samples("row-tp", g[0].dev, g[0].dY, N, K, M, hXt, args.samples, W_at);
    }

    // Cleanup
    for (int r = 0; r < ngpus; ++r)
    {
        CHECK_CUDA(cudaSetDevice(g[r].dev));
        if (g[r].dXsh) CHECK_CUDA(cudaFree(g[r].dXsh));
        if (g[r].dWsh) CHECK_CUDA(cudaFree(g[r].dWsh));
        if (g[r].dY)   CHECK_CUDA(cudaFree(g[r].dY));
        CHECK_CUBLAS(cublasDestroy(g[r].blas));
        CHECK_CUDA(cudaStreamDestroy(g[r].stream));
    }
    for (int r = 0; r < ngpus; ++r) CHECK_NCCL(ncclCommDestroy(comms[r]));

    return 0;
}
