#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>

using namespace std;

// Build:
// nvcc -O3 -std=c++17 -I /home/i.afanasyev/nccl/build/include \
//   tensor_parallel_col.cu -L /home/i.afanasyev/nccl/build/lib -lnccl -lcublas -o tp_col_nccl.bin
//
// Runtime:
// export LD_LIBRARY_PATH=/home/i.afanasyev/nccl/build/lib:$LD_LIBRARY_PATH
//
// Example:
// ./tp_fc_nccl --gpus 2 --n 4096 --k 4096 --m 4096 --iters 20 --warmup 5
// ./tp_fc_nccl --gpus 1 --n 4096 --k 4096 --m 4096 --iters 20 --warmup 5
// ./tp_fc_nccl --gpus 2 --n 256 --k 256 --m 256 --iters 50 --warmup 10 --check 1

#define CHECK_CUDA(cmd)                                              \
    do                                                               \
    {                                                                \
        cudaError_t e = (cmd);                                       \
        if (e != cudaSuccess)                                        \
        {                                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",           \
                         __FILE__, __LINE__, cudaGetErrorString(e)); \
            std::exit(1);                                            \
        }                                                            \
    } while (0)

#define CHECK_CUBLAS(cmd)                                    \
    do                                                       \
    {                                                        \
        cublasStatus_t s = (cmd);                            \
        if (s != CUBLAS_STATUS_SUCCESS)                      \
        {                                                    \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", \
                         __FILE__, __LINE__, (int)s);        \
            std::exit(1);                                    \
        }                                                    \
    } while (0)

struct Args
{
    int gpus = 1;
    int N = 4096;
    int K = 4096;
    int M = 4096;
    int iters = 20;
    int warmup = 5;
    bool check = false;
    int samples = 256; // number of sampled elements to verify (fast)
};

static Args parse_args(int argc, char **argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        auto need = [&](const char *name)
        {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "Missing value for %s\n", name);
                std::exit(1);
            }
        };
        if (!std::strcmp(argv[i], "--gpus"))
        {
            need("--gpus");
            a.gpus = std::atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "--n"))
        {
            need("--n");
            a.N = std::atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "--k"))
        {
            need("--k");
            a.K = std::atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "--m"))
        {
            need("--m");
            a.M = std::atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "--iters"))
        {
            need("--iters");
            a.iters = std::atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "--warmup"))
        {
            need("--warmup");
            a.warmup = std::atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "--check"))
        {
            a.check = true;
        }
        else if (!std::strcmp(argv[i], "--samples"))
        {
            need("--samples");
            a.samples = std::atoi(argv[++i]);
        }
        else
        {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            std::exit(1);
        }
    }
    if (a.gpus < 1)
        a.gpus = 1;
    if (a.iters < 1)
        a.iters = 1;
    if (a.warmup < 0)
        a.warmup = 0;
    if (a.samples < 1)
        a.samples = 1;
    return a;
}

// Fill row-major matrix (rows x cols), ld = cols
static void init_rowmajor(std::vector<float> &a, int rows, int cols)
{
    a.resize((size_t)rows * (size_t)cols);
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            // deterministic, not too large
            a[(size_t)r * cols + c] = (r + 1) * 0.01f + (c + 1) * 0.001f;
        }
    }
}

struct GpuData
{
    int dev = -1;
    cudaStream_t stream{};
    cublasHandle_t blas{};

    float *dX = nullptr;      // will hold Xt (KxN row-major) but interpreted as X (NxK col-major)
    float *dW = nullptr;      // will hold Wt_shard (MshxK row-major) interpreted as Wsh (KxMsh col-major)
    float *dY = nullptr;      // Y_shard (NxMsh) col-major
    float *dY_full = nullptr; // Y_full (NxM) col-major
};

static inline size_t idx_colmajor(int row, int col, int ld_rows)
{
    return (size_t)row + (size_t)col * (size_t)ld_rows;
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0)
    {
        std::cout << "No CUDA devices found\n";
        return 1;
    }

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
              << " samples=" << args.samples
              << "\n";

    // ---- Host storage trick (no per-GPU packing) ----
    // Xt = X^T stored row-major as [K x N]
    // Wt = W^T stored row-major as [M x K]
    // These map to cuBLAS column-major X [N x K] and W [K x M] by raw pointer reuse.
    std::vector<float> hXt; // K x N row-major
    std::vector<float> hWt; // M x K row-major
    init_rowmajor(hXt, K, N);
    init_rowmajor(hWt, M, K);

    // Devices list
    std::vector<int> devs(ngpus);
    for (int i = 0; i < ngpus; ++i)
        devs[i] = i;

    // NCCL init
    std::cout << "[init] ncclCommInitAll...\n";
    std::vector<ncclComm_t> comms(ngpus);
    ncclCommInitAll(comms.data(), ngpus, devs.data());

    // Per-GPU init
    std::vector<GpuData> g(ngpus);
    for (int r = 0; r < ngpus; ++r)
    {
        g[r].dev = devs[r];
        CHECK_CUDA(cudaSetDevice(g[r].dev));
        CHECK_CUDA(cudaStreamCreate(&g[r].stream));
        CHECK_CUBLAS(cublasCreate(&g[r].blas));
        CHECK_CUBLAS(cublasSetStream(g[r].blas, g[r].stream));

        // alloc
        CHECK_CUDA(cudaMalloc(&g[r].dX, (size_t)K * (size_t)N * sizeof(float)));      // Xt
        CHECK_CUDA(cudaMalloc(&g[r].dW, (size_t)Msh * (size_t)K * sizeof(float)));    // Wt shard
        CHECK_CUDA(cudaMalloc(&g[r].dY, (size_t)N * (size_t)Msh * sizeof(float)));    // Y shard
        CHECK_CUDA(cudaMalloc(&g[r].dY_full, (size_t)N * (size_t)M * sizeof(float))); // full
    }

    // H2D copies (one-time)
    for (int r = 0; r < ngpus; ++r)
    {
        CHECK_CUDA(cudaSetDevice(g[r].dev));

        // X is replicated: copy Xt (KxN row-major)
        CHECK_CUDA(cudaMemcpyAsync(
            g[r].dX, hXt.data(),
            (size_t)K * (size_t)N * sizeof(float),
            cudaMemcpyHostToDevice, g[r].stream));

        // W shard: columns of W => rows of Wt, contiguous
        const int m0 = r * Msh; // rows range [m0, m0+Msh) in Wt
        const float *src_wt_shard = hWt.data() + (size_t)m0 * (size_t)K;
        CHECK_CUDA(cudaMemcpyAsync(
            g[r].dW, src_wt_shard,
            (size_t)Msh * (size_t)K * sizeof(float),
            cudaMemcpyHostToDevice, g[r].stream));
    }

    // Ensure copies are done before starting iterations
    for (int r = 0; r < ngpus; ++r)
    {
        CHECK_CUDA(cudaSetDevice(g[r].dev));
        CHECK_CUDA(cudaStreamSynchronize(g[r].stream));
    }
    std::cout << "[init] H2D done\n";

    const float alpha = 1.0f, beta = 0.0f;

    auto one_iteration = [&]()
    {
        // Enqueue GEMMs on each GPU (async)
        for (int r = 0; r < ngpus; ++r)
        {
            CHECK_CUDA(cudaSetDevice(g[r].dev));

            // Interpret:
            // dX memory = Xt (KxN row-major) == X (NxK col-major) with lda=N
            // dW memory = Wt_shard (MshxK row-major) == Wsh (KxMsh col-major) with ldb=K
            // dY is Ysh (NxMsh col-major) with ldc=N
            CHECK_CUBLAS(cublasSgemm(
                g[r].blas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, Msh, K,
                &alpha,
                g[r].dX, N, // lda = N (rows of X in col-major)
                g[r].dW, K, // ldb = K (rows of Wsh in col-major)
                &beta,
                g[r].dY, N // ldc = N
                ));
        }

        // AllGather shards into full Y on each GPU (must be grouped in single-thread multi-GPU)
        ncclGroupStart();
        for (int r = 0; r < ngpus; ++r)
        {
            CHECK_CUDA(cudaSetDevice(g[r].dev));
            ncclAllGather(
                g[r].dY,
                g[r].dY_full,
                (size_t)N * (size_t)Msh,
                ncclFloat,
                comms[r],
                g[r].stream);
        }
        ncclGroupEnd();

        // Sync (ensures iteration finished on all GPUs)
        for (int r = 0; r < ngpus; ++r)
        {
            CHECK_CUDA(cudaSetDevice(g[r].dev));
            CHECK_CUDA(cudaStreamSynchronize(g[r].stream));
        }
    };

    // Warmup
    for (int i = 0; i < args.warmup; ++i)
        one_iteration();

    // Timed iterations
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < args.iters; ++i)
        one_iteration();
    auto t1 = std::chrono::high_resolution_clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double ms_per_iter = (sec * 1000.0) / (double)args.iters;

    // Effective GEMM FLOPs (full GEMM is 2*N*K*M)
    // Note: this includes NCCL AllGather time too, so it's an "end-to-end TP layer" metric.
    long double flops = 2.0L * (long double)N * (long double)K * (long double)M;
    long double tflops = (flops / 1e12L) / ((long double)ms_per_iter / 1000.0L);

    std::cout << "[time] avg " << ms_per_iter << " ms/iter"
              << " | effective " << (double)tflops << " TFLOP/s (GEMM-equivalent)\n";

    // ---- Verification (sampling) ----
    if (args.check)
    {
        std::cout << "[check] sampling verification on GPU0...\n";
        CHECK_CUDA(cudaSetDevice(g[0].dev));

        std::mt19937 rng(123);
        std::uniform_int_distribution<int> dist_n(0, N - 1);
        std::uniform_int_distribution<int> dist_m(0, M - 1);

        int samples = std::min(args.samples, N * M);
        float max_abs = 0.0f;
        float max_rel = 0.0f;

        for (int s = 0; s < samples; ++s)
        {
            int n = dist_n(rng);
            int m = dist_m(rng);

            // Copy one element Y(n,m) from GPU0 (Y is col-major NxM)
            float y_gpu = 0.0f;
            size_t off = idx_colmajor(n, m, N);
            CHECK_CUDA(cudaMemcpy(&y_gpu, g[0].dY_full + off, sizeof(float), cudaMemcpyDeviceToHost));

            // Compute reference: y = sum_k X[n,k] * W[k,m]
            // X[n,k] == Xt[k,n] (Xt is KxN row-major => Xt[k*N + n])
            // W[k,m] == Wt[m,k] (Wt is MxK row-major => Wt[m*K + k])
            double y_ref = 0.0;
            const float *Xt = hXt.data();
            const float *Wt = hWt.data();
            for (int k = 0; k < K; ++k)
            {
                y_ref += (double)Xt[(size_t)k * (size_t)N + (size_t)n] *
                         (double)Wt[(size_t)m * (size_t)K + (size_t)k];
            }

            float abs_err = std::fabs((float)y_ref - y_gpu);
            float denom = std::max(1e-6f, std::fabs((float)y_ref));
            float rel_err = abs_err / denom;

            max_abs = std::max(max_abs, abs_err);
            max_rel = std::max(max_rel, rel_err);
        }

        std::cout << "[check] max_abs=" << max_abs << " max_rel=" << max_rel << "\n";
        std::cout << "        (If you see larger errors: first suspect layout/transpose assumptions.)\n";
    }

    // ---- Cleanup ----
    for (int r = 0; r < ngpus; ++r)
    {
        CHECK_CUDA(cudaSetDevice(g[r].dev));
        if (g[r].dX)
            CHECK_CUDA(cudaFree(g[r].dX));
        if (g[r].dW)
            CHECK_CUDA(cudaFree(g[r].dW));
        if (g[r].dY)
            CHECK_CUDA(cudaFree(g[r].dY));
        if (g[r].dY_full)
            CHECK_CUDA(cudaFree(g[r].dY_full));
        CHECK_CUBLAS(cublasDestroy(g[r].blas));
        CHECK_CUDA(cudaStreamDestroy(g[r].stream));
    }
    for (int r = 0; r < ngpus; ++r)
    {
        ncclCommDestroy(comms[r]);
    }

    return 0;
}
