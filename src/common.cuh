#pragma once

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
#include <sstream>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <cstdint>
#include <cassert>
#include <iomanip>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>

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

#define CHECK_NCCL(cmd) do {                                   \
  ncclResult_t res = (cmd);                                      \
  if (res != ncclSuccess) {                                      \
    std::fprintf(stderr, "NCCL error %s:%d: %s\n",             \
      __FILE__, __LINE__, ncclGetErrorString(res));              \
    std::exit(1);                                              \
  }                                                            \
} while(0)

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

static inline Args parse_args(int argc, char **argv)
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
    if (a.gpus < 1) a.gpus = 1;
    if (a.iters < 1) a.iters = 1;
    if (a.warmup < 0) a.warmup = 0;
    if (a.samples < 1) a.samples = 1;
    return a;
}

// Fill row-major matrix (rows x cols), ld = cols
static inline void init_rowmajor(std::vector<float> &a, int rows, int cols)
{
    a.resize((size_t)rows * (size_t)cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            a[(size_t)r * cols + c] = (r + 1) * 0.01f + (c + 1) * 0.001f;
}

static inline size_t idx_colmajor(int row, int col, int ld_rows)
{
    return (size_t)row + (size_t)col * (size_t)ld_rows;
}

// Sampling-based verification of Y (stored col-major NxM on GPU0).
// Host provides Xt = X^T stored row-major KxN, so X[n,k] == Xt[k*N + n].
// Weight accessor must return W[k,m] for any k,m.
template <typename WeightAccessor>
static inline void verify_samples(
    const char* tag,
    int device,
    const float* dY_colmajor, // NxM col-major
    int N, int K, int M,
    const std::vector<float>& hXt_rowmajor, // KxN row-major
    int samples,
    WeightAccessor W_at // float W_at(int k, int m)
)
{
    std::cout << "[check] " << tag << " sampling verification on GPU" << device << "...\n";
    CHECK_CUDA(cudaSetDevice(device));

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist_n(0, N - 1);
    std::uniform_int_distribution<int> dist_m(0, M - 1);

    samples = std::min(samples, N * M);
    float max_abs = 0.0f;
    float max_rel = 0.0f;

    for (int s = 0; s < samples; ++s)
    {
        int n = dist_n(rng);
        int m = dist_m(rng);

        float y_gpu = 0.0f;
        size_t off = idx_colmajor(n, m, N);
        CHECK_CUDA(cudaMemcpy(&y_gpu, dY_colmajor + off, sizeof(float), cudaMemcpyDeviceToHost));

        double y_ref = 0.0;
        const float* Xt = hXt_rowmajor.data();
        for (int k = 0; k < K; ++k)
        {
            y_ref += (double)Xt[(size_t)k * (size_t)N + (size_t)n] *
                     (double)W_at(k, m);
        }

        float abs_err = std::fabs((float)y_ref - y_gpu);
        float denom = std::max(1e-6f, std::fabs((float)y_ref));
        float rel_err = abs_err / denom;

        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
    }

    std::cout << "[check] max_abs=" << max_abs << " max_rel=" << max_rel << "\n";
}
