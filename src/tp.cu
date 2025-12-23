#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <vector>
#include <cuda_profiler_api.h>

using namespace std;

// nvcc -I /home/i.afanasyev/nccl/build/include/ tp.cu -L /home/i.afanasyev/nccl/build/lib/ -lcublas -lnccl -o tp_fc_nccl
// export LD_LIBRARY_PATH=/home/i.afanasyev/nccl/build/lib/
// for profiling
// export PATH="/home/i.afanasyev/opt/nsys-cli/extract/opt/nvidia/nsight-systems-cli/2025.6.1/bin:$PATH"

#define CHECK_CUDA(cmd) do {                                   \
  cudaError_t e = (cmd);                                       \
  if (e != cudaSuccess) {                                      \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n",             \
      __FILE__, __LINE__, cudaGetErrorString(e));              \
    std::exit(1);                                              \
  }                                                            \
} while(0)

#define CHECK_CUBLAS(cmd) do {                                 \
  cublasStatus_t s = (cmd);                                    \
  if (s != CUBLAS_STATUS_SUCCESS) {                            \
    std::fprintf(stderr, "cuBLAS error %s:%d: %d\n",           \
      __FILE__, __LINE__, (int)s);                             \
    std::exit(1);                                              \
  }                                                            \
} while(0)

#define CHECK_NCCL(cmd) do {                                   \
  ncclResult_t r = (cmd);                                      \
  if (r != ncclSuccess) {                                      \
    std::fprintf(stderr, "NCCL error %s:%d: %s\n",             \
      __FILE__, __LINE__, ncclGetErrorString(r));              \
    std::exit(1);                                              \
  }                                                            \
} while(0)


struct Args {
    std::string mode = "col"; // "col" or "row"
    int gpus = 1;
    int check = 0;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for(int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                std::exit(1);
            }
        };
        if(!std::strcmp(argv[i], "--mode")) { need("--mode"); a.mode = argv[++i]; }
        else if (!std::strcmp(argv[i], "--gpus")) { need("--gpus"); a.gpus = std::atoi(argv[++i]); }
        else if (!std::strcmp(argv[i], "--check")) { need("--check"); a.check = std::atoi(argv[++i]); }
        else {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            std::exit(1);
        }
    }
    if(a.mode != "col" && a.mode != "row") {
        std::fprintf(stderr, "--mode must be 'col' or 'row'\n");
        std::exit(1);
    }
    return a;
}

template <typename T>
void init_mat(vector<T> &data, int rows, int cols, int ld) {
    // Simple deterministic init
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            data[row * ld + col] = (row + 1) * 0.01f + (col + 1) * 0.001f;
        }
    }
}

struct GpuData {
    int dev = -1;
    cudaStream_t stream{};
    cublasHandle_t blas{};
    float* dX = nullptr;
    float* dW = nullptr;
    float* dY = nullptr;
    float* dY_full = nullptr;
};

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if(device_count == 0) {
        std::cout << "No CUDA devices found" << std::endl;
        return 1;
    }

    int ngpus = args.gpus;

    if(ngpus > device_count) {
        std::cout << "Not enough CUDA devices found" << std::endl;
        return 1;
    }

    int size_sq = 4096;
    int N = size_sq;
    int K = size_sq;
    int M = size_sq;

    std::cout << N << " " << K << " " << M << std::endl;

    std::vector<float> hX((size_t)N * K);
    std::vector<float> hW((size_t)K * M);

    init_mat<float>(hX, N, K, K);
    init_mat<float>(hW, K, M, M);

    std::vector<int> devs(ngpus);
    for (int i = 0; i < ngpus; ++i)
        devs[i] = i;

    std::cout << "NCCL init" << std::endl;

    std::vector<ncclComm_t> comms(ngpus);
    CHECK_NCCL(ncclCommInitAll(comms.data(), ngpus, devs.data()));

    std::vector<GpuData> gpu_data(ngpus);
    for (int gpu_id = 0; gpu_id < ngpus; ++gpu_id) {
        gpu_data[gpu_id].dev = devs[gpu_id];
        CHECK_CUDA(cudaSetDevice(gpu_data[gpu_id].dev));
        CHECK_CUDA(cudaStreamCreate(&gpu_data[gpu_id].stream));
        CHECK_CUBLAS(cublasCreate(&gpu_data[gpu_id].blas));
        CHECK_CUBLAS(cublasSetStream(gpu_data[gpu_id].blas, gpu_data[gpu_id].stream));
    }

    cudaProfilerStart();

    const float alpha = 1.0f, beta = 0.0f;
    if (args.mode == "col") {
        std::cout << "[status] Doing col-wise TP" << std::endl;

        // we split hW, which is K rows * M cols, among GPUs col-wise
        // e.g. each M/mgpu cols go to seprate GPU

        int M_shard_size = M / ngpus;

        // move data for each GPU
        for (int gpu_id = 0; gpu_id < ngpus; gpu_id++) {
            CHECK_CUDA(cudaSetDevice(gpu_data[gpu_id].dev));

            CHECK_CUDA(cudaMalloc(&gpu_data[gpu_id].dX, (size_t)N * K * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&gpu_data[gpu_id].dW, (size_t)K * M_shard_size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&gpu_data[gpu_id].dY, (size_t)N * M_shard_size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&gpu_data[gpu_id].dY_full, (size_t)N * M * sizeof(float)));

            // X is replicated on all GPUs in column-parallel
            CHECK_CUDA(cudaMemcpyAsync(gpu_data[gpu_id].dX, hX.data(), (size_t)N * K * sizeof(float),
                                       cudaMemcpyHostToDevice, gpu_data[gpu_id].stream));

            // W shard: columns [r*Msh, (r+1)*Msh)
            int ldW_shard = M_shard_size;
            int ldW = M;
            std::vector<float> hW_shard((size_t)K * M_shard_size);
            for (int col = 0; col < M_shard_size; col++) {
                for (int row = 0; row < K; row++) {
                    hW_shard[row * ldW_shard + col] = hW[row * ldW + col];
                }
            }

            CHECK_CUDA(cudaMemcpyAsync(gpu_data[gpu_id].dW, hW_shard.data(),
                        (size_t)K * M_shard_size * sizeof(float),
                        cudaMemcpyHostToDevice, gpu_data[gpu_id].stream));
        }

        std::cout << "[status] h2d copy done" << std::endl;

        for(int iter = 0; iter < 20; iter++) {
            // compute
            for (int gpu_id = 0; gpu_id < ngpus; gpu_id++) {
                cudaSetDevice(gpu_data[gpu_id].dev);
                // Y_sh = X(NxK) * W_sh(KxMsh) -> Y_sh(NxMsh)
                cublasSgemm(
                    gpu_data[gpu_id].blas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M_shard_size, K,
                    &alpha,
                    gpu_data[gpu_id].dX, N,
                    gpu_data[gpu_id].dW, K,
                    &beta,
                    gpu_data[gpu_id].dY, N
                );
            }

            // materealize output on all GPUS
            ncclGroupStart();
            for (int gpu_id = 0; gpu_id < ngpus; gpu_id++) {
                cudaSetDevice(gpu_data[gpu_id].dev);
                // AllGather shards into full Y on every GPU
                ncclAllGather(
                    gpu_data[gpu_id].dY, // send buf
                    gpu_data[gpu_id].dY_full, // recv buf
                    (size_t)N * M_shard_size, // send size
                    ncclFloat, // dtype
                    comms[gpu_id],
                    gpu_data[gpu_id].stream
                );
            }
            ncclGroupEnd();

            // Sync
            for (int gpu_id = 0; gpu_id < ngpus; gpu_id++) {
                CHECK_CUDA(cudaSetDevice(gpu_data[gpu_id].dev));
                CHECK_CUDA(cudaStreamSynchronize(gpu_data[gpu_id].stream));
            } 
        }

        /*if (args.check) {
            // Copy result from GPU0 and compare
            std::vector<float> hY((size_t)N * M);
            CHECK_CUDA(cudaSetDevice(gpu_data[0].dev));
            CHECK_CUDA(cudaMemcpy(hY.data(), gpu_data[0].dY_full, (size_t)N * M * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            for(int i = 0; i < M; i++) {
                for(int j = 0; j < N; j++) {
                    std::cout << hY[i*N + j] << " "; 
                }
                std::cout << std::endl;
            }
        }*/
    }

    cudaProfilerStop();

    return 0;
}