#include "../common.cuh"


// ---------------------- simple barrier (C++17) ----------------------
struct Barrier
{
    explicit Barrier(int n) : n_(n) {}
    void wait()
    {
        std::unique_lock<std::mutex> lk(m_);
        int my_phase = phase_;
        if (++arrived_ == n_)
        {
            arrived_ = 0;
            phase_++;
            cv_.notify_all();
        }
        else
        {
            cv_.wait(lk, [&]
                     { return phase_ != my_phase; });
        }
    }
    int n_;
    int arrived_{0};
    int phase_{0};
    std::mutex m_;
    std::condition_variable cv_;
};

// ---------------------- device kernels ----------------------

// logits is column-major matrix (E x B): each token is a column of length E.
// This kernel computes softmax over E for each token column + top1 expert.
__global__ void softmax_top1_kernel(float *logits, int E, int B, int *expert_idx)
{
    int t = blockIdx.x; // token column
    if (t >= B)
        return;

    float *col = logits + t * E;

    // block reduction helpers (assume blockDim <= 256)
    __shared__ float sh[256];

    // 1) max
    float local_max = -1e20f;
    for (int i = threadIdx.x; i < E; i += blockDim.x)
    {
        local_max = fmaxf(local_max, col[i]);
    }
    sh[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + stride]);
        __syncthreads();
    }
    float maxv = sh[0];

    // 2) sum exp
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x)
    {
        local_sum += expf(col[i] - maxv);
    }
    sh[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    float sumv = sh[0] + 1e-20f;

    // 3) write softmax back into logits (optional but "real-life-ish")
    for (int i = threadIdx.x; i < E; i += blockDim.x)
    {
        col[i] = expf(col[i] - maxv) / sumv;
    }
    __syncthreads();

    // 4) argmax softmax (top-1)
    if (threadIdx.x == 0)
    {
        int best = 0;
        float bestv = col[0];
        for (int i = 1; i < E; i++)
        {
            float v = col[i];
            if (v > bestv)
            {
                bestv = v;
                best = i;
            }
        }
        expert_idx[t] = best;
    }
}

// counts[e] = number of tokens routed to expert e
__global__ void histogram_experts_kernel(const int *expert_idx, int B, int E, int *counts)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < B)
    {
        int e = expert_idx[t];
        if (0 <= e && e < E)
            atomicAdd(&counts[e], 1);
    }
}

// Pack tokens into send buffer grouped by global expert id.
// Layout: tokens are stored in expert-id order, each token is a contiguous vector of length d.
// X is d x B column-major: token t is at X + t*d.
__global__ void pack_by_expert_kernel(
    const float *X, int d, int B,
    const int *expert_idx,
    const int *expert_offsets, // exclusive offsets, length E+1
    int *expert_counters,      // length E, starts at 0
    float *sendAct,            // size B*d
    int *sendTok,              // size B
    int rank)
{

    int t = blockIdx.x; // one block per token
    if (t >= B)
        return;

    int e = expert_idx[t];

    __shared__ int pos;
    if (threadIdx.x == 0)
    {
        int off = expert_offsets[e];
        int k = atomicAdd(&expert_counters[e], 1);
        pos = off + k;
        sendTok[pos] = rank * B + t; // stable id for demo
    }
    __syncthreads();

    int out_base = pos * d;
    int in_base = t * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x)
    {
        sendAct[out_base + i] = X[in_base + i];
    }
}

// ---------------------- helpers ----------------------

static int get_int_arg(int argc, char **argv, const std::string &key, int def)
{
    for (int i = 1; i + 1 < argc; i++)
    {
        if (argv[i] == key)
            return std::stoi(argv[i + 1]);
    }
    return def;
}

static void fill_random(std::vector<float> &v, uint64_t seed, float scale = 0.1f)
{
    std::mt19937 rng((uint32_t)seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto &x : v)
        x = dist(rng);
}

struct RankCtx
{
    int rank = 0;
    int dev = 0;
    int N = 0;

    int experts_per_gpu = 0;
    int E_total = 0;
    int B = 0;
    int d = 0;

    ncclComm_t comm{};
    cudaStream_t stream{};
    cublasHandle_t cublas{};

    // device buffers
    float *X_d = nullptr;        // B*d
    float *Wgate_d = nullptr;    // d*E_total
    float *logits_d = nullptr;   // E_total*B
    int *expert_idx_d = nullptr; // B

    int *counts_d = nullptr;   // E_total
    int *offsets_d = nullptr;  // E_total+1
    int *counters_d = nullptr; // E_total

    float *sendAct_d = nullptr; // B*d
    int *sendTok_d = nullptr;   // B

    int *allCounts_d = nullptr; // N*E_total

    // recv buffers: allocate worst-case N*B tokens
    float *recvAct_d = nullptr; // (N*B)*d
    int *recvTok_d = nullptr;   // N*B
    float *outAct_d = nullptr;  // (N*B)*d

    // per-rank expert weights (local experts only)
    std::vector<float *> Wexp_d; // experts_per_gpu pointers, each d*d

    // host
    std::vector<int> counts_h;  // E_total
    std::vector<int> offsets_h; // E_total+1

    std::vector<int> allCounts_h; // N*E_total

    std::vector<int> sendCountRank_h; // N
    std::vector<int> sendOffRank_h;   // N+1

    std::vector<int> recvCountRank_h; // N
    std::vector<int> recvOffRank_h;   // N+1

    int total_recv_tokens = 0;
    std::string log;
    double checksum = 0.0;
};

static void compute_exclusive_scan(const std::vector<int> &counts, std::vector<int> &offsets)
{
    offsets.resize(counts.size() + 1);
    offsets[0] = 0;
    for (size_t i = 0; i < counts.size(); i++)
        offsets[i + 1] = offsets[i] + counts[i];
}

static void rank_thread(RankCtx *ctx,
                        const std::vector<float> &Wgate_h,
                        Barrier *bar)
{
    CHECK_CUDA(cudaSetDevice(ctx->dev));

    CHECK_CUDA(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));
    CHECK_CUBLAS(cublasCreate(&ctx->cublas));
    CHECK_CUBLAS(cublasSetStream(ctx->cublas, ctx->stream));

    const int N = ctx->N;
    const int E_total = ctx->E_total;
    const int B = ctx->B;
    const int d = ctx->d;
    const int experts_per_gpu = ctx->experts_per_gpu;

    // Allocate device buffers
    CHECK_CUDA(cudaMalloc(&ctx->X_d, sizeof(float) * B * d));
    CHECK_CUDA(cudaMalloc(&ctx->Wgate_d, sizeof(float) * d * E_total));
    CHECK_CUDA(cudaMalloc(&ctx->logits_d, sizeof(float) * E_total * B));
    CHECK_CUDA(cudaMalloc(&ctx->expert_idx_d, sizeof(int) * B));

    CHECK_CUDA(cudaMalloc(&ctx->counts_d, sizeof(int) * E_total));
    CHECK_CUDA(cudaMalloc(&ctx->offsets_d, sizeof(int) * (E_total + 1)));
    CHECK_CUDA(cudaMalloc(&ctx->counters_d, sizeof(int) * E_total));

    CHECK_CUDA(cudaMalloc(&ctx->sendAct_d, sizeof(float) * B * d));
    CHECK_CUDA(cudaMalloc(&ctx->sendTok_d, sizeof(int) * B));

    CHECK_CUDA(cudaMalloc(&ctx->allCounts_d, sizeof(int) * N * E_total));

    const int max_recv_tokens = N * B;
    CHECK_CUDA(cudaMalloc(&ctx->recvAct_d, sizeof(float) * max_recv_tokens * d));
    CHECK_CUDA(cudaMalloc(&ctx->recvTok_d, sizeof(int) * max_recv_tokens));
    CHECK_CUDA(cudaMalloc(&ctx->outAct_d, sizeof(float) * max_recv_tokens * d));

    // Local experts weights
    ctx->Wexp_d.resize(experts_per_gpu, nullptr);
    for (int le = 0; le < experts_per_gpu; le++)
    {
        CHECK_CUDA(cudaMalloc(&ctx->Wexp_d[le], sizeof(float) * d * d));
    }

    // Host init X
    std::vector<float> X_h(B * d);
    fill_random(X_h, 1234ull + ctx->rank);

    // Copy X and Wgate
    CHECK_CUDA(cudaMemcpyAsync(ctx->X_d, X_h.data(), sizeof(float) * B * d, cudaMemcpyHostToDevice, ctx->stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx->Wgate_d, Wgate_h.data(), sizeof(float) * d * E_total, cudaMemcpyHostToDevice, ctx->stream));

    // Init expert weights per rank
    for (int le = 0; le < experts_per_gpu; le++)
    {
        std::vector<float> W_h(d * d);
        fill_random(W_h, 9999ull + (uint64_t)ctx->rank * 1000ull + (uint64_t)le, 0.05f);
        CHECK_CUDA(cudaMemcpyAsync(ctx->Wexp_d[le], W_h.data(), sizeof(float) * d * d, cudaMemcpyHostToDevice, ctx->stream));
    }

    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    // ------------------ Gating: logits = Wgate^T * X ------------------
    // Wgate: (d x E_total), X: (d x B), logits: (E_total x B) all column-major.
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(
        ctx->cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        E_total, B, d,
        &alpha,
        ctx->Wgate_d, d,
        ctx->X_d, d,
        &beta,
        ctx->logits_d, E_total));

    // softmax + top1 expert per token
    int threads = 128;
    softmax_top1_kernel<<<B, threads, 0, ctx->stream>>>(ctx->logits_d, E_total, B, ctx->expert_idx_d);
    CHECK_CUDA(cudaGetLastError());

    // histogram counts per expert
    CHECK_CUDA(cudaMemsetAsync(ctx->counts_d, 0, sizeof(int) * E_total, ctx->stream));
    histogram_experts_kernel<<<(B + 255) / 256, 256, 0, ctx->stream>>>(ctx->expert_idx_d, B, E_total, ctx->counts_d);
    CHECK_CUDA(cudaGetLastError());

    // copy counts to host
    ctx->counts_h.assign(E_total, 0);
    CHECK_CUDA(cudaMemcpyAsync(ctx->counts_h.data(), ctx->counts_d, sizeof(int) * E_total, cudaMemcpyDeviceToHost, ctx->stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    // offsets (exclusive scan over experts)
    compute_exclusive_scan(ctx->counts_h, ctx->offsets_h);

    // derive send offsets/counts per rank from expert ranges (contiguous by expert id)
    ctx->sendCountRank_h.assign(N, 0);
    ctx->sendOffRank_h.assign(N + 1, 0);

    for (int p = 0; p < N; p++)
    {
        int e0 = p * experts_per_gpu;
        int e1 = (p + 1) * experts_per_gpu;
        int start = ctx->offsets_h[e0];
        int end = ctx->offsets_h[e1];
        ctx->sendOffRank_h[p] = start;
        ctx->sendCountRank_h[p] = end - start;
    }
    ctx->sendOffRank_h[N] = B;

    // pack tokens grouped by expert
    CHECK_CUDA(cudaMemcpyAsync(ctx->offsets_d, ctx->offsets_h.data(),
                               sizeof(int) * (E_total + 1), cudaMemcpyHostToDevice, ctx->stream));
    CHECK_CUDA(cudaMemsetAsync(ctx->counters_d, 0, sizeof(int) * E_total, ctx->stream));

    pack_by_expert_kernel<<<B, 256, 0, ctx->stream>>>(
        ctx->X_d, d, B,
        ctx->expert_idx_d,
        ctx->offsets_d,
        ctx->counters_d,
        ctx->sendAct_d,
        ctx->sendTok_d,
        ctx->rank);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    // Barrier: all ranks ready before NCCL collectives/pt2pt
    bar->wait();

    // ------------------ Share per-expert counts via ncclAllGather ------------------
    // Each rank contributes counts[E_total] -> everyone gets allCounts[N][E_total].
    CHECK_NCCL(ncclAllGather(ctx->counts_d, ctx->allCounts_d, E_total, ncclInt, ctx->comm, ctx->stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    ctx->allCounts_h.assign(N * E_total, 0);
    CHECK_CUDA(cudaMemcpy(ctx->allCounts_h.data(), ctx->allCounts_d, sizeof(int) * N * E_total, cudaMemcpyDeviceToHost));

    // Compute recv counts per sender rank for *my* expert range
    int my_e0 = ctx->rank * experts_per_gpu;
    int my_e1 = (ctx->rank + 1) * experts_per_gpu;

    ctx->recvCountRank_h.assign(N, 0);
    for (int src = 0; src < N; src++)
    {
        int sum = 0;
        const int *row = &ctx->allCounts_h[src * E_total];
        for (int e = my_e0; e < my_e1; e++)
            sum += row[e];
        ctx->recvCountRank_h[src] = sum;
    }

    // recv offsets by sender (we store chunks in sender-rank order 0..N-1)
    ctx->recvOffRank_h.resize(N + 1);
    ctx->recvOffRank_h[0] = 0;
    for (int i = 0; i < N; i++)
        ctx->recvOffRank_h[i + 1] = ctx->recvOffRank_h[i] + ctx->recvCountRank_h[i];
    ctx->total_recv_tokens = ctx->recvOffRank_h[N];

    // Barrier before pt2pt (keeps call ordering tidy)
    bar->wait();

    // ------------------ NCCL Send/Recv activations + token_ids ------------------
    // Self "recv" is a device memcpy from my local slice of sendBuf.
    {
        int self = ctx->rank;

        int send_off = ctx->sendOffRank_h[self];
        int send_cnt = ctx->sendCountRank_h[self];

        int recv_off = ctx->recvOffRank_h[self];
        int recv_cnt = ctx->recvCountRank_h[self];

        // For correctness, these must match:
        // tokens that route to local experts == tokens we "recv from self"
        if (send_cnt != recv_cnt)
        {
            // In this demo they should match; if not, it means our layout assumptions were violated.
            // (They shouldn't be: self range is contiguous in expert order.)
            std::cerr << "[Rank " << ctx->rank << "] WARNING self send_cnt(" << send_cnt
                      << ") != self recv_cnt(" << recv_cnt << ")\n";
        }

        if (send_cnt > 0 && recv_cnt > 0)
        {
            CHECK_CUDA(cudaMemcpyAsync(ctx->recvAct_d + (size_t)recv_off * d,
                                       ctx->sendAct_d + (size_t)send_off * d,
                                       sizeof(float) * (size_t)std::min(send_cnt, recv_cnt) * d,
                                       cudaMemcpyDeviceToDevice, ctx->stream));
            CHECK_CUDA(cudaMemcpyAsync(ctx->recvTok_d + recv_off,
                                       ctx->sendTok_d + send_off,
                                       sizeof(int) * (size_t)std::min(send_cnt, recv_cnt),
                                       cudaMemcpyDeviceToDevice, ctx->stream));
        }
    }

    CHECK_NCCL(ncclGroupStart());
    for (int peer = 0; peer < N; peer++)
    {
        if (peer == ctx->rank)
            continue;

        int send_off = ctx->sendOffRank_h[peer];
        int send_cnt = ctx->sendCountRank_h[peer];

        int recv_off = ctx->recvOffRank_h[peer];
        int recv_cnt = ctx->recvCountRank_h[peer];

        if (send_cnt > 0)
        {
            CHECK_NCCL(ncclSend(ctx->sendAct_d + (size_t)send_off * d,
                                (size_t)send_cnt * d, ncclFloat,
                                peer, ctx->comm, ctx->stream));
            CHECK_NCCL(ncclSend(ctx->sendTok_d + send_off,
                                send_cnt, ncclInt,
                                peer, ctx->comm, ctx->stream));
        }
        if (recv_cnt > 0)
        {
            CHECK_NCCL(ncclRecv(ctx->recvAct_d + (size_t)recv_off * d,
                                (size_t)recv_cnt * d, ncclFloat,
                                peer, ctx->comm, ctx->stream));
            CHECK_NCCL(ncclRecv(ctx->recvTok_d + recv_off,
                                recv_cnt, ncclInt,
                                peer, ctx->comm, ctx->stream));
        }
    }
    CHECK_NCCL(ncclGroupEnd());

    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    // ------------------ Expert compute (local experts only) ------------------
    // recv buffer is partitioned by sender rank. Within each sender chunk, tokens are in *expert order*
    // for my expert range [my_e0..my_e1). So we can walk experts in order and GEMM each slice.
    for (int src = 0; src < N; src++)
    {
        int base_tok = ctx->recvOffRank_h[src];
        int offset_in_chunk = 0;

        const int *row = &ctx->allCounts_h[src * E_total];

        for (int le = 0; le < experts_per_gpu; le++)
        {
            int ge = my_e0 + le;
            int cnt = row[ge];
            if (cnt > 0)
            {
                float *in = ctx->recvAct_d + (size_t)(base_tok + offset_in_chunk) * d;
                float *out = ctx->outAct_d + (size_t)(base_tok + offset_in_chunk) * d;

                // out(d x cnt) = Wexp(d x d) * in(d x cnt)
                CHECK_CUBLAS(cublasSgemm(
                    ctx->cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    d, cnt, d,
                    &alpha,
                    ctx->Wexp_d[le], d,
                    in, d,
                    &beta,
                    out, d));
            }
            offset_in_chunk += cnt;
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    // ------------------ Simple checksum (copy to host and sum) ------------------
    std::vector<float> out_h((size_t)ctx->total_recv_tokens * d);
    if (!out_h.empty())
    {
        CHECK_CUDA(cudaMemcpy(out_h.data(), ctx->outAct_d, sizeof(float) * out_h.size(), cudaMemcpyDeviceToHost));
    }
    double sum = 0.0;
    for (float v : out_h)
        sum += (double)v;
    ctx->checksum = sum;

    // Log some routing info
    std::ostringstream oss;
    oss << "Rank " << ctx->rank << " (dev " << ctx->dev << ")\n";
    oss << "  total recv tokens: " << ctx->total_recv_tokens << "\n";
    oss << "  sendCountRank: ";
    for (int p = 0; p < N; p++)
        oss << ctx->sendCountRank_h[p] << (p + 1 < N ? ", " : "");
    oss << "\n";
    oss << "  recvCountRank: ";
    for (int p = 0; p < N; p++)
        oss << ctx->recvCountRank_h[p] << (p + 1 < N ? ", " : "");
    oss << "\n";
    oss << "  checksum(outAct): " << std::setprecision(12) << ctx->checksum << "\n";
    ctx->log = oss.str();

    // Cleanup
    for (auto p : ctx->Wexp_d)
        CHECK_CUDA(cudaFree(p));
    CHECK_CUDA(cudaFree(ctx->outAct_d));
    CHECK_CUDA(cudaFree(ctx->recvTok_d));
    CHECK_CUDA(cudaFree(ctx->recvAct_d));
    CHECK_CUDA(cudaFree(ctx->allCounts_d));
    CHECK_CUDA(cudaFree(ctx->sendTok_d));
    CHECK_CUDA(cudaFree(ctx->sendAct_d));
    CHECK_CUDA(cudaFree(ctx->counters_d));
    CHECK_CUDA(cudaFree(ctx->offsets_d));
    CHECK_CUDA(cudaFree(ctx->counts_d));
    CHECK_CUDA(cudaFree(ctx->expert_idx_d));
    CHECK_CUDA(cudaFree(ctx->logits_d));
    CHECK_CUDA(cudaFree(ctx->Wgate_d));
    CHECK_CUDA(cudaFree(ctx->X_d));

    CHECK_CUBLAS(cublasDestroy(ctx->cublas));
    CHECK_CUDA(cudaStreamDestroy(ctx->stream));
}

int main(int argc, char **argv)
{
    int ngpus = get_int_arg(argc, argv, "--ngpus", 2);
    int experts_per_gpu = get_int_arg(argc, argv, "--experts-per-gpu", 4);
    int B = get_int_arg(argc, argv, "--batch", 32);
    int d = get_int_arg(argc, argv, "--d", 128);

    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (ngpus > dev_count)
    {
        std::cerr << "Requested ngpus=" << ngpus << " but only " << dev_count << " CUDA devices visible.\n";
        return 1;
    }

    int E_total = ngpus * experts_per_gpu;

    std::cout << "MoE NCCL demo\n"
              << "  ngpus=" << ngpus
              << " experts_per_gpu=" << experts_per_gpu
              << " E_total=" << E_total
              << " batch(B)=" << B
              << " d=" << d << "\n";

    // Host gate matrix Wgate (d x E_total), column-major
    std::vector<float> Wgate_h((size_t)d * E_total);
    fill_random(Wgate_h, 7777ull, 0.05f);

    // NCCL comms
    std::vector<int> devs(ngpus);
    for (int i = 0; i < ngpus; i++)
        devs[i] = i;

    std::vector<ncclComm_t> comms(ngpus);
    CHECK_NCCL(ncclCommInitAll(comms.data(), ngpus, devs.data()));

    Barrier bar(ngpus);

    std::vector<RankCtx> ctxs(ngpus);
    std::vector<std::thread> threads;
    threads.reserve(ngpus);

    for (int r = 0; r < ngpus; r++)
    {
        ctxs[r].rank = r;
        ctxs[r].dev = devs[r];
        ctxs[r].N = ngpus;
        ctxs[r].experts_per_gpu = experts_per_gpu;
        ctxs[r].E_total = E_total;
        ctxs[r].B = B;
        ctxs[r].d = d;
        ctxs[r].comm = comms[r];

        threads.emplace_back(rank_thread, &ctxs[r], std::cref(Wgate_h), &bar);
    }

    for (auto &t : threads)
        t.join();

    // Destroy comms (after threads)
    for (int r = 0; r < ngpus; r++)
    {
        CHECK_NCCL(ncclCommDestroy(comms[r]));
    }

    // Print logs in rank order
    std::cout << "\n--- Per-rank summary ---\n";
    for (int r = 0; r < ngpus; r++)
    {
        std::cout << ctxs[r].log << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}
