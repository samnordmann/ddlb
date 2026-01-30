/**
 * Simple CUDA kernel for Tensor Parallel Row-wise primitive
 * 
 * This implementation uses a simple tiled matrix multiplication kernel
 * without cuBLAS, plus NCCL for reduce-scatter communication.
 * 
 * Operation: C = A @ B where:
 *   - A: [M, K_local] input matrix
 *   - B: [K_local, N] weight matrix  
 *   - C: [M_local, N] output after reduce-scatter
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define TILE_SIZE 16

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define NCCL_CHECK(call) do { \
    ncclResult_t result = call; \
    if (result != ncclSuccess) { \
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(result)); \
    } \
} while(0)

// Global state
static ncclComm_t nccl_comm = nullptr;
static int g_rank = -1;
static int g_world_size = -1;
static int g_m = 0, g_n = 0, g_k = 0;
static bool initialized = false;

/**
 * Simple tiled matrix multiplication kernel
 * C[M,N] = A[M,K] @ B[K,N]
 */
template<typename T>
__global__ void matmul_simple_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int K, int N
) {
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    T sum = 0;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : T(0);
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : T(0);
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Get NCCL data type from torch dtype
 */
ncclDataType_t get_nccl_dtype(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32: return ncclFloat32;
        case torch::kFloat16: return ncclFloat16;
        case torch::kBFloat16: return ncclBfloat16;
        case torch::kFloat64: return ncclFloat64;
        default:
            throw std::runtime_error("Unsupported dtype for NCCL");
    }
}

/**
 * Launch matmul kernel for the appropriate dtype
 */
torch::Tensor matmul_simple(torch::Tensor A, torch::Tensor B, cudaStream_t stream) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        A.scalar_type(), "matmul_simple_kernel", ([&] {
            matmul_simple_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, K, N
            );
        })
    );
    
    CUDA_CHECK(cudaGetLastError());
    return C;
}

/**
 * Get a new NCCL unique ID (call on rank 0 only)
 * Returns the ID as a tensor for easy MPI broadcast in Python
 */
torch::Tensor get_nccl_unique_id() {
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    
    auto tensor = torch::zeros({128}, torch::kUInt8);
    memcpy(tensor.data_ptr(), &id, sizeof(ncclUniqueId));
    return tensor;
}

/**
 * Initialize with communicator info, dimensions, and NCCL ID
 */
void init(int rank, int world_size, int m, int n, int k, torch::Tensor nccl_id_tensor) {
    if (initialized) {
        return;
    }
    
    g_rank = rank;
    g_world_size = world_size;
    g_m = m;
    g_n = n;
    g_k = k;
    
    // Extract NCCL ID from tensor (was broadcast via MPI in Python)
    ncclUniqueId nccl_id;
    memcpy(&nccl_id, nccl_id_tensor.data_ptr(), sizeof(ncclUniqueId));
    
    // Initialize NCCL with the shared ID
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));
    
    initialized = true;
}

/**
 * Run the full TP rowwise operation: matmul + reduce-scatter
 */
torch::Tensor run(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(initialized, "Kernel not initialized. Call init() first.");
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    int M = A.size(0);
    int N = B.size(1);
    int M_local = M / g_world_size;
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Step 1: Simple matrix multiplication (no cuBLAS)
    torch::Tensor local_result = matmul_simple(A, B, stream);
    
    // Step 2: Allocate output for reduce-scatter
    auto output = torch::empty({M_local, N}, A.options());
    
    // Step 3: Reduce-scatter via NCCL
    NCCL_CHECK(ncclReduceScatter(
        local_result.data_ptr(),
        output.data_ptr(),
        M_local * N,
        get_nccl_dtype(A.scalar_type()),
        ncclSum,
        nccl_comm,
        stream
    ));
    
    return output;
}

/**
 * Cleanup resources
 */
void cleanup() {
    if (nccl_comm != nullptr) {
        ncclCommDestroy(nccl_comm);
        nccl_comm = nullptr;
    }
    initialized = false;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Simple TP Row-wise kernel (no cuBLAS) with NCCL communication";
    m.def("get_nccl_unique_id", &get_nccl_unique_id, "Get NCCL unique ID (rank 0 only)");
    m.def("init", &init, "Initialize kernel",
          py::arg("rank"), py::arg("world_size"), py::arg("m"), py::arg("n"), py::arg("k"), py::arg("nccl_id_tensor"));
    m.def("run", &run, "Run matmul + reduce-scatter",
          py::arg("A"), py::arg("B"));
    m.def("cleanup", &cleanup, "Cleanup resources");
}
