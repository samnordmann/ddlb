/**
 * Custom CUDA kernel for Tensor Parallel Row-wise primitive with Sequence Parallelism
 * 
 * This kernel implements the full TP rowwise operation:
 *   1. Local matrix multiplication: C_local = A @ B (using cuBLAS)
 *   2. Reduce-scatter via NCCL to sum and shard results
 * 
 * Operation: C = A @ B where:
 *   - A: [M, K_local] input matrix (full sequence, local K)
 *   - B: [K_local, N] weight matrix (local K, full hidden)
 *   - C: [M_local, N] output after reduce-scatter
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS error"); \
    } \
} while(0)

#define NCCL_CHECK(call) do { \
    ncclResult_t result = call; \
    if (result != ncclSuccess) { \
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(result)); \
    } \
} while(0)

// Global state
static cublasHandle_t cublas_handle = nullptr;
static ncclComm_t nccl_comm = nullptr;
static int g_rank = -1;
static int g_world_size = -1;
static int g_m = 0, g_n = 0, g_k = 0;
static bool initialized = false;

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
 * Perform matrix multiplication using cuBLAS
 */
torch::Tensor matmul_cublas(torch::Tensor A, torch::Tensor B, cudaStream_t stream) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    
    if (A.scalar_type() == torch::kFloat32) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            B.data_ptr<float>(), N,
            A.data_ptr<float>(), K,
            &beta, C.data_ptr<float>(), N
        ));
    } else if (A.scalar_type() == torch::kFloat16) {
        __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        CUBLAS_CHECK(cublasHgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            reinterpret_cast<const __half*>(B.data_ptr<at::Half>()), N,
            reinterpret_cast<const __half*>(A.data_ptr<at::Half>()), K,
            &beta,
            reinterpret_cast<__half*>(C.data_ptr<at::Half>()), N
        ));
    } else if (A.scalar_type() == torch::kBFloat16) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            B.data_ptr(), CUDA_R_16BF, N,
            A.data_ptr(), CUDA_R_16BF, K,
            &beta, C.data_ptr(), CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    } else {
        throw std::runtime_error("Unsupported dtype for cuBLAS GEMM");
    }
    
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
 * Initialize the kernel with communicator info, dimensions, and NCCL ID
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
    
    // Initialize cuBLAS
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // Extract NCCL ID from tensor (was broadcast via MPI in Python)
    ncclUniqueId nccl_id;
    memcpy(&nccl_id, nccl_id_tensor.data_ptr(), sizeof(ncclUniqueId));
    
    // Initialize NCCL with the shared ID
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));
    
    initialized = true;
}

/**
 * Main entry point: run the full TP rowwise operation
 * 
 * Performs: result = reduce_scatter(A @ B)
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
    
    // Step 1: Local matrix multiplication using cuBLAS
    torch::Tensor local_result = matmul_cublas(A, B, stream);
    
    // Step 2: Allocate output buffer for reduce-scatter result
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
    if (cublas_handle != nullptr) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
    if (nccl_comm != nullptr) {
        ncclCommDestroy(nccl_comm);
        nccl_comm = nullptr;
    }
    initialized = false;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TP Row-wise custom kernel (cuBLAS) with NCCL communication";
    m.def("get_nccl_unique_id", &get_nccl_unique_id, "Get NCCL unique ID (rank 0 only)");
    m.def("init", &init, "Initialize kernel",
          py::arg("rank"), py::arg("world_size"), py::arg("m"), py::arg("n"), py::arg("k"), py::arg("nccl_id_tensor"));
    m.def("run", &run, "Run matmul + reduce-scatter",
          py::arg("A"), py::arg("B"));
    m.def("cleanup", &cleanup, "Cleanup resources");
}
