#include <cstdio>
#include <cuda_runtime.h>

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr size_t bytes = n * sizeof(float);

    float *h_a = nullptr;
    float *h_b = nullptr;
    float *h_c = nullptr;

    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);

    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        std::fprintf(
            stderr,
            "Kernel launch failed: %s\n",
            cudaGetErrorString(launch_error)
        );
        return 1;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        if (h_c[i] != 3.0f) {
            std::fprintf(
                stderr,
                "Validation failed at %d: %f\n",
                i,
                h_c[i]
            );
            return 1;
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    std::printf("CUDA_NVCC_SMOKE_TEST_PASSED\n");
    return 0;
}
