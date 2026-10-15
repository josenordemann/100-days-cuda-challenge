#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>

// Memory allocated with malloc is pageable: its pages can be moved out of physical memory by the operating system.
// Memory allocated with cudaMallocHost is pinned: its pages stay in physical memory, allowing more direct transfers between host and device.

const float tolerance = 1e-5f;

#define CUDA_CHECK(call) do {                                                                                                    \
    cudaError_t error = (call);                                                                                                  \
    if (error != cudaSuccess) {                                                                                                  \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));   \
        std::exit(EXIT_FAILURE);                                                                                                \
    }                                                                                                                           \
} while (0)

int main(){
    const int sizes[] = {262144, 1048576, 4194304}; // 1, 4 and 16 MiB
    const int warmup = 20;
    const int repetitions = 100;

    for (int N : sizes) {
        size_t bytes = static_cast<size_t>(N) * sizeof(float);

        float *A = static_cast<float*>(malloc(bytes)); // pageable host memory
        float *B;                                     // pinned host memory
        float *C;                                     // device memory

        if (A == nullptr) {
            std::cout << "Host allocation failed\n";
            return EXIT_FAILURE;
        }

        CUDA_CHECK(cudaMallocHost(&B, bytes));
        CUDA_CHECK(cudaMalloc(&C, bytes));

        for (int i = 0; i < N; i++) {
            A[i] = static_cast<float>(i);
        }

        CUDA_CHECK(cudaMemcpy(C, A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B, C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        bool test = true;

        for (int i = 0; i < N; i++) {
            if (std::abs(B[i] - static_cast<float>(i)) > tolerance) {
                test = false;
            }
        }

        for (int i = 0; i < N; i++) {
            A[i] = -1.0f;
        }

        CUDA_CHECK(cudaMemcpy(C, B, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(A, C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int i = 0; i < N; i++) {
            if (std::abs(A[i] - static_cast<float>(i)) > tolerance) {
                test = false;
            }
        }

        std::cout << "\nN: " << N << ", bytes: " << bytes << '\n';
        std::cout << "The operation was a "
                  << (test ? "success!" : "failure") << '\n';

        if (!test) {
            free(A);
            CUDA_CHECK(cudaFreeHost(B));
            CUDA_CHECK(cudaFree(C));
            return EXIT_FAILURE;
        }

        // warm-up 
        for (int i = 0; i < warmup; i++) {
            CUDA_CHECK(cudaMemcpy(C, A, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(C, B, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(A, C, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(B, C, bytes, cudaMemcpyDeviceToHost));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::steady_clock::now();

        // pageable host -> device
        for (int i = 0; i < repetitions; i++) {
            CUDA_CHECK(cudaMemcpy(C, A, bytes, cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "Pageable H2D: "
                  << duration.count() / repetitions << " ms\n";

        CUDA_CHECK(cudaDeviceSynchronize());
        start = std::chrono::steady_clock::now();

        // pinned host -> device
        for (int i = 0; i < repetitions; i++) {
            CUDA_CHECK(cudaMemcpy(C, B, bytes, cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::steady_clock::now();

        duration = end - start;

        std::cout << "Pinned H2D: "<< duration.count() / repetitions << " ms\n";

        CUDA_CHECK(cudaDeviceSynchronize());
        start = std::chrono::steady_clock::now();

        // device -> pageable host
        for (int i = 0; i < repetitions; i++) {
            CUDA_CHECK(cudaMemcpy(A, C, bytes, cudaMemcpyDeviceToHost));
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::steady_clock::now();

        duration = end - start;

        std::cout << "Pageable D2H: "<< duration.count() / repetitions << " ms\n";

        CUDA_CHECK(cudaDeviceSynchronize());
        start = std::chrono::steady_clock::now();

        // device -> pinned host
        for (int i = 0; i < repetitions; i++) {
            CUDA_CHECK(cudaMemcpy(B, C, bytes, cudaMemcpyDeviceToHost));
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::steady_clock::now();

        duration = end - start;

        std::cout << "Pinned D2H: "<< duration.count() / repetitions << " ms\n";

        free(A);
        CUDA_CHECK(cudaFreeHost(B));
        CUDA_CHECK(cudaFree(C));
    }

    return 0;
}
