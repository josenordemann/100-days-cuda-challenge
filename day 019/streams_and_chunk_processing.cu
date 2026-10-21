#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>

const float tolerance = 1e-5f;
#define threads 64

#define CUDA_CHECK(call) do {                                                                                                    \
    cudaError_t error = (call);                                                                                                  \
    if (error != cudaSuccess) {                                                                                                  \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));   \
        std::exit(EXIT_FAILURE);                                                                                                \
    }                                                                                                                           \
} while (0)

__global__ void transformation(float *A, float* B, int count){
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < count){
        B[i] = 2*A[i] + 1;
    }
}

int main(){
    int seed = 1234;
    std::srand(seed);

    int N = 20000;
    int chunk_size = 16;

    float *A;       //A will have a pinned memory allocation
    float *A_d;
    float *B;
    float *B_d;

    size_t bytes = static_cast<size_t>(sizeof(float)*N);

    cudaStream_t stream;

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMallocHost(&A, bytes));
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMallocHost(&B, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));

    for(int i = 0; i < N; i++){
        A[i] = rand()%20 - rand()%10;
    }

    for(int offset = 0; offset < N; offset += chunk_size){
        int count = min(chunk_size, N - offset);
        size_t chunck_bytes = static_cast<size_t>(count) * sizeof(float);
        int blocks = ((threads + count - 1)/threads);

        CUDA_CHECK(cudaMemcpyAsync(A_d + offset, A + offset, chunck_bytes, cudaMemcpyHostToDevice, stream));   

        transformation<<<blocks, threads, 0, stream>>>(A_d + offset, B_d + offset, count);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(B + offset, B_d + offset, chunck_bytes, cudaMemcpyDeviceToHost, stream));   

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    bool test = true;

    for (int i = 0; i < N; i++) {
        float expected = 2 * A[i] + 1;

        if (std::abs(B[i] - expected) > tolerance) {
            test = false;
        }
    }

    std::cout << "The operation was a "<< (test ? "success!" : "failure") << std::endl;

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(B));
}

