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

    cudaStream_t streams[2];

    for(int i = 0; i < 2; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    for(int offset = 0; offset < N; offset += chunk_size){
        int count = min(chunk_size, N - offset);
        size_t chunk_bytes = static_cast<size_t>(count) * sizeof(float);
        int blocks = ((threads + count - 1)/threads);

        int chunk_index = offset / chunk_size;
        cudaStream_t stream = streams[chunk_index % 2];

        CUDA_CHECK(cudaMemcpyAsync(A_d + offset, A + offset, chunk_bytes, cudaMemcpyHostToDevice, stream));

        transformation<<<blocks, threads, 0, stream>>>(A_d + offset, B_d + offset, count);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(B + offset, B_d + offset, chunk_bytes, cudaMemcpyDeviceToHost, stream));

    }

    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    bool test = true;

    for (int i = 0; i < N; i++) {
        float expected = 2 * A[i] + 1;

        if (std::abs(B[i] - expected) > tolerance) {
            test = false;
        }
    }

    std::cout << "The operation was a "<< (test ? "success!" : "failure") << std::endl;

    for(int i = 0; i < 2; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(B));
}

// basically, we split the vector into smaller parts called chunks.
// offset is where each chunk starts, and count is how many elements it has.
// the last chunk can be smaller than chunk_size.
//
// each stream is a queue of GPU operations.
// for each chunk, we send H2D -> kernel -> D2H to the same stream.
// operations in the same stream follow this order.
// different streams can make progress at the same time, if the GPU supports it.
//
// this works like a pipeline:
// while the GPU calculates chunk 0, it can copy chunk 1 to the device.
// while chunk 0 returns to the host, the GPU can calculate chunk 1.
// this can reduce the total time by overlapping copies and computation.
//
// we alternate streams with chunk_index % 2: 0, 1, 0, 1...
// each chunk uses a different memory region with pointer + offset.
// inside the kernel, index 0 refers to the first element of that chunk.
//
// pinned host memory allows asynchronous transfers that can overlap GPU work.
// we synchronize both streams after sending all chunks, before checking B.
// synchronizing after every chunk would prevent this pipeline.
//
// overlap is not guaranteed, and very small chunks can add too much overhead.