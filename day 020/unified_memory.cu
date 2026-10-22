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

__global__ void transformation(float *A, float* B, int N){
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < N){
        B[i] = 2*A[i] + 1;
    }
}

int main(){
    int seed = 1234;
    std::srand(seed);

    int N = 20000;

    cudaEvent_t start;
    cudaEvent_t stop;

    float t1;
    float t2;
    float *A;
    float *B;

    size_t bytes = static_cast<size_t>(sizeof(float)*N);

    CUDA_CHECK(cudaMallocManaged(&A, bytes));
    CUDA_CHECK(cudaMallocManaged(&B, bytes));
    

    for(int i = 0; i < N; i++){
        A[i] = rand()%20 - rand()%10;
    }

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blocks = (threads + N - 1) / threads;

    CUDA_CHECK(cudaEventRecord(start));

    transformation<<<blocks, threads>>>(A, B, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&t1, start, stop));

    std::cout << "First execution: " << t1 << " ms\n";

    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < 50; i++){
        transformation<<<blocks, threads>>> (A, B, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&t2, start, stop));

    t2 /= 50;

    std::cout << "Mean execution time: " << t2 << " ms\n";

    bool test = true;

    for (int i = 0; i < N; i++) {
        float expected = 2 * A[i] + 1;

        if (std::abs(B[i] - expected) > tolerance) {
            test = false;
        }
    }

    std::cout << "The operation was a "<< (test ? "success!" : "failure") << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
}

