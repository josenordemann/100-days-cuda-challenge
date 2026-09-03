#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define threads 8
#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)

__global__ void race(int *A, int* output, int N){
    int global_id = threadIdx.x + blockDim.x*blockIdx.x;
    int local_id = threadIdx.x;

    __shared__ int shared_data[256];

    if(global_id < N){
        shared_data[local_id] = A[global_id];       //each block will have its own shared memory, that s why we use the local id
    }
    //__syncthreads();
    if (local_id == 0) {
        output[0] = shared_data[128];           // with multiple blocks, thread 0 from each block would try to write to output[0]
    }
}

int main() {
    int N = 1024;
    int A[1024];
    int *A_d;
    int *output;

    for(int i = 0; i < N; i++){
        A[i] = i;
    }

    CUDA_CHECK(cudaMalloc(&A_d, sizeof(int)*N));
    CUDA_CHECK(cudaMalloc(&output, sizeof(int)*N));

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(A_d, A, sizeof(int)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float t = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));

    std::cout<< t<< std::endl;

    race<<<((N + 256 - 1 )/256), 256>>>(A_d, output, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&A, output, sizeof(int)*N, cudaMemcpyDeviceToHost));

  //  for(int i = 0; i<N; i++)
  //      std::cout<< A[i] << std::endl;

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(output));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return EXIT_SUCCESS or EXIT_FAILURE;
}

