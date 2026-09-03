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

__global__ void add(int *A, int *B, int*C, int N){
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<=N)
        C[i] = A[i] + B[i];
}

int main() {
    int N = 1000;
    int A[1024];
    int B[2024];
    int C[1024];
    int *A_d;
    int *B_d;
    int *C_d;
    for(int i = 0; i < N; i++){
        A[i] = i;
        B[i] = 2*i;
    }

    CUDA_CHECK(cudaMalloc(&A_d, sizeof(int)*N));
    CUDA_CHECK(cudaMalloc(&B_d, sizeof(int)*N)); 
    CUDA_CHECK(cudaMalloc(&C_d, sizeof(int)*N)); 

    CUDA_CHECK(cudaMemcpy(A_d, A, sizeof(int)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, sizeof(int)*N, cudaMemcpyHostToDevice));

    add<<<((N + 256 - 1 )/256), 256>>>(A_d, B_d, C_d, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&C, C_d, sizeof(int)*N, cudaMemcpyDeviceToHost));

   int j = 0;
    for(int i = 0; i < N; i++){
        if(C[i] != 3*i)
            j = 1;
    }
    if(j == 0)
        std::cout << "It really worked!"<< std::endl;
    else
        std::cout << "ERROR!" << std::endl;

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return EXIT_SUCCESS or EXIT_FAILURE;
}

