#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
// this code here is just to learn how to make loops and grid concepts
#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)

#define threads 1024

__global__ void vector_addition(int *A, int *B, int *C, int N){
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = blockDim.x*gridDim.x;              //each grid is made of blocks
    for(; i < N; i = i + stride){                   // in this loop, a thread move to the adress of the next element which it has to proccess
        C[i] = A[i] + B[i];                         // in this ex, 0 -> 1024 -> 2028, 1 -> 1025 -> 2029
    }
}

int main(){
    int N = (threads*120000);
    std::vector<int> A;
    std::vector<int> B;
    std::vector<int> C;
    A.resize(N);
    B.resize(N);
    C.resize(N);

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

    CUDA_CHECK(cudaMemcpy(A_d, A.data(), sizeof(int)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B.data(), sizeof(int)*N, cudaMemcpyHostToDevice));

    int blocks = (threads + N - 1)/threads;

    vector_addition<<<blocks, threads>>>(A_d, B_d, C_d, N);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C.data(), C_d, sizeof(int)*N, cudaMemcpyDeviceToHost));

    bool test = true;
    for(int i = 0; i < N; i++){
        if(C[i] != 3*i)
        test = false;
    }
    std::cout << "Everything went alright? " << (test? "Yes!" : "No")<< std::endl;

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return 0;
}