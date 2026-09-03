#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
// this code is just for learning how to use loops and understand grid concepts
#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)

#define threads 256

__global__ void vector_addition(int *A, int *B, int *C, int N){
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = blockDim.x*gridDim.x;              // the stride is the total number of threads in the gri
    for(; i < N; i = i + stride){                   // each thread advances by stride to its next assigned element
        C[i] = A[i] + B[i];                         // in this ex, 0 -> 65536 -> 131072 , 1 -> 65537 -> 131073
    }                                               // here, stride = 65536 = 256 blocks (gridDim) * 256 threads per block (blockDim) = threads in the grid
}                                                   // each thread will proccess 20 values = N/stride

int main(){
    int N = (threads*256*20);
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

    // launch one grid with 256 blocks and 256 threads per block
    // each kernel call "generate" one grid
    vector_addition<<<256, threads>>>(A_d, B_d, C_d, N);

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