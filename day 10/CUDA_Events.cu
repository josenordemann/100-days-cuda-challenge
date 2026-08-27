#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <cstdlib>

// this code is just for learning how to measure CUDA operations with events

// first, we check if the 3D tensor addition is correct. after that, we measure the H2D copy, the kernel execution and the D2H copy separately

// CUDA events return the elapsed time in milliseconds, and we do a warm-up before measuring the kernel because the first executions can be slower

#define threads 8
#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)

__global__ void add(const int *A, const int *B, int *C, int X, int Y, int Z){
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;

    if(x<X && y<Y && z<Z){
        int idx = X*Y*z + X*y + x;
        C[idx] = A[idx] + B[idx];
    }
} 

int main(){
    int X = 10;
    int Y = 16;
    int Z = 32;

    std::size_t elements = static_cast<size_t>(X)*Y*Z;

    std::vector<int> A(elements);
    std::vector<int> B(elements);
    std::vector<int> C(elements);

    int *A_d;
    int *B_d;
    int *C_d;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(threads, threads, threads);
    dim3 grid((block.x + X - 1)/block.x, (block.y + Y -1)/block.y, (block.z + Z - 1)/block.z);

    for(int k = 0; k < Z; k++){
        for(int j = 0; j < Y; j++){
            for (int i = 0; i < X; i++){
                 int idx = X*Y*k + X*j + i;
                 A[idx] = i + j + k;
                 B[idx] = -i + 2*j -3*k;
            }
        }
    }

    //MALLOC

    CUDA_CHECK(cudaMalloc(&A_d, elements*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&B_d, elements*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&C_d, elements*sizeof(int)));

    //MEMCPY

    CUDA_CHECK(cudaMemcpy(A_d, A.data(), sizeof(int)*elements, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B.data(), sizeof(int)*elements, cudaMemcpyHostToDevice));

    //KERNEL 
    
    add<<<grid, block>>>(A_d, B_d, C_d, X, Y, Z);
    CUDA_CHECK(cudaGetLastError());

    //MEMCPY

    CUDA_CHECK(cudaMemcpy(C.data(), C_d, sizeof(int)*elements, cudaMemcpyDeviceToHost));

    //CHECKING RESULTS
    bool test = true;
    for(int k = 0; k < Z; k++){
        for(int j = 0; j < Y; j++){
            for (int i = 0; i < X; i++){
                 int idx = X*Y*k + X*j + i;
                 if (C[idx] - A[idx] - B[idx] != 0)
                 test = false;
            }
        }
    }

    std::cout << "The operation was a "<< (test? "success!" : "failure") << std::endl;

    //MEMCPY
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(A_d, A.data(), sizeof(int)*elements, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B.data(), sizeof(int)*elements, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&h2d, start, stop));

    //WARM UP
    for(int i = 0; i < 20; i++){
        add<<<grid, block>>>(A_d, B_d, C_d, X, Y, Z);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(start));

    //KERNEL
    for(int i = 0; i < 20; i++){    
        add<<<grid, block>>>(A_d, B_d, C_d, X, Y, Z);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));     // the host waits until the stop event has completed

    float kernel_time = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start, stop));     //kernel_time receives the time of execution of the kernel 

    kernel_time = kernel_time/20.0f;

    //MEMCPY

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(C.data(), C_d, sizeof(int)*elements, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&d2h, start, stop));

    std::cout << "Kernel execution time: "<<kernel_time << " ms" << std::endl;
    std::cout << "Memcpy h2d execution time: "<<h2d << " ms" << std::endl;
    std::cout << "Memcpy d2h execution time: "<<d2h << " ms" << std::endl;

    //EVENT DESTROY
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    //CUDA FREE
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return 0;
}

//record start
//operation
//record stop
//synchronize stop
//elapsed time