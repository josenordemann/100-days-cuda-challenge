#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>

#define threads 16
const float tolerance = 1e-5f;

#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)

__global__ void reverse_blocks_shared (const float *A, float *B, const int elements){

    int blockStart = blockIdx.x * blockDim.x;
    int localIndex = threadIdx.x;
    int globalIndex = blockStart + localIndex;
    __shared__ float shared[threads];

    int validCount;
    if(blockDim.x <= elements - blockStart)
        validCount = blockDim.x;
    else
        validCount = elements - blockStart;
        
    int reversedLocalIndex = validCount - 1 - localIndex;

    if(blockStart + localIndex < elements )
        shared[localIndex] = A[globalIndex];

        __syncthreads();    //all threads need to be synchronized, so synchthreads cannot be ifed

    if(localIndex < validCount)
        B[globalIndex] = shared[reversedLocalIndex];
    
}
// blockStart = first global position of the current block
// localIndex = thread position inside the block

// validCount = number of valid elements in the current block
// It is important for the last block, which may not be full.

// reversedLocalIndex = opposite position inside the valid part of the block
// Example with 4 valid elements: 0 -> 3, 1 -> 2, 2 -> 1, 3 -> 0

int main(){
    cudaEvent_t start;
    cudaEvent_t stop;
    float t = 0.0f;
    int row = 5555;
    int col = 9999;
    int elements = row*col;
    int blocks = (threads + elements - 1)/ threads;

    size_t size = sizeof(float)*row*col;

    std::vector<float> A(elements);
    std::vector<float> B(elements);

    float *A_d;
    float *B_d;

    for(int i = 0; i < col; i++){    //X
        for(int j = 0; j < row; j++){    //Y
            int index = j*col + i; //yX + x
            A[index] = i + j / 2.0f;        //same matsum2d
            //A[j][i] = i + j / 2.0f;   row-major
        }
    }

    CUDA_CHECK(cudaMalloc(&A_d, size));
    CUDA_CHECK(cudaMalloc(&B_d, size));
    
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemcpy(A_d, A.data(), size, cudaMemcpyHostToDevice));
    
    reverse_blocks_shared<<<blocks, threads>>>(A_d, B_d, elements);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(B.data(), B_d, size, cudaMemcpyDeviceToHost));

    bool test = true;
    for (int i = 0; i < elements; i++) {
        int blockStart = (i / threads) * threads;
        int remaining = elements - blockStart;

        int validCount = remaining < threads ? remaining : threads;

        int localIndex = i - blockStart;
        int reversedIndex = blockStart + validCount - 1 - localIndex;

        if (std::abs(B[i] - A[reversedIndex]) > tolerance) {
            test = false;
        }
    }
    std::cout << "The operation was a "<< (test? "success!" : "failure") << std::endl;

    for(int i = 0; i < 20; i++)
        reverse_blocks_shared<<<blocks, threads>>>(A_d, B_d, elements); //warm-up

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for(int i = 0; i < 100; i++)
        reverse_blocks_shared<<<blocks, threads>>>(A_d, B_d, elements);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));

    t /= 100;

    std::cout << "Mean time: "<< t << " ms"<< std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    return 0;
}