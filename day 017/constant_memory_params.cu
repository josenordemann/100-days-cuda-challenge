#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>

const float tolerance = 1e-5f;
#define threads 32

#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)

__constant__  float params[2];

__global__ void vector_addition(const float* A, float *B, int N){
    float a = params[0];
    float b = params[1];

    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<N)
        B[i] = a*A[i] + b;
}


int main(){

    int elements = 2500;
    std::vector<float> A(elements);
    std::vector<float> B(elements);

    float *A_d;
    float *B_d;
    float hostParams[2] = {2.0f, 1.0f}; // a, b

    for (int i = 0; i < elements; i++) {
            A[i] = static_cast<float>(i);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(params, hostParams, sizeof(hostParams)));
    CUDA_CHECK(cudaMalloc(&A_d, elements*sizeof(float)));       //device memory allocation
    CUDA_CHECK(cudaMalloc(&B_d, elements*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(A_d, A.data(), elements*sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (threads + elements - 1)/threads;

    vector_addition<<<blocks, threads>>>(A_d, B_d, elements);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(B.data(), B_d, elements*sizeof(float), cudaMemcpyDeviceToHost));

    bool test = true;
    for(int i = 0; i < elements; i++){
        if(std::abs(A[i]*hostParams[0] + hostParams[1] - B[i]) > tolerance)
            test = false;
    }
    std::cout << "The operation was a "<< (test ? "success!" : "failure") << std::endl;

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));

    return 0;
}

//very simple