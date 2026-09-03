#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>

// almost same code as day 3, but SAXPY and using CUDA_CHECK

const int threads = 256;
const float tolerance = 1e-5f; // 0.00001
// when comparing float values, we need to use tolerance
 
const int sizes[] = {255, 257, 301, 1024, 2048};

#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)


__global__ void saxpy(float *X, float *Y, float a, int N){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < N){
        Y[i] = a*X[i] + Y[i]; 
    }
}

int main(){
    std::vector<float> X;
    std::vector<float> Y;
    std::vector<float> Y_cpu;

    int N;
    float *X_d;
    float *Y_d;
    float a;
    int blocks;
    int b = 0;

    for(int i = 0; i<(sizeof(sizes)/sizeof(sizes[0])); i++){
        N = sizes[i];
        a = static_cast<float>(sizes[i]%10);

        blocks = (N + threads - 1)/threads;

        X.resize(N);
        Y.resize(N);
        Y_cpu.resize(N);

        for(int j = 0; j < N; j++){
            X[j] = j;
            Y[j] = 2*j;
            Y_cpu[j] = 2*j;
        }

        CUDA_CHECK(cudaMalloc(&X_d, N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&Y_d, N*sizeof(float)));

        CUDA_CHECK(cudaMemcpy(X_d, X.data(), sizeof(float)*N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(Y_d, Y.data(), sizeof(float)*N, cudaMemcpyHostToDevice));

        saxpy<<<blocks, threads>>>(X_d, Y_d, a, N);

        CUDA_CHECK(cudaGetLastError());                 //we cant use cuda_check in the kernel because it doesnt return cudaError_t type
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(Y.data(), Y_d, sizeof(float)*N, cudaMemcpyDeviceToHost));

        for(int j = 0; j < N; j++){
                Y_cpu[j] = a*X[j] + Y_cpu[j];
                if(std::fabs(Y_cpu[j] - Y[j]) > tolerance){
                    std::cout << "Something went wrong in " << N << std::endl;
                    b = 1;
                }
        }

        CUDA_CHECK(cudaFree(Y_d));
        CUDA_CHECK(cudaFree(X_d));
    }
    if (b == 0)
        std::cout << "Well done!" << std::endl;
}

