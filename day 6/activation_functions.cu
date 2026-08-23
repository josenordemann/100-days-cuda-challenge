#include <iostream>
#include <cuda_runtime.h> // expf e tanhf no kernel
#include <cstdlib>
#include <cstdio>
#include <cmath>        // std::exp e std::tanh na CPU

#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)

#define vector_size 1024

const float tolerance = 1e-5f;

__global__ void reLU(float *V, int N){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < N){
        if(V[i] < 0.0f)
        V[i] = 0.0f;
    }
}

__global__ void leaky_reLU(float *V, float a, int N){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
        if(i < N){
            if(V[i] < 0.0f)
            V[i] = a*V[i];
        }
}

__global__ void sigmoid(float *V, int N){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
        if(i < N)
            V[i] = 1.0f/(1.0f+expf(-V[i]));
}

__global__ void my_tanh(float *V, int N){
    int i = threadIdx.x + blockDim.x*blockIdx.x;    
        if(i < N)
            V[i] = (expf(V[i]) - expf(-V[i]))/(expf(V[i]) + expf(-V[i]));               // there is a tanh(V[i]) function too
}

__global__ void hard_sigmoid(float *V, int N){
    int i = threadIdx.x + blockDim.x*blockIdx.x;    
        if(i < N){
            float x;
            x = ((0.2f*V[i]) + 0.5f);
            if(x < 0.0f)
                x = 0.0f;
            if(x > 1.0f)
                x = 1.0f;
            V[i] = x;
        }
}

int main(){
    //parameters
    int N = vector_size;
    float V[vector_size]; 
    float V_cpu[vector_size];
    int threads = 256;
    int blocks = (threads + vector_size - 1)/threads;
    float a = 0.01f;
    float *V_gpu;

    CUDA_CHECK(cudaMalloc(&V_gpu, N*sizeof(float)));

    //1. reLU
    for(int i = 0; i < N; i++){
       V[i] = static_cast<float>(i%11)- 5.0f;
       V_cpu[i] = V[i];                 
    }

    CUDA_CHECK(cudaMemcpy(V_gpu, V, N*sizeof(float), cudaMemcpyHostToDevice));

    reLU<<<blocks, threads>>>(V_gpu, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(V, V_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost));

    bool passed = true;
    for(int i = 0; i < N; i++){
        if(V_cpu[i] < 0.0f)
            V_cpu[i] = 0.0f;
        if (std::abs(V_cpu[i] - V[i]) > tolerance){
            passed = false;
            break;
        }
    }
    std::cout << "ReLU: "
          << (passed ? "passed" : "failed")
          << std::endl;

    //2. leaky reLU
    for(int i = 0; i < N; i++){
       V[i] = static_cast<float>(i%11)- 5.0f;
       V_cpu[i] = V[i];                 
    }

    CUDA_CHECK(cudaMemcpy(V_gpu, V, N*sizeof(float), cudaMemcpyHostToDevice));

    leaky_reLU<<<blocks, threads>>>(V_gpu, a, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(V, V_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost));

    passed = true;
    for(int i = 0; i < N; i++){
        if(V_cpu[i] < 0.0f)
            V_cpu[i] = a*V_cpu[i];
        if (std::abs(V_cpu[i] - V[i]) > tolerance){
            passed = false;
            break;
        }
    }
    std::cout << "leaky_reLU: "
          << (passed ? "passed" : "failed")
          << std::endl;    


    //3. sigmoid
    for(int i = 0; i < N; i++){
       V[i] = static_cast<float>(i%11)- 5.0f;
       V_cpu[i] = V[i];                 
    }

    CUDA_CHECK(cudaMemcpy(V_gpu, V, N*sizeof(float), cudaMemcpyHostToDevice));

    sigmoid<<<blocks, threads>>>(V_gpu, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(V, V_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost));

    passed = true;
    for(int i = 0; i < N; i++){
        V_cpu[i] = 1.0f/(1.0f+expf(-V_cpu[i]));
        if (std::abs(V_cpu[i] - V[i]) > tolerance){
            passed = false;
            break;
        }
    }
    std::cout << "sigmoid: "
          << (passed ? "passed" : "failed")
          << std::endl;    

    //4. tanh
    for(int i = 0; i < N; i++){
       V[i] = static_cast<float>(i%11)- 5.0f;
       V_cpu[i] = V[i];                 
    }
    CUDA_CHECK(cudaMemcpy(V_gpu, V, N*sizeof(float), cudaMemcpyHostToDevice));

    my_tanh<<<blocks,threads>>>(V_gpu, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(V, V_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost));

    passed = true;
    for(int i = 0; i < N; i++){
        V_cpu[i] = tanh(V_cpu[i]);
        if (std::abs(V_cpu[i] - V[i]) > tolerance){
            passed = false;
            break;
        }
    }
    std::cout << "tanh: "
          << (passed ? "passed" : "failed")
          << std::endl;    

    //5. hard_sigmoid
    for(int i = 0; i < N; i++){
       V[i] = static_cast<float>(i%11)- 5.0f;
       V_cpu[i] = V[i];                 
    }

    CUDA_CHECK(cudaMemcpy(V_gpu, V, N*sizeof(float), cudaMemcpyHostToDevice));

    hard_sigmoid<<<blocks, threads>>>(V_gpu, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(V, V_gpu, sizeof(float)*N, cudaMemcpyDeviceToHost));

    passed = true;
    for(int i = 0; i < N; i++){
        float x;
        x = ((0.2f*V_cpu[i]) + 0.5f);
        if(x < 0.0f)
            x = 0.0f;
        if(x > 1.0f)
            x = 1.0f;
        V_cpu[i] = x;
        if (std::abs(V_cpu[i] - V[i]) > tolerance){
            passed = false;
            break;
        }
    }
    std::cout << "hard_sigmoid: "
          << (passed ? "passed" : "failed")
          << std::endl;    

    //end
    CUDA_CHECK(cudaFree(V_gpu));
    return 0;
}