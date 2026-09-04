#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <string>

#define threads 256
#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)

const float a_tol = 1e-5f;
const float r_tol = 1e-5f;

__global__ void add(float *A, float *B, float*C, int N){
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<N)
        C[i] = A[i] + B[i];
}

int main() {
    int seed = 1234;
    std::srand(seed);
    int N = 1000;
    float A[1024];
    float B[1024];
    float C[1024];
    float *A_d;
    float *B_d;
    float *C_d;
    float t = 0.0f;
    int warm_up = 0;
    int repetitions = 0;
    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    for(int i = 0; i < N; i++){
        A[i] = rand()%20 - rand()%10;
        B[i] = rand()%10 - rand()%20;
    }

    CUDA_CHECK(cudaMalloc(&A_d, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&B_d, sizeof(float)*N)); 
    CUDA_CHECK(cudaMalloc(&C_d, sizeof(float)*N)); 

    CUDA_CHECK(cudaMemcpy(A_d, A, sizeof(float)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, sizeof(float)*N, cudaMemcpyHostToDevice));

    add<<<((N + threads - 1 )/threads), threads>>>(A_d, B_d, C_d, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, C_d, sizeof(float)*N, cudaMemcpyDeviceToHost));

    int j = 0;
    std::string tester = "yes";
    for(int i = 0; i < N; i++){
        if(std::abs((C[i] - (A[i] + B[i]))) > a_tol + r_tol*(std::abs(A[i] + B[i]))){     //harness test
            j++;
            tester = "no";
            std::cout << "error at index " << i << std::endl; 
        }
    }
    if(j == 0)
        std::cout << "It really worked!"<< std::endl;

    // after validation, we write the mean kernel execution time to a CSV file
    while(warm_up <= 0){
        std::cout << "how many warm-ups do you want?"<< std::endl;
        std::cin >> warm_up;
        if(warm_up <= 0)
            std::cout << "insert a valid value" << std::endl;
    }
    
    for(int i = 0; i < warm_up; i++){
        add<<<((N + threads - 1 )/threads), threads>>>(A_d, B_d, C_d, N);       // warm-up
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    while(repetitions <= 0){
        std::cout << "how many repetitions do you want?"<< std::endl;
        std::cin >> repetitions;
        if(repetitions <= 0)
            std::cout << "insert a valid value" << std::endl;
    }

    CUDA_CHECK(cudaEventRecord(start));

    for(int i = 0; i < repetitions; i++)
        add<<<((N + threads - 1 )/threads), threads>>>(A_d, B_d, C_d, N);

    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&t, start, end));

    t = t/repetitions;

    FILE* f;
    f = fopen("test.csv", "w");
    if(f!=nullptr){ 
        fprintf(f, "test_name,N,seed,atol,rtol,warmup,repetitions,kernel_mean_ms,passed,mismatches\n");
        fprintf(f, "harness,%d,%d,%f,%f,%d,%d,%f,%s,%d", N, seed, a_tol, r_tol, warm_up, repetitions, t, tester.c_str(), j);
        fclose(f);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return (j == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

//declaration
//create event
//record start
//operation
//record stop
//synchronize stop
//elapsed time