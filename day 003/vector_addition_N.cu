#include <iostream>
#include <cuda_runtime.h>
#include <vector>

const int sizes[] = {1, 10, 31, 32, 33, 255, 257, 1000};
const int threads= 32;

__global__ void vector_addition(int *A, int *B, int *C, int N){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<N){
        C[i] = A[i] + B[i];
    }
}

int main(){
    int n = sizeof(sizes)/sizeof(int);
    std::vector<int> A;
    std::vector<int> B;
    std::vector<int> C;
    int *A_d;
    int *B_d;
    int *C_d;


    for(int i = 0; i < n; i++ ){

        int N = sizes[i];

        cudaMalloc(&A_d, N*sizeof(int));
        cudaMalloc(&B_d, N*sizeof(int));
        cudaMalloc(&C_d, N*sizeof(int));

        A.resize(N);
        B.resize(N);
        C.resize(N);

        for(int j = 0; j < N; j++){
            A[j] = j;
            B[j] = 2*j;
        }

        cudaMemcpy(A_d, A.data(), N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B.data(), N*sizeof(int), cudaMemcpyHostToDevice);

        int blocks = (N + threads - 1)/threads;                 // the goat formula to know how many blocks you need

        vector_addition<<<blocks, threads>>>(A_d, B_d, C_d, N);
        
        cudaDeviceSynchronize();

        cudaMemcpy(C.data(), C_d, N*sizeof(int), cudaMemcpyDeviceToHost);

        int boolean = 0;
        for(int j = 0; j < N; j++){
            if(C[j] - (3*j) != 0)
                boolean = 1;
        }

        if(boolean == 0)
            std::cout << "Success!" << std::endl;
        else
            std::cout << "Something went wrong!" << std::endl;

        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);

    }    
    return 0;
}

// the flow is: intitialize -> alloc memory -> copy -> kernel -> synchronize -> copy -> check the results -> free 
// most of this code is identical to the day 2 code