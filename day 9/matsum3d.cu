#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>

#define threads 8
const float tolerance = 1e-5f;

#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)


__global__ void matsum3d (const float *A, const float *B, float *C, int X, int Y, int Z){
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;
    int index = z*X*Y + y*X + x;                    //stride x = 1, stride y = X and stride z = X*Y
    if(x < X && y < Y && z < Z){                    //exact same logic of the last exercice. if there was another dimension, we would add i*X*Y*Z to index
        C[index] = A[index] + B[index];
    }
}

int main(){     // tensor[z][y][x]. in CUDA, we store the tensor as a 1D array and use these 3 coordinates to access it. tensors are used a lot in PyTorch
    int X = 32;
    int Y = 24;
    int Z = 10;

    std::size_t elements = static_cast<std::size_t> (X*Y*Z);

    std::vector<float> A(elements);
    std::vector<float> B(elements);
    std::vector<float> C(elements);

    float *A_d;
    float *B_d;
    float *C_d;

    for(int k = 0; k < Z; k++){
        for(int j = 0; j < Y; j++){
            for(int i = 0; i < X; i++){
            A[k*X*Y + j*X + i] = k*1.5f + j*0.5f - 3.5f*i;
            B[k*X*Y + j*X + i] = k*0.5f - j*1.5f + 1.0f*i;
            }
        }
    }

    CUDA_CHECK(cudaMalloc(&A_d, elements*sizeof(float)));       //device memory allocation
    CUDA_CHECK(cudaMalloc(&B_d, elements*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C_d, elements*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(A_d, A.data(), elements*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B.data(), elements*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(threads, threads, threads);   // block represents the number of threads in each dimension. the total number of threads is the product of the dimensions,
    dim3 grid((X + block.x - 1)/block.x,(Y + block.y - 1)/block.y, (Z + block.z - 1)/block.z);  // and it must not exceed the device limit, which is usually 1024 threads.

    //tensor = (10 = Z, 24 = Y, 32 = X) the vector itself is 1D. X is contiguous and changes the fastest in memory
    //block = (8, 8, 8) = (x, y, z)
    //grid = (4, 3, 2) = (x, y, z)

    matsum3d<<<grid, block>>>(A_d, B_d, C_d, X, Y, Z);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C.data(), C_d, sizeof(float)*elements, cudaMemcpyDeviceToHost));
    bool test = true;
    float comparison; 

    for(int k = 0; k < Z; k++){
        for(int j = 0; j < Y; j++){
            for(int i = 0; i < X; i++){
                comparison = k*1.5f + j*0.5f - 3.5f*i + k*0.5f - j*1.5f + 1.0f*i;
                if (std::abs(comparison - C[k*X*Y + j*X + i]) > tolerance)
                    test = false;
            }
        }
    }

    std::cout << "The operation was a "<< (test? "success!" : "failure") << std::endl;

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return 0;
}