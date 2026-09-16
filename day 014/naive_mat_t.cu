#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>

const int threads = 16;
const float tolerance = 1e-5f;

#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)

__global__ void mat_transposition (const float *A, float *B, const int row, const int col){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x < col && y < row){     //col = X and row = Y
        int input = y*col + x;        //yX + x
        int output =  x*row + y;      //xY + y
        B[output] = A[input];       
    }
}

// col = number of columns = matrix width = elements per row = x size
// row = number of rows = matrix height = elements per column = y size

int main(){
    cudaEvent_t start;
    cudaEvent_t stop;
    float t = 0.0f;
    int row = 5555;
    int col = 9999;
    dim3 block(threads, threads);
    int blocksX = (block.x + col - 1)/block.x;
    int blocksY = (block.y + row - 1)/block.y;
    dim3 grid(blocksX, blocksY);

    int elements = row*col;
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
    
    mat_transposition<<<grid, block>>>(A_d, B_d, row, col);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(B.data(), B_d, size, cudaMemcpyDeviceToHost));

    bool test = true;
    for(int i = 0; i < col; i++){    //i = x, col = X  
        for(int j = 0; j < row; j++){   //j = y, row = Y
        int output = i * row + j;   //xY + y
        int input = j * col + i;   // yX + x
        if (std::abs(B[output] - A[input]) > tolerance) {
            test = false;
            }
        }
    }
    std::cout << "The operation was a "<< (test? "success!" : "failure") << std::endl;

    for(int i = 0; i < 20; i++)
        mat_transposition<<<grid, block>>>(A_d, B_d, row, col); //warm-up

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < 100; i++)
        mat_transposition<<<grid, block>>>(A_d, B_d, row, col);
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