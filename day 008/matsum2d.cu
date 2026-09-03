#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>

#define threads 16
#define row 1500
#define col 1000

const float tolerance = 1e-5f; // 0.00001

#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)


__global__ void matsum2d(const float *A, const float *B, float *C, int rows, int cols){         //const because we are just reading them
    int coluna = threadIdx.x + blockDim.x*blockIdx.x;               //think that there is a big 2 dim grid made of blocks, which are made of threads
    int linha = threadIdx.y + blockDim.y*blockIdx.y;                // you calculate the column with the x index and the row with the y index
    if(linha < rows && coluna < cols){                         // to me, it makes a lot of sense. that s why we use dim3, to facilitate our lives
    int i = linha*cols + coluna;                // in a 10x10 matrix, there are 100 elements
        C[i] = A[i] + B[i];         // the 12 index corresponds to row 1 and column 2, so 12 = 1*10 + 2 
    }                               // zero based indexing of course
}

int main(){

    int threadsX = threads;
    int threadsY = threads;
    int cols = col;
    int rows = row;

    std::vector<float> A(rows*cols);
    std::vector<float> B(rows*cols);
    std::vector<float> C(rows*cols);
    
    float *A_d;
    float *B_d;
    float *C_d;

    int index;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            index = i*cols + j;
            A[index] =  i + j / 2.0f;
            B[index] = i * 1.5f + j;
        }
    }

    CUDA_CHECK(cudaMalloc(&A_d, sizeof(float)*row*col));
    CUDA_CHECK(cudaMalloc(&B_d, sizeof(float)*row*col));
    CUDA_CHECK(cudaMalloc(&C_d, sizeof(float)*row*col));

    CUDA_CHECK(cudaMemcpy(A_d, A.data(), sizeof(float)*row*col, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B.data(), sizeof(float)*row*col, cudaMemcpyHostToDevice));

    dim3 block(threadsX, threadsY);     //dim3 is a type to represent elements up to 3 dimensions

    int blocksX = (cols + block.x - 1) / block.x;
    int blocksY = (rows + block.y -1) / block.y;

    // in this case, block.x = threadsX and block.y = threadsY
    // it's the blocks parameters

    dim3 grid(blocksX, blocksY);        //we will need 2 of them, one for blocks and another for the grid.

    // this expression maps a 2D matrix coordinate to a 1D index
    // dim 3 block = (16, 16, 1), for a total of 256 threads per block
    // dim 3 grid = (94, 63, 1), for a total of 5,922 blocks

    matsum2d<<<grid, block>>>(A_d, B_d, C_d, rows, cols);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C.data(), C_d, sizeof(float)*row*col, cudaMemcpyDeviceToHost));

    bool test = true;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            index = i*cols + j;
            float expected = A[index] + B[index];
            if(std::abs(C[index] - expected) > tolerance)
            test = false;
        }
    }

    std::cout << "The operation was a "<< (test? "success!" : "failure") << std::endl;
    
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return 0;

}

// basically, dim3 makes multidimensional grids and blocks easier to configure and understand.