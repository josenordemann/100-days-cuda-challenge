#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>

const float tolerance = 1e-5f;
const int TILE_DIM = 16;

#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)

__global__ void transpose_tiled_shared (const float *A, float *B, int rows, int cols){
    int inputRow = blockIdx.y * TILE_DIM + threadIdx.y;
    int inputCol = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int input_i = inputRow*cols + inputCol;
    
    if (inputRow < rows && inputCol < cols)
        tile[threadIdx.y][threadIdx.x] = A[input_i];

    __syncthreads();    //all threads need to be synchronized, so synchthreads cannot be ifed

    int outputRow = blockIdx.x * TILE_DIM + threadIdx.y;
    int outputCol = blockIdx.y * TILE_DIM + threadIdx.x;

    int output_i = outputRow * rows + outputCol;

    if (outputRow < cols && outputCol < rows)
        B[output_i] = tile[threadIdx.x][threadIdx.y];
}

// basically, we are transposing the matrix: A[row][col] becomes B[col][row]
// inputRow and inputCol use the block position and the thread position to find an element in A
// inputRow * cols + inputCol gives its 1D index, because A has cols elements per row
// each thread stores its value in tile[threadIdx.y][threadIdx.x], using local indices
// after __syncthreads(), each thread reads tile[threadIdx.x][threadIdx.y], swapping the local indices
// we also swap the block coordinates to put the transposed tile in the correct position in B
// outputRow * rows + outputCol gives the output index, because B has rows elements per row
// the extra column in shared memory is padding to reduce bank conflicts

int main(){
    int rows = 35;
    int cols = 265;

    int elements = rows*cols;
    std::vector<float> A(elements);
    std::vector<float> B(elements);

    float *A_d;
    float *B_d;

    int index;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            index = i * cols + j;
            A[index] = i + j / 2.0f;
        }
    }

    CUDA_CHECK(cudaMalloc(&A_d, elements*sizeof(float)));       //device memory allocation
    CUDA_CHECK(cudaMalloc(&B_d, elements*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(A_d, A.data(), elements*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, TILE_DIM);

    int blockX = (cols + block.x - 1)/block.x;
    int blockY = (rows + block.y - 1)/block.y;

    dim3 grid(blockX, blockY);

    transpose_tiled_shared<<<grid, block>>>(A_d, B_d, rows, cols);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(B.data(), B_d, elements*sizeof(float), cudaMemcpyDeviceToHost));

    bool test = true;

    for (int i = 0; i < elements; i++) {
        int row = i / cols;
        int col = i % cols;

        int output = col * rows + row;

        if (std::abs(B[output] - A[i]) > tolerance) {
            test = false;
        }
    }

    std::cout << "The operation was a "<< (test ? "success!" : "failure") << std::endl;

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));

    return 0;
}