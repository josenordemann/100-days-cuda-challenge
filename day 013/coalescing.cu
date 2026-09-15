#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>

const int threads = 256;
const float tolerance = 1e-5f;

#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)


// in this code, we compare two different ways of reading the same row-major matrix
// in the first kernel, neighbor threads read neighbor elements, so the memory access is coalesced
// in the second kernel, neighbor threads read elements from the same column, so the distance between the reads is equal to the number of columns
// the second kernel writes the values consecutively in C, so we can focus the comparison on the reading pattern

__global__ void rowmajor_read (const float *A, float *B, const int row, const int col){
    int col_i = threadIdx.x + blockIdx.x*blockDim.x;
    int row_i = threadIdx.y + blockIdx.y*blockDim.y;

    if(col_i < col && row_i < row){
        int i = row_i*col + col_i;
        B[i] = A[i];
    }
}

__global__ void colmajor_read (const float *A, float *C, const int row, const int col){
    int index = threadIdx.x + blockIdx.x*blockDim.x;

    if(index < (col*row)){
        int row_i = index%row;
        int col_i = index/row;
        int input_index = row_i * col + col_i;
        C[index] = A[input_index];
    }
}

int main(){
    cudaEvent_t start;
    cudaEvent_t stop;
    int rows = 256;
    int cols = 128;
    int threadsX = 32;
    int threadsY = 8;
    dim3 block(threadsX, threadsY);

    int blocksX = (block.x + cols - 1)/block.x;
    int blocksY = (block.y + rows - 1)/block.y;
    dim3 grid(blocksX, blocksY);

    std::vector<float> A(rows*cols);
    std::vector<float> B(rows*cols);
    std::vector<float> C(rows*cols);

    float *A_d;
    float *B_d;
    float *C_d;

    int index;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            index = i*cols + j;
            A[index] =  i + j / 2.5f;
            B[index] = 0.0f;
            C[index] = 0.0f;
        }
    }

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMalloc(&A_d, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMalloc(&B_d, sizeof(float) * rows * cols));
    CUDA_CHECK(cudaMalloc(&C_d, sizeof(float) * rows * cols));

    CUDA_CHECK(cudaMemcpy(A_d, A.data(), sizeof(float)*rows*cols, cudaMemcpyHostToDevice));

    const int warm_up = 20;
    const int repetitions = 100;
    const int blocks1D = (rows * cols + threads - 1) / threads;

    float t1 = 0.0f;
    float t2 = 0.0f;

    // warm-up
    for (int i = 0; i < warm_up; i++) {
        rowmajor_read<<<grid, block>>>(A_d, B_d, rows, cols);
        }
    CUDA_CHECK(cudaGetLastError());

    for (int i = 0; i < warm_up; i++) {
        colmajor_read<<<blocks1D, threads>>>(A_d, C_d, rows, cols);
     }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());


    // row-major measurement
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repetitions; i++) {
        rowmajor_read<<<grid, block>>>(A_d, B_d, rows, cols);
        }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&t1, start, stop));

    t1 /= repetitions;

    // column traversal measurement
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repetitions; i++) {
        colmajor_read<<<blocks1D, threads>>>(A_d, C_d, rows, cols);
        }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&t2, start, stop));

    t2 /= repetitions;

    // copy the results to the host
    CUDA_CHECK(cudaMemcpy(B.data(), B_d, sizeof(float)*rows*cols,cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(C.data(), C_d, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost));

    bool test = true;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
        index = i * cols + j;

        int row_i = index % rows;
        int col_i = index / rows;
        int input_index = row_i * cols + col_i;

        if (std::abs(B[index] - A[index]) > tolerance || std::abs(C[index] - A[input_index]) > tolerance) {
            test = false;
            }
        }
    }

    std::cout << "The operation was a "<< (test? "success!" : "failure") << std::endl;
    std::cout << "Mean row major time: "<< t1 << std::endl;
    std::cout << "Mean column traversal time: "<< t2 << std::endl;
    float s = t2/t1;
    std::cout << "Speed-up: "<< s << std::endl;

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
