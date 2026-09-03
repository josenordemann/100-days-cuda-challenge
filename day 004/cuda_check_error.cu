#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
// in this code, we are implementing a cuda check, a macro to see if something went wrong in our cuda function
// almost all cuda functions return a cudaError_t value, such as cudaSuccess, cudaErrorInvalidValue and cudaErrorMemoryAllocation
// if our cuda function doesnt return cudaSuccess, we will print a message to the dev.

// we will use this cuda check macro in all next codes

#define CUDA_CHECK(call) do {                                                                                                       \
    cudaError_t error = (call);                                                                                                     \
    if (error != cudaSuccess){                                                                                                      \
        fprintf(stderr, "CUDA error at %s, in the line %d. Type of error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));    \
        std::exit(EXIT_FAILURE);                                                                                                    \
    }                                                                                                                               \
}while (0)

// we are using a do while, then we create a cudaError_t variable called error that is going to receive our call, such as a cudaMemcpy
// after this, we check if the error variable is different from cudaSuccess, in other words, if something went wrong.
// in this case, we print the file name where the error occurred, the error line and the error type.

int main(){
    CUDA_CHECK(cudaErrorMemoryAllocation);             // passing an error type just to see if it works
    return 0;
}

// it worked! :)