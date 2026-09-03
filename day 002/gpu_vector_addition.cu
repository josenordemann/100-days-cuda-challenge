#include <iostream>
#include <cuda_runtime.h>
const int N = 256;          //size of the vectors


__global__ void vector_addition(int *A, int *B, int *C){
    int i = blockDim.x*blockIdx.x + threadIdx.x;                 //the blocks have a dimension, that means their number of threads, an index, in this case, there are 8, so the index           
                                                      // goes from 0 to 7, and the threads also have an index inside of their block, then you need to add this too
    if (i<N){                   //here, we  need to confirm that the i index is less than the vector size
    C[i] = A[i] + B[i];
    }
}

int main(){

    int A[N];                 //host variables
    int B[N]; 

    int *C_d, *A_d, *B_d;       //device variables

    for(int i = 0; i < N; i++){         // initialize and assign values to A and B
        A[i] = i;
        B[i] = 2*i;                     // that way, C is supposed to be 3*i
    }

    cudaMalloc(&A_d, N*sizeof(int));        // allocating memory to the device
    cudaMalloc(&B_d, N*sizeof(int));
    cudaMalloc(&C_d, N*sizeof(int));
    
    cudaMemcpy(A_d, A, N*sizeof(int), cudaMemcpyHostToDevice);          //copying values to A_d and B_d
    cudaMemcpy(B_d, B, N*sizeof(int), cudaMemcpyHostToDevice);

    vector_addition<<<8, 32>>>(A_d, B_d, C_d); // here, we are using 8 blocks with 32 threads each, and then passing the a, b and c addresses
    cudaDeviceSynchronize();                  //waiting for the gpu to end

    int C_h[256];               //creating a host variable for C in order to copy the C_d values 

    cudaMemcpy(C_h, C_d, N*sizeof(int), cudaMemcpyDeviceToHost);            //copying C_d to C_h

    /* std::cout << "C = [";              //here, you can print all the values
    for(int i = 0; i < N-1; i++){
        std::cout << C_h[i] << ", ";
    }
    std::cout <<  C_h[255]<< "]" << std::endl;
    */

   int j = 0;                               // but we're going to check if the result is correct for all C elements
    for(int i = 0; i < N; i++){
        if(C_h[i] != 3*i)
            j = 1;
    }
    if(j == 0)
        std::cout << "It really worked!"<< std::endl;
    else
        std::cout << "ERROR!" << std::endl;

    cudaFree(A_d);                // free
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}

// the flow is: intitialize -> alloc memory -> copy -> kernel -> synchronize -> copy -> check the results -> free 