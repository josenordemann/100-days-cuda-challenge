#include <iostream>
#include <cuda_runtime.h>

__global__ void print_42(int*res){
    *res = 42;
}

int main(){
    int res_h = 0;                      //_h for host
    int* res_d = nullptr;               //_d for device
    cudaMalloc(&res_d, sizeof(int));    // alloc memory in the gpu for one int variable
    print_42<<<1,1>>>(res_d);           // to call the function, you may pass the number of blocks and the number of threads per block and than the adress of the variable that is being called
    cudaDeviceSynchronize();            //after calling a gpu function, you need to wait for the gpu to end

    //res_h = *res_d; that doesnt work, because you cant treat the gpu variables as you treat the cpu ones

    cudaMemcpy(&res_h, res_d, sizeof(int), cudaMemcpyDeviceToHost);         //the cudaMemcpy needs a pointer to the cpu variable, a pointer to the gpu variable, the number of byts and the type of the transference (h2d or d2h)
    std::cout<< res_h << std::endl;
    cudaFree(res_d);
    return 0;
}