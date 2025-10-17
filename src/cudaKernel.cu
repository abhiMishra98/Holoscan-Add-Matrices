#include <iostream>
#include <cuda_runtime.h>
#include "cudaKernel.h"


__global__ void add( int N, int *a, int *b){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for(int i = index; i < N; i+=stride){
      int temp = a[i] + b[i];
        for (int j = 0; j < 1000; ++j) {
            temp += j % 5;
        }
        a[i] = temp;
    }
}

void initMatrix(int *a, int *b,int N){
    for(int i=0;i<N;i++){
        a[i]= 1.0f;
        b[i]= 2.0f;
    }
}

void addMatrix(int *a, int *b, int N){

    int blockSize=256;
    int numBlocks = (N+blockSize-1)/blockSize;
    add<<<numBlocks,blockSize>>>(N,a,b);
    //add<<<1,1>>>(N,a,b);

    cudaMemPrefetchAsync(a, N * sizeof(int),0,0);
    cudaMemPrefetchAsync(b, N * sizeof(int),0,0);

    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(b);
    
}