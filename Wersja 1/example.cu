#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void reduce0(float *g_idata, float *g_odata, int size) {

	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = 0;
	if(i < size)
		sdata[tid] = g_idata[i];
	__syncthreads();

	for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
         sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

cudaError_t reduceWithCuda(float *input, int size) {
	float output = 0.0;
	//Liczba w¹tków na blok
	int threadsPerBlock = 256;
	//Liczba bloków (na pocz¹tku)
	int totalBlocks = (size+(threadsPerBlock-1))/threadsPerBlock;

	//Wektor wejœciowy i wyjœciowy device
	float *dev_i, *dev_o;
	cudaError_t cudaStatus;
	
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
	
	// Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_i, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_o, totalBlocks * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_i, input, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	  
	bool turn = true;
	
	// Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    cudaStatus = cudaEventCreate(&start);

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event!\n");
        goto Error;
    }

    cudaEvent_t stop;
    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event!\n");
        goto Error;
    }

    cudaStatus = cudaEventRecord(start, NULL);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event\n");
        goto Error;
    }
	  
	while(true) {	
		if(turn) {  
		  reduce0<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(dev_i, dev_o, size);
		  turn = false;
		} else {
		  reduce0<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(dev_o, dev_i, size);
		  turn = true;
		}
		
		if(totalBlocks == 1) break;
		
		size = totalBlocks;
		totalBlocks = ceil((double)totalBlocks/threadsPerBlock);
	}
	
    cudaStatus = cudaEventRecord(stop, NULL);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event!\n");
        goto Error;
    }

    cudaStatus = cudaEventSynchronize(stop);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event!\n");
        goto Error;
    }
	
	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "reduce0 launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
	
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }
	
	float msecTotal = 0.0f;
    cudaStatus = cudaEventElapsedTime(&msecTotal, start, stop);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
	  
	if(turn) {
		cudaStatus = cudaMemcpy(&output, &dev_i[0], sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	} else {
		cudaStatus = cudaMemcpy(&output, &dev_o[0], sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}
	
	printf("Wynik to: %f, w czasie: %f\n", output, msecTotal);
	
Error:
    cudaFree(dev_i);
    cudaFree(dev_o);
	
	return cudaStatus;
}

int main(void) {
	//Deklaracja rozmiaru
	int size = 939289;
	//Wektor wejœciowy hosta
	float *input = (float*)malloc(size * sizeof(float));
	for(int i = 0; i < size; i++)
		input[i] = 1.0;

	cudaError_t cudaStatus = reduceWithCuda(input, size);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "reduceWithCuda failed!");
        return 1;
    }
	
	cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	
	free(input);

	return 0;
}
