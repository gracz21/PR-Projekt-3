#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <ctime>
#include <time.h>
#include <sstream>
#include <string>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;


__global__ void reduce0(int *g_idata, int *g_odata, int size){

   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
   sdata[tid] = 0;
   if(i<size)
     sdata[tid] = g_idata[i];
   __syncthreads();

	for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main(void){

  int size = 939289;
  thrust::host_vector<int> data_h_i(size, 1);

  //initialize the data, all values will be 1
  //so the final sum will be equal to size

  int threadsPerBlock = 1024;
  int totalBlocks = (size+(threadsPerBlock-1))/threadsPerBlock;
  
  thrust::device_vector<int> data_v_i = data_h_i;
  thrust::device_vector<int> data_v_o(totalBlocks);

  int* output = thrust::raw_pointer_cast(data_v_o.data());
  int* input = thrust::raw_pointer_cast(data_v_i.data());
  
  bool turn = true;
  
	cudaError_t error;
	
	cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	
	while(true) {	
		if(turn) {
			//Odpal kernel (tablica wejœciowa jako input, wyjœciowa jako output
			reduce0<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(input, output, size);
			turn = false;
		} else {
			//Odpal kernel (tablica wyjœciowa jako input, wejœciowa jako output
			reduce0<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(output, input, size);
			turn = true;
		}
		
		//Je¿eli zosta³ jeden blok, to obliczenia zosta³y zakoñczone
		if(totalBlocks == 1) break;
		
		//Korzystaj tylko z zakresu tablicy odpowiadaj¹cemu liczbie bloków z poprzedniej iteracji
		size = totalBlocks;
		//Oblicz now¹ liczbê bloków
		totalBlocks = ceil((double)totalBlocks/threadsPerBlock);
	}
	
	error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	
	//Wektor wyjœciowy hosta
	thrust::host_vector<int> data_h_o;
	  
	//Pobierz wynik
	if(turn)
		//Wynik w tablicy wejœciowej device
		data_h_o = data_v_i;
	else
		//Wynik w tablicy wyjœciowej device
		data_h_o = data_v_o;
	
	//Wyczyœæ wektory
	data_v_i.clear();
	data_v_i.shrink_to_fit();
	  
	data_v_o.clear();
	data_v_o.shrink_to_fit();
	  
	//Wypisz wynik
	cout<< "Wynik: " << data_h_o[0] << endl << "W czasie:" << msecTotal << endl;

	return 0;
}