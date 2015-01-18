#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <ctime>
#include <time.h>
#include <sstream>
#include <string>
#include <fstream>

using namespace std;


__global__ void reduce0(int *g_idata, int *g_odata, int size){

   extern __shared__ int sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
   sdata[tid] = 0;
   if(i<size)
     sdata[tid] = g_idata[i];
   __syncthreads();

	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
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
  
  while(true){
    
    if(turn){
      
      reduce0<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(input, output, size);
      turn = false;
      
    }
    else{
      
      reduce0<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(output, input, size);
      turn = true;
    
    }
    
    if(totalBlocks == 1) break;
    
    size = totalBlocks;
    totalBlocks = ceil((double)totalBlocks/threadsPerBlock);
    
  }

  thrust::host_vector<int> data_h_o;
  
  if(turn)
    data_h_o = data_v_i;
  else 
    data_h_o = data_v_o;
  
  data_v_i.clear();
  data_v_i.shrink_to_fit();
  
  data_v_o.clear();
  data_v_o.shrink_to_fit();
  
  cout<<data_h_o[0]<<endl;


  return 0;

}
