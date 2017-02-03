/* @file: get_gpu_info.cpp
 * @author: Zhang Xiao
 * @date: 2016.09.18
 */
#include <cstdio>

#include <cuda_runtime.h>
//#include <helper_cuda.h>
//#include <helper_functions.h>

void PrintDeviceInfo()
{	
	int devCount;

	//checkCudaErrors(cudaGetDeviceCount(&devCount));
	cudaGetDeviceCount(&devCount);

	if(devCount==0) {
		fprintf(stderr, "There is no device supporting CUDA\n");
	} else {

		for(int dev=0;dev<devCount;dev++) {
			cudaDeviceProp deviceProp;
			
	//		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
			cudaGetDeviceProperties(&deviceProp, dev);
	
			if(dev==0) {
				if(deviceProp.major==9999 && deviceProp.minor==9999) {
					fprintf(stderr, "There is no device supporting CUDA\n");
				} else {
					if(devCount==1) {
						fprintf(stderr, "There is 1 device supporting CUDA\n");
					} else {
						fprintf(stderr, "There are %d devices supporting CUDA\n", devCount);
					}
				}
	
			} // end if
	
			fprintf(stderr, "\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	
			fprintf(stderr, "  Major revision number:                               %d\n", deviceProp.major);
			fprintf(stderr, "  Minor revision number:                               %d\n", deviceProp.minor);
			fprintf(stderr, "  Total amount of global memory:                       %dG bytes\n", deviceProp.totalGlobalMem/(1024*1024*1024));
			fprintf(stderr, "  Number of multiprocessors:                           %d\n", deviceProp.multiProcessorCount);
			fprintf(stderr, "  Number of cores:                                     %d\n", deviceProp.multiProcessorCount*8);
			fprintf(stderr, "  Total amount of constant memory:                     %dK bytes\n", deviceProp.totalConstMem/1024);
			fprintf(stderr, "  Total amount of shared memory per block:             %dK bytes\n", deviceProp.sharedMemPerBlock/1024);
			fprintf(stderr, "  Total number of register available per block:        %d\n", deviceProp.regsPerBlock);
			fprintf(stderr, "  Warp size:                                           %d\n", deviceProp.warpSize);
			fprintf(stderr, "  Maximum number of threads per block:                 %d\n", deviceProp.maxThreadsPerBlock);
			fprintf(stderr, "  Maximum sizes of each dimension of a block:          %d x %d x %d\n", 
					deviceProp.maxThreadsDim[0],
					deviceProp.maxThreadsDim[1],
					deviceProp.maxThreadsDim[2]);
			fprintf(stderr, "  Maximum sizes of each dimension of a grid:           %d x %d x %d\n",
					deviceProp.maxGridSize[0],
					deviceProp.maxGridSize[1],
					deviceProp.maxGridSize[2]);
			fprintf(stderr, "  Maximum memory pitch:                                %u bytes\n", deviceProp.memPitch);
			fprintf(stderr, "  Texture alignment:                                   %u bytes\n", deviceProp.textureAlignment);
			fprintf(stderr, "  Clock rate:                                          %.2f GHz\n", deviceProp.clockRate*1e-6f);
			fprintf(stderr, "  Concurrent copy and execution:                       %s\n", deviceProp.deviceOverlap ? "Yes":"No");

		} // end for

		fprintf(stderr, "\nTest PASSED\n");
	} // end if
}

int main(int argc, char * argv[])
{
	PrintDeviceInfo();

	return 0;
}
