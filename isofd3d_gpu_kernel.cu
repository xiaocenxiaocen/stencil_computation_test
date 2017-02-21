/* @file: isofd3d_gpu_kernel.cu
 * @author: xiaocen
 * @date: 2017.01.13
 */
#include <cstdio>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <iostream>
using namespace std;

#include <cuda_runtime.h>
#include "isofd3d_gpu_kernel.h"

//#define fd8_b0 -2.847222222f
//#define fd8_b1 1.6f
//#define fd8_b2 -0.2f
//#define fd8_b3 0.02539682540f
//#define fd8_b4 -0.001785714286f

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


template<typename T> T * ArrayAlloc(const size_t n0)
{
	T * ptr __attribute__((aligned(16))) = (T*)malloc(sizeof(T) * n0);
	memset(ptr, 0, sizeof(T) * n0);
	return ptr;
}

#define SCHEME_ORDER 8
#define HALF_SCHEME (SCHEME_ORDER / 2)
#define MODULO (HALF_SCHEME - 1)
#define BLOCK_Z 32
#define BLOCK_X 8 
#define ALIGNMENT_PAD (128 / sizeof(float))
#define STENCIL_Z BLOCK_Z + SCHEME_ORDER
#define STENCIL_X BLOCK_X + SCHEME_ORDER
__constant__ float coeff[HALF_SCHEME + 1];
static const float h_coeff[] = { -2.847222222, 1.6, -0.2, 0.02539682540, -0.001785714286};

void paddding(const int nx, const int nz, int & nxpad, int & nzpad)
{
	nzpad = (nz - SCHEME_ORDER) % BLOCK_Z == 0 ? (nz - SCHEME_ORDER) : ((int)((nz - SCHEME_ORDER) / BLOCK_Z) + 1) * BLOCK_Z;
	nxpad = (nx - SCHEME_ORDER) % BLOCK_X == 0 ? (nx - SCHEME_ORDER) : ((int)((nx - SCHEME_ORDER) / BLOCK_X) + 1) * BLOCK_X;
	nxpad += SCHEME_ORDER;
//	nzpad += SCHEME_ORDER;
	nzpad += ALIGNMENT_PAD * 2;
	
//	nzpad = nz % BLOCK_Z == 0 ? nz : ((int)(nz / BLOCK_Z) + 1) * BLOCK_Z;
//	nxpad = nx % BLOCK_X == 0 ? nx : ((int)(nx / BLOCK_X) + 1) * BLOCK_X;
}



__global__ static void inject_source_kernel(float * __restrict__ ucur, const float wav, const int source_index)
{
	ucur[source_index] += wav;
}


// stride_xz = nx * nz
// stride_z = nz;
__global__ static void fd3d_kernel(
float * __restrict__ ucur, float * __restrict__ unext, 
const int stride_xz, const int stride_z, const int ny, 
const float invsqdx, const float invsqdy, const float invsqdz, 
float * __restrict__ vel)
{
//	bool valid = true;
	__shared__ float sdata[STENCIL_X][STENCIL_Z];

	const int ltidz = threadIdx.x;
	const int ltidx = threadIdx.y;
	const int tz = ltidz + HALF_SCHEME;
	const int tx = ltidx + HALF_SCHEME;
	const int iz = blockIdx.x * blockDim.x + tz - HALF_SCHEME + ALIGNMENT_PAD;
	const int ix = blockIdx.y * blockDim.y + tx;
	const int m_idx = ix * stride_z + iz;
//	const int nx = stride_xz / stride_z;

//	if(ix < HALF_SCHEME || ix >= nx - HALF_SCHEME
//	|| iz < HALF_SCHEME || iz >= stride_z - HALF_SCHEME) valid = false; 

	float front[HALF_SCHEME];
	float behind[HALF_SCHEME];

	float * plane = ucur;
	float * plane_output = unext + HALF_SCHEME * stride_xz;
	float * vel_slide = vel + HALF_SCHEME * stride_xz;
	for(int iy = 0; iy < HALF_SCHEME; iy++, plane += stride_xz) {
		front[iy] = plane[m_idx];
		behind[iy] = plane[m_idx + (HALF_SCHEME + 1) * stride_xz];		
	}	

	plane = ucur + HALF_SCHEME * stride_xz;

	float cur = plane[m_idx];
	for(int iy = HALF_SCHEME; iy < ny - HALF_SCHEME; iy++, plane += stride_xz, plane_output += stride_xz, vel_slide += stride_xz) {
		__syncthreads();
		sdata[tx][tz] = cur;

		// top & bottom
		if(ltidx < HALF_SCHEME) {
			sdata[ltidx                           ][tz] = plane[m_idx - HALF_SCHEME * stride_z];
			sdata[ltidx + blockDim.y + HALF_SCHEME][tz] = plane[m_idx + blockDim.y  * stride_z];
		}
		// left & right
		if(ltidz < HALF_SCHEME) {
			sdata[tx][ltidz                           ] = plane[m_idx - HALF_SCHEME];
			sdata[tx][ltidz + blockDim.x + HALF_SCHEME] = plane[m_idx +  blockDim.x];
		}
		__syncthreads();
	
		float val, dev, m_pos;
		
//		val  = coeff[0] * cur * (invsqdx + invsqdy + invsqdz);
//		val += coeff[1] * ((front[(iy - 1) & MODULO] + behind[(iy    ) & MODULO]) * invsqdy 
//				 + (sdata[tx + 1][tz] + sdata[tx - 1][tz]) * invsqdx
//				 + (sdata[tx][tz + 1] + sdata[tx][tz - 1]) * invsqdz);
//
//		val += coeff[2] * ((front[(iy - 2) & MODULO] + behind[(iy + 1) & MODULO]) * invsqdy
//				 + (sdata[tx + 2][tz] + sdata[tx - 2][tz]) * invsqdx 
//				 + (sdata[tx][tz + 2] + sdata[tx][tz - 2]) * invsqdz);	
//
//		val += coeff[3] * ((front[(iy - 3) & MODULO] + behind[(iy + 2) & MODULO]) * invsqdy 
//				 + (sdata[tx + 3][tz] + sdata[tx - 3][tz]) * invsqdx 
//				 + (sdata[tx][tz + 3] + sdata[tx][tz - 3]) * invsqdz);
//
//		val += coeff[4] * ((front[(iy - 4) & MODULO] + behind[(iy + 3) & MODULO]) * invsqdy 
//				 + (sdata[tx + 4][tz] + sdata[tx - 4][tz]) * invsqdx 
//				 + (sdata[tx][tz + 4] + sdata[tx][tz - 4]) * invsqdz);

		m_pos = coeff[0] * cur;

		dev  = m_pos;
		dev += coeff[1] * (front[(iy - 1) & MODULO] + behind[(iy    ) & MODULO])
		     + coeff[2] * (front[(iy - 2) & MODULO] + behind[(iy + 1) & MODULO])
		     + coeff[3] * (front[(iy - 3) & MODULO] + behind[(iy + 2) & MODULO])
		     + coeff[4] * (front[(iy - 4) & MODULO] + behind[(iy + 3) & MODULO]);
		dev *= invsqdy;
		val  = dev;
		
		dev =  m_pos;
		dev += coeff[1] * (sdata[tx + 1][tz] + sdata[tx - 1][tz])
		     + coeff[2] * (sdata[tx + 2][tz] + sdata[tx - 2][tz])
		     + coeff[3] * (sdata[tx + 3][tz] + sdata[tx - 3][tz])
		     + coeff[4] * (sdata[tx + 4][tz] + sdata[tx - 4][tz]);
		dev *= invsqdx;
		val += dev;

		dev =  m_pos;
		dev += coeff[1] * (sdata[tx][tz + 1] + sdata[tx][tz - 1])
		     + coeff[2] * (sdata[tx][tz + 2] + sdata[tx][tz - 2])
		     + coeff[3] * (sdata[tx][tz + 3] + sdata[tx][tz - 3])
		     + coeff[4] * (sdata[tx][tz + 4] + sdata[tx][tz - 4]);
		dev *= invsqdz;
		
		val += dev;
		
//		val  = fd8_b0   * cur * (invsqdx + invsqdy + invsqdz);
//		val += fd8_b1   * ((front[(iy - 1) & MODULO] + behind[(iy + 1) & MODULO]) * invsqdy 
//				 + (sdata[tx + 1][tz] + sdata[tx - 1][tz]) * invsqdx
//				 + (sdata[tx][tz + 1] + sdata[tx][tz - 1]) * invsqdz);
//
//		val += fd8_b2   * ((front[(iy - 2) & MODULO] + behind[(iy + 2) & MODULO]) * invsqdy
//				 + (sdata[tx + 2][tz] + sdata[tx - 2][tz]) * invsqdx 
//				 + (sdata[tx][tz + 2] + sdata[tx][tz - 2]) * invsqdz);	
//
//		val += fd8_b3   * ((front[(iy - 3) & MODULO] + behind[(iy + 3) & MODULO]) * invsqdy 
//				 + (sdata[tx + 3][tz] + sdata[tx - 3][tz]) * invsqdx 
//				 + (sdata[tx][tz + 3] + sdata[tx][tz - 3]) * invsqdz);
//
//		val += fd8_b4   * ((front[(iy - 4) & MODULO] + behind[(iy + 4) & MODULO]) * invsqdy 
//				 + (sdata[tx + 4][tz] + sdata[tx - 4][tz]) * invsqdx 
//				 + (sdata[tx][tz + 4] + sdata[tx][tz - 4]) * invsqdz);
		

//		val  = coeff[0] * cur;
//		val += coeff[1] * ((front[(iy - 1) & MODULO] + behind[(iy + 1) & MODULO]) 
//				 + (sdata[tx + 1][tz] + sdata[tx - 1][tz])
//				 + (sdata[tx][tz + 1] + sdata[tx][tz - 1]));
//
//		val += coeff[2] * ((front[(iy - 2) & MODULO] + behind[(iy + 2) & MODULO])
//				 + (sdata[tx + 2][tz] + sdata[tx - 2][tz]) 
//				 + (sdata[tx][tz + 2] + sdata[tx][tz - 2]));	
//
//		val += coeff[3] * ((front[(iy - 3) & MODULO] + behind[(iy + 3) & MODULO])
//				 + (sdata[tx + 3][tz] + sdata[tx - 3][tz])
//				 + (sdata[tx][tz + 3] + sdata[tx][tz - 3]));
//
//		val += coeff[4] * ((front[(iy - 4) & MODULO] + behind[(iy + 4) & MODULO])
//				 + (sdata[tx + 4][tz] + sdata[tx - 4][tz]) 
//				 + (sdata[tx][tz + 4] + sdata[tx][tz - 4]));

//		if(valid) plane_output[m_idx] = - plane_output[m_idx] + 2.0 * cur + vel_slide[m_idx] * val;
		plane_output[m_idx] = - plane_output[m_idx] + 2.0 * cur + vel_slide[m_idx] * val;
		
		front[iy & MODULO] = cur;
		cur = behind[iy & MODULO];
		behind[iy & MODULO] = plane[m_idx + (HALF_SCHEME + 1) * stride_xz];
	}
}

void fd3d_gpu(const int nx, const int ny, const int nz,
	      const float dx, const float dy, const float dz,
	      float * uo, float * um,
	      const int nt, const float dt, const float * wav, float & gpuTime)
{
	PrintDeviceInfo();	

	int nxpad, nzpad;	
	paddding(nx, nz, nxpad, nzpad);

	const int nxyzpad = nxpad * nzpad * ny;
	
	float ops = (3.0 * 5.0 + 1) * nxyzpad * nt / (1024 * 1024 * 1024.);
	const int nxz  = nx * nz;
	const int stride_xz = nxpad * nzpad;
	const int stride_z = nzpad;

	const int sx = nx / 2 + 0.5;
	const int sz = nz / 2 + 0.5 - HALF_SCHEME + ALIGNMENT_PAD;
	const int sy = ny / 2 + 0.5;

	float * d_uo;
	float * d_um;
	float * d_vel;
	
	float * h_wfd = ArrayAlloc<float>(nxyzpad);
	float * h_vel = ArrayAlloc<float>(nxyzpad);
	for(int i = 0; i < nxyzpad; i++) h_vel[i] = 4 * dt * dt; 

	float mem = 3.0 * 4 * nxyzpad / (1024. * 1024.0 * 1024.0);
	cout << mem << "Gb\n";

	// copy host to device
	cudaMalloc((void**)&d_uo, sizeof(float) * nxyzpad);
	cudaMalloc((void**)&d_um, sizeof(float) * nxyzpad);
	cudaMemset(d_uo, 0, sizeof(float) * nxyzpad);
	cudaMemset(d_um, 0, sizeof(float) * nxyzpad);
	cudaMalloc((void**)&d_vel, sizeof(float) * nxyzpad);
	cudaMemcpy(d_vel, h_vel, sizeof(float) * nxyzpad, cudaMemcpyHostToDevice);	
	cudaMemcpyToSymbol(coeff, (void*)h_coeff, sizeof(float) * (HALF_SCHEME + 1));

	int grid_x = (nzpad - 2 * ALIGNMENT_PAD) / BLOCK_Z;
	int grid_y = (nxpad - SCHEME_ORDER) / BLOCK_X;
//	int grid_x = nzpad / BLOCK_Z;
//	int grid_y = nxpad / BLOCK_X;

	dim3 grids(grid_x, grid_y, 1);
	dim3 threads(BLOCK_Z, BLOCK_X, 1);

	float invsqdz = 1.0 / (dz * dz);
	float invsqdx = 1.0 / (dx * dx);
	float invsqdy = 1.0 / (dy * dy);
	const int sourceIndex = sy * stride_xz + sx * stride_z + sz;

	// kernel execution
	
	cudaEvent_t start;
	cudaEvent_t finish;

	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start,0);

	for(int it = 0; it < nt; it++) {
	
		if(it % 10 == 0) fprintf(stdout, "INFO: %04d time steps of total time steps %04d\n", it, nt);

		inject_source_kernel<<<1, 1>>>(d_uo, wav[it], sourceIndex);

		fd3d_kernel<<<grids, threads>>>(d_uo, d_um, 
						stride_xz, stride_z, ny, 
						invsqdx, invsqdy, invsqdz, 
						d_vel);
		
		{
			float * swap_ptr = d_uo;
			d_uo = d_um;
			d_um = swap_ptr;
		}
	}
	
	cudaEventRecord(finish,0);

	cudaEventSynchronize(finish);
	
	cudaEventElapsedTime(&gpuTime,start,finish);

	cudaEventDestroy(start);
	cudaEventDestroy(finish);

	gpuTime *= 0.001;
	cout << "\nwall time of gpu fd3d: " << gpuTime << "\n";
	gpuTime = ops / gpuTime;

	// copy device to host
	cudaMemcpy(h_wfd, d_uo, sizeof(float) * nxyzpad, cudaMemcpyDeviceToHost);
	for(int iy = 0; iy < ny; iy++) {
		for(int ix = 0; ix < nx; ix++) {
			memcpy(uo + iy * nxz + ix * nz, h_wfd + iy * stride_xz + ix * stride_z + ALIGNMENT_PAD - HALF_SCHEME, sizeof(float) * nz);
		}
	}

	
	cudaFree(d_uo); d_uo = NULL;
	cudaFree(d_um); d_um = NULL;
	cudaFree(d_vel); d_vel = NULL;
	free(h_wfd); h_wfd = NULL;
	free(h_vel); h_vel = NULL;

}
