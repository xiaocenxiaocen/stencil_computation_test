#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <iostream>
#include <complex>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>

#define INTEL_C_COMPILER
#if defined(GNU_C_COMPILER)
extern "C" {
#include "cblas.h"
#include "lapacke.h"
#include "lapacke_mangling.h"
}
#elif defined(INTEL_C_COMPILER)
#include "fftw3.h"
#include "mkl.h"
#endif

#define VERBOSITY
using std::cout;
using std::complex;

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define nullptr NULL

#define SCHEME_RADIUS 4
#define BLOCK_X 32
#define BLOCK_Y 8
#define BLOCK_Z 1
#define MEM_PATTERN_X 32
#define THREAD_X 32
#define THREAD_Y 8
#define UNROLL 8
#define DEBUG

#define safeCall(err) __safeCall(err, __FILE__, __LINE__)

inline void __safeCall(cudaError err, const char * file, const int line)
{
	if(cudaSuccess != err) {
		fprintf(stderr, "ERROR: safeCall() Runtime API error in file <%s>, line %i : %s.\n", file , line, cudaGetErrorString(err));
		exit(-1);
	}
}

class TimerGPU {
public:
	cudaEvent_t start, stop;
	cudaStream_t stream;
	TimerGPU(cudaStream_t stream_ = 0) : stream(stream_) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, stream);
	}
	~TimerGPU() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	float read() {
		cudaEventRecord(stop, stream);
		cudaEventSynchronize(stop);
		float time;
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}
};

class TimerCPU {
	static const int bits = 10;
public:
	long long beg_clock;
	float freq;
	TimerCPU(float freq_) : freq(freq_) { 
		beg_clock = getTSC(bits);
	}
	long long getTSC(int bits) {
#ifdef WIN32
		return __rdtsc();
#else
		unsigned int low, high;
		__asm__(".byte 0x0f, 0x31" :"=a" (low), "=d" (high));
		return ((long long)high<<(32 - bits)) | ((long long)low >> bits);
#endif
	}
	float read() {
		long long end_clock = getTSC(bits);
		long long Kcycles = end_clock - beg_clock;
		float time = (float)(1 << bits) * Kcycles / freq / 1e3f;
		return time;
	}
};

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);

int iDivUp(int a, int b) { return (a % b == 0) ? (a / b) : (a / b + 1); }
int iDivDown(int a, int b) { return a / b; }
int iAlignUp(int a, int b) { return (a % b == 0) ? a : (a - a % b + b); }
int iAlignDown(int a, int b) { return a - a % b; }

__constant__ float coeff[SCHEME_RADIUS + 1];
static const float h_coeff[] = { -2.847222222, 1.6, -0.2, 0.02539682540, -0.001785714286};

template<typename h_T, typename d_T, size_t BX, size_t BY, size_t BZ>
class Wavefield3d {
public:
	Wavefield3d();
	~Wavefield3d();
	void allocate(const int _nx, const int _ny, const int _nz, const int _radius, bool host, d_T * devmem, h_T * hostmem);
	double download();
	double readback();
public:
	int nx, ny, nz;
	int nxpad, nypad, nzpad;
	int radius;
	int padding;
	h_T * h_wf;
	d_T * d_wf;
	bool h_internalAlloc;
	bool d_internalAlloc;
};

#define ALIGNEDMENT_BITS 128
template<typename h_T, typename d_T, size_t BX, size_t BY, size_t BZ>
void Wavefield3d<h_T, d_T, BX, BY, BZ>::allocate(const int _nx, const int _ny, const int _nz, const int _radius, bool host, d_T * devmem, h_T * hostmem)
{
	nx = _nx; ny = _ny; nz = _nz; radius = _radius;
	nxpad = iAlignUp(nx - 2 * radius, BX) + 2 * radius;
	nzpad = iAlignUp(nz - 2 * radius, BZ) + 2 * radius;
	nypad = iAlignUp(ny - 2 * radius, BY) + 2 * radius;
	long long int volumeSize = ((long long int)nxpad) * nypad * nzpad;
#ifdef VERBOSITY
	fprintf(stdout, "INFO: nx = %d, ny = %d, nz = %d.\n", nx, ny, nz);
	fprintf(stdout, "INFO: nxpad = %d, nypad = %d, nzpad = %d.\n", nxpad, nypad, nzpad);
	fflush(stdout);
#endif

	padding = ALIGNEDMENT_BITS / sizeof(h_T) - radius;
	volumeSize += padding;
	
	h_wf = hostmem;
	d_wf = devmem;
	if(d_wf == nullptr) {
		if(volumeSize < 0) {
			fprintf(stderr, "ERROR: cannot allocate %lld bytes from device global memory, file: %s, line: %d\n", volumeSize * sizeof(d_T), __FILE__, __LINE__);
			d_wf = nullptr;
			exit(EXIT_FAILURE);

		}
		safeCall(cudaMalloc((void**)&d_wf, volumeSize * sizeof(d_T))); 
		safeCall(cudaMemset(d_wf, 0, volumeSize * sizeof(d_T)));
		if(d_wf == nullptr) {
			fprintf(stderr, "ERROR: cannot allocate %lld bytes from device global memory, file: %s, line: %d\n", volumeSize * sizeof(d_T), __FILE__, __LINE__);
		}
		d_internalAlloc = true;
	}
	if(host && h_wf == nullptr) {
		long long int h_volumeSize = nx * ny * nz;
		if(h_volumeSize < 0) {
			fprintf(stderr, "ERROR: cannot allocate %lld bytes from host memory, file: %s, line: %d\n", h_volumeSize * sizeof(h_T), __FILE__, __LINE__);
			h_wf = nullptr;
			exit(EXIT_FAILURE);
		}
		h_wf = (float*)malloc(sizeof(h_T) * h_volumeSize);
		memset(h_wf, 0, h_volumeSize * sizeof(h_T));
		h_internalAlloc = true;
	}
}

template<typename h_T, typename d_T, size_t BX, size_t BY, size_t BZ>
Wavefield3d<h_T, d_T, BX, BY, BZ>::Wavefield3d() : nx(0), ny(0), nz(0), radius(0), h_wf(nullptr), d_wf(nullptr), h_internalAlloc(false), d_internalAlloc(false)
{

}

template<typename h_T, typename d_T, size_t BX, size_t BY, size_t BZ>
Wavefield3d<h_T, d_T, BX, BY, BZ>::~Wavefield3d()
{
	if(h_internalAlloc && h_wf != nullptr) free(h_wf);
	h_wf = nullptr;
	if(d_internalAlloc && d_wf != nullptr) safeCall(cudaFree(d_wf));
	d_wf = nullptr;
}

template<typename h_T, typename d_T, size_t BX, size_t BY, size_t BZ>
double Wavefield3d<h_T, d_T, BX, BY, BZ>::download()
{
	TimerGPU timer(0);
	int stride_z = sizeof(d_T) * nzpad;
	int d_stride_y = nxpad * nzpad;
	int h_stride_y = nx * nz;
	if(h_wf != nullptr && d_wf != nullptr) {
		h_T * h_ptr = h_wf;
		d_T * d_ptr = d_wf + padding;
		for(int iy = 0; iy < ny; iy++) {
			safeCall(cudaMemcpy2D(d_ptr, stride_z, h_ptr, sizeof(h_T) * nz, sizeof(h_T) * nz, nx, cudaMemcpyHostToDevice));
			h_ptr += h_stride_y;
			d_ptr += d_stride_y;
		}
	}
	double gpuTime = timer.read();
#ifdef VERBOSITY
	fprintf(stdout, "INFO: download time = %.2fms\n", gpuTime);
	fflush(stdout);
#endif
	return gpuTime;	
}

template<typename h_T, typename d_T, size_t BX, size_t BY, size_t BZ>
double Wavefield3d<h_T, d_T, BX, BY, BZ>::readback()
{
	TimerGPU timer(0);
	int stride_z = sizeof(d_T) * nzpad;
	int d_stride_y = nxpad * nzpad;
	int h_stride_y = nx * nz;
	if(h_wf != nullptr && d_wf != nullptr) {
		h_T * h_ptr = h_wf;
		d_T * d_ptr = d_wf + padding;
		for(int iy = 0; iy < ny; iy++) {
			safeCall(cudaMemcpy2D(h_ptr, sizeof(h_T) * nz, d_ptr, stride_z, sizeof(d_T) * nz, nx, cudaMemcpyDeviceToHost));
			h_ptr += h_stride_y;
			d_ptr += d_stride_y;
		}
	}
	double gpuTime = timer.read();
#ifdef VERBOSITY
	fprintf(stdout, "INFO: download time = %.2fms\n", gpuTime);
	fflush(stdout);
#endif
	return gpuTime;	
}

// 2.5d tiling
// DX >= RADIUS
// DY >= RADIUS
template<size_t BX, size_t BY, size_t TX, size_t TY, size_t DX, size_t DY, size_t RADIUS>
__global__ static void fdtd3d_kernel_template(float * __restrict__ wf_next, float * __restrict__ wf_cur, const int stride_y, const int stride_z, const int ny, 
					const float idx2, const float idy2, const float idz2, float * __restrict__ vel)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	
	const int tid = ty * TX + tx;
	
	const int idx = tid % DX;
	const int idy = tid / DX;

	const int l_tx = idx + RADIUS;
	const int l_ty = idy + RADIUS;

	const int gx = blockIdx.x * blockDim.x;
	const int gy = blockIdx.y * blockDim.y;
	
	__shared__ float s_data[BY + 2 * RADIUS][BX + 2 * RADIUS];

	float front[RADIUS][BY / DY][BX / DX];
	float rear[RADIUS][BY / DY][BX / DX];
	float cur[BY / DY][BX / DX];
	float laplacian[BY / DY][BX / DX];
	
	const int p_idx = (gy + l_ty) * stride_z + gx + l_tx;
	float * plane_input = wf_cur + p_idx;
	float * plane_output = wf_next + RADIUS * stride_y + p_idx;
	float * plane_vel = vel + RADIUS * stride_y + p_idx;
	int iz, ix, iy, i;
	#pragma unroll
	for(iy = 0; iy < RADIUS; iy++) {
		#pragma unroll
		for(ix = 0; ix < BY / DY; ix++) {
			#pragma unroll
			for(iz = 0; iz < BX / DX; iz++) {
				front[iy][ix][iz] = plane_input[ix * DY * stride_z + iz * DX];
				rear[iy][ix][iz] = plane_input[(RADIUS + 1) * stride_y 
							      + ix * DY * stride_z + iz * DX];	
			}
		}
		plane_input += stride_y;
	}
	
	#pragma unroll
	for(ix = 0; ix < BY / DY; ix++) {
		#pragma unroll
		for(iz = 0; iz < BX / DX; iz++) {
			cur[ix][iz] = plane_input[ix * DY * stride_z + iz * DX];
		}
	}

	#pragma unroll UNROLL
	for(iy = RADIUS; iy < ny - RADIUS; iy++) {
		#pragma unroll
		for(ix = 0; ix < BY / DY; ix++) {
			#pragma unroll
			for(iz = 0; iz < BX / DX; iz++) {
				s_data[l_ty + ix * DY][l_tx + iz * DX] = cur[ix][iz];
			}
		}
		
		// top & bottom
		if(idy < RADIUS) {
			#pragma unroll
			for(iz = 0; iz < BX / DX; iz++) {
				s_data[idy              ][l_tx + iz * DX] = plane_input[- RADIUS * stride_z + iz * DX];
				s_data[idy + BY + RADIUS][l_tx + iz * DX] = plane_input[      BY * stride_z + iz * DX];
			}	
		}
		
		// left & right				
		if(idx < RADIUS) {
			#pragma unroll
			for(ix = 0; ix < BY / DY; ix++) {
				s_data[l_ty + ix * DY][idx              ] = plane_input[ix * DY * stride_z - RADIUS];
				s_data[l_ty + ix * DY][idx + BX + RADIUS] = plane_input[ix * DY * stride_z +     BX]; 
			}
		}
		__syncthreads();

		float deriv;
		#pragma unroll
		for(ix = 0; ix < BY / DY; ix++) {
			#pragma unroll
			for(iz = 0; iz < BX / DX; iz++) {
				float m_pos = coeff[0] * cur[ix][iz];
				deriv = m_pos;
				#pragma unroll
				for(i = 1; i <= RADIUS; i++) {
					deriv += coeff[i] * (s_data[l_ty + ix * DY][l_tx + iz * DX + i] + s_data[l_ty + ix * DY][l_tx + iz * DX - i]);
				}
				laplacian[ix][iz]  = deriv * idz2;
				
				deriv = m_pos;
				#pragma unroll
				for(i = 1; i <= RADIUS; i++) {
					deriv += coeff[i] * (s_data[l_ty + ix * DY + i][l_tx + iz * DX] + s_data[l_ty + ix * DY - i][l_tx + iz * DX]);
				}
				laplacian[ix][iz] += deriv * idx2;

				deriv = m_pos;
				#pragma unroll
				for(i = 1; i <= RADIUS; i++) {
					deriv += coeff[i] * (front[RADIUS - i][ix][iz] + rear[i - 1][ix][iz]);
				}
				laplacian[ix][iz] += deriv * idy2;
				
		//		plane_output[ix * DY * stride_z + iz * DX] = - plane_output[ix * DY * stride_z + iz * DX] + cur[ix][iz] + cur[ix][iz] + plane_vel[ix * DY * stride_z + iz * DX] * laplacian[ix][iz];
				plane_output[ix * DY * stride_z + iz * DX] = plane_vel[ix * DY * stride_z + iz * DX];
			}
		}
	
		#pragma unroll
		for(ix = 0; ix < BY / DY; ix++) {
			#pragma unroll
			for(iz = 0; iz < BX / DX; iz++) {
				#pragma unroll
				for(i = 0; i < RADIUS - 1; i++) {
					front[i][ix][iz] = front[i + 1][ix][iz];
				}
				front[RADIUS - 1][ix][iz] = cur[ix][iz];
				cur[ix][iz] = rear[0][ix][iz];
				#pragma unroll
				for(i = 0; i < RADIUS - 1; i++) {
					rear[i][ix][iz] = rear[i + 1][ix][iz];
				}
				rear[RADIUS - 1][ix][iz] = plane_input[(RADIUS + 1) * stride_y + ix * DY * stride_z + iz * DX];
			}
		}	

		plane_vel += stride_y;
		plane_input += stride_y;
		plane_output += stride_y;
		__syncthreads();
	}
}

__device__ float wsinc(float x)
{
	float sinc = 1.f;
	float wind = 0.f;
	float tmp = M_PI * x;
	if(x) sinc = __sinf(tmp) / tmp;
	if(x > -4.f && x < 4.f) wind = 0.5f * (1.f + __cosf(tmp / 4.f));
	return sinc * wind;
}

template<size_t BX, size_t BY, size_t TX, size_t TY, size_t DX, size_t DY, size_t RADIUS>
__global__ void inject_source_kernel_template(
	float * __restrict__ wf_cur, float wavelet, 
	float sx, float sy, float sz, 
	const int x1, const int x2, const int z1, const int z2, const int y1, const int y2,
	const int stride_y, const int stride_z)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	
	const int tid = ty * TX + tx;
	
	const int idx = tid % DX;
	const int idy = tid / DX;

	const int l_tx = idx + RADIUS;
	const int l_ty = idy + RADIUS;

	const int gx = blockIdx.x * blockDim.x + l_tx;
	const int gy = blockIdx.y * blockDim.y + l_ty;
	
	float * ptr_output = wf_cur + gy * stride_z + gx;	

	int iy, ix, iz;
	const float km = 0.68f;
	#pragma unroll
	for(ix = 0; ix < BY / DY; ix++) {
		#pragma unroll
		for(iz = 0; iz < BX / DX; iz++) {
			int x = gx + iz * DX;
			int z = gy + ix * DY;
			if(x >= x1 && x < x2 && z >= z1 && z < z2) {
				float xs = wsinc(km * (sx - x));
				float zs = wsinc(km * (sz - z));
				#pragma unroll	
				for(iy = y1; iy < y2; iy++) {
					float ys = wsinc(km * (sy - iy));
					ptr_output[iy * stride_y + ix * DY * stride_z + iz * DX] += wavelet * xs * ys * zs;
//					if(x == x1 && z == z1 && iy == y1) ptr_output[iy * stride_y + ix * DY * stride_z + iz * DX] += wavelet;
				}		
			}
		}
	}
}
	


// 2.5d tiling + time blocking
//template<size_t BX, size_t BY, size_t DX, size_t DY, >
//__global___ static void fdtd3d_kernel_template()
//{
//	
//}

template<size_t BX, size_t BY, size_t BZ, size_t DX, size_t DY, size_t TX, size_t TY, size_t RADIUS>
double fdtd3d_gpu_wrapper(
	      const int nx, const int ny, const int nz, const int radius,
	      const float dx, const float dy, const float dz,
	      float * wf_cur, float * h_vel, 
	      const int nt, const float dt, const float * wav, const int waveletLength)
{
	cout << DX << "\t" << DY << "\n";
	assert(radius == RADIUS);	

	Wavefield3d<float, float, BY, BZ, BX> d_wf_cur;
	d_wf_cur.allocate(nx, ny, nz, radius, false, nullptr, wf_cur);
	// initialize wavefield
	d_wf_cur.download();	

	Wavefield3d<float, float, BY, BZ, BX> d_wf_next;
	d_wf_next.allocate(nx, ny, nz, radius, false, nullptr, wf_cur);
	// initialize wavefiled
	d_wf_next.download();

	Wavefield3d<float, float, BY, BZ, BX> d_vel;
	d_vel.allocate(nx, ny, nz, radius, false, nullptr, h_vel);
	d_vel.download();

	long long int mpoints = (d_wf_cur.nxpad - 2 * d_wf_cur.radius) * (d_wf_cur.nypad - 2 * d_wf_cur.radius) * (d_wf_cur.nzpad - 2 * d_wf_cur.radius);
	
	cout << 1.0 * mpoints * nt / 1e6 << "Mpoints\n";	
	
	float gflops = (3.f * radius * (1 + 1 + 1) + 2.f + 4.f) * mpoints * nt / 1e9;
	
	cout << 1.0 * gflops << "GFLOP\n";

	const int stride_y = d_wf_cur.nxpad * d_wf_cur.nzpad;
	const int stride_z = d_wf_cur.nzpad;
	const int worky = d_wf_cur.ny;

	const float idx2 = 1.f / (dx * dx);
	const float idy2 = 1.f / (dy * dy);
	const float idz2 = 1.f / (dz * dz);
			
	const float sx = (nx - 1) * dx * 0.5f / dx;
	const float sz = (nz - 1) * dz * 0.5f / dz;
	const float sy = (ny - 1) * dy * 0.5f / dy;
	
	const float wind = 0.05f;
	const int windx = (int)(wind / dx + 0.5f);
	const int windy = (int)(wind / dy + 0.5f);
	const int windz = (int)(wind / dz + 0.5f);	

	const int x1 = sx - windx;
	const int x2 = sx + windx + 1;
	const int z1 = sz - windz;
	const int z2 = sz + windz + 1;
	const int y1 = sy - windy;
	const int y2 = sy + windy + 1;

	float memSize = 3.0 * 4.f * (d_wf_cur.nxpad * d_wf_cur.nypad * d_wf_cur.nzpad + d_wf_cur.padding) / (1024.0 * 1024.0 * 1024.0);
	cout << memSize << "Gb\n";
	
	cudaMemcpyToSymbol(coeff, (void*)h_coeff, sizeof(float) * (radius + 1));

	int grid_x = (d_wf_cur.nzpad - 2 * d_wf_cur.radius) / BX;
	int grid_y = (d_wf_cur.nxpad - 2 * d_wf_cur.radius) / BY;
	dim3 grids(grid_x, grid_y, 1);
	dim3 threads(BX, BY, 1);

	TimerGPU timer(0);
	// kernel execution
	int it = 0;
	for(; it <= nt - 2; it += 2) {
		// from wf_cur -> wf_next
		if(it % 10 == 0) fprintf(stdout, "INFO: %04d time steps of total time steps %04d\n", it, nt);
				
		fdtd3d_kernel_template<BX, BY, TX, TY, DX, DY, RADIUS><<<grids, threads>>>(
			d_wf_next.d_wf + d_wf_next.padding, d_wf_cur.d_wf + d_wf_cur.padding, 
			stride_y, stride_z, worky, 
			idx2, idy2, idz2, d_vel.d_wf + d_vel.padding);
		if(it < waveletLength) {
			inject_source_kernel_template<BX, BY, TX, TY, DX, DY, RADIUS><<<grids, threads>>>(
				d_wf_next.d_wf + d_wf_next.padding, wav[it],
				sx, sy, sz,
				x1, x2, z1, z2, y1, y2, 
				stride_y, stride_z);
		}
	
		// from wf_next -> wf_cur
		if((it + 1)  % 10 == 0) fprintf(stdout, "INFO: %04d time steps of total time steps %04d\n", it, nt);
		fdtd3d_kernel_template<BX, BY, TX, TY, DX, DY, RADIUS><<<grids, threads>>>(
			d_wf_cur.d_wf + d_wf_cur.padding, d_wf_next.d_wf + d_wf_next.padding, 
			stride_y, stride_z, worky, 
			idx2, idy2, idz2, d_vel.d_wf + d_vel.padding);
		if(it + 1 < waveletLength) {
			inject_source_kernel_template<BX, BY, TX, TY, DX, DY, RADIUS><<<grids, threads>>>(
				d_wf_cur.d_wf + d_wf_cur.padding, wav[it + 1], 
				sx, sy, sz,
				x1, x2, z1, z2, y1, y2, 
				stride_y, stride_z);
		}
	}
	
	for(; it < nt; it++) {
		if(it % 10 == 0) fprintf(stdout, "INFO: %04d time steps of total time steps %04d\n", it, nt);
		fdtd3d_kernel_template<BX, BY, TX, TY, DX, DY, RADIUS><<<grids, threads>>>(
			d_wf_next.d_wf + d_wf_next.padding, d_wf_cur.d_wf + d_wf_cur.padding, 
			stride_y, stride_z, worky, 
			idx2, idy2, idz2, d_vel.d_wf + d_vel.padding);
		if(it < waveletLength) {
			inject_source_kernel_template<BX, BY, TX, TY, DX, DY, RADIUS><<<grids, threads>>>(
				d_wf_next.d_wf + d_wf_next.padding, wav[it],
				sx, sy, sz,
				x1, x2, z1, z2, y1, y2, 
				stride_y, stride_z);
		}
	}
	double gpuTime = timer.read();

	fprintf(stdout, "INFO: elapsed time = %.2f ms.\n", gpuTime);
#ifdef VERBOSITY
	fprintf(stdout, "INFO: performance = %.2f GFLOPS.\n", gflops / (1e-3 * gpuTime));
	fprintf(stdout, "INFO: performance = %.2f Mpoints/s.\n", mpoints * nt / (1e6 * gpuTime * 1e-3));
#endif
	fflush(stdout);

	if((nt & 0x1) == 0) d_wf_cur.readback();
	else d_wf_next.readback();

#ifdef DEBUG
	FILE * fp = fopen("/d0/data/zx/wf_gpu.dat", "wb");
	fwrite(wf_cur, sizeof(float), nx * ny * nz, fp);
	fflush(fp);
	fclose(fp);
#endif
	
	return gpuTime;
}

void set_source_wavelet(float * wav, const int nt, const float dt, const float f0)
{
	for(int it = 0; it < nt; it++) {
		float ttime = it * dt;
		float temp = M_PI * M_PI * f0 * f0 * (ttime - 1.0 / f0) * (ttime - 1.0 / f0);
		wav[it] = (1.0 - 2.0 * temp) * expf(- temp);
	}	
}

int main(int argc, char * argv[])
{
	if(argc != 9) {
		fprintf(stderr, "USAGE: nx ny nz dx dy dz nt dt\n");
		return -1;
	}
	
	int nx = atoi(argv[1]);
	int ny = atoi(argv[2]);
	int nz = atoi(argv[3]);
	float dx = atof(argv[4]);
	float dy = atof(argv[5]);
	float dz = atof(argv[6]);
	int nt = atoi(argv[7]);
	float dt = atof(argv[8]);

	const int radius = 4;
	const float f0 = 20.f;
	const int waveletLength = 2 * (int)(1.f / f0 / dt + 0.5f);
	float * wav = (float*)malloc(sizeof(float) * waveletLength);
	set_source_wavelet(wav, waveletLength, dt, f0);
	for(int it = 0; it < waveletLength; it++) {
		wav[it] *= dt * dt / (dx * dy * dz);
	}
	
#ifdef VERBOSITY	
	fprintf(stdout, "INFO: finite difference forward modeling\n");
	fprintf(stdout, "INFO: nx = %d, ny = %d, nz = %d\n", nx, ny, nz);
	fflush(stdout);
#endif

	float * wf_cur = (float*)malloc(sizeof(float) * nx * ny * nz);
	float * wf_next = (float*)malloc(sizeof(float) * nx * ny * nz);
	float * h_vel = (float*)malloc(sizeof(float) * nx * ny * nz);
	float * h_wf_gpu = (float*)malloc(sizeof(float) * nx * ny * nz);
	
	memset(wf_cur, 0, sizeof(float) * nx * ny * nz);
	memset(wf_next, 0, sizeof(float) * nx * ny * nz);
	for(long long int i = 0; i < nx * ny * nz; i++) h_vel[i] = 4.f * 4.f * dt * dt;
	
	fdtd3d_gpu_wrapper<BLOCK_X, BLOCK_Y, BLOCK_Z, MEM_PATTERN_X, THREAD_X * THREAD_Y / MEM_PATTERN_X, THREAD_X, THREAD_Y, SCHEME_RADIUS>(
		nx, ny, nz, radius,
		dx, dy, dz, h_wf_gpu, h_vel,
		nt, dt, wav, waveletLength);

	free(wf_cur); wf_cur = nullptr;
	free(wf_next); wf_next = nullptr;
	free(h_vel); h_vel = nullptr;
	free(h_wf_gpu); h_wf_gpu = nullptr;
	free(wav); wav = nullptr;

	return EXIT_SUCCESS;
}
