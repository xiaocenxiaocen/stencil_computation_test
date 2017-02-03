#include <mkl.h>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <time.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <assert.h>

#include <mmintrin.h> 	// MMX
#include <xmmintrin.h>	// SSE
#include <emmintrin.h>	// SSE2
#include <immintrin.h>	// AVX

#include <cuda_runtime.h>

#include "isofd3d_gpu_kernel.h"

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define ABS(a) ((a) > 0 ? (a) : (-(a)))

#define NUM_THREADS 12

using namespace std;

pthread_mutex_t mutex;

template<typename T> T * ArrayAlloc(const size_t n0)
{
	T * ptr __attribute__((aligned(16))) = (T*)malloc(sizeof(T) * n0);
	memset(ptr, 0, sizeof(T) * n0);
	return ptr;
}

/* @brief: template function for 2d array allocation
 * @param: n0, n1, size of dimensions
 * @retval: buffer, buffer for array
 */
template<typename T> void ArrayAlloc2D(T ***buffer, const int n0, const int n1)
{
	*buffer = (T**)malloc(sizeof(T*) * n0);
	(*buffer)[0] = (T*)malloc(sizeof(T) * n0 * n1);
	for(int i0 = 1; i0 < n0; i0++) (*buffer)[i0] = (*buffer)[0] + i0 * n1;
	return;
}

/* @brief: template function for 3d array allocation
 * @param: n0, n1, n2 size of dimensions
 * @retval: buffer, buffer for array
 */
template<typename T> void ArrayAlloc3D(T ****buffer, const int n0, const int n1, const int n2)
{
	*buffer = (T***)malloc(sizeof(T**) * n0);
	(*buffer)[0] = (T**)malloc(sizeof(T*) * n0 * n1);
	for(int i0 = 1; i0 < n0; i0++) (*buffer)[i0] = (*buffer)[0] + i0 * n1;
	(*buffer)[0][0] = (T*)malloc(sizeof(T) * n0 * n1 * n2);
	for(int i1 = 1; i1 < n1; i1++) (*buffer)[0][i1] = (*buffer)[0][0] + i1 * n2;
	for(int i0 = 1; i0 < n0; i0++) {
		for(int i1 = 0; i1 < n1; i1++) {
			(*buffer)[i0][i1] = (*buffer)[0][0] + i0 * n1 * n2 + i1 * n2;
		}
	}
	return;
}

#define ArrayFree3D(ptr) \
{ \
	free(**(ptr)); \
	free(*(ptr)); \
	free(ptr); \
}

#define ArrayFree2D(ptr) \
{ \
	free(*(ptr)); \
	free(ptr); \
}

#define SCHEME_ORDER 8

#if SCHEME_ORDER == 8
static const float a[] __attribute__((aligned(16))) = { 0.8, -0.2, 0.038095238, -0.0035714286 };
static const float b[] __attribute__((aligned(16))) = { 1.6, -0.2, 0.02539682540, -0.001785714286 };
static const float b0 = -2.847222222;
#endif

__m128 mmx8;
__m128 mmx9;
__m128 mmx10;
__m128 mmx11;
__m128 mmx12;

/* compute 8th order central finite difference of 2rd order x-direction derivative of wfl at location with offset gloIdx */
inline __m128 FD8DDXX(const float *wfl, const int stride, const float dx)
{
	__m128 mmx13;
	__m128 mmx14;
	__m128 mmx15;

	mmx13 = _mm_loadu_ps(wfl);
//	tmp = _mm_set1_ps(b0);
	mmx13 = _mm_mul_ps(mmx13, mmx8);

	mmx14 = _mm_loadu_ps(wfl + 1 * stride);
	mmx15 = _mm_loadu_ps(wfl - 1 * stride);
	mmx15 = _mm_add_ps(mmx14, mmx15);
//	tmp = _mm_set1_ps(b[0]);
	mmx15 = _mm_mul_ps(mmx15, mmx9);
	mmx13 = _mm_add_ps(mmx13, mmx15);

	mmx14 = _mm_loadu_ps(wfl + 2 * stride);
	mmx15 = _mm_loadu_ps(wfl - 2 * stride);
	mmx15 = _mm_add_ps(mmx14, mmx15);
//	tmp = _mm_set1_ps(b[0]);
	mmx15 = _mm_mul_ps(mmx15, mmx10);
	mmx13 = _mm_add_ps(mmx13, mmx15);

	mmx14 = _mm_loadu_ps(wfl + 3 * stride);
	mmx15 = _mm_loadu_ps(wfl - 3 * stride);
	mmx15 = _mm_add_ps(mmx14, mmx15);
//	tmp = _mm_set1_ps(b[0]);
	mmx15 = _mm_mul_ps(mmx15, mmx11);
	mmx13 = _mm_add_ps(mmx13, mmx15);

	mmx14 = _mm_loadu_ps(wfl + 4 * stride);
	mmx15 = _mm_loadu_ps(wfl - 4 * stride);
	mmx15 = _mm_add_ps(mmx14, mmx15);
//	tmp = _mm_set1_ps(b[0]);
	mmx15 = _mm_mul_ps(mmx15, mmx12);
	mmx13 = _mm_add_ps(mmx13, mmx15);

	mmx14 = _mm_set1_ps(dx * dx);
	mmx13 = _mm_div_ps(mmx13, mmx14);

	return mmx13;
}

inline float fd8_dx(const float *  wfl, const float dx, const int stride)
{
	return (a[0] * (*(wfl + 1 * stride) - *(wfl - 1 * stride)) + a[1] * (*(wfl + 2 * stride) - *(wfl - 2 * stride)) + a[2] * (*(wfl + 3 * stride) - *(wfl - 3 * stride)) + a[3] * (*(wfl + 4 * stride) - *(wfl - 4 * stride))) / dx;
}

inline float fd8_dxx(const float *  wfl, const float dx, const int stride)
{
	return (b0 * (*wfl) + b[0] * (*(wfl + 1 * stride) + *(wfl - 1 * stride)) + b[1] * (*(wfl + 2 * stride) + *(wfl - 2 * stride)) + b[2] * (*(wfl + 3 * stride) + *(wfl - 3 * stride)) + b[3] * (*(wfl + 4 * stride) + *(wfl - 4 * stride))) / (dx * dx);
}

void fdtd_naive(const int nx, const int ny, const int nz, const float dx, const float dy, const float dz, float * po, float * pm, const int nt, const float dt, float * wav)
{
	const int sx = nx / 2 + 0.5;
	const int sy = ny / 2 + 0.5;
	const int sz = nz / 2 + 0.5;
	const float v = 2.0 * 2.0 * dt * dt;
	const int nxz = nx * nz;

	for(int it = 0; it < nt; it++) {
		po[sy * nxz + sx * nz + sz] += wav[it];

		for(int iy = SCHEME_ORDER/2; iy < ny - SCHEME_ORDER/2; iy++) {
			float * po_ptr = po + iy * nxz + SCHEME_ORDER/2 * nz;
			float * pm_ptr = pm + iy * nxz + SCHEME_ORDER/2 * nz;
			for(int ix = SCHEME_ORDER/2; ix < nx - SCHEME_ORDER/2; ix++, po_ptr += nz, pm_ptr += nz) {
				for(int iz = SCHEME_ORDER/2; iz < nz - SCHEME_ORDER/2; iz++) {
					float pxx = fd8_dxx(po_ptr + iz, dx, nz);
					float pyy = fd8_dxx(po_ptr + iz, dy, nxz);
					float pzz = fd8_dxx(po_ptr + iz, dz, 1);	
					
					pm_ptr[iz] = 2.0f * po_ptr[iz] - pm_ptr[iz] + v * ( pxx + pyy + pzz);
				}
			}
		}
		
		{
			float * swapPtr = pm;
			pm = po;
			po = swapPtr;
		}
	}
}

void fdtd_openmp_naive(const int nx, const int ny, const int nz, const float dx, const float dy, const float dz, float * po, float * pm, const int nt, const float dt, float * wav)
{
	const int sx = nx / 2 + 0.5;
	const int sy = ny / 2 + 0.5;
	const int sz = nz / 2 + 0.5;
	const float v = 2.0 * 2.0 * dt * dt;
	const int nxz = nx * nz;

	for(int it = 0; it < nt; it++) {
		po[sy * nxz + sx * nz + sz] += wav[it];		

		#ifdef _OPENMP
		#pragma omp parallel for num_threads(NUM_THREADS) shared(po, pm, nx, ny, nz, dx, dy, dz) schedule(dynamic)
		#endif
		for(int iy = SCHEME_ORDER/2; iy < ny - SCHEME_ORDER/2; iy++) {
			float * po_ptr = po + iy * nxz + SCHEME_ORDER/2 * nz;
			float * pm_ptr = pm + iy * nxz + SCHEME_ORDER/2 * nz;
			for(int ix = SCHEME_ORDER/2; ix < nx - SCHEME_ORDER/2; ix++, po_ptr += nz, pm_ptr += nz) {
				for(int iz = SCHEME_ORDER/2; iz < nz - SCHEME_ORDER/2; iz++) {
					float pxx = fd8_dxx(po_ptr + iz, dx, nz);
					float pyy = fd8_dxx(po_ptr + iz, dy, nxz);
					float pzz = fd8_dxx(po_ptr + iz, dz, 1);					
					pm_ptr[iz] = 2.0f * po_ptr[iz] - pm_ptr[iz] + v * (pxx + pyy + pzz);
//					pm_ptr[iz] += v * pyy;
				//	pm_ptr[iz] = 2.0f * po_ptr[iz] - pm_ptr[iz] +  v * (pxx + pzz);
				}
			}
		}
		
		{
			float * swapPtr = pm;
			pm = po;
			po = swapPtr;
		}
	}
}

void fdtd_openmp_vectorize(const int nx, const int ny, const int nz, const float dx, const float dy, const float dz, float * po, float * pm, const int nt, const float dt, float * wav)
{
	const int sx = nx / 2 + 0.5;
	const int sy = ny / 2 + 0.5;
	const int sz = nz / 2 + 0.5;
	const float v = 2.0 * 2.0 * dt * dt;
	const int nxz = nx * nz;
	const int cache_line_size = 64;
	const int thread_block_size = cache_line_size / sizeof(float) - SCHEME_ORDER;

	for(int it = 0; it < nt; it++) {
		po[sy * nxz + sx * nz + sz] += wav[it];		
		
		#ifdef _OPENMP
		#pragma omp parallel for num_threads(NUM_THREADS) shared(po, pm, nx, ny, nz, dx, dy, dz, thread_block_size) schedule(dynamic)
		#endif
		for(int ix = SCHEME_ORDER/2; ix < nx - SCHEME_ORDER/2; ix += thread_block_size) {
			for(int iy = SCHEME_ORDER/2; iy < ny - SCHEME_ORDER/2; iy++) {
				float * po_ptr = po + iy * nxz + ix * nz;
				float * pm_ptr = pm + iy * nxz + ix * nz;
				for(int iix = 0; iix < thread_block_size && ix + iix < nx - SCHEME_ORDER / 2; iix++, po_ptr += nz, pm_ptr += nz) {
					for(int iz = SCHEME_ORDER/2; iz < nz - SCHEME_ORDER/2; iz += 4) {
						__m128 mmx0;
						__m128 mmx1;

						mmx0 = FD8DDXX(po_ptr + iz, nz, dx);
						mmx1 = FD8DDXX(po_ptr + iz, nxz, dy);
						mmx0 = _mm_add_ps(mmx0, mmx1);
						mmx1 = FD8DDXX(po_ptr + iz, 1, dz);
						mmx0 = _mm_add_ps(mmx0, mmx1);
						mmx1 = _mm_set1_ps(v);
						mmx0 = _mm_mul_ps(mmx0, mmx1);

						__m128 mmx3 = _mm_loadu_ps(po_ptr + iz);
						__m128 mmx4 = _mm_loadu_ps(pm_ptr + iz);
						
						mmx3 = _mm_add_ps(mmx3, mmx3);
						mmx4 = _mm_sub_ps(mmx3, mmx4);
						
						mmx4 = _mm_add_ps(mmx4, mmx0);
						
						_mm_storeu_ps(pm_ptr + iz, mmx4);
					}
				}
			}
		}
		
		{
			float * swapPtr = pm;
			pm = po;
			po = swapPtr;
		}
	}
}

struct fdtd_params_t {
	int nx;
	int ny;
	int nz;
	float dx;
	float dy;
	float dz;
	float dt;
	float * po;
	float * pm;
	int ntasks;
	int * taskPool;
	vector<vector<int> > * tasks;
	fdtd_params_t() : po(NULL), pm(NULL), taskPool(NULL), tasks(NULL) { };
	fdtd_params_t(const int nx_, const int ny_, const int nz_, const float dx_, const float dy_, const float dz_, const float dt_, vector<vector<int> > * tasks, const int ntasks) : nx(nx_), ny(ny_), nz(nz_), dx(dx_), dy(dy_), dz(dz_), dt(dt_), tasks(tasks), ntasks(ntasks), po(NULL), pm(NULL), taskPool(NULL) { };
};

void * fdtdThread(void * ptr)
{
	fdtd_params_t * fdtd = (fdtd_params_t*)(ptr);
	vector<vector<int> > * tasks = fdtd->tasks;
	int * taskPool = fdtd->taskPool;
	int ntasks = fdtd->ntasks;
	int nx = fdtd->nx;
	int ny = fdtd->ny;
	int nz = fdtd->nz;
	float dx = fdtd->dx;
	float dy = fdtd->dy;
	float dz = fdtd->dz;
	float dt = fdtd->dt;
	float * po = fdtd->po;
	float * pm = fdtd->pm;
	int nxz = nx * nz;
	float v = 2.0 * 2.0 * dt * dt;

	for(;;) {
		pthread_mutex_lock(&mutex);
		if(*taskPool >= ntasks) {
			pthread_mutex_unlock(&mutex);
			break;
		}
		int taskIdx = *taskPool;
		vector<int> params = tasks->at(taskIdx);
		(*taskPool)++;
		pthread_mutex_unlock(&mutex);
		int l = params[0];
		int r = params[1];
		int b = params[2];
		int t = params[3];
		assert((t - b) % 4 == 0);
		for(int iy = SCHEME_ORDER / 2; iy < ny - SCHEME_ORDER / 2; iy++) {
			float * po_ptr = po + iy * nxz + l * nz;
			float * pm_ptr = pm + iy * nxz + l * nz;
			for(int ix = l; ix < r; ix++, po_ptr += nz, pm_ptr += nz) {
				for(int iz = b; iz < t; iz += 4) {
					__m128 mmx0;
					__m128 mmx1;

					mmx0 = FD8DDXX(po_ptr + iz, nz, dx);
					mmx1 = FD8DDXX(po_ptr + iz, nxz, dy);
					mmx0 = _mm_add_ps(mmx0, mmx1);
					mmx1 = FD8DDXX(po_ptr + iz, 1, dz);
					mmx0 = _mm_add_ps(mmx0, mmx1);
					mmx1 = _mm_set1_ps(v);
					mmx0 = _mm_mul_ps(mmx0, mmx1);

					__m128 mmx3 = _mm_loadu_ps(po_ptr + iz);
					__m128 mmx4 = _mm_loadu_ps(pm_ptr + iz);
				
					mmx3 = _mm_add_ps(mmx3, mmx3);
					mmx4 = _mm_sub_ps(mmx3, mmx4);
					
					mmx4 = _mm_add_ps(mmx4, mmx0);
					
					_mm_storeu_ps(pm_ptr + iz, mmx4);
				}
			}
		}
	}
	pthread_exit(NULL);
}

void fdtd_pthread(const int nx, const int ny, const int nz, const float dx, const float dy, const float dz, float * po, float * pm, const int nt, const float dt, float * wav)
{
	const int sx = nx / 2 + 0.5;
	const int sy = ny / 2 + 0.5;
	const int sz = nz / 2 + 0.5;
	const float v = 2.0 * 2.0 * dt * dt;
	const int nxz = nx * nz;

#define CACHE_LINE_SIZE 64
	const int thread_block_size = CACHE_LINE_SIZE / sizeof(float) - SCHEME_ORDER;
//	const int thread_block_size = 128;

	vector<vector<int> > tasks;
	for(int ix = SCHEME_ORDER / 2; ix < nx - SCHEME_ORDER / 2; ix += thread_block_size) {
//		for(int iz = SCHEME_ORDER / 2; iz < nz - SCHEME_ORDER / 2; iz += thread_block_size) {
			vector<int> temp(4, 0);
			temp[0] = ix;
			temp[1] = min(ix + thread_block_size, nx - SCHEME_ORDER / 2);
	//		temp[2] = iz;
	//		temp[3] = min(iz + thread_block_size, nz - SCHEME_ORDER / 2);
			temp[2] = SCHEME_ORDER / 2; 
			temp[3] = nz - SCHEME_ORDER / 2;
	//		temp[4] = iy;
	//		temp[5] = min(iy + thread_block_size, ny - SCHEME_ORDER / 2);
			tasks.push_back(temp);
//		}
	}
	int ntasks = tasks.size();

	fdtd_params_t fdtd[NUM_THREADS];
	pthread_t thread[NUM_THREADS];
	pthread_attr_t attr;
	size_t stacksize;
	void * status;
	pthread_mutex_init(&mutex, NULL);
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	pthread_attr_getstacksize(&attr, &stacksize);
	fprintf(stderr, "INFO: default stack size = %li\n", stacksize);

	for(int t = 0; t < NUM_THREADS; t++) {
		fdtd[t] = fdtd_params_t(nx, ny, nz, dx, dy, dz, dt, &tasks, ntasks);
	}	
	for(int it = 0; it < nt; it++) {
		po[sy * nxz + sx * nz + sz] += wav[it];
		if(it % 10 == 0) fprintf(stdout, "INFO: %04d time step of total time steps %04d.\n", it, nt);
	
		int tasks = 0;
	
		for(int t = 0; t < NUM_THREADS; t++) {
			fdtd[t].taskPool = &tasks;
			fdtd[t].po = po;
			fdtd[t].pm = pm;	
			int ret = pthread_create(thread + t, &attr, fdtdThread, (void*)(fdtd + t));
			if(ret) {
				fprintf(stderr, "ERROR: return code from pthread_create() is %d",ret);
				exit(-1);
			}
		}
		for(int t = 0; t < NUM_THREADS; t++) {
			int ret = pthread_join(thread[t], &status);
			if(ret) {
				fprintf(stderr, "ERROR: return code from pthread_join() is %d",ret);
				exit(-1);
			}
		}

		{
			float * swapPtr = pm;
			pm = po;
			po = swapPtr;
		}
	}

}


void SetRickerWavlet(float * wav, const int nt, const float dt, const float f0)
{
	for(int it = 0; it < nt; it++) {
		float ttime = it * dt;
		float temp = M_PI * M_PI * f0 * f0 * (ttime - 1.0 / f0) * (ttime - 1.0 / f0);
		wav[it] = (1.0 - 2.0 * temp) * expf(- temp);
	}	
}

template<typename T>
T check_identical(const int nrows, const int ld, T const * __left, T const * __right)
{
	T ret(0.0);
	size_t ninstance = nrows * ld;
	T max_abs_left(0.0);
	for(auto i = 0; i < ninstance; i++) {
		max_abs_left = MAX(max_abs_left, ABS(*(__left + i)));
	}

	for(auto i = 0; i < ninstance; i++) {
	//	ret = MAX(ret, ABS(*(__left + i) - *(__right + i)) / (ABS(*(__left + i)) + 1e-7 * max_abs_left));
		ret = MAX(ret, ABS(*(__left + i) - *(__right + i)));
	}
	return ret;
}

int main(int argc, char * argv[])
{
	FILE * fp;

	int nx = 1024;
	int ny = 1024;
	int nz = 960;
	const float dx = 0.01;
	const float dy = 0.01;
	const float dz = 0.01;
	const float fpeak = 20;
	const float dt = 0.001;
	const float nt = 100;
	int64_t nxyz = nx * ny * nz;
	const int nxz = nx * nz;

	float * wav = ArrayAlloc<float>(nt);
	SetRickerWavlet(wav, nt, dt, fpeak);	

	int nTest = 1;

	float ops = (5 * 3 + 1) * nxyz * nt / (1024 * 1024 * 1024.);

	mmx8  = _mm_set1_ps(b0);
	mmx9  = _mm_set1_ps(b[0]);
	mmx10 = _mm_set1_ps(b[1]);	
	mmx11 = _mm_set1_ps(b[2]);
	mmx12 = _mm_set1_ps(b[3]);

	for(int i_test = 0; i_test < nTest; i_test++, nx *= 2, ny *= 2, nz *= 2) {
		float * po = ArrayAlloc<float>(nxyz);
		float * pm = ArrayAlloc<float>(nxyz);
		float * po1 = ArrayAlloc<float>(nxyz);
		float * pm1 = ArrayAlloc<float>(nxyz);	

		double t;
		clock_t start, finish;	
		float err;	

		cout << "\nTest: " << i_test << "\n";
		cout << "Grid size: ( " << nx << ", " << ny  << ", " << nz << " )\n";
		cout << "Space step: ( " << dx << ", " << dy  << ", " << dz << " )\n";
		cout << "Time samples: " << nt << "\n";
		cout << "Time step: " << dt << "\n";
		cout << "Peak frequency: " << fpeak << "\n";
		
		t = omp_get_wtime();
		start = clock();
		fdtd_naive(nx, ny, nz,
			   dx, dy, dz,
			   po, pm, 
			   nt, dt, wav);		
		finish = clock();
		t = omp_get_wtime() - t;
		cout << "\ncpu clock cycles of naive stencil computation implementation: " << finish - start << "\n";
		cout << "cpu wall time of naive stencil computation implementation: " << t << "\n"; 		
		cout << "performance: " << ops / t << "Gflop/s\n"; 

		memset(po1, 0, sizeof(float) * nxyz);
		memset(pm1, 0, sizeof(float) * nxyz);
		t = omp_get_wtime();
		start = clock();
		fdtd_openmp_naive(nx, ny, nz,
			   	  dx, dy, dz,
			   	  po1, pm1, 
			   	  nt, dt, wav);	
		finish = clock();
		t = omp_get_wtime() - t;
		cout << "\ncpu clock cycles of naive openmp stencil computation implementation: " << finish - start << "\n";
		cout << "cpu wall time of naive openmp stencil computation: " << t << "\n";
		cout << "performance: " << ops / t << "Gflop/s\n";
	
		err = check_identical<float>(ny, nxz, po1, po);
		cout << "error: " << err << "\n";

		memset(po1, 0, sizeof(float) * nxyz);
		memset(pm1, 0, sizeof(float) * nxyz);
		t = omp_get_wtime();
		start = clock();
		fdtd_openmp_vectorize(nx, ny, nz,
			   	      dx, dy, dz,
			   	      po1, pm1, 
			   	      nt, dt, wav);	
		finish = clock();
		t = omp_get_wtime() - t;
		cout << "\ncpu clock cycles of openmp + vectorize stencil computation implementation: " << finish - start << "\n";
		cout << "cpu wall time of openmp + vectorize stencil computation: " << t << "\n";
		cout << "performance: " << ops / t << "Gflop/s\n";
	
		err = check_identical<float>(ny, nxz, po1, po);
		cout << "error: " << err << "\n";
	
		memset(po1, 0, sizeof(float) * nxyz);
		memset(pm1, 0, sizeof(float) * nxyz);
		t = omp_get_wtime();
		start = clock();
		fdtd_pthread(nx, ny, nz,
			     dx, dy, dz,
			     po1, pm1,
			     nt, dt, wav);	
		finish = clock();
		t = omp_get_wtime() - t;
		cout << "\ncpu clock cycles of pthread stencil computation: " << finish - start << "\n";
		cout << "cpu wall time of pthread stencil computation: " << t << "\n";
		cout << "performance: " << ops / t << "Gflop/s\n";
		
		err = check_identical<float>(ny, nxz, po1, po);
		cout << "error: " << err << "\n";

//		fp = fopen("sc_pthread.dat", "wb");
//		fwrite(po1, sizeof(float), nxyz, fp);
//		fclose(fp);

		memset(po1, 0, sizeof(float) * nxyz);
		memset(pm1, 0, sizeof(float) * nxyz);
		float gtime = 0.0;
		t = omp_get_wtime();
		start = clock();
		fd3d_gpu(nx, ny, nz,
			 dx, dy, dz,
			 po1, pm1,
			 nt, dt, wav, gtime);	
		finish = clock();
		t = omp_get_wtime() - t;
		cout << "\ncpu clock cycles of gpu stencil computation: " << finish - start << "\n";
		cout << "cpu wall time of gpu stencil computation: " << t << "\n";
		cout << "performance: " << gtime << "Gflop/s\n";
		
		err = check_identical<float>(ny, nxz, po1, po);
		cout << "error: " << err << "\n";
//
//		fp = fopen("sc_gpu.dat", "wb");
//		fwrite(po1, sizeof(float), nxyz, fp);
//		fclose(fp);

		free(po);
		free(pm);
		free(po1);
		free(pm1);		
	}
}
