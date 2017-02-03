#ifndef ISOFD3D_GPU_KERNEL_H
#define ISOFD3D_GPU_KERNEL_H
extern "C" void fd3d_gpu(const int nx, const int ny, const int nz,
			 const float dx, const float dy, const float dz,
			 float * uo, float * um,
			 const int nt, const float dt, const float * wav, float & gpuTime);
#endif
