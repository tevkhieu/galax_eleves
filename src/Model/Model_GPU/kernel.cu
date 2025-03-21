#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	float3 position_i = positionsGPU[i];
	float3 acceleration_i = {0.0f, 0.0f, 0.0f};
	float3 diff_pos;
	float rdij;

	for (int j = 0; j < n_particles; ++j) {
		diff_pos.x = positionsGPU[j].x - position_i.x;
		diff_pos.y = positionsGPU[j].y - position_i.y;
		diff_pos.z = positionsGPU[j].z - position_i.z;

		rdij = (diff_pos.x * diff_pos.x + diff_pos.y * diff_pos.y + diff_pos.z * diff_pos.z);
		rdij = rsqrtf((rdij * rdij * rdij));
        rdij = fminf(10.0 * rdij, 10.0);
		rdij = rdij * massesGPU[j];

		acceleration_i.x += diff_pos.x * rdij;
		acceleration_i.y += diff_pos.y * rdij;
		acceleration_i.z += diff_pos.z * rdij;
	}

	accelerationsGPU[i] = acceleration_i;

}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	float3 velocities = velocitiesGPU[i];
	float3 particles = positionsGPU[i];
	float3 accelerations = accelerationsGPU[i];
	velocities.x += accelerations.x * 2.0f;
	velocities.y += accelerations.y * 2.0f;
	velocities.z += accelerations.z * 2.0f;
	particles.x += velocities.x * 0.1f;
	particles.y += velocities.y * 0.1f;
	particles.z += velocities.z * 0.1f;

	positionsGPU[i] = particles;
	velocitiesGPU[i] = velocities;

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 32;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU);
}


#endif // GALAX_MODEL_GPU