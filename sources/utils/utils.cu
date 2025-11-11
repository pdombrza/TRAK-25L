#include "utils.h"

__global__ void utils::random::randomInit(curandState* randStates, int x, int y) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= x) || (j >= y)) return;
	int pixelIdx = j * x + i;
	curand_init(1984 + pixelIdx, 0, 0, &randStates[pixelIdx]);
}

__device__ float utils::random::RNG::getRandomUniform() {
	return curand_uniform(state);
}

__device__ glm::vec3 utils::random::RNG::sampleSquare() {
	return glm::vec3(curand_uniform(state) - 0.5f, curand_uniform(state) - 0.5f, 0.0f);
}

__device__ glm::vec3 utils::random::RNG::randomVec3Norm() {
	glm::vec3 res = randomVec3(0.0f, 1.0f);
	return res;
}

__device__ glm::vec3 utils::random::RNG::randomVec3InSphere() {
	while (true) {
		glm::vec3 randomVec = (randomVec3Norm() - 0.5f) * 2.0f;
		if (glm::dot(randomVec, randomVec) < 1.0f)
			return glm::normalize(randomVec);
	}
}

__device__ glm::vec3 utils::random::RNG::randomVec3OnHemisphere(const glm::vec3& normal) {
	glm::vec3 onUnitSphere = randomVec3InSphere();
	onUnitSphere = (glm::dot(onUnitSphere, normal) > 0.0f) ? onUnitSphere : -onUnitSphere;
	return onUnitSphere;
}

__device__ glm::vec3 utils::random::RNG::randomVec3InDisk() {
	while (true) {
		glm::vec3 randomVec{ getRandom(-1.0f, 1.0f), getRandom(-1.0f, 1.0f), 0.0f };
		if (glm::dot(randomVec, randomVec) < 1.0f)
			return randomVec;
	}
}