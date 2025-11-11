#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <random>
#include <curand_kernel.h>
#include <utility>
#include <optional>

#include <cuda/std/limits>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>

#define INF cuda::std::numeric_limits<float>::max()
#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__ )


inline void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

namespace utils {
	namespace random {
		class RNG {
		private:
			curandState* state;
		public:
			__device__ explicit RNG(curandState* state) : state(state) {};
			__device__ ~RNG() = default;
			template<typename T>
			__device__ T getRandom(T min, T max) {
				float generatedRandom = curand_uniform(state);
				return static_cast<float>(min) + (static_cast<float>(max) - static_cast<float>(min)) * generatedRandom;
			};
			__device__ float getRandomUniform();
			__device__ glm::vec3 sampleSquare();
			template<typename T>
			__device__ glm::vec<3, T, glm::defaultp> randomVec3(T min, T max) {
				return glm::vec<3, T, glm::defaultp>(getRandom(min, max), getRandom(min, max), getRandom(min, max));
			};
			__device__ glm::vec3 randomVec3Norm();
			__device__ glm::vec3 randomVec3InSphere();
			__device__ glm::vec3 randomVec3OnHemisphere(const glm::vec3& normal);
			__device__ glm::vec3 randomVec3InDisk();
		};
		
		__global__ void randomInit(curandState* randStates, int x, int y);
	}
}
