#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "utils/utils.h"



class Ray {
private:
	glm::vec3 origin{ 0.0f, 0.0f, 0.0f };
	glm::vec3 direction{ 0.0f, 0.0f, 0.0f };
public:
	__device__ constexpr Ray() = default;
	__device__ Ray(const glm::vec3& origin, const glm::vec3& direction) : origin(origin), direction(glm::normalize(direction)) {};
	__device__ glm::vec3 getOrigin() const;
	__device__ glm::vec3 getDirection() const;
	__device__ void setDirection(const glm::vec3& newDir);
	__device__ void setOrigin(const glm::vec3& newOrigin);
	__device__ glm::vec3 At(const float t) const;
};	