#include <glm/glm.hpp>

#include "ray.h"

__device__ glm::vec3 Ray::getOrigin() const {
	return origin;
}

__device__ glm::vec3 Ray::getDirection() const {
	return direction;
}

__device__ void Ray::setDirection(const glm::vec3& newDir) {
	direction = newDir;
}

__device__ void Ray::setOrigin(const glm::vec3& newOrigin) {
	origin = newOrigin;
}

__device__ glm::vec3 Ray::At(const float t) const {
	return origin + t * direction;
}