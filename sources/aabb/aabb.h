#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "ray/ray.h"

class AABB {
public:
	glm::vec3 minPoint{};
	glm::vec3 maxPoint{};

	__host__ __device__ AABB() = default;
	__host__ __device__ AABB(const glm::vec3& minPoint, const glm::vec3& maxPoint)
		: minPoint(minPoint), maxPoint(maxPoint) {}

    __host__ __device__ inline bool hit(const Ray& ray, float tMin, float tMax) const {
        for (int axis = 0; axis < 3; axis++) {
            float dir = ray.getDirection()[axis];
            float invD = 1.0f / dir;

            float t0 = (minPoint[axis] - ray.getOrigin()[axis]) * invD;
            float t1 = (maxPoint[axis] - ray.getOrigin()[axis]) * invD;

            if (invD < 0.0f) {
                float temp = t0; t0 = t1; t1 = temp;
            }

            tMin = t0 > tMin ? t0 : tMin;
            tMax = t1 < tMax ? t1 : tMax;

            if (tMax <= tMin) {
                return false;
            }
        }
        return true;
    }

    __host__ __device__ inline int longestAxis() const {
        glm::vec3 extents = maxPoint - minPoint;
        if (extents.x > extents.y && extents.x > extents.z) {
            return 0;
        } else if (extents.y > extents.z) {
            return 1;
        } else {
            return 2;
		}
    }

};

__host__ __device__ inline AABB joinAABBs(const AABB& box0, const AABB& box1) {
	glm::vec3 min(fminf(box0.minPoint.x, box1.minPoint.x),
		fminf(box0.minPoint.y, box1.minPoint.y),
		fminf(box0.minPoint.z, box1.minPoint.z));
	glm::vec3 max(fmaxf(box0.maxPoint.x, box1.maxPoint.x),
		fmaxf(box0.maxPoint.y, box1.maxPoint.y),
		fmaxf(box0.maxPoint.z, box1.maxPoint.z));
	return AABB(min, max);
}