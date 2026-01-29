#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "ray/ray.h"

class AABB {
public:
	glm::vec3 minPoint{};
	glm::vec3 maxPoint{};

	__device__ AABB() = default;
	__device__ AABB(const glm::vec3& minPoint, const glm::vec3& maxPoint)
		: minPoint(minPoint), maxPoint(maxPoint) {}

    __device__ inline bool hit(const Ray& ray, float tMin, float tMax) const {
        for (int axis = 0; axis < 3; axis++) {
            float invD = 1.0f / ray.getDirection()[axis];
    
            float tNear = (minPoint[axis] - ray.getOrigin()[axis]) * invD;
            float tFar = (maxPoint[axis] - ray.getOrigin()[axis]) * invD;
            tMin = fmaxf(tMin, fminf(tNear, tFar));
            tMax = fminf(tMax, fmaxf(tNear, tFar));
            if (tMax <= tMin) {
                return false;
            }
        }
        return true;
    }

    __device__ inline int longestAxis() const {
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

__device__ inline AABB joinAABBs(const AABB& box0, const AABB& box1) {
	glm::vec3 min(fminf(box0.minPoint.x, box1.minPoint.x),
		fminf(box0.minPoint.y, box1.minPoint.y),
		fminf(box0.minPoint.z, box1.minPoint.z));
	glm::vec3 max(fmaxf(box0.maxPoint.x, box1.maxPoint.x),
		fmaxf(box0.maxPoint.y, box1.maxPoint.y),
		fmaxf(box0.maxPoint.z, box1.maxPoint.z));
	return AABB(min, max);
}