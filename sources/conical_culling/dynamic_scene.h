#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "conical_culling.h"
#include "aabb/aabb.h"
#include "hittable/hittable.h"


struct DynamicObjectInfo {
    BoundingSphere* boundingSpheres;
    int* dynamicIndices;
    int numDynamic;
    int maxDynamic;

    __host__ DynamicObjectInfo() : boundingSpheres(nullptr), dynamicIndices(nullptr),
                                   numDynamic(0), maxDynamic(0) {}
};

__device__ inline BoundingSphere computeBoundingSphere(const AABB& bbox) {
    glm::vec3 center = (bbox.minPoint + bbox.maxPoint) * 0.5f;
    float radius = glm::length(bbox.maxPoint - center);
    return BoundingSphere(center, radius);
}

__global__ inline void initDynamicObjects(DynamicObjectInfo* d_info, int maxDynamic) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cudaMalloc(&d_info->boundingSpheres, maxDynamic * sizeof(BoundingSphere));
        cudaMalloc(&d_info->dynamicIndices, maxDynamic * sizeof(int));
        d_info->maxDynamic = maxDynamic;
        d_info->numDynamic = 0;
    }
}

__global__ inline void markDynamicObjects(DynamicObjectInfo* d_info, int* dynamicIndices, int numDynamic) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_info->numDynamic = numDynamic;
        for (int i = 0; i < numDynamic; i++) {
            d_info->dynamicIndices[i] = dynamicIndices[i];
        }
    }
}

__global__ inline void updateDynamicObjectBounds(DynamicObjectInfo* d_info,
                                          Hittable** objects,
                                          int* dynamicIndices,
                                          int numDynamic) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numDynamic) return;

    int objIdx = dynamicIndices[idx];
    AABB bbox = objects[objIdx]->boundingBox();
    d_info->boundingSpheres[idx] = computeBoundingSphere(bbox);
}

__global__ inline void cleanupDynamicObjects(DynamicObjectInfo* d_info) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (d_info->boundingSpheres) cudaFree(d_info->boundingSpheres);
        if (d_info->dynamicIndices) cudaFree(d_info->dynamicIndices);
    }
}
