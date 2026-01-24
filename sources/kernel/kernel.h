#pragma once
#include <iostream>

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "ray/ray.h"
#include "hitrec/hitrec.h"
#include "utils/utils.h"
#include "camera/camera.h"
#include "material/material.h"
#include "framebuffer/framebuffer.h"
#include "bvh/bvh.h"
#include "conical_culling/dynamic_scene.h"

// Standard rendering kernel
__global__ void renderScene(Framebuffer* d_Fb, Camera* camera, HittableList* world, int samples, curandState* randState, cudaSurfaceObject_t surfObj = 0);

// Conical ray culling rendering kernel
__global__ void renderSceneWithConicalCulling(
    Framebuffer* d_Fb,
    Camera* camera,
    HittableList* world,
    DynamicObjectInfo* dynamicInfo,
    int samplesPerPixel,
    int maxDepth,
    curandState* randState,
    cudaSurfaceObject_t surfObj = 0
);

// Helper kernels
__global__ void initCamera(Camera* cam, int width, int height);
__global__ void createWorld(Hittable** d_List, HittableList* d_World, BVHNode* d_BVHroot, curandState* randState);
__global__ void destroyWorld(Hittable** d_List, HittableList* d_World, BVHNode* d_BVHroot, int size);