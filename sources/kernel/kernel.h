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
#include "hittable/triangle.h"
#include "bvh/mesh_bvh.h"
#include "bvh/bvh_builder.h"


__global__ void renderScene(Framebuffer* d_Fb, Camera* camera, HittableList* world, int samples, curandState* randState, cudaSurfaceObject_t surfObj = 0);
__global__ void renderSceneWithConicalCulling(Framebuffer* d_Fb, Camera* camera, HittableList* world, 
	DynamicObjectInfo* dynamicInfo, int samplesPerPixel, int maxDepth, curandState* randState, cudaSurfaceObject_t surfObj = 0);
__global__ void initCamera(Camera* cam, int width, int height);
__global__ void createWorld(Hittable** d_List, HittableList* d_World, BVHNode* d_BVHroot, curandState* randState);
__global__ void destroyWorld(Hittable** d_List, HittableList* d_World, BVHNode* d_BVHroot, int size);
__global__ void createMeshTriangles(Hittable** d_List, int startOffset, glm::vec3* d_vertices, int* d_indices, int numTriangles, Material* mat);
__global__ inline void createMeshObject(Hittable** d_List, int index, LinearBVHNode* nodes, glm::vec3* verts, int* indices, Material* mat) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_List[index] = new MeshBVH(nodes, verts, indices, mat);
    }
};

__global__ inline void createMaterial(Material** d_matPtr, glm::vec3 color, bool isMetal) {
    if (threadIdx.x == 0) {
        if (isMetal) *d_matPtr = new Metal(color, 0.1f);
        else *d_matPtr = new Lambertian(color);
    }
}

__global__ inline void createStaticObjects(Hittable** d_List, int maxObjects) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= maxObjects) return;

    if (idx == 0) d_List[0] = new Sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f, new Lambertian(glm::vec3(0.1f, 0.2f, 0.5f)));
    if (idx == 1) d_List[1] = new Sphere(glm::vec3(1.0f, 0.0f, -1.0f), 0.5, new Metal(glm::vec3(0.8f, 0.6f, 0.2f), 0.5f));
    if (idx == 2) d_List[2] = new Sphere(glm::vec3(-1.0f, 0.0f, -1.0f), 0.5, new Dielectric(1.5f));
    if (idx == 3) d_List[3] = new Sphere(glm::vec3(-1.0f, 0.0f, -1.0f), 0.45, new Dielectric(1.0f / 1.5f));
};

__global__ inline void createMaterialWithColor(Material** d_matPtr, glm::vec3 color) {
    if (threadIdx.x == 0) {
        *d_matPtr = new Lambertian(color);
    }
}

__global__ inline void freeSharedMaterial(Material* mat) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete mat;
    }
}
__global__ inline void buildSceneBVH(Hittable** d_List, HittableList* d_World, BVHNode* d_BVHroot, int totalObjects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new(d_BVHroot) BVHNode(d_List, 0, totalObjects);
        d_List[0] = d_BVHroot;
        new(d_World) HittableList(d_List, 1, 1);
    }
}