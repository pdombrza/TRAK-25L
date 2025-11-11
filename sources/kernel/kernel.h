#pragma once
#include <iostream>

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "ray/ray.h"
#include "utils/utils.h"
#include "hittable/hittable.h"
#include "hittablelist/hittablelist.h"


__device__ glm::vec3 color(const Ray& ray, HittableList* world);
__global__ void renderScene(cudaSurfaceObject_t fb, int x, int y, glm::vec3 bottomLeftCorner, glm::vec3 horizontal, glm::vec3 vertical, glm::vec3 origin, HittableList* world);
void launchRenderer(cudaGraphicsResource* glResource, int nx, int ny, int xBlock, int yBlock);
__global__ void createWorld(Hittable** d_List, HittableList* d_World);
__global__ void destroyWorld(Hittable** d_List, HittableList* d_World);