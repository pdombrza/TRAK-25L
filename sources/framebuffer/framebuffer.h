#pragma once
#include <memory>

#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "utils/utils.h"
#include "ray/ray.h"
#include "hittablelist/hittablelist.h"
#include "camera/camera.h"

class Framebuffer {
private:
	int width;
	int height;
	float4* pixels = 0;

public:
	__host__ Framebuffer(int width, int height) : width(width), height(height) { initialize(); };
	__host__ ~Framebuffer() { cleanup(); };
	
	__host__ Framebuffer(const Framebuffer&) = delete;
	__host__ Framebuffer& operator=(const Framebuffer&) = delete;

	__host__ void initialize();
	__host__ void cleanup();
	__device__ void writePixel(int x, int y, const glm::vec3& color);
	__device__ glm::vec3 color(const Ray& ray, HittableList* world, utils::random::RNG& rng);
	__device__ glm::vec3 colorPixel(int i, int j, int nx, int ny, Camera* camera, HittableList* world, utils::random::RNG& rng);
	__host__ __device__ float4* getPixels() const { return pixels; };
	__host__ __device__ int getWidth() const { return width; };
	__host__ __device__ int getHeight() const { return height; };
	__host__ __device__ void setPixels(float4* newPixels) { pixels = newPixels; };
	__host__ std::shared_ptr<glm::vec3[]> getHostPixels() const;
};
