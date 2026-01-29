#pragma once

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <optional>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <execution>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "ray/ray.h"
#include "hitrec/hitrec.h"
#include "hittable/hittable.h"
#include "hittablelist/hittablelist.h"
#include "material/material.h"
#include "camera/camera.h"
#include "kernel/kernel.h"
#include "mesh/mesh.h"
#include "bvh/mesh_bvh.h"
#include "bvh/bvh_builder.h"


struct GPUMeshData {
	LinearBVHNode* d_nodes = nullptr;
	glm::vec3* d_vertices = nullptr;
	int* d_indices = nullptr;
	Material* d_material = nullptr;

	void cleanup() {
		if (d_nodes) cudaFree(d_nodes);
		if (d_vertices) cudaFree(d_vertices);
		if (d_indices) cudaFree(d_indices);
		if (d_material) {
			// Need a kernel to delete material, or track it elsewhere
		}
	}
};

class IRenderer {
public:
	virtual ~IRenderer() = default;
	virtual int render(Camera& camera) = 0;
	virtual void setScene(HittableList* newScene) = 0;
	virtual HittableList* getScene() const = 0;
};

class CudaRenderer : public IRenderer {
private:
	Framebuffer h_Fb;
	Framebuffer* d_Fb = nullptr;
	Camera* d_camera = nullptr;
	Hittable** d_List = nullptr;
	HittableList* d_World = nullptr;
	curandState* d_randStates = nullptr;
	Hittable** d_List_storage = nullptr;
	HittableList* d_World_storage = nullptr;
	BVHNode* d_BVHroot = nullptr;
	cudaGraphicsResource* glResource = nullptr;
	std::vector<GPUMeshData> meshResources;
	glm::vec3* d_meshVertices = nullptr;
	int* d_meshIndices = nullptr;
	int numMeshTriangles = 0;
	DynamicObjectInfo* d_dynamicInfo = nullptr;
	bool conicalCullingEnabled = false;
	int maxDynamicObjects = 10;
protected:
	HittableList* scene;
	int imgWidth = 400;
	int imgHeight = 225;
	int samplesPerPixel = 32;
	int maxDepth = 10;
	int xBlock = 16;
	int yBlock = 16;
	float pixelSamplesScale{};
	virtual void initRenderer();
public:
	CudaRenderer(HittableList* scene) : scene(scene), h_Fb(imgWidth, imgHeight) { initRenderer(); };
	CudaRenderer(HittableList* scene, int imgWidth, int imgHeight, int samplesPerPixel, int maxDepth) : scene(scene), imgWidth(imgWidth), imgHeight(imgHeight), samplesPerPixel(samplesPerPixel), maxDepth(maxDepth), h_Fb(imgWidth, imgHeight) { initRenderer(); };
	CudaRenderer(HittableList* scene, int imgWidth, int imgHeight) : scene(scene), imgWidth(imgWidth), imgHeight(imgHeight), h_Fb(imgWidth, imgHeight) { initRenderer(); };
	~CudaRenderer();
	virtual void setXBlock(int newXBlock) { xBlock = newXBlock; };
	virtual int getXBlock() const { return xBlock; };
	virtual void setYBlock(int newYBlock) { yBlock = newYBlock; };
	virtual int getYBlock() const { return yBlock; };
	virtual void setScene(HittableList* newScene) override { scene = newScene; };
	virtual HittableList* getScene() const override { return scene; };
	virtual void setImgWidth(int newImgWidth) { imgWidth = newImgWidth; };
	virtual int getImgWidth() const { return imgWidth; };
	virtual void setImgHeight(int newImgHeight) { imgHeight = newImgHeight; };
	virtual int getImgHeight() const { return imgHeight; };
	void registerGLTexture(GLuint glTex);
	virtual void setupScene(Camera& camera);
	virtual void updateCamera(Camera& camera) const;
	virtual void destroyScene() const;
	virtual int render(Camera& camera) override;
	virtual void setSamplesPerPixel(int newSamplesPerPixel) { samplesPerPixel = newSamplesPerPixel; };
	std::shared_ptr<glm::vec3[]> getHostPixels() const { return h_Fb.getHostPixels(); };

	// Conical ray culling methods
	void enableConicalCulling(bool enable = true);
	void setMaxDynamicObjects(int maxObjects) { maxDynamicObjects = maxObjects; }
	void markObjectsAsDynamic(const std::vector<int>& dynamicIndices);
	bool isConicalCullingEnabled() const { return conicalCullingEnabled; }
private:
	void initConicalCulling();
	void cleanupConicalCulling();
	void updateDynamicBounds();
};