#include "renderer.h"


void CudaRenderer::initRenderer() {
	checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024));
	checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
	int numPixels = imgWidth * imgHeight;
	checkCudaErrors(cudaMalloc((void**)&d_Fb, sizeof(Framebuffer)));
	checkCudaErrors(cudaMemcpy(d_Fb, &h_Fb, sizeof(Framebuffer), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
	checkCudaErrors(cudaMalloc((void**)&d_List, 5 * sizeof(Hittable*)));
	checkCudaErrors(cudaMalloc((void**)&d_World, sizeof(HittableList)));
	checkCudaErrors(cudaMalloc((void**)&d_BVHroot, sizeof(BVHNode)));
	checkCudaErrors(cudaMalloc((void**)&d_randStates, numPixels * sizeof(curandState)));
	dim3 blocks(imgWidth / xBlock + 1, imgHeight / yBlock + 1);
	dim3 threads(xBlock, yBlock);
	utils::random::randomInit<<<blocks, threads>>>(d_randStates, imgWidth, imgHeight);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}

CudaRenderer::~CudaRenderer() {
	if (conicalCullingEnabled) {
		cleanupConicalCulling();
	}
	checkCudaErrors(cudaFree(d_List));
	d_List = nullptr;
	checkCudaErrors(cudaFree(d_World));
	d_World = nullptr;
	checkCudaErrors(cudaFree(d_Fb));
	d_Fb = nullptr;
	checkCudaErrors(cudaFree(d_camera));
	d_camera = nullptr;
	checkCudaErrors(cudaFree(d_randStates));
	d_randStates = nullptr;
	if (glResource) cudaGraphicsUnregisterResource(glResource);
}

void CudaRenderer::registerGLTexture(GLuint glTex) {
	cudaGraphicsGLRegisterImage(&glResource, glTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

void CudaRenderer::setupScene(Camera& camera) {
	struct MeshConfig {
		std::string name; 
		std::string filepath;
		glm::vec3 position;
		glm::vec3 scale;
		glm::vec3 color;
		bool isMetal;
	};

	std::vector<MeshConfig> meshesToLoad = {
		{ "Floor", ASSETS_PATH "cube.obj", glm::vec3(0, -1.0f, 0), glm::vec3(50.0f, 0.1f, 50.0f), glm::vec3(0.5f, 0.5f, 0.5f), false },
		{ "Box",   ASSETS_PATH "cube.obj", glm::vec3(0, 0.5f, -2.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.8f, 0.2f, 0.2f), false }
	};

	for (auto& m : meshResources) m.cleanup();
	meshResources.clear();

	int staticSphereCount = 4;
	int totalObjects = staticSphereCount + meshesToLoad.size();

	if (d_List) cudaFree(d_List);
	checkCudaErrors(cudaMalloc((void**)&d_List, totalObjects * sizeof(Hittable*)));

	createStaticObjects<<<1, staticSphereCount>>>(d_List, staticSphereCount);
	checkCudaErrors(cudaGetLastError());
	int currentListIndex = staticSphereCount;

	for (const auto& config : meshesToLoad) {
		std::cout << "Processing: " << config.name << "..." << std::endl;
		RawMeshData rawMesh;
		if (!rawMesh.loadObj(config.filepath)) {
			std::cerr << "Failed to load " << config.filepath << std::endl;
			continue;
		}

		for (auto& v : rawMesh.vertices) {
			v = v * config.scale;
			v = v + config.position;
		}

		BVHBuilder builder;
		builder.build(rawMesh.vertices, rawMesh.indices);

		std::vector<int> gpuIndices;
		gpuIndices.reserve(builder.orderedIndices.size() * 3);

		for (int triId : builder.orderedIndices) {
			gpuIndices.push_back(rawMesh.indices[triId + 0]);
			gpuIndices.push_back(rawMesh.indices[triId + 1]);
			gpuIndices.push_back(rawMesh.indices[triId + 2]);
		}

		for (auto& node : builder.nodes) {
			if (node.triCount > 0) {
				node.triIndex *= 3;
			}
		}

		GPUMeshData gpuData;

		size_t nodeSize = builder.nodes.size() * sizeof(LinearBVHNode);
		size_t idxSize = gpuIndices.size() * sizeof(int);
		size_t vertSize = rawMesh.vertices.size() * sizeof(glm::vec3);

		checkCudaErrors(cudaMalloc((void**)&gpuData.d_nodes, nodeSize));
		checkCudaErrors(cudaMalloc((void**)&gpuData.d_indices, idxSize));
		checkCudaErrors(cudaMalloc((void**)&gpuData.d_vertices, vertSize));

		checkCudaErrors(cudaMemcpy(gpuData.d_nodes, builder.nodes.data(), nodeSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(gpuData.d_indices, gpuIndices.data(), idxSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(gpuData.d_vertices, rawMesh.vertices.data(), vertSize, cudaMemcpyHostToDevice));

		Material** d_tempMat;
		checkCudaErrors(cudaMalloc((void**)&d_tempMat, sizeof(Material*)));
		createMaterial<<<1, 1>>>(d_tempMat, config.color, config.isMetal);
		checkCudaErrors(cudaMemcpy(&gpuData.d_material, d_tempMat, sizeof(Material*), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_tempMat));

		createMeshObject<<<1, 1>>>(
			d_List,
			currentListIndex,
			gpuData.d_nodes,
			gpuData.d_vertices,
			gpuData.d_indices,
			gpuData.d_material
			);

		meshResources.push_back(gpuData);
		currentListIndex++;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	buildSceneBVH<<<1, 1>>>(d_List, d_World, d_BVHroot, totalObjects);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void CudaRenderer::updateCamera(Camera& camera) const {
	checkCudaErrors(cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
	initCamera<<<1, 1>>>(d_camera, imgWidth, imgHeight);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

int CudaRenderer::render(Camera& camera) { // TODO: profile this
	// Update dynamic object bounds if conical culling is enabled
	if (conicalCullingEnabled) {
		updateDynamicBounds();
	}

	cudaSurfaceObject_t surfObj = 0;
	if (glResource) {
		cudaArray_t cuArray;
		checkCudaErrors(cudaGraphicsMapResources(1, &glResource));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuArray, glResource, 0, 0));
		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;
		checkCudaErrors(cudaCreateSurfaceObject(&surfObj, &resDesc));
	}

	dim3 blocks(imgWidth / xBlock + 1, imgHeight / yBlock + 1);
	dim3 threads(xBlock, yBlock);

	// Choose rendering kernel based on conical culling setting
	if (conicalCullingEnabled && d_dynamicInfo != nullptr) {
		renderSceneWithConicalCulling<<<blocks, threads>>>(
			d_Fb, d_camera, d_World, d_dynamicInfo,
			10,  // samplesPerPixel for each pixel
			samplesPerPixel,  // maxDepth (reusing samplesPerPixel for depth)
			d_randStates, surfObj
		);
	} else {
		renderScene<<<blocks, threads>>>(d_Fb, d_camera, d_World, samplesPerPixel, d_randStates, surfObj);
	}
	checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaDeviceSynchronize());

	if (glResource) {
		checkCudaErrors(cudaDestroySurfaceObject(surfObj));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &glResource));
	}

	return 0;
}

void CudaRenderer::destroyScene() const {
	destroyWorld<<<1, 1>>>(d_List, d_World, d_BVHroot, 6);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

// ============================================================================
// Conical Ray Culling Implementation
// ============================================================================

void CudaRenderer::initConicalCulling() {
	if (d_dynamicInfo != nullptr) return; // Already initialized

	// Allocate device memory for dynamic object info
	checkCudaErrors(cudaMalloc(&d_dynamicInfo, sizeof(DynamicObjectInfo)));

	// Initialize on device
	initDynamicObjects<<<1, 1>>>(d_dynamicInfo, maxDynamicObjects);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void CudaRenderer::cleanupConicalCulling() {
	if (d_dynamicInfo == nullptr) return;

	cleanupDynamicObjects<<<1, 1>>>(d_dynamicInfo);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(d_dynamicInfo));
	d_dynamicInfo = nullptr;
}

void CudaRenderer::enableConicalCulling(bool enable) {
	if (enable && !conicalCullingEnabled) {
		// Initialize conical culling
		initConicalCulling();
		conicalCullingEnabled = true;
		std::cout << "Conical ray culling enabled" << std::endl;
	} else if (!enable && conicalCullingEnabled) {
		// Cleanup conical culling
		cleanupConicalCulling();
		conicalCullingEnabled = false;
		std::cout << "Conical ray culling disabled" << std::endl;
	}
}

void CudaRenderer::markObjectsAsDynamic(const std::vector<int>& dynamicIndices) {
	if (!conicalCullingEnabled) {
		std::cerr << "Warning: Conical culling not enabled. Call enableConicalCulling() first." << std::endl;
		return;
	}

	if (dynamicIndices.empty()) {
		std::cout << "No dynamic objects marked (all objects static)" << std::endl;
		return;
	}

	// Copy indices to device
	int* d_indices;
	size_t size = dynamicIndices.size() * sizeof(int);
	checkCudaErrors(cudaMalloc(&d_indices, size));
	checkCudaErrors(cudaMemcpy(d_indices, dynamicIndices.data(), size, cudaMemcpyHostToDevice));

	// Mark objects as dynamic
	markDynamicObjects<<<1, 1>>>(d_dynamicInfo, d_indices, dynamicIndices.size());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(d_indices));

	std::cout << "Marked " << dynamicIndices.size() << " objects as dynamic" << std::endl;
}

void CudaRenderer::updateDynamicBounds() {
	if (!conicalCullingEnabled || d_dynamicInfo == nullptr) return;
	DynamicObjectInfo h_info;
	checkCudaErrors(cudaMemcpy(&h_info, d_dynamicInfo, sizeof(DynamicObjectInfo), cudaMemcpyDeviceToHost));

	if (h_info.numDynamic == 0) return;
	int blockSize = 256;
	int numBlocks = (h_info.numDynamic + blockSize - 1) / blockSize;
	updateDynamicObjectBounds<<<numBlocks, blockSize>>>(
		d_dynamicInfo,
		d_List,
		h_info.dynamicIndices,
		h_info.numDynamic
	);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}