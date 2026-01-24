#include "renderer.h"


void CudaRenderer::initRenderer() {
	int numPixels = imgWidth * imgHeight;
	checkCudaErrors(cudaMalloc((void**)&d_Fb, sizeof(Framebuffer)));
	checkCudaErrors(cudaMemcpy(d_Fb, &h_Fb, sizeof(Framebuffer), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera)));
	checkCudaErrors(cudaMalloc((void**)&d_List, 6 * sizeof(Hittable*)));
	checkCudaErrors(cudaMalloc((void**)&d_World, sizeof(HittableList)));
	checkCudaErrors(cudaMalloc((void**)&d_BVHroot, sizeof(BVHNode)));
	checkCudaErrors(cudaMalloc((void**)&d_randStates, numPixels * sizeof(curandState)));
}

CudaRenderer::~CudaRenderer() {
	// Cleanup conical culling if enabled
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
	if (glResource) cudaGraphicsUnregisterResource(glResource); // TODO: consider decoupling gl from renderer - move to separate class
}

void CudaRenderer::registerGLTexture(GLuint glTex) {
	cudaGraphicsGLRegisterImage(&glResource, glTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

void CudaRenderer::setupScene(Camera& camera) const { // TODO: use this in constuctor
	int numPixels = imgWidth * imgHeight;

	checkCudaErrors(cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));
	initCamera<<<1, 1>>>(d_camera, imgWidth, imgHeight);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	dim3 blocks(imgWidth / xBlock + 1, imgHeight / yBlock + 1);
	dim3 threads(xBlock, yBlock);
	utils::random::randomInit<<<blocks, threads>>>(d_randStates, imgWidth, imgHeight);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	createWorld<<<1, 1>>>(d_List, d_World, d_BVHroot, d_randStates);
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

	// Get dynamic info from device
	DynamicObjectInfo h_info;
	checkCudaErrors(cudaMemcpy(&h_info, d_dynamicInfo, sizeof(DynamicObjectInfo), cudaMemcpyDeviceToHost));

	if (h_info.numDynamic == 0) return;

	// Update bounding spheres
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