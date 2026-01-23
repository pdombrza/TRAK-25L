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

	renderScene<<<blocks, threads>>>(d_Fb, d_camera, d_World, samplesPerPixel, d_randStates, surfObj);
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