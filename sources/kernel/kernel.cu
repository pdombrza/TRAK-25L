#include "kernel.h"

__device__ glm::vec3 color(const Ray& ray, HittableList* world) {
	cuda::std::optional<HitRecord> hitRec = world->hit(ray, 0.001f, INF);
	if (hitRec.has_value()) {
		HitRecord hitrec = hitRec.value();
		return 0.5f * (hitrec.normal + glm::vec3(1.0f, 1.0f, 1.0f));
	}

	// gradient
	glm::vec3 direction = ray.getDirection();
	float a = 0.5f * (direction.y + 1.0f);
	return (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
}

__global__ void renderScene(cudaSurfaceObject_t fb, int x, int y, glm::vec3 bottomLeftCorner, glm::vec3 horizontal, glm::vec3 vertical, glm::vec3 origin, HittableList* world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= x) || (j >= y)) return;
    int pixelIdx = j * x + i;
	// TODO: move camera setup to camera class
	float u = float(i) / float(x - 1);
	float v = float(j) / float(y - 1);
	Ray r(origin, bottomLeftCorner + u * horizontal + v * vertical);
	glm::vec3 col = color(r, world);
	uchar4 px = make_uchar4(col.r * 255, col.g * 255, col.b * 255, 255);
	surf2Dwrite(px, fb, i * sizeof(uchar4), j); // TODO: Decouple rendering from GL - allow writing to framebuffer instead

}

void launchRenderer(cudaGraphicsResource* glResource, int nx, int ny, int xBlock, int yBlock) {
	int numPixels = nx * ny;
	cudaSurfaceObject_t fb = 0;
	Hittable** d_List;
	checkCudaErrors(cudaMalloc((void**)&d_List, 2 * sizeof(Hittable*)));
	HittableList* d_World;
	checkCudaErrors(cudaMalloc((void**)&d_World, sizeof(HittableList)));
	createWorld<<<1, 1>>>(d_List, d_World);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	cudaArray_t cuArray;
	checkCudaErrors(cudaGraphicsMapResources(1, &glResource));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuArray, glResource, 0, 0));
	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;
	checkCudaErrors(cudaCreateSurfaceObject(&fb, &resDesc));
	// TODO: move camera setup to camera class
	float aspect = float(nx) / float(ny);
	float viewport_height = 2.0f;
	float viewport_width = aspect * viewport_height;
	glm::vec3 origin = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 horizontal = glm::vec3(viewport_width, 0.0f, 0.0f);
	glm::vec3 vertical   = glm::vec3(0.0f, viewport_height, 0.0f);
	glm::vec3 bottomLeftCorner = glm::vec3(-viewport_width / 2.0f, -viewport_height / 2.0f, -1.0f);

	dim3 blocks(nx / xBlock + 1, ny / yBlock + 1);
	dim3 threads(xBlock, yBlock);
	renderScene<<<blocks, threads>>>(fb, nx, ny,
		bottomLeftCorner,
		horizontal,
		vertical,
		origin,
		d_World
	);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	destroyWorld<<<1, 1>>>(d_List, d_World);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaFree(d_List));
	checkCudaErrors(cudaFree(d_World));
	cudaDeviceReset();
}

__global__ void createWorld(Hittable** d_List, HittableList* d_World) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_List[0] = new Sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f);
		d_List[1] = new Sphere(glm::vec3(0.0f, -100.5f, -1.0f), 100.0f);
		new(d_World) HittableList(d_List, 2);
	}
}

__global__ void destroyWorld(Hittable** d_List, HittableList* d_World) {
	delete d_List[0];
	delete d_List[1];
	delete d_World;
}