#include "kernel.h"

__global__ void renderScene(Framebuffer* d_Fb, Camera* camera, HittableList* world, curandState *randState, cudaSurfaceObject_t surfObj) {
	int x = d_Fb->getWidth();
	int y = d_Fb->getHeight();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= x) || (j >= y)) return;
    int pixelIdx = j * x + i;
	curandState* localRandState = &randState[pixelIdx];
	utils::random::RNG rng(localRandState);
	glm::vec3 col = d_Fb->colorPixel(i, j, x, y, camera, world, rng);
	if (surfObj) {
		uchar4 px = make_uchar4(col.r * 255, col.g * 255, col.b * 255, 255);
		surf2Dwrite(px, surfObj, i * sizeof(uchar4), (y - 1 - j));
	}
	else {
		d_Fb->writePixel(i, j, col);
	}
}

__global__ void initCamera(Camera* cam, int width, int height) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		cam->initialize(width, height);
	}
}

__global__ void createWorld(Hittable** d_List, HittableList* d_World, BVHNode* d_BVHroot, curandState* randState) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_List[0] = new Sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f, new Lambertian(glm::vec3(0.1f, 0.2f, 0.5f)));
		d_List[1] = new Sphere(glm::vec3(0.0f, -100.5f, -1.0f), 100.0f, new Lambertian(glm::vec3(0.8f, 0.8f, 0.0f)));
		d_List[2] = new Sphere(glm::vec3(1.0f, 0.0f, -1.0f), 0.5, new Metal(glm::vec3(0.8f, 0.6f, 0.2f), 0.5f));
		d_List[3] = new Sphere(glm::vec3(-1.0f, 0.0f, -1.0f), 0.5, new Dielectric(1.5f));
		d_List[4] = new Sphere(glm::vec3(-1.0f, 0.0f, -1.0f), 0.45, new Dielectric(1.0f / 1.5f));

		utils::random::RNG rng(&randState[0]);

		new(d_BVHroot) BVHNode(d_List, 0, 5, rng);
		d_List[0] = d_BVHroot;
		new(d_World) HittableList(d_List, 1, 1);
	}
}

__global__ void destroyWorld(Hittable** d_List, HittableList* d_World, BVHNode* d_BVHroot, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
  //      for (int i = 0; i < d_World->getObjCount(); i++) {
		//	delete d_List[i];
		//}
		delete d_List[0]; 
		delete d_BVHroot;
		delete d_World;
    }
}
