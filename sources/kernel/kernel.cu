#include "kernel.h"

__global__ void renderScene(Framebuffer* d_Fb, Camera* camera, HittableList* world, int samples, curandState *randState, cudaSurfaceObject_t surfObj) {
	int x = d_Fb->getWidth();
	int y = d_Fb->getHeight();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= x) || (j >= y)) return;
    int pixelIdx = j * x + i;
	curandState* localRandState = &randState[pixelIdx];
	utils::random::RNG rng(localRandState);
	glm::vec3 col = d_Fb->colorPixel(i, j, x, y, camera, world, samples, rng);
	if (surfObj) {
		uchar4 px = make_uchar4(col.r * 255, col.g * 255, col.b * 255, 255);
		surf2Dwrite(px, surfObj, i * sizeof(uchar4), (y - 1 - j));
	}
	else {
		d_Fb->writePixel(i, j, col);
	}
}

__device__ glm::vec3 colorWithConicalCulling(
    const Ray& ray,
    HittableList* world,
    DynamicObjectInfo* dynamicInfo,
    int maxDepth,
    utils::random::RNG& rng)
{
    Ray currentRay = ray;
    glm::vec3 attenuation(1.0f, 1.0f, 1.0f);

    for (int depth = 0; depth < maxDepth; depth++) {
        HitScatterRecord HSRec = world->hit(currentRay, 0.001f, INF, rng);

        if (HSRec.hitRec.has_value()) {
            HitRecord hitrec = HSRec.hitRec.value();
            ConeSet coneSet;
            constructAndMergeCones(
                hitrec.p,
                hitrec.normal,
                dynamicInfo->boundingSpheres,
                dynamicInfo->numDynamic,
                coneSet
            );

            if (HSRec.scatterRec.has_value()) {
                ScatteringRecord scRec = HSRec.scatterRec.value();
                float maxRayLength = INF;
                bool shouldTrace = shouldTraceRay(
                    glm::normalize(scRec.ray.getDirection()),
                    coneSet,
                    maxRayLength
                );
                attenuation *= scRec.attenuation;
                currentRay = scRec.ray;
            } else {
                return glm::vec3(0.0f);
            }
        } else {
            glm::vec3 direction = glm::normalize(currentRay.getDirection());
            float a = 0.5f * (direction.y + 1.0f);
            glm::vec3 c = (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
            return attenuation * c;
        }
    }
    return glm::vec3(0.0f);
}

__global__ void renderSceneWithConicalCulling(
    Framebuffer* d_Fb,
    Camera* camera,
    HittableList* world,
    DynamicObjectInfo* dynamicInfo,
    int samplesPerPixel,
    int maxDepth,
    curandState* randState,
    cudaSurfaceObject_t surfObj)
{
    int x = d_Fb->getWidth();
    int y = d_Fb->getHeight();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= x) || (j >= y)) return;

    int pixelIdx = j * x + i;
    curandState* localRandState = &randState[pixelIdx];
    utils::random::RNG rng(localRandState);

    glm::vec3 col(0.0f);
    for (int s = 0; s < samplesPerPixel; s++) {
        Ray r = camera->getRay(i, j, rng);
        col += colorWithConicalCulling(r, world, dynamicInfo, maxDepth, rng);
    }

    col /= float(samplesPerPixel);
    col[0] = sqrtf(col[0]);
    col[1] = sqrtf(col[1]);
    col[2] = sqrtf(col[2]);
    if (surfObj) {
        uchar4 px = make_uchar4(
            glm::clamp(col.r, 0.0f, 1.0f) * 255,
            glm::clamp(col.g, 0.0f, 1.0f) * 255,
            glm::clamp(col.b, 0.0f, 1.0f) * 255,
            255
        );
        surf2Dwrite(px, surfObj, i * sizeof(uchar4), (y - 1 - j));
    } else {
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
        auto v0 = glm::vec3(-1.5f, 0.0f, -1.0f);
        auto v1 = glm::vec3(-0.5f, 0.0f, -1.0f);
        auto v2 = glm::vec3(-1.0f, 1.0f, -1.5f);
		d_List[0] = new Sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f, new Lambertian(glm::vec3(0.1f, 0.2f, 0.5f)));
        d_List[1] = new Triangle(v0, v1, v2, new Lambertian(glm::vec3(0.8f, 0.1f, 0.1f)));
		d_List[2] = new Sphere(glm::vec3(1.0f, 0.0f, -1.0f), 0.5, new Metal(glm::vec3(0.8f, 0.6f, 0.2f), 0.5f));
		d_List[3] = new Sphere(glm::vec3(-1.0f, 0.0f, -1.0f), 0.5, new Dielectric(1.5f));
		d_List[4] = new Sphere(glm::vec3(-1.0f, 0.0f, -1.0f), 0.45, new Dielectric(1.0f / 1.5f));

		new(d_BVHroot) BVHNode(d_List, 0, 5);
		d_List[0] = d_BVHroot;
		new(d_World) HittableList(d_List, 1, 1);
	}
}

__global__ void destroyWorld(Hittable** d_List, HittableList* d_World, BVHNode* d_BVHroot, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
		delete d_List[0]; 
		delete d_BVHroot;
		delete d_World;
    }
}

__global__ void createMeshTriangles(Hittable** d_List, int startOffset, glm::vec3* d_vertices, int* d_indices, int numTriangles, Material* mat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numTriangles) return;
    int i0 = d_indices[idx * 3 + 0];
    int i1 = d_indices[idx * 3 + 1];
    int i2 = d_indices[idx * 3 + 2];
    glm::vec3 v0 = d_vertices[i0];
    glm::vec3 v1 = d_vertices[i1];
    glm::vec3 v2 = d_vertices[i2];
    Material* triMat = new Lambertian(glm::vec3(0.5f, 0.5f, 0.5f));
    d_List[startOffset + idx] = new Triangle(v0, v1, v2, triMat);
}