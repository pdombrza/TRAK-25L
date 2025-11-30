#include <iostream>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>
#include <optix_function_table_definition.h>
#include <optix.h>
#include <optix_stubs.h>

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "camera/camera.h"
#include "renderer/renderer.h"
#include "hittable/hittable.h"
#include "hittablelist/hittablelist.h"
#include "material/material.h"
#include "kernel/kernel.h"
#include "shader/shader.h"

#include "Camera.h"

#define OPTIX_CHECK(call) \
    { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::cerr << "Optix call (" #call ") failed: " << res << std::endl; \
            return 1; \
        } \
    }

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = 800.0f / 2.0;
float lastY = 600.0f / 2.0;
bool firstMouse = true;

float deltaTime = 0;
float lastFrame = 0;

HittableList* cudaWorld;
Hittable** cudaList;

void processInput(GLFWwindow* window, float deltaTime);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

int main() {
	{
		int runtimeVersion = 0;
		cudaRuntimeGetVersion(&runtimeVersion);
		std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 1000) / 10 << "\n";

		cudaFree(0);
		OPTIX_CHECK(optixInit());

		// Create OptiX context
		CUcontext cuCtx = 0;
		OptixDeviceContext context = nullptr;
		OptixDeviceContextOptions options = {};
		OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

		std::cout << "OptiX initialized successfully" << std::endl;

		cudaRuntimeGetVersion(&runtimeVersion);
		std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 1000) / 10 << "\n";

		if (!glfwInit()) { // TODO: move OpenGL related code to Window class
			std::cerr << "Failed to initialize GLFW" << std::endl;
			return -1;
		}
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        int width = 1200;
        int height = 800;
        GLFWwindow* window = glfwCreateWindow(width, height, "RT", NULL, NULL);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return -1;
        }

		glfwMakeContextCurrent(window);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwSwapInterval(0);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            glfwDestroyWindow(window);
            glfwTerminate();
            return -1;
        }

        // --- Now initialize CUDA runtime and OptiX (after GL context is current) ---
        {
            int runtimeVersion = 0;
            cudaRuntimeGetVersion(&runtimeVersion);
            std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 1000) / 10 << "\n";

            // Initialize CUDA runtime (this creates the CUDA context for the calling thread)
            cudaFree(0);

            // Initialize OptiX after CUDA runtime is available
            OPTIX_CHECK(optixInit());

            // Create OptiX context (passing 0 will use current CUDA context)
            CUcontext cuCtx = 0;
            OptixDeviceContext context = nullptr;
            OptixDeviceContextOptions options = {};
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

            std::cout << "OptiX initialized successfully" << std::endl;
        }

        // --- build shader, VAO/VBO ---
        Shader shader(SHADERS_PATH "vertex.vert.glsl", SHADERS_PATH "fragment.frag.glsl");

        float vertices[] = {
            -1.f, -1.f,   0.f, 0.f,
             1.f, -1.f,   1.f, 0.f,
             1.f,  1.f,   1.f, 1.f,

            -1.f, -1.f,   0.f, 0.f,
             1.f,  1.f,   1.f, 1.f,
            -1.f,  1.f,   0.f, 1.f
        };

        unsigned int VAO, VBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        // --- Register GL resources with CUDA AFTER context is current and CUDA is initialized ---
        GLuint PBO = 0;
        cudaGraphicsResource* cudaPBOResource = nullptr;

        glGenBuffers(1, &PBO);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
        // Register PBO with CUDA (write discard is typical)
        cudaGraphicsGLRegisterBuffer(&cudaPBOResource, PBO, cudaGraphicsMapFlagsWriteDiscard);

		GLuint glTex;
		glGenTextures(1, &glTex);
		glBindTexture(GL_TEXTURE_2D, glTex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        int xBlock = 16;
        int yBlock = 16;
        std::cerr << "Rendering a " << width << "x" << height << " image " << std::endl;
        std::cerr << "in " << xBlock << "x" << yBlock << " blocks" << std::endl;
        int numPixels = width * height;

		CameraOrientation orientation;
		orientation.lookFrom = glm::vec3(0.0f, 0.0f, 1.0f);
		orientation.lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
		orientation.vUp = glm::vec3(0.0f, 1.0f, 0.0f);
		Camera h_camera(orientation, 90.0f, (float)width / (float)height);
		//h_camera.setVFov(20.0f); Tweak this for defocus disc and fov
		//h_camera.setDefocusAngle(0.6f);
		//h_camera.setFocusDist(10.0f);

		HittableList scene{};

		CudaRenderer renderer(&scene, width, height);
		renderer.setSamplesPerPixel(10);
		renderer.registerGLTexture(glTex);
		renderer.setupScene(h_camera);
		std::cout << "Scene set up completed." << std::endl;
		shader.use();

		glViewport(0, 0, width, height);
		glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) -> void { glViewport(0, 0, width, height); });
		renderer.render(h_camera); // Render once for now - too slow to do multiple frames

		while (!glfwWindowShouldClose(window)) {
			processInput(window);

			glClear(GL_COLOR_BUFFER_BIT);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, glTex);
			glBindVertexArray(VAO);
			glDrawArrays(GL_TRIANGLES, 0, 6);

            // 5. Window
            glfwSwapBuffers(window);
            glfwPollEvents();
        }

		renderer.destroyScene();
	}

	cudaDeviceReset();
	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow* window, float deltaTime) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.ProcessMouseScroll(yoffset);
}
