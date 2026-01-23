#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "camera/camera.h"

class CameraController {
private:
    // Camera orientation vectors
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;

    // Euler angles for rotation
    float yaw = -90.0f;   // Looking towards -Z initially
    float pitch = 0.0f;

    // Movement speed
    float moveSpeed = 2.5f;
    float mouseSensitivity = 0.1f;

    // Mouse state
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    bool firstMouse = true;

    // Delta time for frame-independent movement
    float deltaTime = 0.0f;
    float lastFrame = 0.0f;

    void updateCameraVectors();

public:
    CameraController(const glm::vec3& startPosition = glm::vec3(0.0f, 0.0f, 1.0f));

    void processKeyboard(GLFWwindow* window);
    void processMouse(GLFWwindow* window);
    void updateDeltaTime();

    void updateCamera(Camera& camera);
    CameraOrientation getCameraOrientation() const;

    void setMoveSpeed(float speed) { moveSpeed = speed; }
    void setMouseSensitivity(float sensitivity) { mouseSensitivity = sensitivity; }

    glm::vec3 getPosition() const { return position; }
    glm::vec3 getFront() const { return front; }
};
