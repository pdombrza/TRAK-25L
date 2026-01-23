#include "camera_controller.h"
#include <iostream>

CameraController::CameraController(const glm::vec3& startPosition)
    : position(startPosition),
      front(glm::vec3(0.0f, 0.0f, -1.0f)),
      up(glm::vec3(0.0f, 1.0f, 0.0f))
{
    updateCameraVectors();
}

void CameraController::updateCameraVectors() {
    // Calculate the new front vector from yaw and pitch
    glm::vec3 newFront;
    newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    newFront.y = sin(glm::radians(pitch));
    newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(newFront);

    // Recalculate right and up vectors
    right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
    up = glm::normalize(glm::cross(right, front));
}

void CameraController::processKeyboard(GLFWwindow* window) {
    float velocity = moveSpeed * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        position += front * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        position -= front * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        position -= right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        position += right * velocity;

    // Optional: Q/E for up/down movement
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        position += up * velocity;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        position -= up * velocity;
}

void CameraController::processMouse(GLFWwindow* window) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (firstMouse) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
    }

    double xoffset = xpos - lastMouseX;
    double yoffset = lastMouseY - ypos; // Reversed since y-coordinates go from bottom to top
    lastMouseX = xpos;
    lastMouseY = ypos;

    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;

    yaw += static_cast<float>(xoffset);
    pitch += static_cast<float>(yoffset);

    // Constrain pitch to prevent screen flip
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    updateCameraVectors();
}

void CameraController::updateDeltaTime() {
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
}

void CameraController::updateCamera(Camera& camera) {
    CameraOrientation orientation = getCameraOrientation();
    // Note: Camera's setCameraOrientation is __device__ only, so we need to pass by value
    // and copy the entire camera structure
}

CameraOrientation CameraController::getCameraOrientation() const {
    CameraOrientation orientation;
    orientation.lookFrom = position;
    orientation.lookAt = position + front;
    orientation.vUp = up;
    return orientation;
}
