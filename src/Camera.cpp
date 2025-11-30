#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>

// Constructor
Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
    : Front(glm::vec3(0.0f, 0.0f, -1.0f)),
    MovementSpeed(2.5f),
    MouseSensitivity(0.1f),
    Zoom(45.0f)
{
    Position = position;
    WorldUp = up;
    Yaw = yaw;
    Pitch = pitch;
    updateCameraVectors();
}

// Returns view matrix
glm::mat4 Camera::GetViewMatrix() const {
    return glm::lookAt(Position, Position + Front, Up);
}

// Keyboard WASD input
void Camera::ProcessKeyboard(Camera_Movement direction, float deltaTime) {
    float velocity = MovementSpeed * deltaTime;

    if (direction == FORWARD)
        Position += Front * velocity;
    if (direction == BACKWARD)
        Position -= Front * velocity;
    if (direction == LEFT)
        Position -= Right * velocity;
    if (direction == RIGHT)
        Position += Right * velocity;
}

// Mouse movement input
void Camera::ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
    xoffset *= MouseSensitivity;
    yoffset *= MouseSensitivity;

    Yaw += xoffset;
    Pitch += yoffset;

    // Clamp pitch if needed
    if (constrainPitch) {
        if (Pitch > 89.0f) Pitch = 89.0f;
        if (Pitch < -89.0f) Pitch = -89.0f;
    }

    updateCameraVectors();
}

// Scroll wheel zoom
void Camera::ProcessMouseScroll(float yoffset) {
    Zoom -= yoffset;
    if (Zoom < 1.0f) Zoom = 1.0f;
    if (Zoom > 45.0f) Zoom = 45.0f;
}

// Update direction vectors from Euler angles
void Camera::updateCameraVectors() {
    glm::vec3 front;
    front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    front.y = sin(glm::radians(Pitch));
    front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));

    Front = glm::normalize(front);
    Right = glm::normalize(glm::cross(Front, WorldUp));
    Up = glm::normalize(glm::cross(Right, Front));
}

CameraData cameraToData(const Camera& cam, float aspect, float viewportHeight) {
    CameraData data;
    float viewportWidth = aspect * viewportHeight;

    glm::vec3 w = glm::normalize(cam.Front);
    glm::vec3 u = glm::normalize(glm::cross(w, cam.Up));
    glm::vec3 v = glm::cross(u, w);

    data.origin = cam.Position;
    data.horizontal = u * viewportWidth;
    data.vertical = v * viewportHeight;
    data.lowerLeftCorner = cam.Position + w
        - 0.5f * data.horizontal
        - 0.5f * data.vertical;

    return data;
}
