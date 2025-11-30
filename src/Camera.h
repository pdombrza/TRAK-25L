#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Movement directions
enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

class Camera {
public:
    // Camera attributes
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;

    // Euler angles
    float Yaw;
    float Pitch;

    // Settings
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;

    // Constructor
    Camera(
        glm::vec3 position = glm::vec3(0.0f, 0.0f, 3.0f),
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
        float yaw = -90.0f,
        float pitch = 0.0f
    );

    // Returns the view matrix from LookAt
    glm::mat4 GetViewMatrix() const;

    // Input handlers
    void ProcessKeyboard(Camera_Movement direction, float deltaTime);
    void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);
    void ProcessMouseScroll(float yoffset);

private:
    // Recalculate the camera basis vectors
    void updateCameraVectors();
};

struct CameraData {
    glm::vec3 origin;
    glm::vec3 horizontal;
    glm::vec3 vertical;
    glm::vec3 lowerLeftCorner;
};

CameraData cameraToData(const Camera& cam, float aspect, float viewportHeight = 2.0f);