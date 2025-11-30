#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>


class Shader {
public:
	unsigned int ID;
	Shader(const char* vertexPath, const char* fragmentPath);
	void use();
};