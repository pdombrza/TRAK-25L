#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <glm/glm.hpp>

struct RawMeshData {
    std::vector<glm::vec3> vertices;
    std::vector<int> indices;

    bool loadObj(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open mesh file: " << filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string prefix;
            ss >> prefix;

            if (prefix == "v") {
                glm::vec3 v;
                ss >> v.x >> v.y >> v.z;
                vertices.push_back(v);
            }
            else if (prefix == "f") {
                std::string vertexStr;
                for (int i = 0; i < 3; i++) {
                    ss >> vertexStr;
                    size_t slashPos = vertexStr.find('/');
                    int index = 0;
                    if (slashPos != std::string::npos) {
                        index = std::stoi(vertexStr.substr(0, slashPos));
                    }
                    else {
                        index = std::stoi(vertexStr);
                    }
                    indices.push_back(index - 1);
                }
            }
        }

        std::cout << "Loaded Mesh: " << vertices.size() << " verts, "
            << indices.size() / 3 << " triangles." << std::endl;
        return true;
    }
};
