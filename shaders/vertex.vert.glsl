#version 450 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 v_uv;
out vec2 f_uv;

void main() {
    f_uv = v_uv;
    gl_Position = vec4(position, 0.0, 1.0);
}