#version 450 core
in vec2 f_uv;
uniform sampler2D s;
out vec4 fragColor;

void main() {
    fragColor = texture(s, f_uv);
}