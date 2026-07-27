#version 450

layout(location = 0) out vec4 out_param_0;

void main() {
    vec2 corner = vec2(
        float((gl_VertexIndex << 1) & 2),
        float(gl_VertexIndex & 2));
    gl_Position = vec4(
        corner * vec2(1.0, -1.0) + vec2(-1.0, 1.0),
        0.0, 1.0);
    out_param_0 = vec4(corner * 0.5, 0.0, 1.0);
}
