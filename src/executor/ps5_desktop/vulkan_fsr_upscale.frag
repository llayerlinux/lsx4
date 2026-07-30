#version 450

layout(set = 0, binding = 0) uniform sampler2D Source;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

layout(push_constant) uniform UpscaleParameters {
    vec2 inverseSourceSize;
    vec2 inverseOutputSize;
} params;

float luma(vec3 value) {
    return dot(value, vec3(0.299, 0.587, 0.114));
}

void main() {
    // A compact EASU-style edge-adaptive reconstruction.  The local luma
    // gradient chooses the long sampling axis, while the cross samples limit
    // ringing on alpha-tested/UI edges.  This is intentionally one pass so the
    // mobile presenter keeps a single render submit.
    vec2 texel = params.inverseSourceSize;
    vec4 center = texture(Source, uv);
    vec4 left = texture(Source, uv - vec2(texel.x, 0.0));
    vec4 right = texture(Source, uv + vec2(texel.x, 0.0));
    vec4 up = texture(Source, uv - vec2(0.0, texel.y));
    vec4 down = texture(Source, uv + vec2(0.0, texel.y));

    float horizontal = abs(luma(right.rgb) - luma(left.rgb));
    float vertical = abs(luma(down.rgb) - luma(up.rgb));
    vec2 edgeAxis = horizontal > vertical
        ? vec2(0.0, texel.y)
        : vec2(texel.x, 0.0);
    vec4 along0 = texture(Source, uv - edgeAxis * 0.5);
    vec4 along1 = texture(Source, uv + edgeAxis * 0.5);
    vec4 reconstructed = (center * 2.0 + along0 + along1) * 0.25;

    vec4 localMinimum = min(center, min(min(left, right), min(up, down)));
    vec4 localMaximum = max(center, max(max(left, right), max(up, down)));
    vec4 sharpened = reconstructed +
        (reconstructed - (left + right + up + down) * 0.25) * 0.18;
    color = clamp(sharpened, localMinimum, localMaximum);
}
