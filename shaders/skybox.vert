#version 330 core
layout(location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;       // rotation-only (translation stripped in CPU code)

void main()
{
    TexCoords = aPos;
    // Set z = w so the depth value becomes 1.0 after perspective divide
    // (skybox is always at the far plane, behind everything else)
    vec4 pos = projection * view * vec4(aPos, 1.0);
    gl_Position = pos.xyww;
}
