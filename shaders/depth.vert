#version 330 core
// Shadow-pass vertex shader – outputs clip-space position from the light's POV.
// Only position attribute is used; texture-coords and normals are ignored.
layout(location = 0) in vec3 aPos;

uniform mat4 lightSpace;
uniform mat4 model;

void main()
{
    gl_Position = lightSpace * model * vec4(aPos, 1.0);
}
