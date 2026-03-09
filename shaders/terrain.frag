#version 330 core
in vec2  TexCoord;
in vec3  Normal;
in vec3  FragPos;
in float Height;

out vec4 FragColor;

uniform sampler2D texture1;
uniform bool      heightBlend;  // true for terrain mesh, false for flat ground

uniform vec3 lightDir;    // direction light is pointing (already normalised)
uniform vec3 lightColor;

void main()
{
    vec3 texColor = texture(texture1, TexCoord).rgb;

    vec3 finalColor;
    if (heightBlend)
    {
        // Blend texture with a height-dependent colour so low areas look sandy
        // and higher hills look green.
        float t         = clamp(Height / 3.0, 0.0, 1.0);
        vec3  lowColor  = vec3(0.55, 0.40, 0.20);  // sandy / brown
        vec3  highColor = vec3(0.20, 0.65, 0.15);  // grass green
        vec3  hColor    = mix(lowColor, highColor, t);
        finalColor      = mix(texColor, hColor, 0.40);
    }
    else
    {
        finalColor = texColor;
    }

    // Simple Lambertian diffuse + ambient
    vec3  norm    = normalize(Normal);
    float diff    = max(dot(norm, -lightDir), 0.0);
    vec3  ambient = 0.35 * lightColor;
    vec3  diffuse = diff * 0.65 * lightColor;

    FragColor = vec4((ambient + diffuse) * finalColor, 1.0);
}
