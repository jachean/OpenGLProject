#version 330 core
// P3: terrain/scene fragment shader with directional sun + 4 streetlight shadows.

in vec2  TexCoord;
in vec3  Normal;
in vec3  FragPos;
in float Height;

out vec4 FragColor;

// ── Albedo texture ──────────────────────────────────────────────────────────
uniform sampler2D texture1;
uniform bool      heightBlend;   // true = terrain height colour blend

// ── Directional sun ─────────────────────────────────────────────────────────
uniform vec3      lightDir;          // already normalised, points away from sun
uniform vec3      lightColor;
uniform mat4      lightSpaceMatrix;
uniform sampler2D shadowMap;         // bound to texture unit 1

// ── 4 streetlights ──────────────────────────────────────────────────────────
// GLSL 3.30 does not guarantee dynamic sampler array indexing, so we use
// four separate uniforms instead.
uniform vec3      spotPos0,        spotPos1,        spotPos2,        spotPos3;
uniform vec3      spotColor0,      spotColor1,      spotColor2,      spotColor3;
uniform mat4      spotLightSpace0,  spotLightSpace1,  spotLightSpace2,  spotLightSpace3;
uniform sampler2D spotShadow0,     spotShadow1,     spotShadow2,     spotShadow3;

// ── PCF 3×3 shadow test ─────────────────────────────────────────────────────
float shadowFactor(sampler2D shadowTex, vec4 fragLS)
{
    vec3 proj = fragLS.xyz / fragLS.w;
    proj = proj * 0.5 + 0.5;
    if (proj.z > 1.0) return 0.0;       // beyond far plane → not in shadow

    float shadow = 0.0;
    float bias   = 0.005;
    vec2  ts     = 1.0 / vec2(textureSize(shadowTex, 0));
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y) {
            float d = texture(shadowTex, proj.xy + vec2(x, y) * ts).r;
            shadow += (proj.z - bias > d) ? 1.0 : 0.0;
        }
    return shadow / 9.0;
}

// ── Point-light contribution (attenuation + shadow) ─────────────────────────
vec3 spotContrib(vec3 pos, vec3 col, mat4 lsm, sampler2D sm)
{
    vec3  L    = pos - FragPos;
    float dist = length(L);
    vec3  Ln   = L / dist;

    // Soft quadratic attenuation – keeps ~75 % intensity at 4.5 units (lamp height)
    // and still reaches ~35 % at 12 units for a wide, visible pool of light.
    float att  = 1.0 / (1.0 + 0.04 * dist + 0.008 * dist * dist);
    float diff = max(dot(normalize(Normal), Ln), 0.0);
    float sh   = shadowFactor(sm, lsm * vec4(FragPos, 1.0));

    return (1.0 - sh) * att * diff * col;
}

void main()
{
    // ── Base colour ─────────────────────────────────────────────────────────
    vec3 texColor = texture(texture1, TexCoord).rgb;
    if (heightBlend) {
        float t  = clamp(Height / 3.0, 0.0, 1.0);
        vec3  hc = mix(vec3(0.55, 0.40, 0.20), vec3(0.20, 0.65, 0.15), t);
        texColor = mix(texColor, hc, 0.40);
    }

    vec3 norm = normalize(Normal);

    // ── Directional sun ─────────────────────────────────────────────────────
    float sunSh  = shadowFactor(shadowMap, lightSpaceMatrix * vec4(FragPos, 1.0));
    float diff   = max(dot(norm, -lightDir), 0.0);
    vec3  ambient = 0.25 * lightColor;
    vec3  sunDiff = (1.0 - sunSh) * diff * 0.65 * lightColor;

    // ── Streetlights ────────────────────────────────────────────────────────
    vec3 spots =
          spotContrib(spotPos0, spotColor0, spotLightSpace0, spotShadow0)
        + spotContrib(spotPos1, spotColor1, spotLightSpace1, spotShadow1)
        + spotContrib(spotPos2, spotColor2, spotLightSpace2, spotShadow2)
        + spotContrib(spotPos3, spotColor3, spotLightSpace3, spotShadow3);

    FragColor = vec4((ambient + sunDiff + spots) * texColor, 1.0);
}
