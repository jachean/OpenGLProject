// =============================================================================
// GPS Project  –  P3: Camera control, lighting and shadows
//
// Builds on P2 (street circuit + static objects).
//
// P3 additions:
//   • Full 6-DOF camera  –  WASD (XZ), QE (Y), mouse (yaw+pitch),
//     Z / X keys (roll CW / CCW).
//   • Directional sun shadow  –  2048×2048 depth FBO, PCF 3×3.
//   • 4 streetlights  –  each has its own 1024×1024 shadow FBO; they cast
//     multiple overlapping shadows and are rendered as pole + lamp head.
//
// Controls:
//   W / A / S / D   – forward / left / back / right
//   Q / E           – move down / up
//   Z / X           – roll clockwise / counter-clockwise
//   Mouse           – look (yaw + pitch)
//   Scroll          – zoom (FOV)
//   ESC             – quit
//
// Dependencies (NuGet):
//   nupengl.core  (GLFW 3, GLEW, freeglut)    glm
// =============================================================================

#define _USE_MATH_DEFINES
#include <cmath>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// Window / render settings
// ─────────────────────────────────────────────────────────────────────────────
static const int   SCR_W = 1280;
static const int   SCR_H = 720;
static const char* TITLE = "GPS – P3: Camera Control, Lighting & Shadows";

// ─────────────────────────────────────────────────────────────────────────────
// Camera  (yaw + pitch + roll,  XYZ translation)
// ─────────────────────────────────────────────────────────────────────────────
struct Camera
{
    glm::vec3 position;
    glm::vec3 front, up, right;
    glm::vec3 worldUp;

    float yaw   = -90.0f;   // look along -Z initially
    float pitch = -10.0f;
    float roll  =   0.0f;   // ← P3: roll around front axis

    float speed       = 8.0f;
    float sensitivity = 0.10f;
    float fov         = 60.0f;

    explicit Camera(glm::vec3 pos = glm::vec3(0.0f, 2.5f, 10.0f))
        : position(pos), worldUp(0.0f, 1.0f, 0.0f)
    {
        updateVectors();
    }

    void updateVectors()
    {
        // 1. Standard yaw + pitch → front direction
        front.x = std::cos(glm::radians(yaw))   * std::cos(glm::radians(pitch));
        front.y = std::sin(glm::radians(pitch));
        front.z = std::sin(glm::radians(yaw))   * std::cos(glm::radians(pitch));
        front   = glm::normalize(front);

        // 2. Derive right and up without roll
        right = glm::normalize(glm::cross(front, worldUp));
        up    = glm::normalize(glm::cross(right, front));

        // 3. Apply roll: rotate right and up around the front axis
        if (roll != 0.0f) {
            float r  = glm::radians(roll);
            float c  = std::cos(r), s = std::sin(r);
            glm::vec3 nr =  c * right + s * up;
            glm::vec3 nu = -s * right + c * up;
            right = glm::normalize(nr);
            up    = glm::normalize(nu);
        }
    }

    glm::mat4 view() const
    {
        return glm::lookAt(position, position + front, up);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Global input state
// ─────────────────────────────────────────────────────────────────────────────
static Camera g_cam;
static bool   g_firstMouse = true;
static float  g_lastX = SCR_W / 2.0f, g_lastY = SCR_H / 2.0f;
static float  g_dt = 0.0f, g_lastFrame = 0.0f;

// ─────────────────────────────────────────────────────────────────────────────
// Shader utilities
// ─────────────────────────────────────────────────────────────────────────────
static std::string readFile(const char* path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[SHADER] Cannot open file: " << path << "\n";
        return "";
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static GLuint compileShader(GLenum type, const std::string& src)
{
    GLuint id = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(id, 1, &c, nullptr);
    glCompileShader(id);

    GLint ok = 0;
    glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(id, sizeof(log), nullptr, log);
        std::cerr << "[SHADER] Compile error:\n" << log << "\n";
    }
    return id;
}

static GLuint createProgram(const char* vsPath, const char* fsPath)
{
    GLuint vs   = compileShader(GL_VERTEX_SHADER,   readFile(vsPath));
    GLuint fs   = compileShader(GL_FRAGMENT_SHADER, readFile(fsPath));
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        std::cerr << "[SHADER] Link error:\n" << log << "\n";
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

static void setMat4(GLuint prog, const char* name, const glm::mat4& m)
{
    glUniformMatrix4fv(glGetUniformLocation(prog, name), 1, GL_FALSE, glm::value_ptr(m));
}
static void setVec3(GLuint prog, const char* name, const glm::vec3& v)
{
    glUniform3fv(glGetUniformLocation(prog, name), 1, glm::value_ptr(v));
}
static void setInt(GLuint prog, const char* name, int val)
{
    glUniform1i(glGetUniformLocation(prog, name), val);
}
static void setBool(GLuint prog, const char* name, bool val)
{
    glUniform1i(glGetUniformLocation(prog, name), val ? 1 : 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Procedural texture generation  (no external image files needed)
// ─────────────────────────────────────────────────────────────────────────────
static float noise2(int x, int y)
{
    int n = x + y * 57;
    n = (n << 13) ^ n;
    return 1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f;
}

static GLuint makeTexture2D(int w, int h, const std::vector<unsigned char>& px)
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    return tex;
}

static GLuint makeGrassTexture()
{
    const int W = 256, H = 256;
    std::vector<unsigned char> px(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float n      = 0.5f + 0.5f * noise2(x, y);
            float stripe = 0.90f + 0.10f * std::sin((x + y) * 0.25f);
            int r = (int)(( 35 + 15 * n) * stripe);
            int g = (int)((120 + 40 * n) * stripe);
            int b = (int)(( 20 + 15 * n) * stripe);
            px[(y*W+x)*3+0] = (unsigned char)std::min(r, 255);
            px[(y*W+x)*3+1] = (unsigned char)std::min(g, 255);
            px[(y*W+x)*3+2] = (unsigned char)std::min(b, 255);
        }
    return makeTexture2D(W, H, px);
}

static GLuint makeTerrainTexture()
{
    const int W = 256, H = 256;
    std::vector<unsigned char> px(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float n = 0.5f + 0.5f * noise2(x * 3, y * 3 + 17);
            int r = (int)(130 + 30 * n);
            int g = (int)(100 + 25 * n);
            int b = (int)( 60 + 20 * n);
            px[(y*W+x)*3+0] = (unsigned char)std::min(r, 255);
            px[(y*W+x)*3+1] = (unsigned char)std::min(g, 255);
            px[(y*W+x)*3+2] = (unsigned char)std::min(b, 255);
        }
    return makeTexture2D(W, H, px);
}

static GLuint makeRoadTexture()
{
    const int W = 512, H = 128;
    std::vector<unsigned char> px(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float fv = (float)y / H;
            float n  = 0.5f + 0.5f * noise2(x * 7, y * 7 + 31);
            int r = (int)(36 + 10 * n);
            int g = (int)(36 + 10 * n);
            int b = (int)(40 + 10 * n);
            if (std::abs(fv - 0.5f) < 0.04f && (x % 32) < 20)
                r = g = b = 230;
            if (fv < 0.07f || fv > 0.93f) {
                float et = (fv < 0.5f) ? (1.0f - fv / 0.07f) : ((fv - 0.93f) / 0.07f);
                r = (int)(36 + et * 190);
                g = (int)(36 + et * 190);
                b = (int)(40 + et * 190);
            }
            px[(y*W+x)*3+0] = (unsigned char)std::min(r, 255);
            px[(y*W+x)*3+1] = (unsigned char)std::min(g, 255);
            px[(y*W+x)*3+2] = (unsigned char)std::min(b, 255);
        }
    return makeTexture2D(W, H, px);
}

static GLuint makeBuildingTexture()
{
    const int W = 256, H = 256;
    std::vector<unsigned char> px(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float n      = 0.5f + 0.5f * noise2(x, y + 500);
            bool winRow  = ((y % 32) >= 4  && (y % 32) <= 24);
            bool winCol  = ((x % 24) >= 3  && (x % 24) <= 18);
            if (winRow && winCol) {
                px[(y*W+x)*3+0] = (unsigned char)(70  + (int)(20 * n));
                px[(y*W+x)*3+1] = (unsigned char)(95  + (int)(30 * n));
                px[(y*W+x)*3+2] = (unsigned char)(135 + (int)(40 * n));
            } else {
                int v = (int)(165 + 22 * n);
                px[(y*W+x)*3+0] = (unsigned char)std::min(v,     255);
                px[(y*W+x)*3+1] = (unsigned char)std::min(v,     255);
                px[(y*W+x)*3+2] = (unsigned char)std::min(v - 5, 255);
            }
        }
    return makeTexture2D(W, H, px);
}

static GLuint makeLeafTexture()
{
    const int W = 128, H = 128;
    std::vector<unsigned char> px(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float n = 0.5f + 0.5f * noise2(x * 3, y * 3 + 200);
            px[(y*W+x)*3+0] = (unsigned char)(22  + (int)(18 * n));
            px[(y*W+x)*3+1] = (unsigned char)(100 + (int)(55 * n));
            px[(y*W+x)*3+2] = (unsigned char)(18  + (int)(14 * n));
        }
    return makeTexture2D(W, H, px);
}

static GLuint makeBarkTexture()
{
    const int W = 64, H = 128;
    std::vector<unsigned char> px(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float n = 0.5f + 0.5f * noise2(x, y * 2 + 700);
            float s = 0.5f + 0.5f * std::sin(y * 0.9f + n * 2.0f);
            px[(y*W+x)*3+0] = (unsigned char)(78  + (int)(42 * s + 14 * n));
            px[(y*W+x)*3+1] = (unsigned char)(52  + (int)(28 * s + 10 * n));
            px[(y*W+x)*3+2] = (unsigned char)(28  + (int)(16 * s +  8 * n));
        }
    return makeTexture2D(W, H, px);
}

// Lamp-head texture – warm yellow/white to suggest an emissive glow
static GLuint makeLampTexture()
{
    const int W = 4, H = 4;
    std::vector<unsigned char> px(W * H * 3);
    for (int i = 0; i < W * H; i++) {
        px[i*3+0] = 255;
        px[i*3+1] = 240;
        px[i*3+2] = 150;
    }
    return makeTexture2D(W, H, px);
}

// Cubemap skybox
static GLuint makeSkyboxCubemap()
{
    const int W = 512, H = 512;
    std::vector<unsigned char> face(W * H * 3);

    auto set = [&](int x, int y, int r, int g, int b) {
        face[(y * W + x) * 3 + 0] = (unsigned char)std::min(r, 255);
        face[(y * W + x) * 3 + 1] = (unsigned char)std::min(g, 255);
        face[(y * W + x) * 3 + 2] = (unsigned char)std::min(b, 255);
    };

    GLuint cubemap;
    glGenTextures(1, &cubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);

    const GLenum faceTargets[6] = {
        GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
    };

    for (int fi = 0; fi < 6; fi++) {
        bool isTop    = (fi == 2);
        bool isBottom = (fi == 3);
        float phaseShift = fi * 1.3f;

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float u = (float)x / (W - 1);
                float v = (float)y / (H - 1);

                if (isTop) {
                    float cx   = 2.0f * u - 1.0f;
                    float cy   = 2.0f * v - 1.0f;
                    float dist = std::sqrt(cx * cx + cy * cy);
                    float t    = std::min(dist / 1.2f, 1.0f);
                    set(x, y, (int)(40 + t * 100), (int)(90 + t * 100), (int)(185 + t * 50));
                }
                else if (isBottom) {
                    set(x, y, 15, 12, 10);
                }
                else {
                    // v=0 → looking UP (sky), v=0.5 → horizon, v=1 → looking DOWN
                    float ridgeV = 0.42f
                        + 0.06f * std::sin(u * (float)M_PI * 4.0f + phaseShift)
                        + 0.04f * std::sin(u * (float)M_PI * 9.0f + phaseShift * 1.7f)
                        + 0.03f * std::cos(u * (float)M_PI * 6.5f + phaseShift * 0.9f);
                    ridgeV = std::max(0.30f, std::min(ridgeV, 0.58f));

                    if (v < ridgeV) {
                        float t = v / ridgeV;
                        set(x, y, (int)(50 + t * 95), (int)(100 + t * 95), (int)(195 + t * 40));
                    } else {
                        float t = (v - ridgeV) / (1.0f - ridgeV);
                        float n = 0.5f + 0.5f * noise2(x + fi * 512, y);
                        set(x, y,
                            (int)((88 + t * 45) * (0.85f + 0.15f * n)),
                            (int)((75 + t * 30) * (0.85f + 0.15f * n)),
                            (int)((58 + t * 18) * (0.85f + 0.15f * n)));
                    }
                }
            }
        }
        glTexImage2D(faceTargets[fi], 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, face.data());
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    return cubemap;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mesh data types
// ─────────────────────────────────────────────────────────────────────────────
struct SkyboxMesh { GLuint vao, vbo; };

struct Mesh {
    GLuint vao, vbo, ebo;
    int    indexCount;
};

// ─────────────────────────────────────────────────────────────────────────────
// Shadow map  (depth FBO + depth texture)
// ─────────────────────────────────────────────────────────────────────────────
struct ShadowMap {
    GLuint fbo, tex;
    int    width, height;
};

static ShadowMap createShadowMap(int w, int h)
{
    ShadowMap sm;
    sm.width = w;  sm.height = h;

    glGenTextures(1, &sm.tex);
    glBindTexture(GL_TEXTURE_2D, sm.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                 w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float border[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);

    glGenFramebuffers(1, &sm.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, sm.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, sm.tex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return sm;
}

// ─────────────────────────────────────────────────────────────────────────────
// Streetlight
// ─────────────────────────────────────────────────────────────────────────────
struct Streetlight {
    glm::vec3 position;    // world position of the lamp head
    glm::vec3 color;
    ShadowMap shadow;
    glm::mat4 lightSpace;
};

// ─────────────────────────────────────────────────────────────────────────────
// Skybox mesh  (unit cube, positions only, 36 vertices)
// ─────────────────────────────────────────────────────────────────────────────
static SkyboxMesh createSkyboxMesh()
{
    static const float v[] = {
        -1.0f,  1.0f, -1.0f,  -1.0f, -1.0f, -1.0f,   1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,   1.0f,  1.0f, -1.0f,  -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,  -1.0f, -1.0f, -1.0f,  -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,  -1.0f,  1.0f,  1.0f,  -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,   1.0f, -1.0f,  1.0f,   1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,   1.0f,  1.0f, -1.0f,   1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,  -1.0f,  1.0f,  1.0f,   1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,   1.0f, -1.0f,  1.0f,  -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,   1.0f,  1.0f, -1.0f,   1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,  -1.0f,  1.0f,  1.0f,  -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,  -1.0f, -1.0f,  1.0f,   1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,  -1.0f, -1.0f,  1.0f,   1.0f, -1.0f,  1.0f
    };

    SkyboxMesh m{};
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ground plane  (pos + texcoord + normal, 8 floats per vertex)
// ─────────────────────────────────────────────────────────────────────────────
static Mesh createGroundMesh(float halfSize)
{
    float s = halfSize, r = halfSize / 4.0f;
    float verts[] = {
        -s, 0.0f, -s,  0.0f,  r,   0.0f, 1.0f, 0.0f,
         s, 0.0f, -s,  r,     r,   0.0f, 1.0f, 0.0f,
         s, 0.0f,  s,  r,     0.0f, 0.0f, 1.0f, 0.0f,
        -s, 0.0f,  s,  0.0f,  0.0f, 0.0f, 1.0f, 0.0f,
    };
    unsigned int idx[] = { 0, 1, 2,  2, 3, 0 };

    Mesh m{};
    m.indexCount = 6;
    glGenVertexArrays(1, &m.vao); glGenBuffers(1, &m.vbo); glGenBuffers(1, &m.ebo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);              glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(5*sizeof(float))); glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Terrain mesh  (height-map, rolling hills)
// ─────────────────────────────────────────────────────────────────────────────
static Mesh createTerrainMesh(int gridSize, float scale, float amplitude, float freq)
{
    int V = gridSize + 1;

    auto getH = [&](float x, float z) -> float {
        return amplitude * (
            0.50f * std::sin(x * freq)              * std::cos(z * freq) +
            0.30f * std::sin(x * freq * 2.1f + 0.40f) +
            0.20f * std::cos(z * freq * 1.7f + 0.80f)
        );
    };

    std::vector<float> verts(V * V * 8);
    for (int j = 0; j <= gridSize; j++) {
        for (int i = 0; i <= gridSize; i++) {
            float u = (float)i / gridSize, w = (float)j / gridSize;
            float x = (u - 0.5f) * 2.0f * scale;
            float z = (w - 0.5f) * 2.0f * scale;
            float y = getH(x, z);
            int base = (j * V + i) * 8;
            verts[base+0]=x; verts[base+1]=y; verts[base+2]=z;
            verts[base+3]=u*8.0f; verts[base+4]=w*8.0f;
            const float eps = 0.05f;
            float dhdx = (getH(x+eps,z) - getH(x-eps,z)) / (2.0f*eps);
            float dhdz = (getH(x,z+eps) - getH(x,z-eps)) / (2.0f*eps);
            glm::vec3 n = glm::normalize(glm::vec3(-dhdx, 1.0f, -dhdz));
            verts[base+5]=n.x; verts[base+6]=n.y; verts[base+7]=n.z;
        }
    }

    std::vector<unsigned int> indices;
    indices.reserve(gridSize * gridSize * 6);
    for (int j = 0; j < gridSize; j++)
        for (int i = 0; i < gridSize; i++) {
            unsigned int a=j*V+i, b=j*V+i+1, c=(j+1)*V+i, d=(j+1)*V+i+1;
            indices.push_back(a); indices.push_back(b); indices.push_back(d);
            indices.push_back(a); indices.push_back(d); indices.push_back(c);
        }

    Mesh m{}; m.indexCount=(int)indices.size();
    glGenVertexArrays(1,&m.vao); glGenBuffers(1,&m.vbo); glGenBuffers(1,&m.ebo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER,m.vbo);
    glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(verts.size()*sizeof(float)),verts.data(),GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,(GLsizeiptr)(indices.size()*sizeof(unsigned int)),indices.data(),GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)0);              glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(5*sizeof(float))); glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Oval street circuit mesh
// ─────────────────────────────────────────────────────────────────────────────
static Mesh createCircuitMesh(float semiX, float semiZ, float roadHalfWidth, int segments)
{
    float totalLen = 0.0f;
    {
        float px = semiX, pz = 0.0f;
        for (int i = 1; i <= segments; i++) {
            float t  = 2.0f*(float)M_PI*i/segments;
            float cx = semiX*std::cos(t), cz = semiZ*std::sin(t);
            float dx = cx-px, dz = cz-pz;
            totalLen += std::sqrt(dx*dx+dz*dz);
            px=cx; pz=cz;
        }
    }

    std::vector<float>        verts;
    std::vector<unsigned int> indices;
    float arcLen=0.0f, prevCx=semiX, prevCz=0.0f;

    for (int i = 0; i <= segments; i++) {
        float t  = 2.0f*(float)M_PI*i/segments;
        float cx = semiX*std::cos(t), cz = semiZ*std::sin(t);
        if (i > 0) {
            float dx=cx-prevCx, dz=cz-prevCz;
            arcLen += std::sqrt(dx*dx+dz*dz);
        }
        float u = arcLen / (roadHalfWidth*2.0f);
        float nx=std::cos(t)/semiX, nz=std::sin(t)/semiZ;
        float nl=std::sqrt(nx*nx+nz*nz); nx/=nl; nz/=nl;
        float ix=cx-nx*roadHalfWidth, iz=cz-nz*roadHalfWidth;
        verts.insert(verts.end(), {ix,0.02f,iz, u,0.0f, 0.0f,1.0f,0.0f});
        float ox=cx+nx*roadHalfWidth, oz=cz+nz*roadHalfWidth;
        verts.insert(verts.end(), {ox,0.02f,oz, u,1.0f, 0.0f,1.0f,0.0f});
        prevCx=cx; prevCz=cz;
    }
    for (int i = 0; i < segments; i++) {
        unsigned int a=i*2,b=i*2+1,c=(i+1)*2,d=(i+1)*2+1;
        indices.push_back(a); indices.push_back(c); indices.push_back(b);
        indices.push_back(b); indices.push_back(c); indices.push_back(d);
    }

    Mesh m{}; m.indexCount=(int)indices.size();
    glGenVertexArrays(1,&m.vao); glGenBuffers(1,&m.vbo); glGenBuffers(1,&m.ebo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER,m.vbo);
    glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(verts.size()*sizeof(float)),verts.data(),GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,(GLsizeiptr)(indices.size()*sizeof(unsigned int)),indices.data(),GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)0);              glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(5*sizeof(float))); glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Box mesh  (buildings, trunks, poles).  Origin at centre of base (y=0..h).
// ─────────────────────────────────────────────────────────────────────────────
static Mesh createBoxMesh(float w, float h, float d)
{
    float hw=w*0.5f, hd=d*0.5f;
    std::vector<float> verts = {
        -hw,0, hd, 0,0, 0,0,1,   hw,0, hd, 1,0, 0,0,1,   hw,h, hd, 1,1, 0,0,1,  -hw,h, hd, 0,1, 0,0,1,
         hw,0,-hd, 0,0, 0,0,-1, -hw,0,-hd, 1,0, 0,0,-1, -hw,h,-hd, 1,1, 0,0,-1,  hw,h,-hd, 0,1, 0,0,-1,
        -hw,0,-hd, 0,0,-1,0,0, -hw,0, hd, 1,0,-1,0,0, -hw,h, hd, 1,1,-1,0,0, -hw,h,-hd, 0,1,-1,0,0,
         hw,0, hd, 0,0, 1,0,0,  hw,0,-hd, 1,0, 1,0,0,  hw,h,-hd, 1,1, 1,0,0,  hw,h, hd, 0,1, 1,0,0,
        -hw,h, hd, 0,0, 0,1,0,  hw,h, hd, 1,0, 0,1,0,  hw,h,-hd, 1,1, 0,1,0, -hw,h,-hd, 0,1, 0,1,0,
        -hw,0,-hd, 0,0, 0,-1,0,  hw,0,-hd, 1,0, 0,-1,0,  hw,0, hd, 1,1, 0,-1,0, -hw,0, hd, 0,1, 0,-1,0,
    };
    unsigned int idx[36];
    for (int f=0;f<6;f++){unsigned int b=f*4; idx[f*6+0]=b;idx[f*6+1]=b+1;idx[f*6+2]=b+2;idx[f*6+3]=b;idx[f*6+4]=b+2;idx[f*6+5]=b+3;}
    Mesh m{}; m.indexCount=36;
    glGenVertexArrays(1,&m.vao); glGenBuffers(1,&m.vbo); glGenBuffers(1,&m.ebo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER,m.vbo);
    glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(verts.size()*sizeof(float)),verts.data(),GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(idx),idx,GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)0);              glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(5*sizeof(float))); glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cone mesh  (tree crown).  Apex at (0, height, 0), base at y=0.
// ─────────────────────────────────────────────────────────────────────────────
static Mesh createConeMesh(float radius, float height, int segments)
{
    std::vector<float>        verts;
    std::vector<unsigned int> indices;
    for (int i=0;i<segments;i++){
        float a0=2.0f*(float)M_PI*i/segments, a1=2.0f*(float)M_PI*(i+1)/segments;
        float amid=(a0+a1)*0.5f;
        float x0=radius*std::cos(a0),z0=radius*std::sin(a0);
        float x1=radius*std::cos(a1),z1=radius*std::sin(a1);
        glm::vec3 n=glm::normalize(glm::vec3(height*std::cos(amid),radius,height*std::sin(amid)));
        unsigned int base=(unsigned int)verts.size()/8;
        verts.insert(verts.end(),{0.0f,height,0.0f,(a0+a1)/(2.0f*(float)M_PI),1.0f,n.x,n.y,n.z});
        verts.insert(verts.end(),{x0,0.0f,z0,(float)i/segments,0.0f,n.x,n.y,n.z});
        verts.insert(verts.end(),{x1,0.0f,z1,(float)(i+1)/segments,0.0f,n.x,n.y,n.z});
        indices.push_back(base); indices.push_back(base+2); indices.push_back(base+1);
    }
    unsigned int ci=(unsigned int)verts.size()/8;
    verts.insert(verts.end(),{0.0f,0.0f,0.0f,0.5f,0.5f,0.0f,-1.0f,0.0f});
    for(int i=0;i<segments;i++){
        float a=2.0f*(float)M_PI*i/segments;
        verts.insert(verts.end(),{radius*std::cos(a),0.0f,radius*std::sin(a),
            0.5f+0.5f*std::cos(a),0.5f+0.5f*std::sin(a),0.0f,-1.0f,0.0f});
    }
    for(int i=0;i<segments;i++){indices.push_back(ci);indices.push_back(ci+1+i);indices.push_back(ci+1+(i+1)%segments);}

    Mesh m{}; m.indexCount=(int)indices.size();
    glGenVertexArrays(1,&m.vao); glGenBuffers(1,&m.vbo); glGenBuffers(1,&m.ebo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER,m.vbo);
    glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(verts.size()*sizeof(float)),verts.data(),GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,(GLsizeiptr)(indices.size()*sizeof(unsigned int)),indices.data(),GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)0);              glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,8*sizeof(float),(void*)(5*sizeof(float))); glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// GLFW callbacks
// ─────────────────────────────────────────────────────────────────────────────
static void cbFramebufferSize(GLFWwindow*, int w, int h)
{
    glViewport(0, 0, w, h);
}

static void cbScroll(GLFWwindow*, double /*dx*/, double dy)
{
    g_cam.fov -= (float)dy * 2.0f;
    g_cam.fov  = std::max(15.0f, std::min(120.0f, g_cam.fov));
}

static void cbMouse(GLFWwindow*, double xpos, double ypos)
{
    if (g_firstMouse) {
        g_lastX = (float)xpos;
        g_lastY = (float)ypos;
        g_firstMouse = false;
    }
    float dx = ((float)xpos - g_lastX) * g_cam.sensitivity;
    float dy = (g_lastY - (float)ypos) * g_cam.sensitivity;
    g_lastX = (float)xpos;
    g_lastY = (float)ypos;

    g_cam.yaw   += dx;
    g_cam.pitch  = std::max(-89.0f, std::min(89.0f, g_cam.pitch + dy));
    g_cam.updateVectors();
}

static void processInput(GLFWwindow* win)
{
    if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(win, true);

    float spd = g_cam.speed * g_dt;
    if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) g_cam.position += g_cam.front * spd;
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) g_cam.position -= g_cam.front * spd;
    if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) g_cam.position -= g_cam.right * spd;
    if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) g_cam.position += g_cam.right * spd;
    if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) g_cam.position -= g_cam.up    * spd;
    if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) g_cam.position += g_cam.up    * spd;

    // Z / X  –  roll clockwise / counter-clockwise  (P3)
    float rollSpd = 45.0f * g_dt;   // degrees per second
    if (glfwGetKey(win, GLFW_KEY_Z) == GLFW_PRESS) { g_cam.roll += rollSpd; g_cam.updateVectors(); }
    if (glfwGetKey(win, GLFW_KEY_X) == GLFW_PRESS) { g_cam.roll -= rollSpd; g_cam.updateVectors(); }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    // ── GLFW init ────────────────────────────────────────────────────────────
    if (!glfwInit()) { std::cerr << "Failed to init GLFW\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(SCR_W, SCR_H, TITLE, nullptr, nullptr);
    if (!win) { std::cerr << "Failed to create GLFW window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);
    glfwSetFramebufferSizeCallback(win, cbFramebufferSize);
    glfwSetCursorPosCallback(win,       cbMouse);
    glfwSetScrollCallback(win,          cbScroll);
    glfwSetInputMode(win, GLFW_CURSOR,  GLFW_CURSOR_DISABLED);

    // ── GLEW init ────────────────────────────────────────────────────────────
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "Failed to init GLEW\n"; return -1; }

    glEnable(GL_DEPTH_TEST);

    // ── Shaders ──────────────────────────────────────────────────────────────
    GLuint skyboxProg  = createProgram("shaders/skybox.vert",  "shaders/skybox.frag");
    GLuint terrainProg = createProgram("shaders/terrain.vert", "shaders/terrain.frag");
    GLuint depthProg   = createProgram("shaders/depth.vert",   "shaders/depth.frag");

    // ── Meshes ───────────────────────────────────────────────────────────────
    SkyboxMesh skyboxMesh  = createSkyboxMesh();
    Mesh       groundMesh  = createGroundMesh(18.0f);
    Mesh       terrainMesh = createTerrainMesh(64, 6.0f, 1.5f, 0.35f);
    Mesh       circuitMesh = createCircuitMesh(12.0f, 8.0f, 1.5f, 120);
    Mesh       boxMesh     = createBoxMesh(1.0f, 1.0f, 1.0f);
    Mesh       coneMesh    = createConeMesh(1.0f, 1.0f, 16);

    // ── Textures ─────────────────────────────────────────────────────────────
    GLuint skyboxTex   = makeSkyboxCubemap();
    GLuint grassTex    = makeGrassTexture();
    GLuint terrainTex  = makeTerrainTexture();
    GLuint roadTex     = makeRoadTexture();
    GLuint buildingTex = makeBuildingTexture();
    GLuint leafTex     = makeLeafTexture();
    GLuint barkTex     = makeBarkTexture();
    GLuint lampTex     = makeLampTexture();

    // ── Scene object data (used by both shadow and main passes) ──────────────
    struct BldgDef { float x, z, w, h, d; };
    const BldgDef buildings[] = {
        { -14.5f, -4.0f,  2.5f, 5.5f, 2.0f },
        { -14.5f,  0.0f,  2.5f, 7.0f, 2.0f },
        { -14.5f,  4.0f,  2.5f, 5.0f, 2.0f },
        {  15.0f, -3.0f,  4.0f, 3.0f, 3.5f },
        {  15.0f,  3.0f,  4.0f, 3.0f, 3.5f },
        {   0.0f, 11.5f,  1.5f, 8.0f, 1.5f },
    };
    struct TreePos { float x, z; };
    const TreePos trees[] = {
        {  3.0f,  2.0f }, { -4.0f,  3.5f }, {  2.0f, -4.0f },
        { -3.5f, -3.0f }, {  5.0f,  0.5f },
    };

    // ── Lighting ─────────────────────────────────────────────────────────────
    glm::vec3 lightDir   = glm::normalize(glm::vec3(-0.4f, -1.0f, -0.5f));
    glm::vec3 lightColor = glm::vec3(1.0f, 0.97f, 0.90f);

    // ── Sun shadow map  (2048×2048 orthographic) ─────────────────────────────
    ShadowMap sunShadow = createShadowMap(2048, 2048);
    glm::vec3 sunPos    = -lightDir * 40.0f;   // place light far opposite to lightDir
    glm::mat4 sunLightView  = glm::lookAt(sunPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 sunLightProj  = glm::ortho(-28.0f, 28.0f, -28.0f, 28.0f, 0.1f, 150.0f);
    glm::mat4 sunLightSpace = sunLightProj * sunLightView;

    // ── 4 streetlights (perspective shadow maps, 1024×1024) ─────────────────
    // Placed at the four corners of the oval, well clear of all buildings:
    //   grandstands  –  x≈-14.5, z ∈ [-5, 5]
    //   pits         –  x≈15,    z ∈ [-4.75, 4.75]
    //   control tower–  x≈0,     z ∈ [10.75, 12.25]
    // Corner positions (|x|=13.5, |z|=7.5) are outside the road and avoid
    // every building footprint in both X and Z.
    const int NUM_LIGHTS = 4;
    Streetlight streetlights[NUM_LIGHTS];
    const glm::vec3 slPos[NUM_LIGHTS] = {
        {  13.5f, 4.5f, -7.5f },   // back-right corner
        { -13.5f, 4.5f, -7.5f },   // back-left corner
        {  13.5f, 4.5f,  7.5f },   // front-right corner
        { -13.5f, 4.5f,  7.5f },   // front-left corner
    };
    // Intensity > 1 so the contribution can visibly add on top of the sun.
    // The shader sums (ambient + sunDiff + spots) before multiplying texColor,
    // so values > 1 create a clearly visible warm halo at ground level.
    const glm::vec3 slColor(3.0f, 2.64f, 1.95f);   // warm sodium-lamp yellow, 3× intensity
    {
        glm::mat4 spotProj = glm::perspective(glm::radians(120.0f), 1.0f, 0.5f, 30.0f);
        for (int i = 0; i < NUM_LIGHTS; i++) {
            streetlights[i].position  = slPos[i];
            streetlights[i].color     = slColor;
            streetlights[i].shadow    = createShadowMap(1024, 1024);
            // Looking straight down; use (1,0,0) as up since dir is (0,-1,0)
            glm::mat4 sv = glm::lookAt(slPos[i],
                                       slPos[i] + glm::vec3(0.0f, -1.0f, 0.0f),
                                       glm::vec3(1.0f,  0.0f, 0.0f));
            streetlights[i].lightSpace = spotProj * sv;
        }
    }

    // ── Set constant terrain-shader uniforms (done once before loop) ─────────
    glUseProgram(terrainProg);
    setInt (terrainProg, "texture1",    0);   // albedo  → unit 0
    setInt (terrainProg, "shadowMap",   1);   // sun     → unit 1
    setInt (terrainProg, "spotShadow0", 2);   // light 0 → unit 2
    setInt (terrainProg, "spotShadow1", 3);
    setInt (terrainProg, "spotShadow2", 4);
    setInt (terrainProg, "spotShadow3", 5);
    setVec3(terrainProg, "lightDir",    lightDir);
    setVec3(terrainProg, "lightColor",  lightColor);
    setMat4(terrainProg, "lightSpaceMatrix", sunLightSpace);
    // Streetlight world positions, colours, and light-space matrices
    const char* spNames[] = { "spotPos0","spotPos1","spotPos2","spotPos3" };
    const char* scNames[] = { "spotColor0","spotColor1","spotColor2","spotColor3" };
    const char* slNames[] = { "spotLightSpace0","spotLightSpace1","spotLightSpace2","spotLightSpace3" };
    for (int i = 0; i < NUM_LIGHTS; i++) {
        setVec3(terrainProg, spNames[i], streetlights[i].position);
        setVec3(terrainProg, scNames[i], streetlights[i].color);
        setMat4(terrainProg, slNames[i], streetlights[i].lightSpace);
    }

    // ── Shadow render helper: draws all opaque geometry (no textures) ────────
    // Used for both the sun pass and each streetlight shadow pass.
    auto shadowRender = [&](GLuint prog)
    {
        // Ground
        setMat4(prog, "model", glm::mat4(1.0f));
        glBindVertexArray(groundMesh.vao);
        glDrawElements(GL_TRIANGLES, groundMesh.indexCount, GL_UNSIGNED_INT, nullptr);

        // Terrain (offset matches main pass)
        setMat4(prog, "model", glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.05f, 0.0f)));
        glBindVertexArray(terrainMesh.vao);
        glDrawElements(GL_TRIANGLES, terrainMesh.indexCount, GL_UNSIGNED_INT, nullptr);

        // Circuit
        setMat4(prog, "model", glm::mat4(1.0f));
        glBindVertexArray(circuitMesh.vao);
        glDrawElements(GL_TRIANGLES, circuitMesh.indexCount, GL_UNSIGNED_INT, nullptr);

        // Buildings
        for (const auto& b : buildings) {
            setMat4(prog, "model",
                glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(b.x, 0.0f, b.z)),
                           glm::vec3(b.w, b.h, b.d)));
            glBindVertexArray(boxMesh.vao);
            glDrawElements(GL_TRIANGLES, boxMesh.indexCount, GL_UNSIGNED_INT, nullptr);
        }

        // Trees (trunk + crown)
        for (const auto& tr : trees) {
            glm::vec3 base(tr.x, 0.0f, tr.z);
            setMat4(prog, "model",
                glm::scale(glm::translate(glm::mat4(1.0f), base), glm::vec3(0.25f, 1.5f, 0.25f)));
            glBindVertexArray(boxMesh.vao);
            glDrawElements(GL_TRIANGLES, boxMesh.indexCount, GL_UNSIGNED_INT, nullptr);
            setMat4(prog, "model",
                glm::scale(glm::translate(glm::mat4(1.0f), base + glm::vec3(0.0f, 1.5f, 0.0f)),
                           glm::vec3(0.9f, 1.8f, 0.9f)));
            glBindVertexArray(coneMesh.vao);
            glDrawElements(GL_TRIANGLES, coneMesh.indexCount, GL_UNSIGNED_INT, nullptr);
        }

        // Streetlight poles (cast shadows; lamp heads are too small to matter)
        for (int i = 0; i < NUM_LIGHTS; i++) {
            glm::vec3 poleBase(streetlights[i].position.x, 0.0f, streetlights[i].position.z);
            setMat4(prog, "model",
                glm::scale(glm::translate(glm::mat4(1.0f), poleBase),
                           glm::vec3(0.18f, streetlights[i].position.y, 0.18f)));
            glBindVertexArray(boxMesh.vao);
            glDrawElements(GL_TRIANGLES, boxMesh.indexCount, GL_UNSIGNED_INT, nullptr);
        }
        glBindVertexArray(0);
    };

    // ── Render loop ──────────────────────────────────────────────────────────
    while (!glfwWindowShouldClose(win))
    {
        float now   = (float)glfwGetTime();
        g_dt        = now - g_lastFrame;
        g_lastFrame = now;

        processInput(win);

        // ────────────────────────────────────────────────────────────────────
        // SHADOW PASSES  (geometry drawn from each light's POV)
        // ────────────────────────────────────────────────────────────────────
        glUseProgram(depthProg);

        // Sun shadow pass
        glViewport(0, 0, sunShadow.width, sunShadow.height);
        glBindFramebuffer(GL_FRAMEBUFFER, sunShadow.fbo);
        glClear(GL_DEPTH_BUFFER_BIT);
        setMat4(depthProg, "lightSpace", sunLightSpace);
        shadowRender(depthProg);

        // Streetlight shadow passes
        for (int i = 0; i < NUM_LIGHTS; i++) {
            glViewport(0, 0, streetlights[i].shadow.width, streetlights[i].shadow.height);
            glBindFramebuffer(GL_FRAMEBUFFER, streetlights[i].shadow.fbo);
            glClear(GL_DEPTH_BUFFER_BIT);
            setMat4(depthProg, "lightSpace", streetlights[i].lightSpace);
            shadowRender(depthProg);
        }

        // ────────────────────────────────────────────────────────────────────
        // MAIN PASS
        // ────────────────────────────────────────────────────────────────────
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, SCR_W, SCR_H);
        glClearColor(0.05f, 0.07f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 view = g_cam.view();
        glm::mat4 proj = glm::perspective(glm::radians(g_cam.fov),
                                          (float)SCR_W / (float)SCR_H,
                                          0.1f, 500.0f);

        // ── Skybox (before everything; no depth write) ────────────────────
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);
        glUseProgram(skyboxProg);
        setMat4(skyboxProg, "projection", proj);
        setMat4(skyboxProg, "view",       glm::mat4(glm::mat3(view)));
        setInt (skyboxProg, "skybox",     0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTex);
        glBindVertexArray(skyboxMesh.vao);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);

        // ── Bind all shadow maps (units 1-5, stay bound for entire scene) ──
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, sunShadow.tex);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, streetlights[0].shadow.tex);
        glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, streetlights[1].shadow.tex);
        glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, streetlights[2].shadow.tex);
        glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, streetlights[3].shadow.tex);

        // ── Scene geometry with terrainProg ────────────────────────────────
        glUseProgram(terrainProg);
        setMat4(terrainProg, "view",       view);
        setMat4(terrainProg, "projection", proj);

        // Convenience lambdas for drawing a box or cone
        auto drawBox = [&](const glm::mat4& m, GLuint tex) {
            setMat4(terrainProg, "model", m);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex);
            glBindVertexArray(boxMesh.vao);
            glDrawElements(GL_TRIANGLES, boxMesh.indexCount, GL_UNSIGNED_INT, nullptr);
        };
        auto drawCone = [&](const glm::mat4& m, GLuint tex) {
            setMat4(terrainProg, "model", m);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex);
            glBindVertexArray(coneMesh.vao);
            glDrawElements(GL_TRIANGLES, coneMesh.indexCount, GL_UNSIGNED_INT, nullptr);
        };

        // Ground plane
        setMat4(terrainProg, "model", glm::mat4(1.0f));
        setBool(terrainProg, "heightBlend", false);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, grassTex);
        glBindVertexArray(groundMesh.vao);
        glDrawElements(GL_TRIANGLES, groundMesh.indexCount, GL_UNSIGNED_INT, nullptr);

        // Terrain hills
        setMat4(terrainProg, "model", glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.05f, 0.0f)));
        setBool(terrainProg, "heightBlend", true);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, terrainTex);
        glBindVertexArray(terrainMesh.vao);
        glDrawElements(GL_TRIANGLES, terrainMesh.indexCount, GL_UNSIGNED_INT, nullptr);

        // Circuit
        setMat4(terrainProg, "model", glm::mat4(1.0f));
        setBool(terrainProg, "heightBlend", false);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, roadTex);
        glBindVertexArray(circuitMesh.vao);
        glDrawElements(GL_TRIANGLES, circuitMesh.indexCount, GL_UNSIGNED_INT, nullptr);

        // Buildings
        setBool(terrainProg, "heightBlend", false);
        for (const auto& b : buildings) {
            drawBox(glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(b.x, 0.0f, b.z)),
                               glm::vec3(b.w, b.h, b.d)), buildingTex);
        }

        // Trees
        for (const auto& tr : trees) {
            glm::vec3 base(tr.x, 0.0f, tr.z);
            drawBox(glm::scale(glm::translate(glm::mat4(1.0f), base),
                               glm::vec3(0.25f, 1.5f, 0.25f)), barkTex);
            drawCone(glm::scale(glm::translate(glm::mat4(1.0f), base + glm::vec3(0.0f, 1.5f, 0.0f)),
                                glm::vec3(0.9f, 1.8f, 0.9f)), leafTex);
        }

        // Streetlight poles + lamp heads
        for (int i = 0; i < NUM_LIGHTS; i++) {
            glm::vec3 poleBase(streetlights[i].position.x, 0.0f, streetlights[i].position.z);
            float     lampH = streetlights[i].position.y;

            // Pole (thin vertical box from ground to lamp height)
            drawBox(glm::scale(glm::translate(glm::mat4(1.0f), poleBase),
                               glm::vec3(0.18f, lampH, 0.18f)), buildingTex);

            // Lamp head (flat box at top of pole)
            drawBox(glm::scale(glm::translate(glm::mat4(1.0f),
                               glm::vec3(streetlights[i].position.x, lampH, streetlights[i].position.z)),
                               glm::vec3(0.55f, 0.20f, 0.55f)), lampTex);
        }

        glBindVertexArray(0);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    glDeleteVertexArrays(1, &skyboxMesh.vao); glDeleteBuffers(1, &skyboxMesh.vbo);
    glDeleteVertexArrays(1, &groundMesh.vao);  glDeleteBuffers(1, &groundMesh.vbo);  glDeleteBuffers(1, &groundMesh.ebo);
    glDeleteVertexArrays(1, &terrainMesh.vao); glDeleteBuffers(1, &terrainMesh.vbo); glDeleteBuffers(1, &terrainMesh.ebo);
    glDeleteVertexArrays(1, &circuitMesh.vao); glDeleteBuffers(1, &circuitMesh.vbo); glDeleteBuffers(1, &circuitMesh.ebo);
    glDeleteVertexArrays(1, &boxMesh.vao);     glDeleteBuffers(1, &boxMesh.vbo);     glDeleteBuffers(1, &boxMesh.ebo);
    glDeleteVertexArrays(1, &coneMesh.vao);    glDeleteBuffers(1, &coneMesh.vbo);    glDeleteBuffers(1, &coneMesh.ebo);

    glDeleteTextures(1, &skyboxTex);
    glDeleteTextures(1, &grassTex);
    glDeleteTextures(1, &terrainTex);
    glDeleteTextures(1, &roadTex);
    glDeleteTextures(1, &buildingTex);
    glDeleteTextures(1, &leafTex);
    glDeleteTextures(1, &barkTex);
    glDeleteTextures(1, &lampTex);

    glDeleteFramebuffers(1, &sunShadow.fbo); glDeleteTextures(1, &sunShadow.tex);
    for (int i = 0; i < NUM_LIGHTS; i++) {
        glDeleteFramebuffers(1, &streetlights[i].shadow.fbo);
        glDeleteTextures(1,    &streetlights[i].shadow.tex);
    }

    glDeleteProgram(skyboxProg);
    glDeleteProgram(terrainProg);
    glDeleteProgram(depthProg);

    glfwTerminate();
    return 0;
}
