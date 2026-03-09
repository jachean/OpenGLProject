// =============================================================================
// GPS Project  –  P1: Scene within a cube
//
// What is rendered:
//   1. Skybox  – large cube surrounding the whole scene.
//               Top face  : sky (gradient from horizon-blue to zenith-blue).
//               Side faces: horizon (sky gradient above a mountain silhouette).
//               Bottom face: dark (not visible under the ground plane).
//   2. Ground  – flat 36×36-unit grass plane at y = 0.
//   3. Terrain – 60×60 height-map mesh with rolling hills, inside the scene.
//
// All textures are generated procedurally at startup – no image files are
// needed to run the project.  To replace them with real images, see the
// texture-loading functions at the bottom of this file.
//
// Controls:
//   W / A / S / D   – move forward / left / back / right
//   Q / E           – move down / up
//   Mouse           – look around (cursor is hidden/captured)
//   Scroll wheel    – zoom (change FOV)
//   ESC             – quit
//
// Dependencies (installed via NuGet):
//   nupengl.core   (GLFW 3, GLEW, freeglut – headers + libs auto-linked)
//   glm            (math)
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
static const char* TITLE = "GPS – P2: Street Circuit & Static Objects";

// ─────────────────────────────────────────────────────────────────────────────
// Camera
// ─────────────────────────────────────────────────────────────────────────────
struct Camera
{
    glm::vec3 position;
    glm::vec3 front, up, right;
    glm::vec3 worldUp;

    float yaw   = -90.0f;   // look along -Z initially
    float pitch = -10.0f;
    float speed = 8.0f;
    float sensitivity = 0.10f;
    float fov   = 60.0f;

    explicit Camera(glm::vec3 pos = glm::vec3(0.0f, 2.5f, 10.0f))
        : position(pos), worldUp(0.0f, 1.0f, 0.0f)
    {
        updateVectors();
    }

    void updateVectors()
    {
        front.x = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
        front.y = std::sin(glm::radians(pitch));
        front.z = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
        front  = glm::normalize(front);
        right  = glm::normalize(glm::cross(front, worldUp));
        up     = glm::normalize(glm::cross(right, front));
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

// Convenience: set uniforms by name
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
// Procedural texture generation
//
// All textures are built entirely on the CPU and uploaded to OpenGL.
// No external image files are required.
//
// To use real image files instead (e.g. PNG / JPEG), include stb_image.h
// (public domain, https://github.com/nothings/stb) and replace these
// functions with stbi_load() calls.
// ─────────────────────────────────────────────────────────────────────────────

// Simple deterministic hash noise in [0, 1] for integer inputs
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

// Grass texture (ground plane)
static GLuint makeGrassTexture()
{
    const int W = 256, H = 256;
    std::vector<unsigned char> px(W * H * 3);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float n = 0.5f + 0.5f * noise2(x, y);
            // Green channel dominant, slight stripe variation
            float stripe = 0.90f + 0.10f * std::sin((x + y) * 0.25f);
            int r = (int)(( 35 + 15 * n) * stripe);
            int g = (int)((120 + 40 * n) * stripe);
            int b = (int)(( 20 + 15 * n) * stripe);
            px[(y * W + x) * 3 + 0] = (unsigned char)std::min(r, 255);
            px[(y * W + x) * 3 + 1] = (unsigned char)std::min(g, 255);
            px[(y * W + x) * 3 + 2] = (unsigned char)std::min(b, 255);
        }
    }
    return makeTexture2D(W, H, px);
}

// Terrain / soil texture (used as base for the height-map terrain)
static GLuint makeTerrainTexture()
{
    const int W = 256, H = 256;
    std::vector<unsigned char> px(W * H * 3);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float n = 0.5f + 0.5f * noise2(x * 3, y * 3 + 17);
            int r = (int)(130 + 30 * n);
            int g = (int)(100 + 25 * n);
            int b = (int)( 60 + 20 * n);
            px[(y * W + x) * 3 + 0] = (unsigned char)std::min(r, 255);
            px[(y * W + x) * 3 + 1] = (unsigned char)std::min(g, 255);
            px[(y * W + x) * 3 + 2] = (unsigned char)std::min(b, 255);
        }
    }
    return makeTexture2D(W, H, px);
}

// Road / asphalt texture – dark gray with a dashed centre line and edge markings.
// U direction = along the road, V direction = inner(0) to outer(1) edge.
static GLuint makeRoadTexture()
{
    const int W = 512, H = 128;
    std::vector<unsigned char> px(W * H * 3);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float fv = (float)y / H;
            float n  = 0.5f + 0.5f * noise2(x * 7, y * 7 + 31);
            int r = (int)(36 + 10 * n);
            int g = (int)(36 + 10 * n);
            int b = (int)(40 + 10 * n);
            // Dashed centre line
            if (std::abs(fv - 0.5f) < 0.04f && (x % 32) < 20)
                r = g = b = 230;
            // Edge stripes
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
    }
    return makeTexture2D(W, H, px);
}

// Building texture – concrete wall with a grid of tinted windows.
static GLuint makeBuildingTexture()
{
    const int W = 256, H = 256;
    std::vector<unsigned char> px(W * H * 3);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float n = 0.5f + 0.5f * noise2(x, y + 500);
            bool winRow = ((y % 32) >= 4  && (y % 32) <= 24);
            bool winCol = ((x % 24) >= 3  && (x % 24) <= 18);
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
    }
    return makeTexture2D(W, H, px);
}

// Leaf texture – varied greens for the tree crown cone.
static GLuint makeLeafTexture()
{
    const int W = 128, H = 128;
    std::vector<unsigned char> px(W * H * 3);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float n = 0.5f + 0.5f * noise2(x * 3, y * 3 + 200);
            px[(y*W+x)*3+0] = (unsigned char)(22  + (int)(18 * n));
            px[(y*W+x)*3+1] = (unsigned char)(100 + (int)(55 * n));
            px[(y*W+x)*3+2] = (unsigned char)(18  + (int)(14 * n));
        }
    }
    return makeTexture2D(W, H, px);
}

// Bark texture – brown streaks for the tree trunk box.
static GLuint makeBarkTexture()
{
    const int W = 64, H = 128;
    std::vector<unsigned char> px(W * H * 3);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float n = 0.5f + 0.5f * noise2(x, y * 2 + 700);
            float s = 0.5f + 0.5f * std::sin(y * 0.9f + n * 2.0f);
            px[(y*W+x)*3+0] = (unsigned char)(78  + (int)(42 * s + 14 * n));
            px[(y*W+x)*3+1] = (unsigned char)(52  + (int)(28 * s + 10 * n));
            px[(y*W+x)*3+2] = (unsigned char)(28  + (int)(16 * s +  8 * n));
        }
    }
    return makeTexture2D(W, H, px);
}

// Cubemap skybox:
//   +Y (top)     – sky gradient (light blue near horizon, deep blue at zenith)
//   -Y (bottom)  – dark (never visible)
//   ±X, ±Z (sides) – horizon scene: sky gradient above a procedural mountain ridge
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
        bool isTop    = (fi == 2);  // +Y
        bool isBottom = (fi == 3);  // -Y

        // The horizontal offset shifts the mountain profile on each face so
        // they don't look identical.
        float phaseShift = fi * 1.3f;

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float u = (float)x / (W - 1);   // [0, 1] horizontal
                float v = (float)y / (H - 1);   // [0, 1] vertical
                // OpenGL cubemap face memory layout (side faces):
                //   v = 0  →  bottom of pixel array  →  looking UP   (sky)
                //   v = 0.5 →  middle               →  horizon
                //   v = 1  →  top of pixel array    →  looking DOWN  (ground)

                if (isTop) {
                    // +Y face: radial gradient, darkest at centre (zenith),
                    // lighter toward edges where it meets the side faces.
                    float cx   = 2.0f * u - 1.0f;
                    float cy   = 2.0f * v - 1.0f;
                    float dist = std::sqrt(cx * cx + cy * cy);
                    float t    = std::min(dist / 1.2f, 1.0f);  // 0=zenith, 1=edge
                    int r = (int)( 40 + t * 100);
                    int g = (int)( 90 + t * 100);
                    int b = (int)(185 + t *  50);
                    set(x, y, r, g, b);
                }
                else if (isBottom) {
                    set(x, y, 15, 12, 10);   // dark, never visible under ground
                }
                else {
                    // Side faces  (horizon panorama):
                    //   v < ridgeV  →  sky   (camera looking upward)
                    //   v = ridgeV  →  mountain ridge (at / just above horizon)
                    //   v > ridgeV  →  mountain rock / ground (below horizon)
                    //
                    // ridgeV ≈ 0.42 places the ridge just above the v=0.5 horizon
                    // so it is clearly visible when looking straight ahead.
                    float ridgeV = 0.42f
                        + 0.06f * std::sin(u * (float)M_PI * 4.0f + phaseShift)
                        + 0.04f * std::sin(u * (float)M_PI * 9.0f + phaseShift * 1.7f)
                        + 0.03f * std::cos(u * (float)M_PI * 6.5f + phaseShift * 0.9f);
                    ridgeV = std::max(0.30f, std::min(ridgeV, 0.58f));

                    if (v < ridgeV) {
                        // Sky zone: from v=0 (zenith) to the mountain ridge
                        float t  = v / ridgeV;           // 0=zenith, 1=ridge
                        int r = (int)( 50 + t * 95);     // deep blue → horizon blue
                        int g = (int)(100 + t * 95);
                        int b = (int)(195 + t * 40);
                        set(x, y, r, g, b);
                    }
                    else {
                        // Mountain / ground zone: from ridge down to v=1
                        float t  = (v - ridgeV) / (1.0f - ridgeV);  // 0=peak, 1=base
                        float n  = 0.5f + 0.5f * noise2(x + fi * 512, y);
                        int r = (int)((88 + t * 45) * (0.85f + 0.15f * n));
                        int g = (int)((75 + t * 30) * (0.85f + 0.15f * n));
                        int b = (int)((58 + t * 18) * (0.85f + 0.15f * n));
                        set(x, y, r, g, b);
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
// Skybox mesh  (unit cube, 36 vertices, positions only)
// ─────────────────────────────────────────────────────────────────────────────
static SkyboxMesh createSkyboxMesh()
{
    // Standard skybox cube from LearnOpenGL.com (CC0)
    static const float v[] = {
        // +X
        -1.0f,  1.0f, -1.0f,  -1.0f, -1.0f, -1.0f,   1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,   1.0f,  1.0f, -1.0f,  -1.0f,  1.0f, -1.0f,
        // -X
        -1.0f, -1.0f,  1.0f,  -1.0f, -1.0f, -1.0f,  -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,  -1.0f,  1.0f,  1.0f,  -1.0f, -1.0f,  1.0f,
        // +Y
         1.0f, -1.0f, -1.0f,   1.0f, -1.0f,  1.0f,   1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,   1.0f,  1.0f, -1.0f,   1.0f, -1.0f, -1.0f,
        // -Y
        -1.0f, -1.0f,  1.0f,  -1.0f,  1.0f,  1.0f,   1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,   1.0f, -1.0f,  1.0f,  -1.0f, -1.0f,  1.0f,
        // Top
        -1.0f,  1.0f, -1.0f,   1.0f,  1.0f, -1.0f,   1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,  -1.0f,  1.0f,  1.0f,  -1.0f,  1.0f, -1.0f,
        // Bottom
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
// Ground plane  (flat quad, pos + texcoord + normal)
// ─────────────────────────────────────────────────────────────────────────────
static Mesh createGroundMesh(float halfSize)
{
    float s = halfSize;
    float r = halfSize / 4.0f;   // texture repeat count (tile)

    // Layout per vertex:  pos.xyz (3)  texcoord.uv (2)  normal.xyz (3)
    float verts[] = {
        -s, 0.0f, -s,   0.0f,  r,   0.0f, 1.0f, 0.0f,
         s, 0.0f, -s,   r,     r,   0.0f, 1.0f, 0.0f,
         s, 0.0f,  s,   r,     0.0f, 0.0f, 1.0f, 0.0f,
        -s, 0.0f,  s,   0.0f,  0.0f, 0.0f, 1.0f, 0.0f,
    };
    unsigned int idx[] = { 0, 1, 2,  2, 3, 0 };

    Mesh m{};
    m.indexCount = 6;
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glGenBuffers(1, &m.ebo);

    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Terrain mesh  (height-map grid with rolling hills)
//
//   gridSize  – number of quads per axis  (total vertices = (gridSize+1)^2)
//   scale     – half-extent of the mesh in world units
//   amplitude – maximum hill height in world units
//   freq      – spatial frequency of the sine waves
// ─────────────────────────────────────────────────────────────────────────────
static Mesh createTerrainMesh(int gridSize, float scale,
                               float amplitude, float freq)
{
    int V = gridSize + 1;   // vertices per axis

    auto getH = [&](float x, float z) -> float {
        return amplitude * (
            0.50f * std::sin(x * freq)        * std::cos(z * freq) +
            0.30f * std::sin(x * freq * 2.1f + 0.40f) +
            0.20f * std::cos(z * freq * 1.7f + 0.80f)
        );
    };

    // Build vertex buffer: pos(3), texcoord(2), normal(3)
    std::vector<float> verts(V * V * 8);

    for (int j = 0; j <= gridSize; j++) {
        for (int i = 0; i <= gridSize; i++) {
            float u = (float)i / gridSize;
            float w = (float)j / gridSize;
            float x = (u - 0.5f) * 2.0f * scale;
            float z = (w - 0.5f) * 2.0f * scale;
            float y = getH(x, z);

            int base = (j * V + i) * 8;
            verts[base + 0] = x;
            verts[base + 1] = y;
            verts[base + 2] = z;
            verts[base + 3] = u * 8.0f;   // texture tiling (8×)
            verts[base + 4] = w * 8.0f;

            // Compute normal from finite differences
            const float eps = 0.05f;
            float dhdx = (getH(x + eps, z) - getH(x - eps, z)) / (2.0f * eps);
            float dhdz = (getH(x, z + eps) - getH(x, z - eps)) / (2.0f * eps);
            glm::vec3 n = glm::normalize(glm::vec3(-dhdx, 1.0f, -dhdz));
            verts[base + 5] = n.x;
            verts[base + 6] = n.y;
            verts[base + 7] = n.z;
        }
    }

    // Build index buffer (two triangles per quad, CCW winding)
    std::vector<unsigned int> indices;
    indices.reserve(gridSize * gridSize * 6);
    for (int j = 0; j < gridSize; j++) {
        for (int i = 0; i < gridSize; i++) {
            unsigned int a = j       * V + i;
            unsigned int b = j       * V + i + 1;
            unsigned int c = (j + 1) * V + i;
            unsigned int d = (j + 1) * V + i + 1;
            indices.push_back(a); indices.push_back(b); indices.push_back(d);
            indices.push_back(a); indices.push_back(d); indices.push_back(c);
        }
    }

    Mesh m{};
    m.indexCount = (int)indices.size();
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glGenBuffers(1, &m.ebo);

    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(verts.size()   * sizeof(float)),        verts.data(),   GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (GLsizeiptr)(indices.size() * sizeof(unsigned int)),  indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Oval street circuit mesh
// ─────────────────────────────────────────────────────────────────────────────
static Mesh createCircuitMesh(float semiX, float semiZ,
                               float roadHalfWidth, int segments)
{
    float totalLen = 0.0f;
    {
        float px = semiX, pz = 0.0f;
        for (int i = 1; i <= segments; i++) {
            float t  = 2.0f * (float)M_PI * i / segments;
            float cx = semiX * std::cos(t), cz = semiZ * std::sin(t);
            float dx = cx - px, dz = cz - pz;
            totalLen += std::sqrt(dx*dx + dz*dz);
            px = cx; pz = cz;
        }
    }

    std::vector<float>        verts;
    std::vector<unsigned int> indices;
    float arcLen = 0.0f, prevCx = semiX, prevCz = 0.0f;

    for (int i = 0; i <= segments; i++) {
        float t  = 2.0f * (float)M_PI * i / segments;
        float cx = semiX * std::cos(t), cz = semiZ * std::sin(t);
        if (i > 0) {
            float dx = cx - prevCx, dz = cz - prevCz;
            arcLen += std::sqrt(dx*dx + dz*dz);
        }
        float u = arcLen / (roadHalfWidth * 2.0f);

        float nx = std::cos(t) / semiX, nz = std::sin(t) / semiZ;
        float nl = std::sqrt(nx*nx + nz*nz);
        nx /= nl;  nz /= nl;

        float ix = cx - nx * roadHalfWidth, iz = cz - nz * roadHalfWidth;
        verts.insert(verts.end(), {ix, 0.02f, iz,  u, 0.0f,  0.0f, 1.0f, 0.0f});
        float ox = cx + nx * roadHalfWidth, oz = cz + nz * roadHalfWidth;
        verts.insert(verts.end(), {ox, 0.02f, oz,  u, 1.0f,  0.0f, 1.0f, 0.0f});

        prevCx = cx;  prevCz = cz;
    }

    for (int i = 0; i < segments; i++) {
        unsigned int a = i*2, b = i*2+1, c = (i+1)*2, d = (i+1)*2+1;
        indices.push_back(a); indices.push_back(c); indices.push_back(b);
        indices.push_back(b); indices.push_back(c); indices.push_back(d);
    }

    Mesh m{};
    m.indexCount = (int)indices.size();
    glGenVertexArrays(1, &m.vao); glGenBuffers(1, &m.vbo); glGenBuffers(1, &m.ebo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size()*sizeof(float)), verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(indices.size()*sizeof(unsigned int)), indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);              glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(5*sizeof(float))); glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Box mesh – buildings and tree trunks.  Origin at centre of base (y=0 to h).
// ─────────────────────────────────────────────────────────────────────────────
static Mesh createBoxMesh(float w, float h, float d)
{
    float hw = w*0.5f, hd = d*0.5f;
    std::vector<float> verts = {
        -hw,0, hd, 0,0, 0,0,1,   hw,0, hd, 1,0, 0,0,1,   hw,h, hd, 1,1, 0,0,1,  -hw,h, hd, 0,1, 0,0,1,
         hw,0,-hd, 0,0, 0,0,-1, -hw,0,-hd, 1,0, 0,0,-1, -hw,h,-hd, 1,1, 0,0,-1,  hw,h,-hd, 0,1, 0,0,-1,
        -hw,0,-hd, 0,0,-1,0,0, -hw,0, hd, 1,0,-1,0,0, -hw,h, hd, 1,1,-1,0,0, -hw,h,-hd, 0,1,-1,0,0,
         hw,0, hd, 0,0, 1,0,0,  hw,0,-hd, 1,0, 1,0,0,  hw,h,-hd, 1,1, 1,0,0,  hw,h, hd, 0,1, 1,0,0,
        -hw,h, hd, 0,0, 0,1,0,  hw,h, hd, 1,0, 0,1,0,  hw,h,-hd, 1,1, 0,1,0, -hw,h,-hd, 0,1, 0,1,0,
        -hw,0,-hd, 0,0, 0,-1,0,  hw,0,-hd, 1,0, 0,-1,0,  hw,0, hd, 1,1, 0,-1,0, -hw,0, hd, 0,1, 0,-1,0,
    };
    unsigned int idx[36];
    for (int f=0; f<6; f++) {
        unsigned int b=f*4;
        idx[f*6+0]=b; idx[f*6+1]=b+1; idx[f*6+2]=b+2;
        idx[f*6+3]=b; idx[f*6+4]=b+2; idx[f*6+5]=b+3;
    }
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
// Cone mesh – tree crown.  Apex at (0, height, 0), base at y=0.
// ─────────────────────────────────────────────────────────────────────────────
static Mesh createConeMesh(float radius, float height, int segments)
{
    std::vector<float>        verts;
    std::vector<unsigned int> indices;

    for (int i = 0; i < segments; i++) {
        float a0 = 2.0f*(float)M_PI*i/segments, a1 = 2.0f*(float)M_PI*(i+1)/segments;
        float amid = (a0+a1)*0.5f;
        float x0=radius*std::cos(a0), z0=radius*std::sin(a0);
        float x1=radius*std::cos(a1), z1=radius*std::sin(a1);
        glm::vec3 n = glm::normalize(glm::vec3(height*std::cos(amid), radius, height*std::sin(amid)));
        unsigned int base = (unsigned int)verts.size()/8;
        verts.insert(verts.end(), {0.0f,height,0.0f, (a0+a1)/(2.0f*(float)M_PI),1.0f, n.x,n.y,n.z});
        verts.insert(verts.end(), {x0,0.0f,z0, (float)i/segments,0.0f, n.x,n.y,n.z});
        verts.insert(verts.end(), {x1,0.0f,z1, (float)(i+1)/segments,0.0f, n.x,n.y,n.z});
        indices.push_back(base); indices.push_back(base+2); indices.push_back(base+1);
    }
    // base cap
    unsigned int ci = (unsigned int)verts.size()/8;
    verts.insert(verts.end(), {0.0f,0.0f,0.0f, 0.5f,0.5f, 0.0f,-1.0f,0.0f});
    for (int i=0; i<segments; i++) {
        float a=2.0f*(float)M_PI*i/segments;
        verts.insert(verts.end(), {radius*std::cos(a),0.0f,radius*std::sin(a),
            0.5f+0.5f*std::cos(a), 0.5f+0.5f*std::sin(a), 0.0f,-1.0f,0.0f});
    }
    for (int i=0; i<segments; i++) {
        indices.push_back(ci);
        indices.push_back(ci+1+i);
        indices.push_back(ci+1+(i+1)%segments);
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

    float dx =  ((float)xpos - g_lastX) * g_cam.sensitivity;
    float dy =  (g_lastY - (float)ypos) * g_cam.sensitivity;  // inverted Y
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
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    // ── GLFW init ────────────────────────────────────────────────────────────
    if (!glfwInit()) {
        std::cerr << "Failed to initialise GLFW\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(SCR_W, SCR_H, TITLE, nullptr, nullptr);
    if (!win) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(win);
    glfwSetFramebufferSizeCallback(win, cbFramebufferSize);
    glfwSetCursorPosCallback(win,       cbMouse);
    glfwSetScrollCallback(win,          cbScroll);
    glfwSetInputMode(win, GLFW_CURSOR,  GLFW_CURSOR_DISABLED);

    // ── GLEW init ────────────────────────────────────────────────────────────
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialise GLEW\n";
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    // ── Shaders ──────────────────────────────────────────────────────────────
    GLuint skyboxProg  = createProgram("shaders/skybox.vert",  "shaders/skybox.frag");
    GLuint terrainProg = createProgram("shaders/terrain.vert", "shaders/terrain.frag");

    // ── Meshes ───────────────────────────────────────────────────────────────
    SkyboxMesh skyboxMesh = createSkyboxMesh();
    Mesh       groundMesh = createGroundMesh(18.0f);
    Mesh       terrainMesh = createTerrainMesh(64, 6.0f, 1.5f, 0.35f);  // scaled to fit inside circuit
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

    // ── Constant uniforms ────────────────────────────────────────────────────
    glm::vec3 lightDir   = glm::normalize(glm::vec3(-0.4f, -1.0f, -0.5f));
    glm::vec3 lightColor = glm::vec3(1.0f, 0.97f, 0.90f);

    // ── Render loop ──────────────────────────────────────────────────────────
    while (!glfwWindowShouldClose(win))
    {
        float now = (float)glfwGetTime();
        g_dt       = now - g_lastFrame;
        g_lastFrame = now;

        processInput(win);

        glClearColor(0.05f, 0.07f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 view = g_cam.view();
        glm::mat4 proj = glm::perspective(
            glm::radians(g_cam.fov),
            (float)SCR_W / (float)SCR_H,
            0.1f, 500.0f
        );

        // ── Skybox ───────────────────────────────────────────────────────────
        // Draw before everything else with depth writes off so it sits behind
        // all scene objects.  Strip translation from the view matrix so the
        // camera always stays at the centre of the sky cube.
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);

        glUseProgram(skyboxProg);
        glm::mat4 skyView = glm::mat4(glm::mat3(view));  // remove translation
        setMat4(skyboxProg, "projection", proj);
        setMat4(skyboxProg, "view",       skyView);
        setInt (skyboxProg, "skybox",     0);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTex);
        glBindVertexArray(skyboxMesh.vao);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);

        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);

        // ── Ground plane ─────────────────────────────────────────────────────
        glm::mat4 model = glm::mat4(1.0f);

        glUseProgram(terrainProg);
        setMat4(terrainProg, "model",      model);
        setMat4(terrainProg, "view",       view);
        setMat4(terrainProg, "projection", proj);
        setVec3(terrainProg, "lightDir",   lightDir);
        setVec3(terrainProg, "lightColor", lightColor);
        setInt (terrainProg, "texture1",   0);
        setBool(terrainProg, "heightBlend", false);  // flat ground → no height blend

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, grassTex);
        glBindVertexArray(groundMesh.vao);
        glDrawElements(GL_TRIANGLES, groundMesh.indexCount, GL_UNSIGNED_INT, nullptr);

        // ── Terrain (infield hills) ───────────────────────────────────────────
        glm::mat4 terrainModel = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.05f, 0.0f));
        setMat4(terrainProg, "model",       terrainModel);
        setBool(terrainProg, "heightBlend", true);
        glBindTexture(GL_TEXTURE_2D, terrainTex);
        glBindVertexArray(terrainMesh.vao);
        glDrawElements(GL_TRIANGLES, terrainMesh.indexCount, GL_UNSIGNED_INT, nullptr);

        // ── Circuit ───────────────────────────────────────────────────────────
        setMat4(terrainProg, "model",       glm::mat4(1.0f));
        setBool(terrainProg, "heightBlend", false);
        glBindTexture(GL_TEXTURE_2D, roadTex);
        glBindVertexArray(circuitMesh.vao);
        glDrawElements(GL_TRIANGLES, circuitMesh.indexCount, GL_UNSIGNED_INT, nullptr);

        // ── Static objects ────────────────────────────────────────────────────
        auto drawBox = [&](const glm::mat4& m, GLuint tex) {
            setMat4(terrainProg, "model", m);
            glBindTexture(GL_TEXTURE_2D, tex);
            glBindVertexArray(boxMesh.vao);
            glDrawElements(GL_TRIANGLES, boxMesh.indexCount, GL_UNSIGNED_INT, nullptr);
        };
        auto drawCone = [&](const glm::mat4& m, GLuint tex) {
            setMat4(terrainProg, "model", m);
            glBindTexture(GL_TEXTURE_2D, tex);
            glBindVertexArray(coneMesh.vao);
            glDrawElements(GL_TRIANGLES, coneMesh.indexCount, GL_UNSIGNED_INT, nullptr);
        };
        setBool(terrainProg, "heightBlend", false);

        // 6 buildings
        struct BldgDef { float x, z, w, h, d; };
        static const BldgDef buildings[] = {
            { -14.5f, -4.0f,  2.5f, 5.5f, 2.0f },
            { -14.5f,  0.0f,  2.5f, 7.0f, 2.0f },
            { -14.5f,  4.0f,  2.5f, 5.0f, 2.0f },
            {  15.0f, -3.0f,  4.0f, 3.0f, 3.5f },
            {  15.0f,  3.0f,  4.0f, 3.0f, 3.5f },
            {   0.0f, 11.5f,  1.5f, 8.0f, 1.5f },
        };
        for (const auto& b : buildings) {
            glm::mat4 bm = glm::scale(
                glm::translate(glm::mat4(1.0f), glm::vec3(b.x, 0.0f, b.z)),
                glm::vec3(b.w, b.h, b.d));
            drawBox(bm, buildingTex);
        }

        // 5 trees (trunk + crown)
        struct TreePos { float x, z; };
        static const TreePos trees[] = {
            {  3.0f,  2.0f }, { -4.0f,  3.5f }, {  2.0f, -4.0f },
            { -3.5f, -3.0f }, {  5.0f,  0.5f },
        };
        for (const auto& tr : trees) {
            glm::vec3 base(tr.x, 0.0f, tr.z);
            drawBox(glm::scale(glm::translate(glm::mat4(1.0f), base),
                               glm::vec3(0.25f, 1.5f, 0.25f)), barkTex);
            drawCone(glm::scale(glm::translate(glm::mat4(1.0f), base + glm::vec3(0.0f, 1.5f, 0.0f)),
                                glm::vec3(0.9f, 1.8f, 0.9f)), leafTex);
        }

        glBindVertexArray(0);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    glDeleteVertexArrays(1, &skyboxMesh.vao);
    glDeleteBuffers(1, &skyboxMesh.vbo);
    glDeleteVertexArrays(1, &groundMesh.vao);
    glDeleteBuffers(1, &groundMesh.vbo);
    glDeleteBuffers(1, &groundMesh.ebo);
    glDeleteVertexArrays(1, &terrainMesh.vao);
    glDeleteBuffers(1, &terrainMesh.vbo);
    glDeleteBuffers(1, &terrainMesh.ebo);
    glDeleteTextures(1, &skyboxTex);
    glDeleteTextures(1, &grassTex);
    glDeleteTextures(1, &terrainTex);
    glDeleteTextures(1, &roadTex);
    glDeleteTextures(1, &buildingTex);
    glDeleteTextures(1, &leafTex);
    glDeleteTextures(1, &barkTex);
    glDeleteVertexArrays(1, &circuitMesh.vao); glDeleteBuffers(1, &circuitMesh.vbo); glDeleteBuffers(1, &circuitMesh.ebo);
    glDeleteVertexArrays(1, &boxMesh.vao);     glDeleteBuffers(1, &boxMesh.vbo);     glDeleteBuffers(1, &boxMesh.ebo);
    glDeleteVertexArrays(1, &coneMesh.vao);    glDeleteBuffers(1, &coneMesh.vbo);    glDeleteBuffers(1, &coneMesh.ebo);
    glDeleteProgram(skyboxProg);
    glDeleteProgram(terrainProg);

    glfwTerminate();
    return 0;
}
