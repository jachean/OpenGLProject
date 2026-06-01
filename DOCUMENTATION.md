# GPS Project – Full Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [File Structure](#2-file-structure)
3. [How OpenGL Works (Primer)](#3-how-opengl-works-primer)
4. [main.cpp – Section by Section](#4-maincpp--section-by-section)
   - [Window Constants](#41-window-constants)
   - [Camera System](#42-camera-system)
   - [Global State](#43-global-state)
   - [Shader Utilities](#44-shader-utilities)
   - [Uniform Helpers](#45-uniform-helpers)
   - [Procedural Textures](#46-procedural-textures)
   - [Mesh Data Types](#47-mesh-data-types)
   - [Shadow Map](#48-shadow-map)
   - [Streetlight](#49-streetlight)
   - [Mesh Creation Functions](#410-mesh-creation-functions)
   - [GLFW Callbacks](#411-glfw-callbacks)
   - [Input Processing](#412-input-processing)
   - [main() – Initialization](#413-main--initialization)
   - [main() – Render Loop](#414-main--render-loop)
5. [Shader Files](#5-shader-files)
   - [skybox.vert / skybox.frag](#51-skyboxvert--skyboxfrag)
   - [terrain.vert](#52-terrainvert)
   - [terrain.frag](#53-terrainfrag)
   - [depth.vert / depth.frag](#54-depthvert--depthfrag)
6. [Shadow Mapping Explained](#6-shadow-mapping-explained)
7. [Controls Reference](#7-controls-reference)

---

## 1. Project Overview

This is an OpenGL 3.3 real-time 3D scene built for the Graphics Processing Systems course.
It is written in C++ using:

| Library | What it provides |
|---------|-----------------|
| **GLFW** | Creates the OS window and OpenGL context; receives keyboard/mouse events |
| **GLEW** | Loads OpenGL function pointers at runtime (required on Windows) |
| **GLM** | Math library (vectors, matrices, trigonometry) used for 3D transforms |
| **opengl32.lib** | The base Windows OpenGL library |

All textures are generated **procedurally on the CPU** — no image files are needed.
The project is split across three lab milestones:

| Branch | Contents |
|--------|----------|
| `master` | P1 – Skybox, ground plane, terrain hills, FPS camera |
| `P2` | P2 – Oval street circuit, 6 buildings, 5 trees |
| `P3` | P3 – Camera roll, directional sun shadow, 4 streetlights with shadows |

---

## 2. File Structure

```
GPSproject/
├── main.cpp                     ← All C++ code lives here
├── shaders/
│   ├── skybox.vert              ← Vertex shader for the sky cube
│   ├── skybox.frag              ← Fragment shader for the sky cube
│   ├── terrain.vert             ← Vertex shader for all scene geometry
│   ├── terrain.frag             ← Fragment shader – lighting + shadows
│   ├── depth.vert               ← Vertex shader for shadow passes
│   └── depth.frag               ← Fragment shader for shadow passes
├── GPSproject.vcxproj           ← Visual Studio project file
├── GPSproject.vcxproj.filters   ← VS Solution Explorer groupings
├── packages.config              ← NuGet package list
└── DOCUMENTATION.md             ← This file
```

---

## 3. How OpenGL Works (Primer)

Before reading the code, it helps to understand a few OpenGL concepts.

### The GPU Pipeline

When you draw something, the GPU runs it through a fixed pipeline:

```
CPU sends vertex data
        ↓
[ Vertex Shader ]   ← runs once per vertex, outputs clip-space position
        ↓
[ Rasterization ]   ← GPU fills in pixels between vertices (interpolates)
        ↓
[ Fragment Shader ] ← runs once per pixel, outputs final colour
        ↓
[ Depth Test ]      ← discards pixels hidden behind closer objects
        ↓
[ Framebuffer ]     ← pixels written to the screen (or a texture)
```

### VAO, VBO, EBO

- **VBO (Vertex Buffer Object)** – a block of memory on the GPU holding raw vertex data (positions, texture coordinates, normals).
- **EBO (Element Buffer Object)** – a list of indices that say which vertices form each triangle. Lets you reuse vertices without duplicating them.
- **VAO (Vertex Array Object)** – remembers the layout of a VBO (which floats mean position, which mean texture coord, etc.). You bind the VAO once to describe the layout, then just bind it again to draw.

### Uniforms

A **uniform** is a variable you set in C++ that is readable inside a shader. It is the same value for every vertex/fragment in a single draw call. Examples: the camera matrix, the light direction, the texture sampler.

### Texture Units

OpenGL has numbered "slots" called **texture units** (GL_TEXTURE0, GL_TEXTURE1, …). You bind a texture to a unit, then tell the shader which unit to sample from. This lets a single draw call read multiple textures at once (e.g., the albedo texture on unit 0, the shadow map on unit 1).

### Framebuffer Objects (FBO)

By default, drawing goes to the screen. An **FBO** redirects drawing into a texture instead. Shadow mapping works by drawing the scene into a depth-only FBO from the light's point of view.

---

## 4. main.cpp – Section by Section

### 4.1 Window Constants

```cpp
static const int   SCR_W = 1280;
static const int   SCR_H = 720;
static const char* TITLE = "GPS – P3: ...";
```

These set the pixel dimensions of the window and the title bar text. They are used when creating the GLFW window and when computing the aspect ratio for the perspective projection matrix.

---

### 4.2 Camera System

```cpp
struct Camera { ... };
```

The Camera struct represents the player's viewpoint. It stores:

| Field | Type | Meaning |
|-------|------|---------|
| `position` | vec3 | World-space coordinates of the camera (x, y, z) |
| `front` | vec3 | Unit vector pointing in the direction the camera looks |
| `up` | vec3 | Unit vector pointing "upward" from the camera's perspective |
| `right` | vec3 | Unit vector pointing to the camera's right |
| `worldUp` | vec3 | Fixed world "up" direction, always (0, 1, 0) |
| `yaw` | float | Horizontal rotation in degrees (-90 = look along -Z) |
| `pitch` | float | Vertical rotation in degrees (negative = look slightly down) |
| `roll` | float | Roll rotation in degrees (tilts the horizon) |
| `speed` | float | Movement speed in world units per second |
| `sensitivity` | float | How many degrees of rotation per pixel of mouse movement |
| `fov` | float | Field of view in degrees (zoom). 60° is a natural human view. |

#### `updateVectors()`

Called every time yaw, pitch, or roll changes. It recalculates `front`, `right`, and `up`.

**Step 1 – Compute `front` from yaw and pitch:**
```
front.x = cos(yaw) * cos(pitch)
front.y = sin(pitch)
front.z = sin(yaw) * cos(pitch)
```
This converts spherical coordinates (yaw=longitude, pitch=latitude) into a Cartesian direction vector.

**Step 2 – Derive `right` and `up` without roll:**
```
right = normalize(cross(front, worldUp))
up    = normalize(cross(right, front))
```
`cross(A, B)` gives a vector perpendicular to both A and B. This gives us the camera's local axes.

**Step 3 – Apply roll:**
If `roll != 0`, the `right` and `up` vectors are rotated around the `front` axis using a 2D rotation formula:
```
newRight =  cos(roll)*right + sin(roll)*up
newUp    = -sin(roll)*right + cos(roll)*up
```
This tilts the camera sideways without changing where it looks.

#### `view()`

Returns a 4×4 **view matrix** using `glm::lookAt(position, position + front, up)`.

The view matrix transforms world coordinates into camera-relative coordinates. It is the mathematical equivalent of "move the world so the camera is at the origin, looking along -Z".

---

### 4.3 Global State

```cpp
static Camera g_cam;
static bool   g_firstMouse = true;
static float  g_lastX, g_lastY;
static float  g_dt, g_lastFrame;
```

- `g_cam` – the single global camera instance.
- `g_firstMouse` – prevents a large jump on the first mouse movement (before any `lastX/Y` value has been recorded).
- `g_lastX / g_lastY` – the mouse position from the previous frame, used to calculate how far the mouse moved.
- `g_dt` – **delta time**: the number of seconds since the last frame. Used to make movement speed frame-rate independent (multiplying speed by `g_dt` gives the same distance per second regardless of FPS).
- `g_lastFrame` – the timestamp of the previous frame.

These are global (not inside `main`) because they need to be accessible from GLFW callback functions, which cannot receive custom parameters.

---

### 4.4 Shader Utilities

#### `readFile(path)`

Opens a text file (a `.vert` or `.frag` shader file) and returns its entire content as a `std::string`. If the file cannot be opened, it prints an error and returns an empty string.

#### `compileShader(type, src)`

1. Creates an empty shader object on the GPU with `glCreateShader(type)` where type is either `GL_VERTEX_SHADER` or `GL_FRAGMENT_SHADER`.
2. Uploads the GLSL source code text to the GPU.
3. Compiles it with `glCompileShader`.
4. Checks for compile errors and prints them if any.
5. Returns the shader's ID (a number the GPU uses to identify it).

#### `createProgram(vsPath, fsPath)`

A **shader program** links a vertex shader and fragment shader together into a complete GPU pipeline.

1. Reads and compiles the vertex shader from `vsPath`.
2. Reads and compiles the fragment shader from `fsPath`.
3. Creates a program with `glCreateProgram()`.
4. Attaches both shaders.
5. Links them together (`glLinkProgram`) — this connects outputs of the vertex shader to inputs of the fragment shader.
6. Checks for link errors.
7. Deletes the now-unnecessary individual shader objects (they're baked into the program).
8. Returns the program ID.

---

### 4.5 Uniform Helpers

These are small convenience wrappers around OpenGL's verbose uniform-setting functions.

```cpp
setMat4(prog, "model", matrix)   // sets a 4×4 matrix uniform
setVec3(prog, "lightDir", vec)   // sets a 3-float vector uniform
setInt (prog, "texture1", 0)     // sets an integer uniform (used for texture units)
setBool(prog, "heightBlend", true) // sets a bool as integer (0 or 1)
```

Each works the same way:
1. `glGetUniformLocation(prog, name)` – looks up the integer location of the named uniform in the shader.
2. `glUniform*` – sends the value to the GPU.

The program must be currently active (`glUseProgram(prog)`) for these to work.

---

### 4.6 Procedural Textures

All textures are generated entirely by C++ code at startup. No image files are loaded from disk.

#### `noise2(x, y)`

A simple deterministic **hash function** that takes two integers and returns a float in the range [0, 1]. Given the same inputs it always returns the same output, but values for nearby coordinates look random. Used to add variation to textures without patterns.

The formula uses prime number multiplications and bit-shifting to scramble the input. It is not cryptographic — just visually convincing.

#### `makeTexture2D(w, h, pixels)`

Takes a CPU-side array of RGB bytes and uploads it to a GPU texture. Steps:

1. `glGenTextures` – allocates a texture slot on the GPU, returns an ID.
2. `glBindTexture(GL_TEXTURE_2D, id)` – makes this the "current" 2D texture.
3. `glTexImage2D` – uploads the pixel data.
4. `glGenerateMipmap` – automatically creates smaller versions of the texture (mipmaps) for when the object is far away. Prevents shimmering.
5. `glTexParameteri` – sets sampling rules:
   - `WRAP_S/T = GL_REPEAT` – the texture tiles infinitely in U and V directions.
   - `MIN_FILTER = GL_LINEAR_MIPMAP_LINEAR` – smooth blending between mipmap levels.
   - `MAG_FILTER = GL_LINEAR` – smooth when zoomed in.

#### `makeGrassTexture()`

Creates a 256×256 green texture. For each pixel:
- Adds random noise via `noise2` to vary the shade of green.
- Adds a subtle diagonal stripe pattern with `sin((x+y)*0.25)` to simulate mown-grass rows.

#### `makeTerrainTexture()`

Creates a 256×256 brown/sandy texture (used as the base of the terrain hills, then blended with height-based colour in the shader).

#### `makeRoadTexture()`

Creates a 512×128 asphalt texture. The long axis (512) maps along the road direction; the short axis (128) maps across the road width. Features:
- Dark gray base with subtle noise.
- A **dashed centre line**: white pixels where `abs(v - 0.5) < 0.04` AND `(x % 32) < 20` (every 32 texels, a 20-texel white dash, then a 12-texel gap).
- **Edge markings**: a fade to white near `v=0` and `v=1` (the inner and outer road edges).

#### `makeBuildingTexture()`

Creates a 256×256 concrete-and-glass texture. For each pixel:
- Checks if it falls inside a "window" grid using modular arithmetic (`x%24`, `y%32`).
- Windows are blue-gray; walls are light concrete gray.

#### `makeLeafTexture()`

Creates a 128×128 varied-green texture for the tree crowns. Just noise in the green channel.

#### `makeBarkTexture()`

Creates a 64×128 wood-bark texture with vertical streaks, using a combination of noise and a sine wave along the Y axis to simulate bark grain.

#### `makeLampTexture()`

Creates a tiny 4×4 solid warm-yellow texture (RGB 255, 240, 150). Used for the lamp head boxes at the top of each streetlight pole to make them look like glowing bulbs.

#### `makeSkyboxCubemap()`

Creates a **cubemap** — a special texture made of 6 square images that together wrap the inside of a cube. The camera is always at the centre of this cube, so it appears to be an infinitely distant sky.

Each of the 6 faces is generated differently:

- **+Y (top face)**: A radial gradient — dark blue at the centre (zenith, directly overhead) blending to lighter blue at the edges (horizon).
- **-Y (bottom face)**: Near-black. Never visible (the ground covers it).
- **4 side faces** (±X, ±Z): A horizon panorama. Each pixel's V coordinate maps to a vertical angle:
  - `v = 0` → looking **up** (toward zenith)
  - `v = 0.5` → looking straight ahead (horizon)
  - `v = 1` → looking **down**

  A `ridgeV` value (~0.42, plus a sine wave for irregularity) marks the mountain ridge line:
  - Pixels with `v < ridgeV` are sky (blue gradient).
  - Pixels with `v >= ridgeV` are mountain rock (brown-gray).

  Each face uses a different `phaseShift` value so the mountains don't look identical on every side.

---

### 4.7 Mesh Data Types

```cpp
struct SkyboxMesh { GLuint vao, vbo; };

struct Mesh {
    GLuint vao, vbo, ebo;
    int    indexCount;
};
```

- `SkyboxMesh` – only needs a VAO and VBO (no index buffer; 36 vertices written explicitly).
- `Mesh` – the general type for all other geometry. Has a VAO, VBO, EBO, and the number of indices to pass to `glDrawElements`.

All scene geometry uses a fixed **vertex layout** of 8 floats per vertex:
```
[pos.x, pos.y, pos.z,  tex.u, tex.v,  normal.x, normal.y, normal.z]
  0       1      2       3      4        5          6          7
```
This layout is registered in every mesh's VAO using three calls to `glVertexAttribPointer`, binding attributes 0 (position), 1 (texcoord), and 2 (normal).

---

### 4.8 Shadow Map

```cpp
struct ShadowMap {
    GLuint fbo, tex;
    int    width, height;
};
```

A shadow map is a depth texture rendered from a light's point of view.

#### `createShadowMap(w, h)`

1. **Create depth texture**: A `GL_DEPTH_COMPONENT` texture — it stores only a single float per pixel (the depth value). Unlike colour textures, it uses `GL_NEAREST` filtering (no blending between samples) and `GL_CLAMP_TO_BORDER` wrapping (areas outside the light's view default to depth=1.0, meaning "not in shadow").
2. **Create FBO**: A Framebuffer Object that renders into the depth texture instead of the screen. `glDrawBuffer(GL_NONE)` and `glReadBuffer(GL_NONE)` tell OpenGL we only write depth, not colour.

---

### 4.9 Streetlight

```cpp
struct Streetlight {
    glm::vec3 position;    // world position of the lamp head
    glm::vec3 color;       // RGB colour (can be > 1.0 for extra brightness)
    ShadowMap shadow;      // this light's depth FBO
    glm::mat4 lightSpace;  // projection * view from this light's perspective
};
```

Each streetlight has its own shadow map and its own light-space matrix. The `lightSpace` matrix transforms a world-space position into the light's clip space, which is needed both when rendering the shadow map and when sampling it in the fragment shader.

---

### 4.10 Mesh Creation Functions

#### `createSkyboxMesh()`

Creates a unit cube (each side = 2 units, centred at the origin) using 36 manually listed vertices (6 faces × 2 triangles × 3 vertices). Only positions are stored — no texcoords or normals, because the skybox shader uses the vertex position itself as the texture lookup direction.

#### `createGroundMesh(halfSize)`

Creates a flat quad (two triangles forming a rectangle) at Y=0. The quad spans from `-halfSize` to `+halfSize` in both X and Z. Texture coordinates are scaled by `halfSize/4` so the grass texture tiles rather than stretching.

Normals are all pointing straight up (0, 1, 0) since the ground is perfectly flat.

#### `createTerrainMesh(gridSize, scale, amplitude, freq)`

Creates a height-map terrain grid.

**Parameters:**
- `gridSize` – number of quads per axis. A 64×64 grid produces 65×65 = 4,225 vertices.
- `scale` – half the world-space extent of the mesh. With scale=6, the mesh goes from -6 to +6 in X and Z.
- `amplitude` – maximum hill height in world units.
- `freq` – how tightly the hills oscillate. Higher = more hills squeezed in.

**Height function `getH(x, z)`:**

The Y position of each vertex is computed from three overlapping sine/cosine waves at different frequencies. This creates natural-looking rolling hills:
```
y = amplitude * (
    0.50 * sin(x*freq) * cos(z*freq)   ← primary hill shape
  + 0.30 * sin(x*freq*2.1 + 0.4)      ← secondary variation
  + 0.20 * cos(z*freq*1.7 + 0.8)      ← tertiary variation
)
```

**Normal computation:**

Normals cannot be a simple (0,1,0) because the surface is curved. Instead they are computed using **finite differences** — sampling the height function slightly left/right and front/back:
```
dhdx = (getH(x+eps, z) - getH(x-eps, z)) / (2*eps)   ← slope in X
dhdz = (getH(x, z+eps) - getH(x, z-eps)) / (2*eps)   ← slope in Z
normal = normalize(-dhdx, 1, -dhdz)
```
This gives a vector perpendicular to the terrain surface at each point, which the lighting shader needs to calculate how directly the light hits that point.

**Index buffer:**

Each quad is split into two triangles using CCW (counter-clockwise) winding:
```
a --- b
|  \ |
c --- d
Triangles: (a,b,d) and (a,d,c)
```

#### `createCircuitMesh(semiX, semiZ, roadHalfWidth, segments)`

Creates the oval road as a ring-shaped mesh.

**Parameters:**
- `semiX, semiZ` – semi-axes of the ellipse (the road centreline). With semiX=12, semiZ=8, the oval is 24 units wide and 16 units deep.
- `roadHalfWidth` – half the road width. With 1.5, the full road is 3 units wide.
- `segments` – how many quads go around the oval. 120 gives a smooth-looking curve.

**How it works:**

The oval is parameterised by angle `t` going from 0 to 2π. For each step:
1. Compute the **centre point** of the road at angle t: `(semiX*cos(t), 0, semiZ*sin(t))`.
2. Compute the **outward normal** (direction pointing away from the centre of the ellipse): `normalize(cos(t)/semiX, 0, sin(t)/semiZ)`. This formula accounts for the fact that the normal of an ellipse is not the same as the radial direction.
3. The **inner edge** vertex = centre − normal × roadHalfWidth.
4. The **outer edge** vertex = centre + normal × roadHalfWidth.
5. U texture coordinate increases with arc length (distance travelled around the oval), so the road texture doesn't stretch or compress at the curves.
6. V = 0 for inner edge, V = 1 for outer edge, so the road texture maps correctly across the width.

#### `createBoxMesh(w, h, d)`

Creates a rectangular box with 6 faces. The origin is at the **centre of the base** (not the centre of the box), so placing the box at world position (x, 0, z) makes it sit on the ground.

Each face has 4 vertices (a quad split into 2 triangles = 6 indices per face × 6 faces = 36 indices total). Each face has its own set of 4 vertices so normals can be face-flat (pointing directly outward from each face), rather than averaged across edges.

Used for: buildings, tree trunks, streetlight poles, and streetlight lamp heads.

#### `createConeMesh(radius, height, segments)`

Creates a cone with the **apex at (0, height, 0)** and the **base at Y=0**. Used for tree crowns.

For each angular segment:
1. Compute two base points at angles `a0` and `a1`.
2. Add a triangle from the apex to those two base points.
3. The normal for each triangle is computed at the midpoint angle using the cone's surface equation: `normalize(height*cos(amid), radius, height*sin(amid))`.

After the side faces, a **base cap** is added: a centre vertex and `segments` outer vertices, forming a fan of triangles to close the bottom of the cone.

---

### 4.11 GLFW Callbacks

These functions are registered with GLFW and called automatically when events happen.

#### `cbFramebufferSize(window, w, h)`

Called when the window is resized. Updates the OpenGL viewport — the rectangular area of the window that OpenGL draws into — to match the new window size with `glViewport(0, 0, w, h)`.

#### `cbScroll(window, dx, dy)`

Called on mouse scroll. `dy` is positive when scrolling up (zoom in) and negative when scrolling down (zoom out). Adjusts `g_cam.fov` and clamps it between 15° and 120°.

A smaller FOV magnifies the view (telephoto); a larger FOV creates a fish-eye effect.

#### `cbMouse(window, xpos, ypos)`

Called every time the mouse moves (the cursor is hidden and locked in place).

1. On the first call (`g_firstMouse == true`), initialises `lastX/Y` to the current position to avoid a sudden jump.
2. Computes `dx = (xpos - lastX) * sensitivity` and `dy = (lastY - ypos) * sensitivity`. Note Y is inverted because screen Y goes down but pitch-up should correspond to moving the mouse up.
3. Adds `dx` to `yaw` and `dy` to `pitch`, clamping pitch to ±89° to prevent flipping upside down.
4. Calls `updateVectors()` to recompute the camera axes.

---

### 4.12 Input Processing

#### `processInput(window)`

Called once per frame (inside the render loop). Uses `glfwGetKey` to poll keyboard state.

| Key | Effect |
|-----|--------|
| ESC | Closes the window |
| W | Move forward along `front` direction |
| S | Move backward |
| A | Strafe left (along `-right`) |
| D | Strafe right (along `+right`) |
| Q | Move downward (along `-up`) |
| E | Move upward (along `+up`) |
| Z | Roll clockwise (+roll), then calls `updateVectors()` |
| X | Roll counter-clockwise (-roll), then calls `updateVectors()` |

All movement is multiplied by `g_cam.speed * g_dt` so that speed is measured in world units per **second**, not per frame.

---

### 4.13 main() – Initialization

#### GLFW Setup

1. `glfwInit()` – starts the GLFW library.
2. `glfwWindowHint` calls configure the OpenGL version (3.3 Core Profile). Core Profile means deprecated legacy functions are unavailable.
3. `glfwCreateWindow(SCR_W, SCR_H, TITLE, nullptr, nullptr)` – creates the OS window and an OpenGL context.
4. `glfwMakeContextCurrent(win)` – activates the context on the current thread. All OpenGL calls after this apply to this window.
5. Callback registrations: `glfwSetFramebufferSizeCallback`, `glfwSetCursorPosCallback`, `glfwSetScrollCallback`.
6. `glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED)` – hides the cursor and locks it to the window centre. Raw mouse delta is then reported via the cursor callback.

#### GLEW Setup

`glewInit()` loads all OpenGL function pointers. On Windows, functions like `glBindVertexArray` are not available at link time — GLEW finds their addresses in the OpenGL driver at runtime.

`glewExperimental = GL_TRUE` ensures GLEW loads functions for Core Profile contexts.

#### OpenGL State

`glEnable(GL_DEPTH_TEST)` turns on the depth buffer. Without this, objects drawn later would always appear on top of earlier ones regardless of distance.

#### Shader Programs

Three shader programs are compiled:
- `skyboxProg` – used only for the skybox cube.
- `terrainProg` – used for all scene geometry (ground, terrain, road, buildings, trees, streetlight poles).
- `depthProg` – used for all 5 shadow passes (no colour output, depth only).

#### Meshes and Textures

All mesh and texture creation functions are called once at startup. The results (GPU IDs stored in structs) are kept for the duration of the program. Creating meshes/textures is expensive; using them is cheap.

#### Scene Object Data

```cpp
struct BldgDef { float x, z, w, h, d; };
const BldgDef buildings[] = { ... };

struct TreePos { float x, z; };
const TreePos trees[] = { ... };
```

These arrays describe where buildings and trees are placed. They are defined as local `const` arrays in `main()` so they can be captured by the `shadowRender` lambda (see below). The building scale values `(w, h, d)` are passed directly to `glm::scale` to stretch the unit box mesh.

#### Lighting Setup

```cpp
glm::vec3 lightDir   = normalize(-0.4, -1.0, -0.5);
glm::vec3 lightColor = (1.0, 0.97, 0.90);
```

`lightDir` is the direction the sun's rays travel — not where the sun is, but where the light points. A slight negative X and Z component means the sun is to the upper-right-back of the scene.

**Sun shadow map setup:**
```cpp
glm::vec3 sunPos       = -lightDir * 40.0f;
glm::mat4 sunLightView = lookAt(sunPos, (0,0,0), (0,1,0));
glm::mat4 sunLightProj = ortho(-28, 28, -28, 28, 0.1, 150);
glm::mat4 sunLightSpace = sunLightProj * sunLightView;
```

The sun is modelled as a directional light (infinitely far away, rays are parallel). For shadow mapping, a virtual camera is placed far away in the opposite direction of the light, looking at the scene origin. An **orthographic** projection is used (no perspective — distant objects are the same size as near ones), covering a 56×56 world-unit area.

#### Streetlight Setup

Four streetlights are created, one at each corner of the oval. Each:
1. Gets a position at `(±13.5, 4.5, ±7.5)` — outside the road curves, clear of all buildings.
2. Gets a 1024×1024 shadow FBO.
3. Gets a **perspective** light-space matrix — `glm::perspective(120°, 1.0, 0.5, 30.0)` looking straight down (`lookAt(pos, pos+(0,-1,0), (1,0,0))`). Perspective is used for spotlights because they radiate outward in a cone. 120° FOV gives wide ground coverage from 4.5 units up.

#### Constant Uniforms (Set Once Before the Loop)

Before the render loop starts, uniforms that never change are set once:
```cpp
glUseProgram(terrainProg);
setInt(terrainProg, "texture1",    0);  // albedo texture → unit 0
setInt(terrainProg, "shadowMap",   1);  // sun shadow     → unit 1
setInt(terrainProg, "spotShadow0", 2);  // light 0 shadow → unit 2
// ... etc for spotShadow1,2,3 → units 3,4,5
setVec3(terrainProg, "lightDir", lightDir);
setVec3(terrainProg, "lightColor", lightColor);
setMat4(terrainProg, "lightSpaceMatrix", sunLightSpace);
// ... streetlight positions, colours, light-space matrices
```

Setting uniforms is a CPU→GPU communication step. Doing it once avoids redundant work each frame for values that never change.

#### `shadowRender` Lambda

```cpp
auto shadowRender = [&](GLuint prog) { ... };
```

A **lambda** (anonymous function) that draws all scene geometry using whatever shader program `prog` is currently active. It is used for both the shadow passes (with `depthProg`) and is structured identically to the main-pass geometry drawing.

For each object it calls `setMat4(prog, "model", ...)` to position the object, binds the correct VAO, and calls `glDrawElements`. The depth shader ignores textures and normals — it only reads `aPos` (attribute 0), so no textures are bound during shadow passes.

---

### 4.14 main() – Render Loop

The render loop runs until the user closes the window. Each iteration represents one frame.

#### Timing

```cpp
float now   = (float)glfwGetTime();
g_dt        = now - g_lastFrame;
g_lastFrame = now;
```

`glfwGetTime()` returns seconds since the program started. Subtracting the last frame's time gives `g_dt` — the time this frame took. Used in `processInput` to make movement speed frame-rate independent.

#### Shadow Passes

Before drawing anything to the screen, the scene is rendered 5 times into depth textures (one per light source):

```
For each light (sun + 4 streetlights):
  1. Resize viewport to shadow map resolution
  2. Bind the light's FBO
  3. Clear the depth buffer
  4. Activate depthProg, set lightSpace matrix
  5. Call shadowRender(depthProg) — draws all geometry
```

The GPU writes only depth values (how far each pixel is from the light). The resulting depth texture records "what is closest to this light at each angle."

#### Main Pass

After all shadow maps are filled:

1. **Restore the screen framebuffer**: `glBindFramebuffer(GL_FRAMEBUFFER, 0)` and `glViewport(0, 0, SCR_W, SCR_H)`.
2. **Clear**: `glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)` wipes the previous frame.
3. **Compute matrices**: The view matrix comes from `g_cam.view()`. The projection matrix from `glm::perspective(fov, aspect, near, far)` — `near=0.1` (minimum visible distance), `far=500` (maximum visible distance).

**Skybox drawing:**

```cpp
glDepthFunc(GL_LEQUAL);
glDepthMask(GL_FALSE);
// draw skybox
glDepthMask(GL_TRUE);
glDepthFunc(GL_LESS);
```

The skybox is drawn with `GL_DEPTH_MASK = false` so it writes no depth values (it should always be behind everything). `GL_LEQUAL` allows it to pass the depth test at depth=1.0 (the maximum). The view matrix has its translation stripped (`glm::mat4(glm::mat3(view))`) so the skybox doesn't move as the camera moves — it always surrounds the player.

**Binding shadow textures:**

```cpp
glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, sunShadow.tex);
glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, streetlights[0].shadow.tex);
// ... etc
```

The 5 depth textures are bound to units 1-5 and stay there for the entire scene rendering. As the fragment shader runs for each pixel, it samples these textures to determine shadowing.

**Scene geometry:**

For each object:
1. Set the `model` matrix (translates and scales the unit mesh into world space).
2. Bind the object's albedo texture to unit 0.
3. Set `heightBlend` — true only for the terrain, which blends between sandy and green based on height.
4. Call `glDrawElements` with the object's index count.

**Double buffering:**

```cpp
glfwSwapBuffers(win);
glfwPollEvents();
```

The window has two buffers: the **back buffer** (where OpenGL draws) and the **front buffer** (what is displayed). `glfwSwapBuffers` exchanges them, making the completed frame visible instantly without tearing. `glfwPollEvents` processes OS events (keyboard, mouse, resize) and triggers the registered callbacks.

---

## 5. Shader Files

Shaders are small programs that run on the GPU. They are written in **GLSL** (OpenGL Shading Language), which looks like C.

### 5.1 skybox.vert / skybox.frag

**skybox.vert:**
```glsl
layout(location = 0) in vec3 aPos;
out vec3 TexCoords;
uniform mat4 projection, view;

void main() {
    TexCoords = aPos;
    vec4 pos = projection * view * vec4(aPos, 1.0);
    gl_Position = pos.xyww;   // ← key trick
}
```

`gl_Position = pos.xyww` instead of the usual `pos.xyzw`. This sets the depth of every skybox pixel to `w/w = 1.0` — the maximum possible depth — so the skybox is always drawn behind everything else.

`TexCoords = aPos` uses the vertex position as the cubemap texture direction. Since the cube is centred at the origin, each vertex position is also a direction vector pointing outward, which is exactly what a cubemap lookup needs.

**skybox.frag:**
```glsl
uniform samplerCube skybox;
void main() { FragColor = texture(skybox, TexCoords); }
```

Samples the cubemap in the direction `TexCoords`. The GPU automatically selects the correct face and pixel.

---

### 5.2 terrain.vert

Used for all scene geometry (ground, terrain, road, buildings, trees, poles).

```glsl
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aTexCoord;
layout(location=2) in vec3 aNormal;
```

Receives the 3 attributes from each vertex (matching the 8-float layout).

```glsl
uniform mat4 model, view, projection;
```

Three matrices:
- **model** – positions/scales/rotates the object in the world.
- **view** – transforms world space into camera space (what the camera sees).
- **projection** – applies perspective (farther = smaller).

The full transform is: `gl_Position = projection * view * model * vec4(aPos, 1.0)`.

```glsl
FragPos = vec3(model * vec4(aPos, 1.0));
Normal  = mat3(transpose(inverse(model))) * aNormal;
Height  = aPos.y;
```

- `FragPos` – the vertex position in world space (not camera space). Needed by the fragment shader to compute light direction and sample shadow maps.
- The normal transform uses `transpose(inverse(model))` instead of just `model` because non-uniform scaling distorts normals. This is the mathematically correct normal matrix.
- `Height` – the raw local Y position, used by the fragment shader to blend terrain colours.

---

### 5.3 terrain.frag

The most complex shader — handles albedo texturing, height blending, directional sun lighting + shadow, and 4 point-light streetlights + shadows.

**Inputs from vertex shader:** `TexCoord`, `Normal`, `FragPos`, `Height`.

**Uniforms:** All the textures, light properties, and matrices described in section 4.13.

#### `shadowFactor(shadowTex, fragLS)`

Takes a shadow map texture and the current fragment's position in that light's clip space (`fragLS`). Returns a value from 0.0 (fully lit) to 1.0 (fully in shadow).

Steps:
1. **Perspective divide**: `proj = fragLS.xyz / fragLS.w` – converts from clip space to NDC space (range -1 to 1).
2. **Remap to [0,1]**: `proj = proj * 0.5 + 0.5` – matches the shadow map's texture coordinate range.
3. **Beyond far plane check**: If `proj.z > 1.0`, the fragment is outside the shadow map's range. Return 0 (no shadow).
4. **PCF (Percentage Closer Filtering)**: Sample the shadow map at a 3×3 grid of nearby texels (9 samples). For each sample, compare the stored depth against the current fragment's depth minus a small `bias` (0.005) to prevent self-shadowing artefacts. Average the 9 binary comparisons.

The bias is necessary because floating-point imprecision makes a surface appear to shadow itself (called "shadow acne").

PCF gives soft shadow edges by averaging multiple samples at slightly different positions. A single sample would produce hard, aliased edges.

#### `spotContrib(pos, col, lsm, sm)`

Computes how much a single streetlight illuminates the current fragment.

1. `L = pos - FragPos` – vector from fragment to light.
2. `dist = length(L)` – distance to light.
3. `Ln = L / dist` – normalised direction to light.
4. **Attenuation**: `att = 1 / (1 + 0.04*dist + 0.008*dist²)` – quadratic falloff. At dist=4.5 (directly below the lamp), att≈0.75 (75% strength). At dist=12, att≈0.38.
5. **Diffuse factor**: `diff = max(dot(Normal, Ln), 0)` – how directly the surface faces the light.
6. **Shadow**: `sh = shadowFactor(sm, lsm * vec4(FragPos, 1.0))` – is this point in the light's shadow?
7. Returns `(1 - sh) * att * diff * col` — no contribution if in shadow.

#### `void main()`

1. Sample albedo texture.
2. If `heightBlend`, blend albedo with a sandy→green height-based colour.
3. Compute sun shadow and diffuse: `sunSh` via `shadowFactor`, `diff` via dot product with `lightDir`.
4. `ambient = 0.25 * lightColor` – base light that fills all surfaces regardless of shadowing.
5. `sunDiff = (1 - sunSh) * diff * 0.65 * lightColor` – zero in sun shadow, up to 65% in full sun.
6. Sum all 4 streetlight contributions: `spots = spotContrib(...) * 4`.
7. Final output: `FragColor = vec4((ambient + sunDiff + spots) * texColor, 1.0)`.

The reason colours can sum to > 1.0 (e.g., `ambient + sunDiff + spots = 2.0`) is that OpenGL automatically clamps the final output to [0, 1]. This allows physically brighter light sources (like the streetlight set to intensity 3.0) to dominate and visibly brighten surfaces.

---

### 5.4 depth.vert / depth.frag

The shadow pass shaders are minimal.

**depth.vert:**
```glsl
layout(location = 0) in vec3 aPos;
uniform mat4 lightSpace, model;

void main() {
    gl_Position = lightSpace * model * vec4(aPos, 1.0);
}
```

Transforms the vertex into the light's clip space. Attributes 1 and 2 (texcoord, normal) exist in the VAO but are simply ignored.

**depth.frag:**
```glsl
void main() { }
```

Does nothing. The GPU automatically writes the depth value to the FBO's depth attachment. No colour output is needed.

---

## 6. Shadow Mapping Explained

Shadow mapping is a two-pass technique.

### Pass 1 – Depth pass (from the light's point of view)

The scene is rendered from the light's position/direction using an orthographic (sun) or perspective (streetlight) projection. The result is stored in a depth texture. Each texel stores the distance to the closest surface visible from that light.

```
Light position
      |
      v
[Depth texture pixel = 3.2]  ← closest thing is 3.2 units away
```

### Pass 2 – Main pass (from the camera's point of view)

For each fragment, we ask: "Is this point in shadow?"

1. Transform the fragment's world position into the light's clip space using `lightSpaceMatrix`.
2. Use the XY as texture coordinates to look up the stored depth.
3. Compare the stored depth (closest surface seen by the light) with the fragment's depth in light space:
   - If `fragment depth > stored depth + bias` → the light could not "see" this point → **in shadow**.
   - Otherwise → **lit**.

```
Fragment at depth 5.8 in light space
Stored depth at same XY = 3.2
5.8 > 3.2 + 0.005 → TRUE → in shadow (something at 3.2 blocked the light)
```

### Why 5 passes?

One pass per light source. Each light has its own unique view of the scene:
- 1 sun (directional, orthographic)
- 4 streetlights (point-ish, perspective, looking down)

Each produces its own shadow map. The fragment shader samples all 5 maps for every pixel.

---

## 7. Controls Reference

| Input | Action |
|-------|--------|
| **W** | Move forward |
| **S** | Move backward |
| **A** | Strafe left |
| **D** | Strafe right |
| **Q** | Move down |
| **E** | Move up |
| **Z** | Roll camera clockwise |
| **X** | Roll camera counter-clockwise |
| **Mouse move** | Look around (yaw + pitch) |
| **Scroll wheel** | Zoom in/out (change FOV) |
| **ESC** | Quit |
