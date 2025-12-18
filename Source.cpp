#include <GL/glew.h>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/Graphics/Image.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#pragma region Base

void ShaderLog(GLuint shader)
{
    GLint infologLen = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLen);
    if (infologLen > 1)
    {
        std::vector<char> infoLog(infologLen);
        GLsizei charsWritten = 0;
        glGetShaderInfoLog(shader, infologLen, &charsWritten, infoLog.data());
        std::cout << "Shader log:\n" << infoLog.data() << std::endl;
    }
}

void ProgramLog(GLuint prog)
{
    GLint infologLen = 0;
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infologLen);
    if (infologLen > 1)
    {
        std::vector<char> infoLog(infologLen);
        GLsizei charsWritten = 0;
        glGetProgramInfoLog(prog, infologLen, &charsWritten, infoLog.data());
        std::cout << "Program log:\n" << infoLog.data() << std::endl;
    }
}

GLuint CompileShader(GLenum type, const char* src)
{
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    ShaderLog(sh);
    return sh;
}

GLuint LinkProgram(GLuint vert, GLuint frag)
{
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);
    GLint success = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success)
        ProgramLog(prog);
    return prog;
}

#pragma endregion

#pragma region Vec2 Vec3 Mat4

struct Vec2
{
    float x, y;

    Vec2() : x(0), y(0) {}
    Vec2(float x_, float y_) : x(x_), y(y_) {}
};

struct Vec3
{
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
};

Vec3 operator+(const Vec3& a, const Vec3& b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
Vec3 operator-(const Vec3& a, const Vec3& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
Vec3 operator*(const Vec3& a, float s) { return { a.x * s, a.y * s, a.z * s }; }

float Dot(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 Cross(const Vec3& a, const Vec3& b)
{
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

float Length(const Vec3& v)
{
    return std::sqrt(Dot(v, v));
}

Vec3 Normalize(const Vec3& v)
{
    float len = Length(v);
    if (len <= 1e-6f) return v;
    return v * (1.0f / len);
}

struct Mat4
{
    float m[16] = { 0 };

    static Mat4 Identity()
    {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    static Mat4 Translation(float x, float y, float z)
    {
        Mat4 r = Identity();
        r.m[12] = x;
        r.m[13] = y;
        r.m[14] = z;
        return r;
    }

    static Mat4 Scale(float x, float y, float z)
    {
        Mat4 r;
        r.m[0] = x;
        r.m[5] = y;
        r.m[10] = z;
        r.m[15] = 1.0f;
        return r;
    }

    static Mat4 RotationX(float angleRad)
    {
        Mat4 r = Identity();
        float c = std::cos(angleRad);
        float s = std::sin(angleRad);

        r.m[5] = c;
        r.m[6] = s;
        r.m[9] = -s;
        r.m[10] = c;

        return r;
    }
    static Mat4 RotationY(float angleRad)
    {
        Mat4 r = Identity();
        float c = std::cos(angleRad);
        float s = std::sin(angleRad);
        r.m[0] = c;
        r.m[2] = s;
        r.m[8] = -s;
        r.m[10] = c;
        return r;
    }
    static Mat4 RotationZ(float angleRad)
    {
        Mat4 r = Identity();
        float c = std::cos(angleRad);
        float s = std::sin(angleRad);

        r.m[0] = c;
        r.m[1] = s;
        r.m[4] = -s;
        r.m[5] = c;

        return r;
    }

    static Mat4 Perspective(float fovyRad, float aspect, float zNear, float zFar)
    {
        Mat4 r;
        float tanHalfFovy = std::tan(fovyRad / 2.0f);

        r.m[0] = 1.0f / (aspect * tanHalfFovy);
        r.m[5] = 1.0f / tanHalfFovy;
        r.m[10] = -(zFar + zNear) / (zFar - zNear);
        r.m[11] = -1.0f;
        r.m[14] = -(2.0f * zFar * zNear) / (zFar - zNear);
        return r;
    }

    static Mat4 LookAt(const Vec3& eye, const Vec3& center, const Vec3& up)
    {
        Vec3 f = Normalize(center - eye);
        Vec3 s = Normalize(Cross(f, up));
        Vec3 u = Cross(s, f);

        Mat4 r = Identity();
        r.m[0] = s.x;
        r.m[4] = s.y;
        r.m[8] = s.z;

        r.m[1] = u.x;
        r.m[5] = u.y;
        r.m[9] = u.z;

        r.m[2] = -f.x;
        r.m[6] = -f.y;
        r.m[10] = -f.z;

        r.m[12] = -Dot(s, eye);
        r.m[13] = -Dot(u, eye);
        r.m[14] = Dot(f, eye);
        return r;
    }
};

Mat4 operator*(const Mat4& a, const Mat4& b)
{
    Mat4 r;
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            r.m[col * 4 + row] =
                a.m[0 * 4 + row] * b.m[col * 4 + 0] +
                a.m[1 * 4 + row] * b.m[col * 4 + 1] +
                a.m[2 * 4 + row] * b.m[col * 4 + 2] +
                a.m[3 * 4 + row] * b.m[col * 4 + 3];
        }
    }
    return r;
}

#pragma endregion

#pragma region Load Obj n Texture

GLuint LoadTextureFromFile(const std::string& filename)
{
    sf::Image img;
    if (!img.loadFromFile(filename))
    {
        std::cout << "Failed to load texture: " << filename << std::endl;
        return 0;
    }
    img.flipVertically();

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
        img.getSize().x, img.getSize().y,
        0, GL_RGBA, GL_UNSIGNED_BYTE, img.getPixelsPtr());

    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

bool LoadOBJ(const std::string& filename, std::vector<float>& outVertices)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cout << "Failed to open OBJ: " << filename << std::endl;
        return false;
    }

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<sf::Vector2f> texcoords;

    outVertices.clear();

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        // -------- VERTEX POSITION --------
        if (prefix == "v")
        {
            float x, y, z;
            iss >> x >> y >> z;
            positions.emplace_back(x, y, z);
        }
        // -------- TEXCOORD --------
        else if (prefix == "vt")
        {
            float u, v;
            iss >> u >> v;
            texcoords.emplace_back(u, v);
        }
        // -------- NORMAL --------
        else if (prefix == "vn")
        {
            float x, y, z;
            iss >> x >> y >> z;
            normals.emplace_back(x, y, z);
        }
        // -------- FACE --------
        else if (prefix == "f")
        {
            std::vector<std::string> tokens;
            std::string tok;
            while (iss >> tok)
                tokens.push_back(tok);

            if (tokens.size() < 3)
                continue;

            auto parseIndex = [&](const std::string& s, int& vi, int& ti, int& ni)
                {
                    vi = ti = ni = 0;

                    size_t p1 = s.find('/');
                    size_t p2 = s.find('/', p1 + 1);

                    vi = std::stoi(s.substr(0, p1));

                    if (p1 != std::string::npos)
                    {
                        if (p2 == std::string::npos)
                        {
                            if (p1 + 1 < s.size())
                                ti = std::stoi(s.substr(p1 + 1));
                        }
                        else
                        {
                            if (p2 > p1 + 1)
                                ti = std::stoi(s.substr(p1 + 1, p2 - p1 - 1));
                            if (p2 + 1 < s.size())
                                ni = std::stoi(s.substr(p2 + 1));
                        }
                    }
                };

            auto pushVertex = [&](int vi, int ti, int ni)
                {
                    Vec3 pos = positions[vi - 1];
                    Vec3 norm = (ni > 0 && ni <= (int)normals.size()) ? normals[ni - 1] : Vec3(0, 1, 0);
                    sf::Vector2f uv = (ti > 0 && ti <= (int)texcoords.size())
                        ? texcoords[ti - 1]
                        : sf::Vector2f(0.f, 0.f);

                    outVertices.push_back(pos.x);
                    outVertices.push_back(pos.y);
                    outVertices.push_back(pos.z);

                    outVertices.push_back(norm.x);
                    outVertices.push_back(norm.y);
                    outVertices.push_back(norm.z);

                    outVertices.push_back(uv.x);
                    outVertices.push_back(uv.y);
                };

            // TRIANGULATION (fan)
            int v0, t0, n0;
            parseIndex(tokens[0], v0, t0, n0);

            for (size_t i = 1; i + 1 < tokens.size(); ++i)
            {
                int v1, t1, n1;
                int v2, t2, n2;

                parseIndex(tokens[i], v1, t1, n1);
                parseIndex(tokens[i + 1], v2, t2, n2);

                pushVertex(v0, t0, n0);
                pushVertex(v1, t1, n1);
                pushVertex(v2, t2, n2);
            }
        }
    }

    if (outVertices.empty())
    {
        std::cout << "OBJ has no vertices: " << filename << std::endl;
        return false;
    }

    std::cout << "OBJ loaded: " << filename
        << " | vertices: " << outVertices.size() / 8 << std::endl;

    return true;
}

#pragma endregion

#pragma region Mesh

struct Mesh
{
    GLuint VAO = 0;
    GLuint VBO = 0;
    GLsizei vertexCount = 0;
};

struct Instance
{
    Vec3 pos;
    float rotY;
    float scale;
};

Mesh CreateMeshFromInterleaved(const std::vector<float>& data)
{
    Mesh m;
    m.vertexCount = static_cast<GLsizei>(data.size() / 8); // 8 float на вершину

    glGenVertexArrays(1, &m.VAO);
    glGenBuffers(1, &m.VBO);

    glBindVertexArray(m.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m.VBO);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);

    // layout (location=0) vec3 position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // layout (location=1) vec3 normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // layout (location=2) vec2 texcoord
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return m;
}

#pragma endregion

#pragma region Terrain & Heightmap

struct Heightmap
{
    int w, h;
    std::vector<float> data; // 0..1
};

Heightmap LoadHeightmap(const char* path)
{
    Heightmap hm;
    int channels;
    unsigned char* img = stbi_load(path, &hm.w, &hm.h, &channels, 1);

    if (!img)
        throw std::runtime_error("Failed to load heightmap");

    hm.data.resize(hm.w * hm.h);

    for (int i = 0; i < hm.w * hm.h; ++i)
        hm.data[i] = img[i] / 255.0f;

    stbi_image_free(img);
    return hm;
}

struct TerrainMesh
{
    GLuint VAO = 0;
    GLuint VBO = 0;
    GLuint EBO = 0;
    GLsizei indexCount = 0;
};

TerrainMesh CreateTerrainMesh(const std::vector<float>& vertices, const std::vector<unsigned>& indices)
{
    TerrainMesh m;
    m.indexCount = static_cast<GLsizei>(indices.size());

    glGenVertexArrays(1, &m.VAO);
    glGenBuffers(1, &m.VBO);
    glGenBuffers(1, &m.EBO);

    glBindVertexArray(m.VAO);

    glBindBuffer(GL_ARRAY_BUFFER, m.VBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        vertices.size() * sizeof(float),
        vertices.data(),
        GL_STATIC_DRAW
    );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.EBO);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        indices.size() * sizeof(unsigned),
        indices.data(),
        GL_STATIC_DRAW
    );

    // position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // texcoord
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    return m;
}

TerrainMesh CreateTerrainFromHeightmap(const Heightmap& hm,float sizeXZ,float heightScale,float uvTiling)
{
    int w = hm.w;
    int h = hm.h;

    std::vector<Vec3> pos(w * h);
    std::vector<Vec3> nrm(w * h, Vec3(0, 0, 0));
    std::vector<Vec2> uv(w * h);

    float dx = sizeXZ / (w - 1);
    float dz = sizeXZ / (h - 1);

    for (int z = 0; z < h; ++z)
    {
        for (int x = 0; x < w; ++x)
        {
            int i = z * w + x;

            float y = hm.data[i] * heightScale;

            pos[i] = {
                x * dx - sizeXZ * 0.5f,
                y,
                z * dz - sizeXZ * 0.5f
            };

            uv[i] = {
                (float)x / (w - 1) * uvTiling,
                (float)z / (h - 1) * uvTiling
            };
        }
    }

    std::vector<unsigned> indices;
    indices.reserve((w - 1) * (h - 1) * 6);

    for (int z = 0; z < h - 1; ++z)
    {
        for (int x = 0; x < w - 1; ++x)
        {
            int i0 = z * w + x;
            int i1 = z * w + x + 1;
            int i2 = (z + 1) * w + x;
            int i3 = (z + 1) * w + x + 1;

            indices.push_back(i0);
            indices.push_back(i2);
            indices.push_back(i1);

            indices.push_back(i1);
            indices.push_back(i2);
            indices.push_back(i3);
        }
    }

    for (size_t i = 0; i < indices.size(); i += 3)
    {
        unsigned a = indices[i];
        unsigned b = indices[i + 1];
        unsigned c = indices[i + 2];

        Vec3 e1 = pos[b] - pos[a];
        Vec3 e2 = pos[c] - pos[a];
        Vec3 nn = Normalize(Cross(e1, e2));

        nrm[a] = nrm[a] + nn;
        nrm[b] = nrm[b] + nn;
        nrm[c] = nrm[c] + nn;
    }

    for (auto& n : nrm)
        n = Normalize(n);

    std::vector<float> data;
    data.reserve(w * h * 8);

    for (int i = 0; i < w * h; ++i)
    {
        data.push_back(pos[i].x);
        data.push_back(pos[i].y);
        data.push_back(pos[i].z);

        data.push_back(nrm[i].x);
        data.push_back(nrm[i].y);
        data.push_back(nrm[i].z);

        data.push_back(uv[i].x);
        data.push_back(uv[i].y);
    }

    return CreateTerrainMesh(data, indices);
}

#pragma endregion

#pragma region Shaders

const char* vertexShaderSrc = R"(
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTex;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

uniform float uTime;
uniform float uSwayStrength;   // 0 = выкл
uniform float uSwaySpeed;

out vec2 vTex;
out vec3 vNormal;

void main()
{
    vec3 pos = aPos;

    //сдвиги
    if (uSwayStrength > 0.0)
    {
        float phase =
            pos.y * 2.0 +
            uTime * uSwaySpeed;

        float sway =
            sin(phase) *
            uSwayStrength;

        pos.x += sway;
    }

    vTex = aTex;
    vNormal = mat3(uModel) * aNormal;
    gl_Position = uProj * uView * uModel * vec4(pos, 1.0);
}

)";
const char* fragmentShaderSrc = R"(
#version 330 core

in vec2 vTex;
in vec3 vNormal;

out vec4 FragColor;

uniform sampler2D uTexture;
uniform vec3 uLightDir;
uniform vec3 uLightColor;

void main()
{
    vec4 tex = texture(uTexture, vTex);
    if (tex.a < 0.5) discard;

    vec3 N = normalize(vNormal);
    vec3 L = normalize(-uLightDir);

    float diff = max(dot(N, L), 0.0);

    // небольшая засветка
    vec3 ambient = 0.25 * tex.rgb;
    vec3 diffuse = diff * tex.rgb * uLightColor;

    FragColor = vec4(ambient + diffuse, 1.0);
}

)";

#pragma endregion

int main()
{
    setlocale(LC_ALL, "ru_RU.utf8");
    bool aimCamera = false;
    bool aimKeyPressed = false;

    std::srand(static_cast<unsigned>(std::time(nullptr)));
    auto RandRange = [](float a, float b)
        {
            return a + (b - a) * (std::rand() / float(RAND_MAX));
        };

    sf::Window window(
        sf::VideoMode({ 1200u, 900u }),
        "Airship Scene",
        sf::Style::Default
    );
    window.setFramerateLimit(60);
    window.setActive(true);

    glewInit();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    // ===== ШЕЙДЕРЫ =====
    GLuint vert = CompileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint frag = CompileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint prog = LinkProgram(vert, frag);
    glDeleteShader(vert);
    glDeleteShader(frag);

    GLint uModelLoc = glGetUniformLocation(prog, "uModel");
    GLint uViewLoc = glGetUniformLocation(prog, "uView");
    GLint uProjLoc = glGetUniformLocation(prog, "uProj");
    GLint uTexLoc = glGetUniformLocation(prog, "uTexture");
    GLint uLightDirLoc = glGetUniformLocation(prog, "uLightDir");
    GLint uLightColorLoc = glGetUniformLocation(prog, "uLightColor");

    GLint uTimeLoc = glGetUniformLocation(prog, "uTime");
    GLint uSwayStrengthLoc = glGetUniformLocation(prog, "uSwayStrength");
    GLint uSwaySpeedLoc = glGetUniformLocation(prog, "uSwaySpeed");

    // ===== ДИРИЖАБЛЬ =====
    std::vector<float> airshipData;
    LoadOBJ("airship.obj", airshipData);
    Mesh airshipMesh = CreateMeshFromInterleaved(airshipData);
    GLuint airshipTex = LoadTextureFromFile("airship_diffuse.png");

    Vec3 airshipPos(-20, 40, 0);
    float airshipYaw = 0.0f;
    float airshipSpeed = 12.0f;

    // ===== ЁЛКА =====
    std::vector<float> treeData;
    LoadOBJ("tree.obj", treeData);
    Mesh treeMesh = CreateMeshFromInterleaved(treeData);
    GLuint treeTex = LoadTextureFromFile("tree_diffuse.png");

    Vec3 treePos(0, 0, 0);

    // ===== TERRAIN =====
    Heightmap hm = LoadHeightmap("heightmap.png");

    TerrainMesh terrain =
        CreateTerrainFromHeightmap(
            hm,
            300.0f,   // размер по XZ
            10.0f,    // высота
            20.0f     // тайлинг текстуры
        );

    GLuint terrainTex = LoadTextureFromFile("ground_diffuse.png");

    // ===== Домики-цели =====
    std::vector<float> houseData;
    LoadOBJ("house.obj", houseData);
    Mesh houseMesh = CreateMeshFromInterleaved(houseData);
    GLuint houseTex = LoadTextureFromFile("house_diffuse.png");
    std::vector<Instance> houses;
    for (int i = 0; i < 10; ++i)
    {
        Instance h;
        h.pos.x = RandRange(-80.0f, 80.0f);
        h.pos.z = RandRange(-80.0f, 80.0f);
        h.pos.y = 0.0f;

        h.rotY = RandRange(0.0f, 6.28f);
        h.scale = RandRange(1.5f, 2.5f);

        houses.push_back(h);
    }

    // ===== Декорации =====
    std::vector<float> decor1Data;
    LoadOBJ("decor1.obj", decor1Data);
    Mesh decor1Mesh = CreateMeshFromInterleaved(decor1Data);
    GLuint decor1Tex = LoadTextureFromFile("decor1_diffuse.png");
    std::vector<float> decor2Data;
    LoadOBJ("decor2.obj", decor2Data);
    Mesh decor2Mesh = CreateMeshFromInterleaved(decor2Data);
    GLuint decor2Tex = LoadTextureFromFile("decor2_diffuse.png");
    std::vector<Instance> decors1;
    std::vector<Instance> decors2;

    for (int i = 0; i < 10; ++i)
    {
        Instance t;
        t.pos = { RandRange(-100,100), 0.0f, RandRange(-100,100) };
        t.rotY = RandRange(0, 6.28f);
        t.scale = RandRange(1.8f, 2.5f);
        decors1.push_back(t);
    }

    for (int i = 0; i < 10; ++i)
    {
        Instance r;
        r.pos = { RandRange(-100,100), 0.0f, RandRange(-100,100) };
        r.rotY = RandRange(0, 6.28f);
        r.scale = RandRange(1.0f, 1.8f);
        decors2.push_back(r);
    }

    // ===== ТУЧИ =====
    std::vector<float> cloudData;
    LoadOBJ("cloud.obj", cloudData);
    Mesh cloudMesh = CreateMeshFromInterleaved(cloudData);
    GLuint cloudTex = LoadTextureFromFile("cloud_diffuse.png");
    std::vector<Instance> clouds;

    for (int i = 0; i < 10; ++i)
    {
        Instance c;
        c.pos.x = RandRange(-120.0f, 120.0f);
        c.pos.z = RandRange(-120.0f, 120.0f);
        c.pos.y = RandRange(25.0f, 30.0f);

        c.rotY = RandRange(0, 6.28f);
        c.scale = RandRange(7.0f, 14.0f);

        clouds.push_back(c);
    }

    // ===== ШАРЫ =====
    std::vector<float> balloonData;
    LoadOBJ("balloon.obj", balloonData);
    Mesh balloonMesh = CreateMeshFromInterleaved(balloonData);
    GLuint balloonTex = LoadTextureFromFile("balloon_diffuse.png");
    std::vector<Instance> balloons;

    for (int i = 0; i < 10; ++i)
    {
        Instance b;
        b.pos.x = RandRange(-100.0f, 100.0f);
        b.pos.z = RandRange(-100.0f, 100.0f);
        b.pos.y = RandRange(28.0f, 35.0f);

        b.rotY = RandRange(0, 6.28f);
        b.scale = RandRange(4.6f, 6.5f);

        balloons.push_back(b);
    }

    // ===== ПРОЕКЦИЯ =====
    auto deg2rad = [](float d) { return d * (float)M_PI / 180.0f; };
    auto makeProj = [&](unsigned w, unsigned h)
        {
            float a = (h == 0) ? 1.0f : (float)w / (float)h;
            return Mat4::Perspective(deg2rad(60.0f), a, 0.5f, 500.0f);
        };
    Mat4 proj = makeProj(window.getSize().x, window.getSize().y);

    sf::Clock clock;
    static float globalTime = 0.0f;

    while (window.isOpen())
    {
        float dt = clock.restart().asSeconds();
        globalTime += dt;

        while (auto e = window.pollEvent())
        {
            if (e->is<sf::Event::Closed>())
                window.close();
            if (auto* r = e->getIf<sf::Event::Resized>())
            {
                glViewport(0, 0, r->size.x, r->size.y);
                proj = makeProj(r->size.x, r->size.y);
            }
        }

        // ===== УПРАВЛЕНИЕ ДИРИЖАБЛЕМ =====
        Vec3 forward(std::cos(airshipYaw), 0, std::sin(airshipYaw));
        forward = Normalize(forward);

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W))
            airshipPos = airshipPos + forward * airshipSpeed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S))
            airshipPos = airshipPos - forward * airshipSpeed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
            airshipYaw -= 1.5f * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
            airshipYaw += 1.5f * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space))
            airshipPos.y += airshipSpeed * dt;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LShift))
            airshipPos.y -= airshipSpeed * dt;

        if (airshipPos.y < 2) airshipPos.y = 2;
        if (airshipPos.y > 40) airshipPos.y = 40;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::C))
        {
            if (!aimKeyPressed)
            {
                aimCamera = !aimCamera;
                aimKeyPressed = true;
            }
        }
        else
        {
            aimKeyPressed = false;
        }

        // ===== КАМЕРА =====
        Vec3 camPos;
        Vec3 camTarget;
        Vec3 upVec;

        if (!aimCamera)
        {
            camPos = airshipPos - forward * 6.0f + Vec3(0, 4, 0);
            camTarget = airshipPos;
            upVec = Vec3(0, 1, 0);
        }
        else
        {
            camPos = airshipPos + Vec3(0, 0.0f, 0);
            camTarget = airshipPos + Vec3(0, -10.0f, 0);

            // ❗ up не должен быть (0,1,0)
            upVec = forward;
        }

        Mat4 view = Mat4::LookAt(camPos, camTarget, upVec);


        // ===== РЕНДЕР =====
        glClearColor(0.6f, 0.8f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);
        glUniformMatrix4fv(uViewLoc, 1, GL_FALSE, view.m);
        glUniformMatrix4fv(uProjLoc, 1, GL_FALSE, proj.m);
        glUniform3f(uLightDirLoc, -0.3f, -1.0f, -0.2f);
        glUniform3f(uLightColorLoc, 1, 1, 1);
        glUniform1i(uTexLoc, 0);

        glUniform1f(uTimeLoc, globalTime);
        glUniform1f(uSwayStrengthLoc, 0.0f);
        glUniform1f(uSwaySpeedLoc, 0.0f);

        // ===== TERRAIN =====
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, terrainTex);

        Mat4 terrainModel = Mat4::Identity();
        glUniformMatrix4fv(uModelLoc, 1, GL_FALSE, terrainModel.m);

        glBindVertexArray(terrain.VAO);
        glDrawElements(GL_TRIANGLES, terrain.indexCount, GL_UNSIGNED_INT, 0);


        // ===== ДОМИКИ =====
        for (const auto& h : houses)
        {
            Mat4 model =
                Mat4::Translation(h.pos.x, h.pos.y, h.pos.z) *
                Mat4::RotationY(h.rotY) *
                Mat4::Scale(h.scale, h.scale, h.scale);

            glUniformMatrix4fv(uModelLoc, 1, GL_FALSE, model.m);
            glBindTexture(GL_TEXTURE_2D, houseTex);
            glBindVertexArray(houseMesh.VAO);
            glDrawArrays(GL_TRIANGLES, 0, houseMesh.vertexCount);
        }

        // ===== ДЕКОР1 =====
        for (const auto& i : decors1)
        {
            Mat4 model =
                Mat4::Translation(i.pos.x, i.pos.y, i.pos.z) *
                Mat4::RotationY(i.rotY) *
                Mat4::Scale(i.scale, i.scale, i.scale);

            glUniformMatrix4fv(uModelLoc, 1, GL_FALSE, model.m);
            glBindTexture(GL_TEXTURE_2D, decor1Tex);
            glBindVertexArray(decor1Mesh.VAO);
            glDrawArrays(GL_TRIANGLES, 0, decor1Mesh.vertexCount);
        }

        // ===== ДЕКОР2 =====
        for (const auto& i : decors2)
        {
            Mat4 model =
                Mat4::Translation(i.pos.x, i.pos.y, i.pos.z) *
                Mat4::RotationY(i.rotY) *
                Mat4::Scale(i.scale, i.scale, i.scale);

            glUniformMatrix4fv(uModelLoc, 1, GL_FALSE, model.m);
            glBindTexture(GL_TEXTURE_2D, decor2Tex);
            glBindVertexArray(decor2Mesh.VAO);
            glDrawArrays(GL_TRIANGLES, 0, decor2Mesh.vertexCount);
        }

        glUniform1f(uSwayStrengthLoc, 0.05f);
        glUniform1f(uSwaySpeedLoc, 1.5f);//активируем причину тряски

        // ===== ТУЧИ =====
        for (const auto& c : clouds)
        {
            Mat4 model =
                Mat4::Translation(c.pos.x, c.pos.y, c.pos.z) *
                Mat4::RotationY(c.rotY) *
                Mat4::Scale(c.scale, c.scale, c.scale);

            glUniformMatrix4fv(uModelLoc, 1, GL_FALSE, model.m);
            glBindTexture(GL_TEXTURE_2D, cloudTex);
            glBindVertexArray(cloudMesh.VAO);
            glDrawArrays(GL_TRIANGLES, 0, cloudMesh.vertexCount);
        }

        // ===== ШАРЫ =====
        for (const auto& b : balloons)
        {
            Mat4 model =
                Mat4::Translation(b.pos.x, b.pos.y, b.pos.z) *
                Mat4::RotationY(b.rotY) *
                Mat4::Scale(b.scale, b.scale, b.scale);

            glUniformMatrix4fv(uModelLoc, 1, GL_FALSE, model.m);
            glBindTexture(GL_TEXTURE_2D, balloonTex);
            glBindVertexArray(balloonMesh.VAO);
            glDrawArrays(GL_TRIANGLES, 0, balloonMesh.vertexCount);
        }

        // ===== ЁЛКА =====       
        glBindTexture(GL_TEXTURE_2D, treeTex);
        Mat4 treeModel =
            Mat4::Translation(treePos.x, treePos.y, treePos.z) *
            Mat4::Scale(4.0f, 4.0f, 4.0f);
        glUniformMatrix4fv(uModelLoc, 1, GL_FALSE, treeModel.m);
        glBindVertexArray(treeMesh.VAO);
        glDrawArrays(GL_TRIANGLES, 0, treeMesh.vertexCount);

        // ===== ДИРИЖАБЛЬ =====
        glBindTexture(GL_TEXTURE_2D, airshipTex);
        Mat4 airshipModel =
            Mat4::Translation(airshipPos.x, airshipPos.y, airshipPos.z) *
            Mat4::RotationY(airshipYaw + M_PI) *
            Mat4::Scale(1, 1, 1);
        glUniformMatrix4fv(uModelLoc, 1, GL_FALSE, airshipModel.m);
        glBindVertexArray(airshipMesh.VAO);
        glDrawArrays(GL_TRIANGLES, 0, airshipMesh.vertexCount);

        glBindVertexArray(0);
        glUseProgram(0);

        window.display();
    }

    return 0;
}
