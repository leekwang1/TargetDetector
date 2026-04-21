#pragma once
#include <array>
#include <cmath>
#include <limits>

struct RawPoint
{
    float horizontalAngle, verticalAngle, distance, intensity;
};
struct Point3D
{
    float x, y, z;
};
struct CartesianPoint
{
    float x, y, z, intensity;
};
struct CartesianPointRGB
{
    float x, y, z, r, g, b, intensity;
};

struct PolarPoint
{
    float hAngle, vAngle, distance, intensity;

    CartesianPoint toCartesian() const
    {
        CartesianPoint cartesian;
        // opengl(BLK SDK 공식 문서 반영)
        cartesian.x = -distance * sin(vAngle) * sin(hAngle);
        cartesian.y = -distance * cos(vAngle);
        cartesian.z = -distance * sin(vAngle) * cos(hAngle);
        cartesian.intensity = intensity;
        return cartesian;
    }

    bool isInvalid() const
    {
        return std::abs(hAngle) <= std::numeric_limits<float>::epsilon() && std::abs(vAngle) <= std::numeric_limits<float>::epsilon() && std::abs(distance) <= std::numeric_limits<float>::epsilon() && std::abs(intensity) <= std::numeric_limits<float>::epsilon();
    }
};

struct Vec3
{
    double x, y, z;
};

struct Mat3
{
    double m[3][3]; // row-major
};

static Vec3 normalize(const Vec3 &v)
{
    double len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len <= 1e-12)
        return {0.0, 0.0, 0.0};
    return {v.x / len, v.y / len, v.z / len};
}

static Vec3 cross(const Vec3 &a, const Vec3 &b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

static double dot(const Vec3 &a, const Vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 mat3MulVec3(const Mat3 &M, const Vec3 &v)
{
    return {
        M.m[0][0] * v.x + M.m[0][1] * v.y + M.m[0][2] * v.z,
        M.m[1][0] * v.x + M.m[1][1] * v.y + M.m[1][2] * v.z,
        M.m[2][0] * v.x + M.m[2][1] * v.y + M.m[2][2] * v.z};
}

using Matrix4x4 = std::array<std::array<double, 4>, 4>;
