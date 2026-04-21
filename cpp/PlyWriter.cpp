#include "PlyWriter.h"
#include "logging.h"
#include <cstdint>
#include <iomanip>
#include <limits>
#include <vector>
#include <cmath>
#include <cstring>
#include <sstream>

#define LOG_TAG "PLY"

// -------------------------------------------------------------------------
// 일반 데이터 저장
// -------------------------------------------------------------------------
//  점군 카운트를 바로 저장(메모리에 모든 점군 올리는 방식) - Ascii, binary 동일(header는 Ascii로 저장)
//  헤더에 distance(옵션) 포함
static void _writePlyHeader(std::ofstream &outFile,
                            size_t pointCount,
                            bool includeDistance)
{
    outFile << "ply\n";
    outFile << "format binary_little_endian 1.0\n";
    outFile << "element vertex " << pointCount << "\n";
    outFile << "property float x\n";
    outFile << "property float y\n";
    outFile << "property float z\n";
    outFile << "property uchar red\n";
    outFile << "property uchar green\n";
    outFile << "property uchar blue\n";
    outFile << "property float intensity\n";

    // 추가: distance 필드 (옵션)
    if (includeDistance)
        outFile << "property float distance\n";

    outFile << "end_header\n";
}

static inline float _calcDistance(const CartesianPointRGB &p)
{
    return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

// 바이너리 포인트 출력
void _writeColorPointBinary(std::ofstream &outFile,
                            const CartesianPointRGB &point,
                            bool includeDistance)
{
    outFile.write(reinterpret_cast<const char *>(&point.x), sizeof(float));
    outFile.write(reinterpret_cast<const char *>(&point.y), sizeof(float));
    outFile.write(reinterpret_cast<const char *>(&point.z), sizeof(float));

    uint8_t r = static_cast<uint8_t>(point.r);
    uint8_t g = static_cast<uint8_t>(point.g);
    uint8_t b = static_cast<uint8_t>(point.b);

    outFile.write(reinterpret_cast<const char *>(&r), sizeof(uint8_t));
    outFile.write(reinterpret_cast<const char *>(&g), sizeof(uint8_t));
    outFile.write(reinterpret_cast<const char *>(&b), sizeof(uint8_t));

    outFile.write(reinterpret_cast<const char *>(&point.intensity), sizeof(float));

    if (includeDistance)
    {
        float dist = _calcDistance(point);
        outFile.write(reinterpret_cast<const char *>(&dist), sizeof(float));
    }
}

// 바이너리 점군 로드 (distance 필드 있으면 4바이트 스킵)
static std::vector<CartesianPointRGB> _loadBinaryPointCloud(std::ifstream &file,
                                                            size_t headerBytes)
{
    std::vector<CartesianPointRGB> points;
    points.reserve(1024);

    // 1) 헤더 부분 문자열로 확보해서 distance 유무 판정
    file.clear();
    file.seekg(0, std::ios::beg);
    const size_t sniff = std::min<size_t>(headerBytes + 64, 4096);
    std::string header(sniff, '\0');
    file.read(header.data(), sniff);
    bool hasDistance = (header.find("property float distance") != std::string::npos);

    // 2) 본문 위치로 이동
    file.clear();
    file.seekg(headerBytes, std::ios::beg);

    while (true)
    {
        CartesianPointRGB pt{};
        // x,y,z
        if (!file.read(reinterpret_cast<char *>(&pt.x), sizeof(float)))
            break;
        if (!file.read(reinterpret_cast<char *>(&pt.y), sizeof(float)))
            break;
        if (!file.read(reinterpret_cast<char *>(&pt.z), sizeof(float)))
            break;

        // r,g,b (uchar)
        uint8_t r = 0, g = 0, b = 0;
        if (!file.read(reinterpret_cast<char *>(&r), sizeof(uint8_t)))
            break;
        if (!file.read(reinterpret_cast<char *>(&g), sizeof(uint8_t)))
            break;
        if (!file.read(reinterpret_cast<char *>(&b), sizeof(uint8_t)))
            break;
        pt.r = static_cast<float>(r);
        pt.g = static_cast<float>(g);
        pt.b = static_cast<float>(b);

        // intensity
        if (!file.read(reinterpret_cast<char *>(&pt.intensity), sizeof(float)))
            break;

        // distance(옵션) — 있으면 읽고 버리기
        if (hasDistance)
        {
            float dummy = 0.0f;
            if (!file.read(reinterpret_cast<char *>(&dummy), sizeof(float)))
                break;
        }

        points.push_back(pt);
    }
    return points;
}

// ply 헤더 검사(아스키, 바이너리 공용)
bool checkPlyHeaderBasicValidity(std::ifstream &file,
                                 size_t &headerBytes)
{
    constexpr size_t MAX_HEADER_SIZE = 2048;
    char buffer[MAX_HEADER_SIZE];

    file.clear();
    file.seekg(0, std::ios::beg);
    file.read(buffer, MAX_HEADER_SIZE);
    std::streamsize readSize = file.gcount();

    std::string header(buffer, readSize);
    size_t endHeaderPos = header.find("end_header");
    if (endHeaderPos == std::string::npos)
        return false;

    headerBytes = endHeaderPos + std::string("end_header").size();

    bool hasX = header.find("property float x") != std::string::npos;
    bool hasY = header.find("property float y") != std::string::npos;
    bool hasZ = header.find("property float z") != std::string::npos;

    bool hasIntensity = header.find("property float intensity") != std::string::npos;

    bool hasRed = header.find("property uchar red") != std::string::npos;
    bool hasGreen = header.find("property uchar green") != std::string::npos;
    bool hasBlue = header.find("property uchar blue") != std::string::npos;

    bool hasColor = hasRed && hasGreen && hasBlue;

    file.clear();
    file.seekg(0, std::ios::beg);

    return (hasX && hasY && hasZ) && (hasIntensity || hasColor);
}

// 바이너리 포인트 저장
bool savePointCloud(const std::string &filePath,
                    const std::vector<CartesianPointRGB> &points,
                    bool includeDistance)
{
    std::ofstream outFile(filePath, std::ios::binary);
    if (!outFile.is_open())
    {
        SENDLOGF_TAG(LOG_TAG, "Failed to open file for write: %s", filePath.c_str());
        return false;
    }

    // 헤더는 ASCII로 작성
    _writePlyHeader(outFile, points.size(), includeDistance);

    for (const auto &pt : points)
        _writeColorPointBinary(outFile, pt, includeDistance);

    outFile.close();
    SENDLOGF_TAG(LOG_TAG, "PLY saved (distance %s): %s (N=%zu)", includeDistance ? "ON" : "OFF", filePath.c_str(), static_cast<size_t>(points.size()));
    return true;
}

// 점군 로드
std::vector<CartesianPointRGB> loadPointCloud(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        return {};

    // 헤더 검사
    size_t headerBytes = 0;
    if (!checkPlyHeaderBasicValidity(file, headerBytes))
        return {};

    {
        std::string line;
        size_t headerBytes = 0;

        while (std::getline(file, line))
        {
            headerBytes += line.size() + 1; // +1 for newline
            if (line == "end_header")
                break;
        }

        return _loadBinaryPointCloud(file, headerBytes);
    }
}

// -------------------------------------------------------------------------
// raw 데이터 저장
// -------------------------------------------------------------------------
static void _writeRawPlyHeader(std::ofstream &outFile,
                               size_t pointCount)
{
    outFile << "ply\n";
    outFile << "format binary_little_endian 1.0\n";
    outFile << "element vertex " << pointCount << "\n";

    outFile << "property float horizontal_angle\n";
    outFile << "property float vertical_angle\n";
    outFile << "property float distance\n";
    outFile << "property float intensity\n";

    outFile << "end_header\n";
}

static void _writeRawData(std::ofstream &outFile,
                          const RawPoint &point)
{

    outFile.write(reinterpret_cast<const char *>(&point.horizontalAngle), sizeof(float));
    outFile.write(reinterpret_cast<const char *>(&point.verticalAngle), sizeof(float));
    outFile.write(reinterpret_cast<const char *>(&point.distance), sizeof(float));
    outFile.write(reinterpret_cast<const char *>(&point.intensity), sizeof(float));
}

static std::vector<RawPoint> _loadBinaryRawData(std::ifstream &file, size_t headerBytes)
{
    std::vector<RawPoint> points;
    points.reserve(1024);

    file.clear();
    file.seekg(static_cast<std::streamoff>(headerBytes), std::ios::beg);

    while (true)
    {
        RawPoint pt{};
        if (!file.read(reinterpret_cast<char *>(&pt.horizontalAngle), sizeof(float)))
            break;
        if (!file.read(reinterpret_cast<char *>(&pt.verticalAngle), sizeof(float)))
            break;
        if (!file.read(reinterpret_cast<char *>(&pt.distance), sizeof(float)))
            break;
        if (!file.read(reinterpret_cast<char *>(&pt.intensity), sizeof(float)))
            break;

        points.push_back(pt);
    }

    return points;
}

bool checkRawPlyHeader(std::ifstream &file, size_t &headerBytes)
{
    constexpr size_t MAX_HEADER_SIZE = 4096;
    char buffer[MAX_HEADER_SIZE];

    file.clear();
    file.seekg(0, std::ios::beg);
    file.read(buffer, MAX_HEADER_SIZE);
    std::streamsize readSize = file.gcount();

    std::string header(buffer, static_cast<size_t>(readSize));
    size_t endHeaderPos = header.find("end_header");
    if (endHeaderPos == std::string::npos)
    {
        SENDLOGF_TAG(LOG_TAG, "checkRawPlyHeader - no end_header");
        return false;
    }

    headerBytes = endHeaderPos + std::string("end_header").size() + 1;

    bool hasH = header.find("property float horizontal_angle") != std::string::npos;
    bool hasV = header.find("property float vertical_angle") != std::string::npos;
    bool hasD = header.find("property float distance") != std::string::npos;
    bool hasI = header.find("property float intensity") != std::string::npos;

    if (!hasH || !hasV || !hasD || !hasI)
    {
        SENDLOGF_TAG(LOG_TAG, "checkRawPlyHeader - missing RAW properties");
        return false;
    }

    return true;
}

bool saveRawPointCloud(const std::string &filePath,
                       const std::vector<RawPoint> &points)
{
    std::ofstream outFile(filePath, std::ios::binary);
    if (!outFile.is_open())
    {
        SENDLOGF_TAG(LOG_TAG, "Failed to open file for Binary write: %s", filePath.c_str());
        return false;
    }

    _writeRawPlyHeader(outFile, points.size());

    for (const auto &pt : points)
        _writeRawData(outFile, pt);

    outFile.close();
    SENDLOGF_TAG(LOG_TAG, "RAW PLY saved: %s (N=%zu)",
                 filePath.c_str(),
                 static_cast<size_t>(points.size()));
    return true;
}

std::vector<RawPoint> loadRawData(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        return {};

    size_t headerBytes = 0;

    if (!checkRawPlyHeader(file, headerBytes))
        return {};

    return _loadBinaryRawData(file, headerBytes);
}

// -------------------------------------------------------------------------
// front 축 정렬
// -------------------------------------------------------------------------
// facemarker 3D center를 기준으로 정면(Z축) 정렬 회전행렬 생성
// facemarker 3D center를 기준으로 "수평(yaw)만" 정렬하는 회전행렬
Mat3 computeFaceAlignmentMatrix(const cv::Point3f &faceCenter)
{
    // 1) XZ 평면에 투영: 수직 성분(y)은 버림
    Vec3 dir = {faceCenter.x, 0.0f, faceCenter.z};

    dir = normalize(dir); // XZ 평면상의 정규화 방향

    // 2) 새 Z축: 정면(-Z) 방향에 맞추기
    //    현재 dir이 카메라에서 facemarker로 가는 수평 방향이라면,
    //    newZ = -dir 으로 두면 "facemarker 쪽이 정면"이 됨.
    Vec3 newZ = {-dir.x, 0.0f, -dir.z};

    // 3) world up은 고정: (0,1,0)
    Vec3 worldUp = {0.0f, 1.0f, 0.0f};

    // 4) 새 X/Y 축 구성 (여기서 up은 항상 (0,1,0)에 가깝게 유지됨)
    Vec3 newX = normalize(cross(worldUp, newZ));
    Vec3 newY = normalize(cross(newZ, newX)); // 이 newY는 거의 (0,1,0)

    Mat3 R{};
    // row-major: 각 행 = newX, newY, newZ
    R.m[0][0] = newX.x;
    R.m[0][1] = newX.y;
    R.m[0][2] = newX.z;
    R.m[1][0] = newY.x;
    R.m[1][1] = newY.y;
    R.m[1][2] = newY.z;
    R.m[2][0] = newZ.x;
    R.m[2][1] = newZ.y;
    R.m[2][2] = newZ.z;

    return R;
}

void rotatePointCloudInPlace(std::vector<CartesianPointRGB> &points,
                             const Mat3 &R)
{

    if (points.empty())
    {
        SENDLOGF_TAG(LOG_TAG, "[rotatePointCloudInPlace] points is empty. Skip rotation.");
        return;
    }
    for (auto &p : points)
    {
        Vec3 v{p.x, p.y, p.z};
        Vec3 r = mat3MulVec3(R, v);
        p.x = static_cast<float>(r.x);
        p.y = static_cast<float>(r.y);
        p.z = static_cast<float>(r.z);
    }
}

// 정면방향 점군 필터링
std::vector<CartesianPointRGB> filterByFrontRangeMinusZ(
    const std::vector<CartesianPointRGB> &points,
    float backMax,  // 뒤쪽 최대 거리 (예: 5.0f)
    float frontMax) // 정면 최대 거리 (예: 10.0f)
{
    const float zMin = -frontMax; // 예: -10
    const float zMax = backMax;   // 예: 5

    std::vector<CartesianPointRGB> out;
    out.reserve(points.size());

    for (const auto &p : points)
    {
        if (p.z >= zMin && p.z <= zMax)
            out.push_back(p);
    }
    return out;
}
