#include "PointCloudRegistration.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include "CenterEstimation.h"
#include "PlyWriter.h"
#include "logging.h"

namespace fs = std::filesystem;
using nlohmann::json;

#define LOG_TAG "PointCloudRegistration"

namespace
{
std::tuple<int, std::string, std::string> SplitDetectionKey(const std::string &key)
{
    const size_t split = key.find('_');
    if (split == std::string::npos)
        return {std::numeric_limits<int>::max(), key, key};

    int index = std::numeric_limits<int>::max();
    try
    {
        index = std::stoi(key.substr(0, split));
    }
    catch (...)
    {
        index = std::numeric_limits<int>::max();
    }

    return {index, key.substr(split + 1), key};
}

template <typename T, typename KeyGetter>
void SortByDetectionKey(std::vector<T> &items, const KeyGetter &keyGetter)
{
    std::sort(items.begin(), items.end(), [&](const T &lhs, const T &rhs)
              { return SplitDetectionKey(keyGetter(lhs)) < SplitDetectionKey(keyGetter(rhs)); });
}

bool StartsWith(const std::string &value, const std::string &prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::string DetectionFolder(const std::string &blastPath, const std::string &targetType)
{
    return (fs::path(blastPath) / "DetectTarget" / targetType).string();
}

std::string RegistrationFolder(const std::string &blastPath, const std::string &targetType)
{
    return (fs::path(blastPath) / "Registration" / targetType).string();
}

std::string ToJsonKey(const std::string &fileName, const std::string &prefix)
{
    fs::path path(fileName);
    const std::string stem = path.stem().string();
    if (!StartsWith(stem, prefix))
        return {};
    return stem.substr(prefix.size());
}

json ArucoConfigToJson(const ArucoDetectorConfig &config)
{
    return {
        {"adaptiveThreshWinSizeMin", config.adaptiveThreshWinSizeMin},
        {"adaptiveThreshWinSizeMax", config.adaptiveThreshWinSizeMax},
        {"adaptiveThreshWinSizeStep", config.adaptiveThreshWinSizeStep},
        {"adaptiveThreshConstant", config.adaptiveThreshConstant},
        {"minMarkerPerimeterRate", config.minMarkerPerimeterRate},
        {"maxMarkerPerimeterRate", config.maxMarkerPerimeterRate},
        {"cornerRefinementMethod", config.cornerRefinementMethod},
        {"cornerRefinementWinSize", config.cornerRefinementWinSize},
        {"detectInvertedMarker", config.detectInvertedMarker},
        {"perspectiveRemovePixelPerCell", config.perspectiveRemovePixelPerCell},
        {"minOtsuStdDev", config.minOtsuStdDev},
        {"perspectiveRemoveIgnoredMarginPerCell", config.perspectiveRemoveIgnoredMarginPerCell},
        {"errorCorrectionRate", config.errorCorrectionRate},
        {"maxErroneousBitsInBorderRate", config.maxErroneousBitsInBorderRate},
        {"useAruco3Detection", config.useAruco3Detection},
    };
}

std::string ToUpperCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch)
                   { return static_cast<char>(std::toupper(ch)); });
    return value;
}

int ParseCornerRefinementMethod(const std::string &method)
{
    const std::string upper = ToUpperCopy(method);
    if (upper == "NONE" || upper == "CORNER_REFINE_NONE")
        return cv::aruco::CORNER_REFINE_NONE;
    if (upper == "CONTOUR" || upper == "CORNER_REFINE_CONTOUR")
        return cv::aruco::CORNER_REFINE_CONTOUR;
    if (upper == "APRILTAG" || upper == "CORNER_REFINE_APRILTAG")
        return cv::aruco::CORNER_REFINE_APRILTAG;
    return cv::aruco::CORNER_REFINE_SUBPIX;
}

void ApplyDetectorConfig(cv::aruco::DetectorParameters &params,
                         const ArucoDetectorConfig &config)
{
    params.adaptiveThreshWinSizeMin = config.adaptiveThreshWinSizeMin;
    params.adaptiveThreshWinSizeMax = config.adaptiveThreshWinSizeMax;
    params.adaptiveThreshWinSizeStep = config.adaptiveThreshWinSizeStep;
    params.adaptiveThreshConstant = config.adaptiveThreshConstant;
    params.minMarkerPerimeterRate = config.minMarkerPerimeterRate;
    params.maxMarkerPerimeterRate = config.maxMarkerPerimeterRate;
    params.cornerRefinementMethod =
        static_cast<cv::aruco::CornerRefineMethod>(ParseCornerRefinementMethod(config.cornerRefinementMethod));
    params.cornerRefinementWinSize = config.cornerRefinementWinSize;
    params.detectInvertedMarker = config.detectInvertedMarker;
    params.perspectiveRemovePixelPerCell = config.perspectiveRemovePixelPerCell;
    params.minOtsuStdDev = config.minOtsuStdDev;
    params.perspectiveRemoveIgnoredMarginPerCell = config.perspectiveRemoveIgnoredMarginPerCell;
    params.errorCorrectionRate = config.errorCorrectionRate;
    params.maxErroneousBitsInBorderRate = config.maxErroneousBitsInBorderRate;
    params.useAruco3Detection = config.useAruco3Detection;
}

std::vector<MarkerDetectionResult> BuildMarkerResults(const std::vector<std::vector<cv::Point2f>> &corners,
                                                      const std::vector<int> &ids)
{
    std::vector<MarkerDetectionResult> result;
    result.reserve(ids.size());

    for (size_t i = 0; i < ids.size(); ++i)
    {
        cv::Point2f center(0.0f, 0.0f);
        for (const auto &pt : corners[i])
            center += pt;
        center *= 0.25f;

        MarkerDetectionResult item;
        item.id = ids[i];
        item.center = center;
        item.corners = corners[i];
        result.push_back(item);
    }

    return result;
}

bool SaveDetectionMetaJson(const std::string &detectFolder,
                           bool isAruco,
                           const ArucoDetectorConfig &config)
{
    json meta;
    meta["schema_version"] = "detect_meta_v1";
    meta["target_type"] = fs::path(detectFolder).filename().string();
    meta["detector_type"] = isAruco ? "aruco" : "custom";
    meta["dictionary"] = isAruco ? "DICT_4X4_50" : "CUSTOM_4X4";
    meta["parameters"] = ArucoConfigToJson(config);

    const fs::path metaPath = fs::path(detectFolder) / "detect_meta.json";
    std::ofstream ofs(metaPath);
    if (!ofs.is_open())
        return false;

    ofs << meta.dump(2);
    return true;
}

cv::aruco::Dictionary createCustomDictionaryFromBits()
{
    constexpr int markerSize = 4;
    constexpr int maxCorrectionBits = 0;

    cv::Mat markerBits(markerSize, markerSize, CV_8UC1);
    uchar data[16] = {
        0, 0, 1, 1,
        0, 0, 1, 1,
        1, 1, 0, 0,
        1, 1, 0, 0};
    std::memcpy(markerBits.data, data, sizeof(data));

    cv::Mat byteList = cv::aruco::Dictionary::getByteListFromBits(markerBits);
    cv::Mat fullList(10, byteList.cols, byteList.type());
    for (int i = 0; i < fullList.rows; ++i)
        byteList.copyTo(fullList.row(i));

    cv::aruco::Dictionary dict;
    dict.bytesList = fullList;
    dict.markerSize = markerSize;
    dict.maxCorrectionBits = maxCorrectionBits;
    return dict;
}

std::vector<MarkerDetectionResult> findAllArucoMarkerCentersById(const cv::Mat &image,
                                                                 const ArucoDetectorConfig &config,
                                                                 size_t *outRejectedCount = nullptr)
{
    cv::aruco::Dictionary dictionary =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    cv::aruco::DetectorParameters params;
    ApplyDetectorConfig(params, config);

    cv::aruco::ArucoDetector detector(dictionary, params);

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> rejected;
    detector.detectMarkers(image, corners, ids, rejected);

    if (outRejectedCount != nullptr)
        *outRejectedCount = rejected.size();

    SENDLOGF_TAG(LOG_TAG, "Detected %d aruco markers", static_cast<int>(ids.size()));
    return BuildMarkerResults(corners, ids);
}

std::vector<MarkerDetectionResult> findCustomMarkerCenters(const cv::Mat &image,
                                                           const ArucoDetectorConfig &config,
                                                           size_t *outRejectedCount = nullptr)
{
    cv::aruco::Dictionary dict = createCustomDictionaryFromBits();
    cv::aruco::DetectorParameters params;
    ApplyDetectorConfig(params, config);

    cv::aruco::ArucoDetector detector(dict, params);

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> rejected;
    detector.detectMarkers(image, corners, ids, rejected);

    if (outRejectedCount != nullptr)
        *outRejectedCount = rejected.size();

    SENDLOGF_TAG(LOG_TAG, "Detected %d custom markers", static_cast<int>(ids.size()));
    return BuildMarkerResults(corners, ids);
}

cv::Vec3f pixelToRayWorld(int pixelX,
                          int pixelY,
                          int imageWidth,
                          int imageHeight,
                          const Blk360G2_ImageCalibration &calib)
{
    const float fx = static_cast<float>(calib.projection.elements[0]);
    const float fy = static_cast<float>(calib.projection.elements[5]);
    const float cx = static_cast<float>(calib.projection.elements[8]);
    const float cy = static_cast<float>(calib.projection.elements[9]);

    const float ndcX = 2.0f * (pixelX + 0.5f) / std::max(1, imageWidth) - 1.0f;
    const float ndcY = 1.0f - 2.0f * (pixelY + 0.5f) / std::max(1, imageHeight);

    const float x = (ndcX - cx) / fx;
    const float y = (ndcY - cy) / fy;
    const float z = -1.0f;

    cv::Vec3f rayCam = cv::normalize(cv::Vec3f(x, y, z));

    cv::Matx44f transform;
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
            transform(row, col) = static_cast<float>(calib.transformation.elements[col * 4 + row]);
    }

    const cv::Vec4f rayWorldHomo = transform * cv::Vec4f(rayCam[0], rayCam[1], rayCam[2], 0.0f);
    return cv::normalize(cv::Vec3f(rayWorldHomo[0], rayWorldHomo[1], rayWorldHomo[2]));
}

CartesianPointRGB findClosestPointOnRay(const cv::Vec3f &rayDir,
                                        const std::vector<CartesianPointRGB> &points,
                                        float maxAngleThresholdDeg = 1.0f,
                                        float minDepthThreshold = 0.5f)
{
    CartesianPointRGB closestPoint{};
    float minDist = std::numeric_limits<float>::max();

    for (const auto &p : points)
    {
        cv::Vec3f dir(p.x, p.y, p.z);
        const float norm = cv::norm(dir);
        if (norm < minDepthThreshold)
            continue;

        const float cosine = std::clamp(dir.dot(rayDir) / norm, -1.0f, 1.0f);
        const float angleDeg = std::acos(cosine) * 180.0f / static_cast<float>(CV_PI);
        if (angleDeg > maxAngleThresholdDeg)
            continue;

        if (rayDir.dot(dir) <= 0.0f)
            continue;

        const float dist = cv::norm(dir - rayDir * norm);
        if (dist < minDist)
        {
            minDist = dist;
            closestPoint = p;
        }
    }

    return closestPoint;
}

CartesianPointRGB findPointByDepthProjection(const cv::Vec3f &rayDir,
                                             const std::vector<CartesianPointRGB> &points,
                                             float maxAngleThresholdDeg = 1.0f,
                                             float minDepthThreshold = 0.5f)
{
    const CartesianPointRGB closestPoint =
        findClosestPointOnRay(rayDir, points, maxAngleThresholdDeg, minDepthThreshold);

    const float closestNorm = cv::norm(cv::Vec3f(closestPoint.x, closestPoint.y, closestPoint.z));
    if (closestNorm <= 1e-6f || std::isnan(closestNorm))
        return {};

    const cv::Vec3f refined = rayDir * closestNorm;

    CartesianPointRGB result{};
    result.x = refined[0];
    result.y = refined[1];
    result.z = refined[2];
    result.intensity = closestPoint.intensity;
    return result;
}

std::vector<CartesianPointRGB> findPointsInRadius(const cv::Vec3f &center,
                                                  const std::vector<CartesianPointRGB> &points,
                                                  float radius)
{
    std::vector<CartesianPointRGB> result;
    const float r2 = radius * radius;

    for (const auto &p : points)
    {
        const float dx = p.x - center[0];
        const float dy = p.y - center[1];
        const float dz = p.z - center[2];
        const float dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 <= r2)
            result.push_back(p);
    }

    return result;
}

bool SaveMergedRegCloud(const std::string &outputPath,
                        const std::vector<std::vector<CartesianPointRGB>> &allGroups,
                        float mergeDistance = 0.005f)
{
    std::vector<CartesianPointRGB> mergedPoints;

    auto isClose = [mergeDistance](const CartesianPointRGB &a, const CartesianPointRGB &b)
    {
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        const float dz = a.z - b.z;
        return (dx * dx + dy * dy + dz * dz) < (mergeDistance * mergeDistance);
    };

    for (const auto &group : allGroups)
    {
        for (const auto &pt : group)
        {
            bool duplicate = false;
            for (const auto &existing : mergedPoints)
            {
                if (isClose(pt, existing))
                {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate)
                mergedPoints.push_back(pt);
        }
    }

    if (!savePointCloud(outputPath, mergedPoints))
    {
        SENDLOGF_TAG(LOG_TAG, "Failed to save merged point cloud: %s", outputPath.c_str());
        return false;
    }

    return true;
}

bool SaveDetectionJson(const std::string &jsonPath,
                       const std::vector<MarkerDetectionResult> &markers)
{
    fs::create_directories(fs::path(jsonPath).parent_path());

    json j;
    j["markers"] = json::array();

    for (const auto &mk : markers)
    {
        json corners = json::array();
        for (const auto &pt : mk.corners)
            corners.push_back({pt.x, pt.y});

        j["markers"].push_back({
            {"id", mk.id},
            {"pixel", {mk.center.x, mk.center.y}},
            {"corners", corners},
        });
    }

    std::ofstream ofs(jsonPath, std::ios::binary);
    if (!ofs.is_open())
        return false;

    ofs << j.dump(2);
    return true;
}

bool LoadPixelsFromJson(const std::string &jsonPath,
                        std::vector<std::pair<int, cv::Point2f>> &outMarkers)
{
    outMarkers.clear();

    std::ifstream ifs(jsonPath, std::ios::binary);
    if (!ifs.is_open())
        return false;

    json j;
    try
    {
        ifs >> j;
    }
    catch (...)
    {
        return false;
    }

    if (!j.contains("markers") || !j["markers"].is_array())
        return false;

    for (const auto &marker : j["markers"])
    {
        if (!marker.contains("id") || !marker.contains("pixel"))
            continue;
        if (!marker["pixel"].is_array() || marker["pixel"].size() != 2)
            continue;

        outMarkers.emplace_back(
            marker["id"].get<int>(),
            cv::Point2f(static_cast<float>(marker["pixel"][0].get<double>()),
                        static_cast<float>(marker["pixel"][1].get<double>())));
    }

    return true;
}

bool SaveDetectedCentersJson(const std::string &jsonPath,
                             const std::vector<std::pair<int, cv::Point3f>> &centers)
{
    try
    {
        json out = json::array();
        for (const auto &center : centers)
        {
            char idBuffer[16];
            std::snprintf(idBuffer, sizeof(idBuffer), "T%03d", center.first);
            out.push_back({
                {"id", std::string(idBuffer)},
                {"center", {center.second.x, center.second.y, center.second.z}},
            });
        }

        std::ofstream ofs(jsonPath);
        if (!ofs.is_open())
            return false;

        ofs << out.dump(2);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool SaveFacemarkerJson(const std::string &jsonPath,
                        const std::vector<std::pair<int, cv::Point2f>> &detections,
                        const cv::Point3f *faceCenter3D = nullptr)
{
    try
    {
        json root;
        root["facemarkers"] = json::array();

        for (const auto &detection : detections)
        {
            root["facemarkers"].push_back({
                {"id", detection.first},
                {"pixel", {detection.second.x, detection.second.y}},
            });
        }

        if (faceCenter3D != nullptr)
        {
            root["facemarkers3D"] = json::array({
                {
                    {"id", 0},
                    {"center", {faceCenter3D->x, faceCenter3D->y, faceCenter3D->z}},
                },
            });
        }

        std::ofstream ofs(jsonPath);
        if (!ofs.is_open())
            return false;

        ofs << root.dump(2);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

void DrawMarkerDetections(cv::Mat &vis,
                          const std::vector<MarkerDetectionResult> &markerResults)
{
    for (const auto &marker : markerResults)
    {
        cv::drawMarker(vis, marker.center, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 24, 2);
        for (const auto &corner : marker.corners)
            cv::circle(vis, corner, 4, cv::Scalar(255, 0, 0), -1);
        cv::putText(vis,
                    std::to_string(marker.id),
                    cv::Point(static_cast<int>(marker.center.x) + 8, static_cast<int>(marker.center.y) - 8),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(0, 0, 255),
                    2);
    }
}

} // namespace

bool DetectMarkerImageStandalone(const std::string &imagePath,
                                 bool isAruco,
                                 const ArucoDetectorConfig &config,
                                 const std::string &outputImagePath,
                                 SingleImageDetectionSummary &outSummary)
{
    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        SENDLOGF_TAG(LOG_TAG, "Failed to load image: %s", imagePath.c_str());
        return false;
    }

    size_t rejectedCount = 0;
    std::vector<MarkerDetectionResult> markerResults =
        isAruco ? findAllArucoMarkerCentersById(image, config, &rejectedCount)
                : findCustomMarkerCenters(image, config, &rejectedCount);

    cv::Mat vis = image.clone();
    DrawMarkerDetections(vis, markerResults);

    fs::path savedPath = outputImagePath.empty()
                             ? (fs::path(imagePath).parent_path() / (fs::path(imagePath).stem().string() + "_aruco_result.jpg"))
                             : fs::path(outputImagePath);
    if (!savedPath.is_absolute())
        savedPath = fs::absolute(savedPath);
    fs::create_directories(savedPath.parent_path());

    if (!cv::imwrite(savedPath.string(), vis))
    {
        SENDLOGF_TAG(LOG_TAG, "Failed to save marked image: %s", savedPath.string().c_str());
        return false;
    }

    outSummary.imagePath = imagePath;
    outSummary.outputImagePath = savedPath.string();
    outSummary.imageWidth = image.cols;
    outSummary.imageHeight = image.rows;
    outSummary.rejectedCount = rejectedCount;
    outSummary.detections = std::move(markerResults);
    return true;
}

namespace
{

bool DetectMarkersAndSaveImages(const std::string &blastPath,
                                const std::vector<CartesianPointRGB> &pointCloud,
                                const std::string &targetType,
                                bool isAruco,
                                const ArucoDetectorConfig &config)
{
    std::vector<DetectionImageInput> inputs;
    if (!DiscoverDetectionInputs(blastPath, targetType, inputs))
    {
        SENDLOGF_TAG(LOG_TAG, "No detection inputs found for targetType=%s", targetType.c_str());
        return false;
    }

    const std::string detectFolder = DetectionFolder(blastPath, targetType);
    fs::create_directories(detectFolder);
    SaveDetectionMetaJson(detectFolder, isAruco, config);

    std::vector<std::pair<int, cv::Point2f>> facemarkerDetections;

    for (const auto &input : inputs)
    {
        const cv::Mat image = cv::imread(input.imagePath, cv::IMREAD_COLOR);
        if (image.empty())
        {
            SENDLOGF_TAG(LOG_TAG, "Failed to load image: %s", input.imagePath.c_str());
            continue;
        }

        std::vector<MarkerDetectionResult> markerResults =
            isAruco ? findAllArucoMarkerCentersById(image, config)
                    : findCustomMarkerCenters(image, config);

        cv::Mat vis = image.clone();
        DrawMarkerDetections(vis, markerResults);
        for (const auto &marker : markerResults)
        {
            if (marker.id == 0)
                facemarkerDetections.emplace_back(marker.id, marker.center);
        }

        const fs::path detectPath = fs::path(detectFolder) / ("detect_" + input.key + ".json");
        const fs::path markedPath = fs::path(detectFolder) / ("target_marked_" + input.key + ".jpg");

        if (!SaveDetectionJson(detectPath.string(), markerResults))
            SENDLOGF_TAG(LOG_TAG, "Failed to save detection json: %s", detectPath.string().c_str());

        if (!cv::imwrite(markedPath.string(), vis))
            SENDLOGF_TAG(LOG_TAG, "Failed to save marked image: %s", markedPath.string().c_str());
    }

    if (!facemarkerDetections.empty() && !pointCloud.empty())
    {
        cv::Point3f faceCenter;
        std::vector<std::pair<int, cv::Point2f>> markers2D;
        if (!ExtractFaceMarkerCenter_Standalone(blastPath, targetType, pointCloud, faceCenter, markers2D))
        {
            SENDLOGF_TAG(LOG_TAG, "Facemarker extraction failed after 2D detection");
        }
    }
    else if (!facemarkerDetections.empty())
    {
        const fs::path facemarkerPath = fs::path(detectFolder) / "detect_facemarker.json";
        SaveFacemarkerJson(facemarkerPath.string(), facemarkerDetections);
    }

    return true;
}

std::vector<fs::path> DiscoverDetectionJsonPaths(const std::string &blastPath,
                                                 const std::string &targetType)
{
    std::vector<fs::path> jsonPaths;
    const fs::path detectDir = fs::path(DetectionFolder(blastPath, targetType));
    if (!fs::exists(detectDir))
        return jsonPaths;

    for (const auto &entry : fs::directory_iterator(detectDir))
    {
        if (!entry.is_regular_file())
            continue;

        const std::string name = entry.path().filename().string();
        if (!StartsWith(name, "detect_"))
            continue;
        if (name == "detect_meta.json" || name == "detect_facemarker.json")
            continue;
        if (entry.path().extension() != ".json")
            continue;

        jsonPaths.push_back(entry.path());
    }

    SortByDetectionKey(jsonPaths, [](const fs::path &path)
                       { return ToJsonKey(path.filename().string(), "detect_"); });
    return jsonPaths;
}

std::vector<CartesianPointRGB> ConvertRawToCartesian(const std::vector<RawPoint> &rawPoints)
{
    std::vector<CartesianPointRGB> out;
    out.reserve(rawPoints.size());

    for (const auto &raw : rawPoints)
    {
        PolarPoint polar{};
        polar.hAngle = raw.horizontalAngle;
        polar.vAngle = raw.verticalAngle;
        polar.distance = raw.distance;
        polar.intensity = raw.intensity;

        const CartesianPoint cart = polar.toCartesian();
        CartesianPointRGB point{};
        point.x = cart.x;
        point.y = cart.y;
        point.z = cart.z;
        point.intensity = cart.intensity;
        const float scaled = std::clamp(cart.intensity * 255.0f, 0.0f, 255.0f);
        point.r = scaled;
        point.g = scaled;
        point.b = scaled;
        out.push_back(point);
    }

    return out;
}

Mat3 IdentityMat3()
{
    Mat3 identity{};
    identity.m[0][0] = 1.0;
    identity.m[1][1] = 1.0;
    identity.m[2][2] = 1.0;
    return identity;
}
} // namespace

bool DiscoverDetectionInputs(const std::string &blastPath,
                             const std::string &targetType,
                             std::vector<DetectionImageInput> &outInputs)
{
    outInputs.clear();

    const fs::path detectDir = fs::path(blastPath) / "DetectTarget" / targetType;
    if (!fs::exists(detectDir))
        return false;

    for (const auto &entry : fs::directory_iterator(detectDir))
    {
        if (!entry.is_regular_file())
            continue;

        const std::string name = entry.path().filename().string();
        if (!StartsWith(name, "orign_"))
            continue;

        const std::string extension = entry.path().extension().string();
        if (extension != ".jpg" && extension != ".jpeg" && extension != ".png" &&
            extension != ".JPG" && extension != ".JPEG" && extension != ".PNG")
            continue;

        const std::string key = entry.path().stem().string().substr(std::string("orign_").size());
        const fs::path calibPath = detectDir / ("calib_" + key + ".json");
        if (!fs::exists(calibPath))
        {
            SENDLOGF_TAG(LOG_TAG, "Skip image without calib json: %s", entry.path().string().c_str());
            continue;
        }

        outInputs.push_back({
            entry.path().string(),
            calibPath.string(),
            key,
        });
    }

    SortByDetectionKey(outInputs, [](const DetectionImageInput &input)
                       { return input.key; });
    return !outInputs.empty();
}

bool LoadCalibrationFromJson(const std::string &jsonPath,
                             int &width,
                             int &height,
                             Blk360G2_ImageCalibration &calib)
{
    width = 0;
    height = 0;
    calib = {};

    std::ifstream ifs(jsonPath, std::ios::binary);
    if (!ifs.is_open())
        return false;

    json j;
    try
    {
        ifs >> j;
    }
    catch (...)
    {
        return false;
    }

    if (!j.contains("width") || !j.contains("height") ||
        !j.contains("projection") || !j.contains("transformation"))
        return false;

    if (!j["projection"].is_array() || !j["transformation"].is_array() ||
        j["projection"].size() != 16 || j["transformation"].size() != 16)
        return false;

    width = j["width"].get<int>();
    height = j["height"].get<int>();

    for (int i = 0; i < 16; ++i)
    {
        calib.projection.elements[i] = j["projection"][i].get<double>();
        calib.transformation.elements[i] = j["transformation"][i].get<double>();
    }

    return width > 0 && height > 0;
}

bool LoadTargetPointCloud(const std::string &blastPath,
                          const std::string &targetType,
                          std::vector<CartesianPointRGB> &outPoints)
{
    const fs::path pointCloudPath = fs::path(blastPath) / (targetType + ".ply");
    outPoints = loadPointCloud(pointCloudPath.string());
    if (!outPoints.empty())
        return true;

    const fs::path rawPath = fs::path(blastPath) / "Raw" / (targetType + "_raw.ply");
    const std::vector<RawPoint> rawPoints = loadRawData(rawPath.string());
    if (rawPoints.empty())
        return false;

    outPoints = ConvertRawToCartesian(rawPoints);
    return !outPoints.empty();
}

bool LoadFaceMarkerCenter3D(const std::string &jsonPath,
                            cv::Point3f &outCenter)
{
    std::ifstream ifs(jsonPath);
    if (!ifs.is_open())
        return false;

    json j;
    try
    {
        ifs >> j;
    }
    catch (...)
    {
        return false;
    }

    if (!j.contains("facemarkers3D") || !j["facemarkers3D"].is_array() || j["facemarkers3D"].empty())
        return false;

    const auto &entry = j["facemarkers3D"].front();
    if (!entry.contains("center") || !entry["center"].is_array() || entry["center"].size() != 3)
        return false;

    outCenter = cv::Point3f(entry["center"][0].get<float>(),
                            entry["center"][1].get<float>(),
                            entry["center"][2].get<float>());
    return true;
}

bool Markerdetection(const std::string &blastPath,
                     std::vector<CartesianPointRGB> &points,
                     const std::string &targetType,
                     bool isAruco,
                     const ArucoDetectorConfig &config)
{
    const auto start = std::chrono::steady_clock::now();

    if (!DetectMarkersAndSaveImages(blastPath, points, targetType, isAruco, config))
        return false;

    const auto end = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    const auto sec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    SENDLOGF_TAG("ProcessTime", "Markerdetection time: %lld ms (%lld sec)", ms, sec);
    return true;
}

bool ExtractAndSaveAndMergePointClouds_Standalone(const std::string &blastPath,
                                                  const std::string &targetType,
                                                  const std::vector<CartesianPointRGB> &pointCloud)
{
    std::vector<std::pair<int, cv::Point3f>> detectedCenters;

    const fs::path regDir = RegistrationFolder(blastPath, targetType);
    fs::create_directories(regDir);

    std::vector<std::vector<CartesianPointRGB>> allTargetGroups;
    std::unordered_map<int, std::vector<CartesianPointRGB>> idBuckets;

    const auto detectJsons = DiscoverDetectionJsonPaths(blastPath, targetType);
    for (const auto &jsonPath : detectJsons)
    {
        const std::string key = ToJsonKey(jsonPath.filename().string(), "detect_");
        const fs::path calibPath = fs::path(DetectionFolder(blastPath, targetType)) / ("calib_" + key + ".json");

        std::vector<std::pair<int, cv::Point2f>> markers;
        if (!LoadPixelsFromJson(jsonPath.string(), markers) || markers.empty())
        {
            SENDLOGF_TAG(LOG_TAG, "No markers found in %s", jsonPath.string().c_str());
            continue;
        }

        int width = 0;
        int height = 0;
        Blk360G2_ImageCalibration calib{};
        if (!LoadCalibrationFromJson(calibPath.string(), width, height, calib))
        {
            SENDLOGF_TAG(LOG_TAG, "Failed to load calib: %s", calibPath.string().c_str());
            continue;
        }

        std::vector<CartesianPointRGB> nearbyAll;
        nearbyAll.reserve(markers.size() * 2000);

        for (const auto &[id, pixel] : markers)
        {
            const cv::Vec3f ray = pixelToRayWorld(static_cast<int>(pixel.x),
                                                  static_cast<int>(pixel.y),
                                                  width,
                                                  height,
                                                  calib);

            const CartesianPointRGB targetPos = findPointByDepthProjection(ray, pointCloud);
            const cv::Vec3f center(targetPos.x, targetPos.y, targetPos.z);
            if (cv::norm(center) <= 1e-6f)
                continue;

            detectedCenters.emplace_back(id, cv::Point3f(targetPos.x, targetPos.y, targetPos.z));

            auto nearby = findPointsInRadius(center, pointCloud, 0.15f);
            nearbyAll.insert(nearbyAll.end(), nearby.begin(), nearby.end());

            auto &bucket = idBuckets[id];
            bucket.insert(bucket.end(), nearby.begin(), nearby.end());
        }

        const fs::path perImagePly = regDir / ("target_" + key + ".ply");
        if (!nearbyAll.empty())
        {
            if (!savePointCloud(perImagePly.string(), nearbyAll))
                return false;
            allTargetGroups.push_back(std::move(nearbyAll));
        }
    }

    {
        const float mergeThresholdM = 0.05f;
        const int minVotes = 2;

        std::unordered_map<int, std::vector<cv::Point3f>> groupedCenters;
        for (const auto &entry : detectedCenters)
            groupedCenters[entry.first].push_back(entry.second);

        auto distance = [](const cv::Point3f &a, const cv::Point3f &b)
        {
            const float dx = a.x - b.x;
            const float dy = a.y - b.y;
            const float dz = a.z - b.z;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        };

        auto supportCount = [&](const cv::Point3f &center, float radiusM)
        {
            const float r2 = radiusM * radiusM;
            int count = 0;
            for (const auto &point : pointCloud)
            {
                const float dx = point.x - center.x;
                const float dy = point.y - center.y;
                const float dz = point.z - center.z;
                if (dx * dx + dy * dy + dz * dz <= r2)
                    ++count;
            }
            return count;
        };

        std::vector<std::pair<int, cv::Point3f>> filtered;
        for (auto &[id, centers] : groupedCenters)
        {
            struct Cluster
            {
                cv::Point3f mean;
                int count = 0;
            };

            std::vector<Cluster> clusters;
            for (const auto &center : centers)
            {
                int bestIndex = -1;
                float bestDistance = mergeThresholdM;
                for (int i = 0; i < static_cast<int>(clusters.size()); ++i)
                {
                    const float d = distance(center, clusters[i].mean);
                    if (d < bestDistance)
                    {
                        bestDistance = d;
                        bestIndex = i;
                    }
                }

                if (bestIndex < 0)
                {
                    clusters.push_back({center, 1});
                }
                else
                {
                    Cluster &cluster = clusters[bestIndex];
                    const float count = static_cast<float>(cluster.count);
                    cluster.mean.x = (cluster.mean.x * count + center.x) / (count + 1.0f);
                    cluster.mean.y = (cluster.mean.y * count + center.y) / (count + 1.0f);
                    cluster.mean.z = (cluster.mean.z * count + center.z) / (count + 1.0f);
                    cluster.count += 1;
                }
            }

            int bestClusterIndex = -1;
            int bestClusterCount = 0;
            for (int i = 0; i < static_cast<int>(clusters.size()); ++i)
            {
                if (clusters[i].count > bestClusterCount)
                {
                    bestClusterCount = clusters[i].count;
                    bestClusterIndex = i;
                }
            }

            if (bestClusterIndex >= 0 && bestClusterCount >= minVotes)
            {
                filtered.emplace_back(id, clusters[bestClusterIndex].mean);
            }
            else
            {
                const float supportRadiusM = 0.04f;
                int bestIndex = -1;
                int bestSupport = -1;
                for (int i = 0; i < static_cast<int>(centers.size()); ++i)
                {
                    const int support = supportCount(centers[i], supportRadiusM);
                    if (support > bestSupport)
                    {
                        bestSupport = support;
                        bestIndex = i;
                    }
                }

                if (bestIndex >= 0)
                    filtered.emplace_back(id, centers[bestIndex]);
            }
        }

        detectedCenters.swap(filtered);
    }

    const fs::path centersJson = regDir / "targets3d.json";
    SaveDetectedCentersJson(centersJson.string(), detectedCenters);

    std::vector<CartesianPointRGB> centerPoints;
    centerPoints.reserve(detectedCenters.size());
    for (const auto &entry : detectedCenters)
    {
        CartesianPointRGB point{};
        point.x = entry.second.x;
        point.y = entry.second.y;
        point.z = entry.second.z;
        point.r = 255;
        point.g = 0;
        point.b = 0;
        centerPoints.push_back(point);
    }
    savePointCloud((regDir / "targets3d.ply").string(), centerPoints);

    if (!allTargetGroups.empty())
    {
        SaveMergedRegCloud((regDir / (targetType + "_reg.ply")).string(), allTargetGroups);

        std::vector<std::vector<CartesianPointRGB>> idGroups;
        idGroups.reserve(idBuckets.size());
        for (const auto &[id, points] : idBuckets)
        {
            const fs::path idPath = regDir / ("id_" + std::to_string(id) + ".ply");
            savePointCloud(idPath.string(), points);
            idGroups.push_back(points);
        }

        if (!idGroups.empty())
            SaveMergedRegCloud((regDir / (targetType + "_reg_byId.ply")).string(), idGroups);
    }

    return true;
}

bool ExtractFaceMarkerCenter_Standalone(const std::string &blastPath,
                                        const std::string &targetType,
                                        const std::vector<CartesianPointRGB> &pointCloud,
                                        cv::Point3f &outFaceCenter,
                                        std::vector<std::pair<int, cv::Point2f>> &outMarkers2D)
{
    outMarkers2D.clear();
    std::vector<cv::Point3f> faceCandidates3D;

    const auto detectJsons = DiscoverDetectionJsonPaths(blastPath, targetType);
    for (const auto &jsonPath : detectJsons)
    {
        const std::string key = ToJsonKey(jsonPath.filename().string(), "detect_");
        const fs::path calibPath = fs::path(DetectionFolder(blastPath, targetType)) / ("calib_" + key + ".json");

        std::vector<std::pair<int, cv::Point2f>> markers;
        if (!LoadPixelsFromJson(jsonPath.string(), markers))
            continue;

        int width = 0;
        int height = 0;
        Blk360G2_ImageCalibration calib{};
        if (!LoadCalibrationFromJson(calibPath.string(), width, height, calib))
            continue;

        for (const auto &[id, pixel] : markers)
        {
            if (id != 0)
                continue;

            const cv::Vec3f ray = pixelToRayWorld(static_cast<int>(pixel.x),
                                                  static_cast<int>(pixel.y),
                                                  width,
                                                  height,
                                                  calib);
            const CartesianPointRGB targetPos = findPointByDepthProjection(ray, pointCloud);
            const cv::Vec3f center(targetPos.x, targetPos.y, targetPos.z);
            if (cv::norm(center) <= 1e-6f)
                continue;

            faceCandidates3D.emplace_back(targetPos.x, targetPos.y, targetPos.z);
            outMarkers2D.emplace_back(id, pixel);
        }
    }

    if (faceCandidates3D.empty())
        return false;

    outFaceCenter = faceCandidates3D.front();
    const fs::path facemarkerPath = fs::path(DetectionFolder(blastPath, targetType)) / "detect_facemarker.json";
    return SaveFacemarkerJson(facemarkerPath.string(), outMarkers2D, &outFaceCenter);
}

bool FindFaceMarkerCenterFromCenters(const std::vector<std::pair<std::string, cv::Vec3f>> &centers,
                                     cv::Point3f &outCenter)
{
    for (const auto &entry : centers)
    {
        const std::string &idString = entry.first;
        const char *digits = idString.c_str();
        if (!idString.empty() && (idString[0] == 'T' || idString[0] == 't'))
            digits = idString.c_str() + 1;

        if (std::atoi(digits) == 0)
        {
            outCenter = cv::Point3f(entry.second[0], entry.second[1], entry.second[2]);
            return true;
        }
    }

    return false;
}

bool FindCenter3DPt(const std::string &blastPath,
                    const std::string &targetType,
                    std::vector<CartesianPointRGB> &points,
                    float targetSize,
                    bool usefacemarker,
                    Mat3 &outFrontDir)
{
    if (!ExtractAndSaveAndMergePointClouds_Standalone(blastPath, targetType, points))
        return false;

    return ComputeIntensityCenters(blastPath,
                                   targetType,
                                   points,
                                   outFrontDir,
                                   usefacemarker,
                                   targetSize,
                                   0.15f,
                                   50.0f);
}

bool RunDetectionPipeline(const std::string &blastPath,
                          const std::string &targetType,
                          bool isAruco,
                          bool usefacemarker,
                          float targetSize,
                          Mat3 &outFrontDir,
                          const ArucoDetectorConfig &config)
{
    outFrontDir = IdentityMat3();

    std::vector<CartesianPointRGB> points;
    if (!LoadTargetPointCloud(blastPath, targetType, points))
    {
        SENDLOGF_TAG(LOG_TAG, "Point cloud not found for %s", targetType.c_str());
        return false;
    }

    if (!Markerdetection(blastPath, points, targetType, isAruco, config))
        return false;

    return FindCenter3DPt(blastPath, targetType, points, targetSize, usefacemarker, outFrontDir);
}
