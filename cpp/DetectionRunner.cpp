#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "CenterEstimation.h"
#include "PointCloudRegistration.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace
{
Mat3 IdentityMat3()
{
    Mat3 value{};
    value.m[0][0] = 1.0;
    value.m[1][1] = 1.0;
    value.m[2][2] = 1.0;
    return value;
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

json DetectionSummaryToJson(const SingleImageDetectionSummary &summary,
                            const ArucoDetectorConfig &config,
                            bool isAruco)
{
    json detections = json::array();
    json ids = json::array();

    for (const auto &detection : summary.detections)
    {
        ids.push_back(detection.id);

        json corners = json::array();
        for (const auto &corner : detection.corners)
            corners.push_back({corner.x, corner.y});

        detections.push_back({
            {"id", detection.id},
            {"center", {detection.center.x, detection.center.y}},
            {"corners", corners},
        });
    }

    return {
        {"detector_type", isAruco ? "aruco" : "custom"},
        {"image_path", summary.imagePath},
        {"output_image_path", summary.outputImagePath},
        {"image_width", summary.imageWidth},
        {"image_height", summary.imageHeight},
        {"num_detected", summary.detections.size()},
        {"num_rejected", summary.rejectedCount},
        {"ids", ids},
        {"detections", detections},
        {"parameters", ArucoConfigToJson(config)},
    };
}

void PrintUsage()
{
    std::cout
        << "Usage: DetectionRunner <stage> [blastPath] [targetType] [options]\n"
        << "  stage      : detect | merge | adjust | all | summary | detect-image\n"
        << "  blastPath  : defaults to sample_set\n"
        << "  targetType : defaults to reference\n"
        << "Options:\n"
        << "  --custom            use custom marker detector instead of ArUco\n"
        << "  --use-face-marker   align final centers using face marker id 0\n"
        << "  --target-size <cm>  target size in cm for intensity refinement (default: 13)\n"
        << "  --output-image <path>                         save detect-image visualization\n"
        << "  --adaptive-thresh-win-size-min <int>         detect-image parameter\n"
        << "  --adaptive-thresh-win-size-max <int>         detect-image parameter\n"
        << "  --adaptive-thresh-win-size-step <int>        detect-image parameter\n"
        << "  --adaptive-thresh-constant <float>           detect-image parameter\n"
        << "  --min-marker-perimeter-rate <float>          detect-image parameter\n"
        << "  --max-marker-perimeter-rate <float>          detect-image parameter\n"
        << "  --corner-refinement-method <NONE|SUBPIX|CONTOUR|APRILTAG>\n"
        << "  --corner-refinement-win-size <int>           detect-image parameter\n"
        << "  --detect-inverted-marker                     detect-image parameter\n"
        << "  --no-detect-inverted-marker                  detect-image parameter\n"
        << "  --perspective-remove-pixel-per-cell <int>    detect-image parameter\n"
        << "  --min-otsu-std-dev <float>                   detect-image parameter\n"
        << "  --perspective-remove-ignored-margin-per-cell <float>\n"
        << "                                               detect-image parameter\n"
        << "  --error-correction-rate <float>              detect-image parameter\n"
        << "  --max-erroneous-bits-in-border-rate <float>  detect-image parameter\n"
        << "  --use-aruco3-detection                       detect-image parameter\n"
        << "  --no-use-aruco3-detection                    detect-image parameter\n";
}

void PrintSummary(const fs::path &blastPath, const std::string &targetType)
{
    const fs::path detectDir = blastPath / "DetectTarget" / targetType;
    const fs::path regDir = blastPath / "Registration" / targetType;

    auto countMatching = [](const fs::path &dir, const std::string &prefix, const std::string &suffix)
    {
        size_t count = 0;
        if (!fs::exists(dir))
            return count;

        for (const auto &entry : fs::directory_iterator(dir))
        {
            if (!entry.is_regular_file())
                continue;

            const std::string name = entry.path().filename().string();
            if (name.rfind(prefix, 0) == 0 && entry.path().extension() == suffix)
                ++count;
        }
        return count;
    };

    std::cout << "blast_path: " << blastPath.string() << "\n";
    std::cout << "target_type: " << targetType << "\n";
    std::cout << "detect_dir: " << detectDir.string() << "\n";
    std::cout << "registration_dir: " << regDir.string() << "\n";
    std::cout << "input_images: " << countMatching(detectDir, "orign_", ".jpg") << "\n";
    std::cout << "detect_json: " << countMatching(detectDir, "detect_", ".json") << "\n";
    std::cout << "marked_images: " << countMatching(detectDir, "target_marked_", ".jpg") << "\n";
    std::cout << "targets3d_json: " << ((regDir / "targets3d.json").string()) << "\n";
    std::cout << "targets3d_adjust_json: " << ((regDir / "targets3d_adjust.json").string()) << "\n";
    std::cout << "debug_dir: " << (regDir / "debug").string() << "\n";
}
} // namespace

int main(int argc, char **argv)
{
    std::vector<std::string> positional;
    bool isAruco = true;
    bool useFaceMarker = false;
    float targetSize = 13.0f;
    ArucoDetectorConfig arucoConfig;
    std::string outputImagePath;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--custom")
        {
            isAruco = false;
        }
        else if (arg == "--use-face-marker")
        {
            useFaceMarker = true;
        }
        else if (arg == "--target-size")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            targetSize = std::stof(argv[++i]);
        }
        else if (arg == "--output-image")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            outputImagePath = argv[++i];
        }
        else if (arg == "--adaptive-thresh-win-size-min")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.adaptiveThreshWinSizeMin = std::stoi(argv[++i]);
        }
        else if (arg == "--adaptive-thresh-win-size-max")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.adaptiveThreshWinSizeMax = std::stoi(argv[++i]);
        }
        else if (arg == "--adaptive-thresh-win-size-step")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.adaptiveThreshWinSizeStep = std::stoi(argv[++i]);
        }
        else if (arg == "--adaptive-thresh-constant")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.adaptiveThreshConstant = std::stod(argv[++i]);
        }
        else if (arg == "--min-marker-perimeter-rate")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.minMarkerPerimeterRate = std::stod(argv[++i]);
        }
        else if (arg == "--max-marker-perimeter-rate")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.maxMarkerPerimeterRate = std::stod(argv[++i]);
        }
        else if (arg == "--corner-refinement-method")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.cornerRefinementMethod = argv[++i];
        }
        else if (arg == "--corner-refinement-win-size")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.cornerRefinementWinSize = std::stoi(argv[++i]);
        }
        else if (arg == "--detect-inverted-marker")
        {
            arucoConfig.detectInvertedMarker = true;
        }
        else if (arg == "--no-detect-inverted-marker")
        {
            arucoConfig.detectInvertedMarker = false;
        }
        else if (arg == "--perspective-remove-pixel-per-cell")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.perspectiveRemovePixelPerCell = std::stoi(argv[++i]);
        }
        else if (arg == "--min-otsu-std-dev")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.minOtsuStdDev = std::stod(argv[++i]);
        }
        else if (arg == "--perspective-remove-ignored-margin-per-cell")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.perspectiveRemoveIgnoredMarginPerCell = std::stod(argv[++i]);
        }
        else if (arg == "--error-correction-rate")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.errorCorrectionRate = std::stod(argv[++i]);
        }
        else if (arg == "--max-erroneous-bits-in-border-rate")
        {
            if (i + 1 >= argc)
            {
                PrintUsage();
                return 1;
            }
            arucoConfig.maxErroneousBitsInBorderRate = std::stod(argv[++i]);
        }
        else if (arg == "--use-aruco3-detection")
        {
            arucoConfig.useAruco3Detection = true;
        }
        else if (arg == "--no-use-aruco3-detection")
        {
            arucoConfig.useAruco3Detection = false;
        }
        else if (arg == "--help" || arg == "-h")
        {
            PrintUsage();
            return 0;
        }
        else
        {
            positional.push_back(arg);
        }
    }

    const std::string stage = positional.size() > 0 ? positional[0] : "all";

    if (stage == "detect-image")
    {
        if (positional.size() < 2)
        {
            PrintUsage();
            return 1;
        }

        fs::path imagePath = positional[1];
        if (!imagePath.is_absolute())
            imagePath = fs::absolute(imagePath);

        SingleImageDetectionSummary summary;
        const bool ok = DetectMarkerImageStandalone(imagePath.string(),
                                                    isAruco,
                                                    arucoConfig,
                                                    outputImagePath,
                                                    summary);
        std::cout << DetectionSummaryToJson(summary, arucoConfig, isAruco).dump(2) << "\n";
        return ok ? 0 : 1;
    }

    fs::path blastPath = positional.size() > 1 ? fs::path(positional[1]) : fs::path("sample_set");
    const std::string targetType = positional.size() > 2 ? positional[2] : "reference";

    if (!blastPath.is_absolute())
        blastPath = fs::absolute(blastPath);

    Mat3 outFrontDir = IdentityMat3();
    std::vector<CartesianPointRGB> points;
    const bool hasPointCloud = LoadTargetPointCloud(blastPath.string(), targetType, points);

    bool ok = false;

    if (stage == "detect")
    {
        ok = Markerdetection(blastPath.string(), points, targetType, isAruco, arucoConfig);
    }
    else if (stage == "merge")
    {
        if (!hasPointCloud)
        {
            std::cerr << "Point cloud not found for merge stage.\n";
            return 1;
        }
        ok = ExtractAndSaveAndMergePointClouds_Standalone(blastPath.string(), targetType, points);
    }
    else if (stage == "adjust")
    {
        if (!hasPointCloud)
        {
            std::cerr << "Point cloud not found for adjust stage.\n";
            return 1;
        }
        ok = ComputeIntensityCenters(blastPath.string(),
                                     targetType,
                                     points,
                                     outFrontDir,
                                     useFaceMarker,
                                     targetSize,
                                     0.15f,
                                     50.0f);
    }
    else if (stage == "all")
    {
        ok = RunDetectionPipeline(blastPath.string(), targetType, isAruco, useFaceMarker, targetSize, outFrontDir, arucoConfig);
    }
    else if (stage == "summary")
    {
        PrintSummary(blastPath, targetType);
        return 0;
    }
    else
    {
        PrintUsage();
        return 1;
    }

    PrintSummary(blastPath, targetType);
    return ok ? 0 : 1;
}
