#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "CenterEstimation.h"
#include "PointCloudRegistration.h"

namespace fs = std::filesystem;

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

void PrintUsage()
{
    std::cout
        << "Usage: DetectionRunner <stage> [blastPath] [targetType] [options]\n"
        << "  stage      : detect | merge | adjust | all | summary\n"
        << "  blastPath  : defaults to sample_set\n"
        << "  targetType : defaults to reference\n"
        << "Options:\n"
        << "  --custom            use custom marker detector instead of ArUco\n"
        << "  --use-face-marker   align final centers using face marker id 0\n"
        << "  --target-size <cm>  target size in cm for intensity refinement (default: 13)\n";
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
        ok = Markerdetection(blastPath.string(), points, targetType, isAruco);
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
        ok = RunDetectionPipeline(blastPath.string(), targetType, isAruco, useFaceMarker, targetSize, outFrontDir);
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
