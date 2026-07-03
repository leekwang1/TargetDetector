#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "PointStructures.h"
#include "blk360/BLK360G2.h"
#include <opencv2/core.hpp>

struct MarkerDetectionResult
{
    int id = -1;
    cv::Point2f center;
    std::vector<cv::Point2f> corners;
};

struct DetectionImageInput
{
    std::string imagePath;
    std::string calibPath;
    std::string key;
};

struct ArucoDetectorConfig
{
    int adaptiveThreshWinSizeMin = 3;
    int adaptiveThreshWinSizeMax = 23;
    int adaptiveThreshWinSizeStep = 10;
    double adaptiveThreshConstant = 7.0;
    double minMarkerPerimeterRate = 0.01;
    double maxMarkerPerimeterRate = 4.0;
    std::string cornerRefinementMethod = "SUBPIX";
    int cornerRefinementWinSize = 5;
    bool detectInvertedMarker = false;
    int perspectiveRemovePixelPerCell = 4;
    double minOtsuStdDev = 5.0;
    double perspectiveRemoveIgnoredMarginPerCell = 0.13;
    double errorCorrectionRate = 0.6;
    double maxErroneousBitsInBorderRate = 0.35;
    bool useAruco3Detection = false;
};

struct SingleImageDetectionSummary
{
    std::string imagePath;
    std::string outputImagePath;
    int imageWidth = 0;
    int imageHeight = 0;
    size_t rejectedCount = 0;
    std::vector<MarkerDetectionResult> detections;
};

bool DiscoverDetectionInputs(const std::string &blastPath,
                             const std::string &targetType,
                             std::vector<DetectionImageInput> &outInputs);

bool LoadCalibrationFromJson(const std::string &jsonPath,
                             int &width,
                             int &height,
                             Blk360G2_ImageCalibration &calib);

bool LoadTargetPointCloud(const std::string &blastPath,
                          const std::string &targetType,
                          std::vector<CartesianPointRGB> &outPoints);

bool LoadFaceMarkerCenter3D(const std::string &jsonPath,
                            cv::Point3f &outCenter);

bool DetectMarkerImageStandalone(const std::string &imagePath,
                                 bool isAruco,
                                 const ArucoDetectorConfig &config,
                                 const std::string &outputImagePath,
                                 SingleImageDetectionSummary &outSummary);

bool Markerdetection(const std::string &blastPath,
                     std::vector<CartesianPointRGB> &points,
                     const std::string &targetType,
                     bool isAruco = true,
                     const ArucoDetectorConfig &config = ArucoDetectorConfig());

bool ExtractAndSaveAndMergePointClouds_Standalone(const std::string &blastPath,
                                                  const std::string &targetType,
                                                  const std::vector<CartesianPointRGB> &pointCloud);

bool ExtractFaceMarkerCenter_Standalone(const std::string &blastPath,
                                        const std::string &targetType,
                                        const std::vector<CartesianPointRGB> &pointCloud,
                                        cv::Point3f &outFaceCenter,
                                        std::vector<std::pair<int, cv::Point2f>> &outMarkers2D);

bool FindFaceMarkerCenterFromCenters(const std::vector<std::pair<std::string, cv::Vec3f>> &centers,
                                     cv::Point3f &outCenter);

bool FindCenter3DPt(const std::string &blastPath,
                    const std::string &targetType,
                    std::vector<CartesianPointRGB> &points,
                    float targetSize,
                    bool usefacemarker,
                    Mat3 &outFrontDir);

bool RunDetectionPipeline(const std::string &blastPath,
                          const std::string &targetType,
                          bool isAruco,
                          bool usefacemarker,
                          float targetSize,
                          Mat3 &outFrontDir,
                          const ArucoDetectorConfig &config = ArucoDetectorConfig());
