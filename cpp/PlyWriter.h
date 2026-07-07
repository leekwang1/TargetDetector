#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "PointStructures.h"
// -------------------------------------------------------------------------
// 일반 데이터 저장
// -------------------------------------------------------------------------
//  점군 카운트를 바로 저장(메모리에 모든 점군 올리는 방식) - (header는 Ascii로 저장)
//  distance 속성 선택적으로 포함하는 헤더
static void _writePlyHeader(std::ofstream &outFile,
                            size_t pointCount,
                            bool includeDistance);

// 바이너리 포인트 출력
static void _writePlyHeaderBinary(std::ofstream &outFile,
                                  size_t pointCount,
                                  bool includeDistance);

// 바이너리 점군 로드
static std::vector<CartesianPointRGB> _loadBinaryPointCloud(std::ifstream &file,
                                                            size_t headerBytes);

// ply 헤더 검사
bool checkPlyHeaderBasicValidity(std::ifstream &file,
                                 size_t &headerBytes);

// 점군 저장
bool savePointCloud(const std::string &filePath,
                    const std::vector<CartesianPointRGB> &points,
                    bool includeDistance = false);

// 점군 로드
std::vector<CartesianPointRGB> loadPointCloud(const std::string &path);

// -------------------------------------------------------------------------
// raw 데이터 저장
// -------------------------------------------------------------------------
static void _writeRawPlyHeader(std::ofstream &outFile,
                               size_t pointCount);

static void _writeRawData(std::ofstream &outFile,
                          const RawPoint &point);

static std::vector<RawPoint> _loadBinaryRawData(std::ifstream &file,
                                                size_t headerBytes);

bool saveRawPointCloud(const std::string &filePath,
                       const std::vector<RawPoint> &points);

bool checkRawPlyHeader(std::ifstream &file,
                       size_t &headerBytes);

std::vector<RawPoint> loadRawData(const std::string &path);

// -------------------------------------------------------------------------
// front 축 정렬
// -------------------------------------------------------------------------

// facemarker 3D center를 기준으로 정면(Z축) 정렬 회전행렬 생성
Mat3 computeFaceAlignmentMatrix(const cv::Point3f &faceCenter);

// 점군 회전 적용
void rotatePointCloudInPlace(std::vector<CartesianPointRGB> &points,
                             const Mat3 &R);

void transformPointCloudInPlace(std::vector<CartesianPointRGB> &points,
                                const Matrix4x4 &M);

// 정면방향 점군 필터링
std::vector<CartesianPointRGB> filterByFrontRangeMinusZ(const std::vector<CartesianPointRGB> &points,
                                                        float backMax,        // 뒤쪽 최대 거리 (예: 5.0f)
                                                        float frontMax = 30); // 정면 최대 거리 (예: 10.0f)
