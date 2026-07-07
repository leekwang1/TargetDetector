#pragma once

#include <cmath>
#include <string>
#include "blk360/BLK360G2.h"
#include "PointStructures.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

// 점군 정합 - ArUco 타겟은 타겟 센터점을 이용해 강체변환, custom ArUco 타겟은 타겟탐지 점군을 이용한 ICP
// [단계별 파이프라인]
// 1. 이미지에서 ArUco 마커 감지 & 2D 센터 계산
//      OpenCV ArUco로 마커를 찾고, 검출된 네 모서리(corners)의 평균으로 픽셀 단위 중심점(2D) 를 구함.
//      즉 center = (p0+p1+p2+p3)/4 로 계산함. 2D 중심은 id → (x,y) 맵으로 보관.
// 2. 픽셀 → 레이(ray) 변환(월드 방향)
//      파노라마/캘리브레이션에서 얻은 projection, transformation을 사용해 pixelToRayWorld()로 월드공간 단위 방향벡터(rayDir) 를 구함.
//      내부적으로 fx, fy, cx, cy를 추출하고, NDC → 카메라 → 변환행렬(T) 적용으로 월드 방향을 얻음.
// 3. 점군(reference/compared)에서 레이 근처 점 찾기
//      방법 A(정밀): findClosestPointOnRay()
//      — 레이 방향과 각도/거리 조건으로 가장 가까운 기존 점군의 점을 선택. 결과는 “기존 점군의 한 점”(x,y,z).
//      방법 B(현행 기본 로직): findPointByDepthProjection() - 방법 A 포함
//      — 먼저 위 방식으로 가장 가까운 점을 찾고, 그 점까지의 깊이(norm) 를 레이에 투영해서 rayDir * depth 로 신규 좌표를 생성. 즉 최종 결과는 “새로 계산한 좌표(레이 위 점)” 임.
// 4. 여러 이미지에 잡힌 동일 ID의 3D 센터 평균화(옵션)
//      같은 마커 ID가 여러 번 검출되면 누적해 평균 3D 센터를 만든 뒤, targets3d.json/targets3d.ply로 저장.
// 5. 레퍼런스 vs 컴페어드 페어링 & 강체변환 초기값
//      targets3d.json(ref/cmp)에서 같은 ID끼리 페어를 만들고 pairs.json 작성 → Kabsch(SVD) 로 R,t 폐형식 추정 → (필요 시) 짧은 ICP 리파인.

// [상세 내용]
// 1) 스캔 데이터 준비 & 파노라마/컬러라이저
//      최신 Setup 선택 → 오피스 이미지 수신/스티칭 → 파노라마 생성/적용 → 포인트클라우드 저장(PLY) 까지 진행. 여기서 생성되는 기본 점군 파일이 {blastPath}/{targetType}.ply. 이후 마커 탐지에서 이 점군을 참조함.

// 2) 타겟 탐지(이미지 단계) — “id + pixel” JSON 생성
//      함수: Markerdetection(blastPath, workflow, images, targetType, binary)
//      폴더 준비: DetectTarget/{targetType}.
//      각 파노라마/오피스 이미지에서 아루코(또는 커스텀) 타겟을 검출하고, 마커 ID와 픽셀 중심만 저장하는 JSON(detect_*.json)과 보정값 JSON(calib_*.json)을 기록. 또한 시각화된 검출 이미지(target_marked_*.jpg)도 저장. (헤더에 JSON 저장/로드용 시그니처가 정의되어 있고, 실제 호출은 DetectMarkersAndSaveImages 경유)
//      산출물(예):
//      DetectTarget/{targetType}/detect_{index}_{campos}.json ← { id, pixel:[x,y] } 리스트
//      DetectTarget/{targetType}/calib_{index}_{campos}.json ← width/height + projection[16] + transformation[16]
//      DetectTarget/{targetType}/target_marked_{index}_{campos}.jpg

// 3) JSON만으로 3D 추출 & 머지(테스트 버튼/뷰어에서 별도 실행)
//      함수: RunMergeFromSavedJson(blastPath, targetType, binary)
//      단계: ExtractAndSaveAndMergePointClouds_Standalone(...) 호출
//      동작: DetectTarget/{targetType} 폴더의 모든 detect_*.json을 읽고, 대응 calib_*.json을 로드 → 각 마커 픽셀을 ray로 재투영하여 {blastPath}/{targetType}.ply에서 반경 점군을 추출 → 이미지별 추출 PLY(Registration/{targetType}/target_{index}_{campos}.ply)를 저장하고 전체 머지.
//      추가 산출물(머지 쪽 개선사항 반영):
//      이미지별 PLY: Registration/{targetType}/target_{i}_{campos}.ply
//      전체 머지: Registration/{targetType}/{targetType}_reg.ply
//      ID별 버킷 저장: Registration/{targetType}/id_{ID}.ply
//      ID기준 통합본: Registration/{targetType}/{targetType}_reg_byId.ply ← id 그룹별로 합쳐 놓은 버전
//      또한, 이 과정에서 이미지들로부터 마커의 3D 센터(평균 등) 를 계산하여 Registration/{targetType}/targets3d.json에도 저장하도록 구성되어 있음(코멘트/시그니처 존재). 이후 페어링/ICP 초기 정렬에서 사용.

// 4) 레퍼런스 vs 컴페어드: ID 페어링 & 페어 파일 생성
//      함수: RunFindCenterAndMergePt(...) 또는 RunFindCenterAndMergePtCloud(...)
//      내부에서 참조/비교 각각에 대해 3)의 머지를 수행한 뒤,
//      같은 마커 ID끼리만 페어를 만들고(없는 ID는 버림), 페어 정보(JSON) 및 두 집합의 페어된 점군 PLY를 만든다(함수명으로 보면 RunPairTargetsFromJsonAndPt/RunPairTargetsFromJson가 역할). 이렇게 하면 같은 타겟끼리만 정합 대상으로 쓸 수 있음.
//      산출물(예):
//      Registration/pairs.json ← { pairs:[ {id, ref:[x,y,z], comp:[x,y,z]} ... ] } 형태(코드상 소비 형식)
//      Registration/reference/reference_reg(_byId).ply
//      Registration/compared/compared_reg(_byId).ply
// 5) ICP 실행 — 페어 기반 “초기정렬” + (필요 시) 회전 스윕 백업
//      JNI에서 runICP(target, source, result, resultTxt) 호출 → Run_ICPProcess(...) 수행.
//      Run_ICPProcess는 먼저 pairs.json을 경로에서 추정해 읽고(Registration 상위가 blastPath), Kabsch(SVD)로 R,t를 폐형식 추정한다.
//      페어 수가 3개 이상이면, 이 R,t를 그대로 최종 변환으로 사용하고, 회전 스윕(0~360°) 은 스킵한다(= 빠른 초기정렬).
//      페어가 부족하면, 예전처럼 Z-축 회전 스윕 + ICP로 최소 오차를 찾는 백업 루틴으로 간다.
//      결과:
//      변환 적용된 source 점군을 result_file(PLY)로 저장(기존 PLY의 intensity는 복원해 넣음).
//      4×4 변환 행렬은 result_txt에 기록

// ArUco 마커 타겟 탐지
static std::map<int, cv::Point2f> findAllArucoMarkerCentersById(
    const cv::Mat &image,
    int maxMarkerId = 15);

// Custom ArUco 마커 타겟 탐지
static std::map<int, cv::Point2f> findCustomMarkerCenters(
    const cv::Mat &image);

// 픽셀 좌표와 4x4 캘리브레이션 행렬을 사용해 방향 벡터(ray)를 계산
static cv::Vec3f pixelToRayWorld(int pixelX,
                                 int pixelY,
                                 int imageWidth,
                                 int imageHeight,
                                 const Blk360G2_ImageCalibration &calib);

// ray 방향과 가장 유사한 방향을 가진 포인트를 점군에서 검출 (빠름:테스트용) - 현재 미사용
static CartesianPointRGB findClosestPointByDirection(const cv::Vec3f &rayDir,                       // 입력: 기준 방향 벡터 (월드 공간 ray)
                                                     const std::vector<CartesianPointRGB> &points); // 입력: 검사할 포인트 리스트

// ray 방향에 따라 거리 및 각도 조건을 기반으로 가장 가까운 점군 검출 (정밀: 현재사용)
static CartesianPointRGB findClosestPointOnRay(const cv::Vec3f &rayDir,                      // 입력: 기준 방향 벡터 (단위 벡터)
                                               const std::vector<CartesianPointRGB> &points, // 입력: 비교할 점 목록
                                               float maxAngleThresholdDeg = 1.0f,            // 레이 방향(rayDir)과 점(point)의 방향 벡터 간의 허용 최대 각도(도 단위)
                                               float minDepthThreshold = 0.5f);              // 원점으로부터 점까지의 거리가 이 값보다 짧으면 무시 : 50cm

// ray 방향에 따라 거리 및 각도 조건을 기반으로 가장 가까운 점군 검출하고 depth값만 활용(Ray + Depth)
static CartesianPointRGB findPointByDepthProjection(const cv::Vec3f &rayDir,                      // 입력: 기준 방향 벡터 (정규화된 단위 벡터)
                                                    const std::vector<CartesianPointRGB> &points, // 입력: 비교 대상이 되는 점 목록
                                                    float maxAngleThresholdDeg = 1.0f,            // 레이 방향(rayDir)과 점(point)의 방향 벡터 간의 허용 최대 각도(도 단위)
                                                    float minDepthThreshold = 0.5f);              // 원점으로부터 점까지의 거리가 이 값보다 짧으면 무시 : 50cm

// 점군으로부터 이미지 픽셀 위치 계산
static cv::Point2f worldToPixel(const cv::Vec3f &worldPoint,
                                const Blk360G2_ImageCalibration &imgCalib,
                                int imageWidth,
                                int imageHeight);

// 보조함수 - OpenCV 변환 함수
cv::Mat convertImageToCvMat(Blk360G2_ImageHandle image);

// 보조함수 - 반경 점 필터링
static std::vector<CartesianPointRGB> findPointsInRadius(const cv::Vec3f &center,
                                                         const std::vector<CartesianPointRGB> &points,
                                                         float radius);

// 중복을 제거한 ICP용 기준점군(.ply) 저장 함수
// - allGroups: 각 이미지에서 추출한 타겟 주변 점군 리스트 (복수 그룹)
// - mergeDistance: 두 점이 이 거리 이하일 경우 동일한 점으로 간주
static bool SaveMergedRegCloud(const std::string &outputPath,
                               const std::vector<std::vector<CartesianPointRGB>> &allGroups,
                               float mergeDistance = 0.005f);

// save face marker to JSON
static bool SaveFacemarkerJson(const std::string &jsonPath,
                               const std::vector<std::pair<int, cv::Point2f>> &detections,
                               const cv::Point3f *faceCenter3D = nullptr);

// load face marker to JSON
bool LoadFaceMarkerCenter3D(const std::string &jsonPath,
                            cv::Point3f &outCenter);

// save id + pixel to JSON
static bool SaveDetectionJson(const std::string &jsonPath,
                              const std::vector<std::pair<int, cv::Point2f>> &markers);

// load id + pixel from JSON
static bool LoadPixelsFromJson(const std::string &jsonPath,
                               std::vector<std::pair<int, cv::Point2f>> &outMarkers);

// 캘리브레이션 JSON 저장: width/height + projection(16) + transformation(16)
static bool SaveCalibrationJson(const std::string &jsonPath,
                                int width, int height,
                                const Blk360G2_ImageCalibration &calib);

// 캘리브레이션 JSON 로드: width, height, projection[16], transformation[16]
static bool LoadCalibrationJson(const std::string &jsonPath,
                                int &width, int &height,
                                float proj[16], float xf[16]);

// 타겟 중심 좌표 저장
static bool SaveDetectedCentersJson(const std::string &jsonPath,
                                    const std::vector<std::pair<int, cv::Point3f>> &centers);

// 타겟 중심 좌표 로드
static bool LoadDetectedCentersJson(const std::string &jsonPath,
                                    std::map<int, cv::Point3f> &outCenters);

// 페어링 및 pair.json 저장
bool PairTargetsFromJsonAndPt(const std::string &blastPath,
                              const std::string &targetNameCmp, // "compared"
                              const std::string &targetNameRef, // "reference"
                              bool overwrite);

// isAruco == true  → ArUco 마커 탐지(findAllArucoMarkerCentersById)
// isAruco == false → 커스텀 마커 탐지(findCustomMarkerCenters)
// 출력: detect_*.json(id+pixel), calib_*.json, 표시 이미지 저장
static bool DetectMarkersAndSaveImages(const std::string &blastPath,
                                       Blk360G2_DataManipulationWorkflowHandle dataManipulationWorkflow,
                                       const std::vector<Blk360G2_ImageHandle> &images,
                                       const std::vector<CartesianPointRGB> &pointCloud,
                                       const std::string &targetType,
                                       bool isAruco = true);

// JSON만으로 점군추출 머지
static bool ExtractAndSaveAndMergePointClouds_Standalone(const std::string &blastPath,
                                                         const std::string &targetType,
                                                         const std::vector<CartesianPointRGB> &pointCloud);

// JSON + 캘리브레이션만으로 facemarker(0번)의 3D center를 추출하는 경량 함수
bool ExtractFaceMarkerCenter_Standalone(const std::string &blastPath,
                                        const std::string &targetType,
                                        const std::vector<CartesianPointRGB> &pointCloud,
                                        cv::Point3f &outFaceCenter,
                                        std::vector<std::pair<int, cv::Point2f>> &outMarkers2D);

// 이미지들에서 타겟만 탐지/저장하는 단계
// 이후 점군 추출/머지는 JSON 기반 함수에서 처리 (탐지 모드와 무관)
bool Markerdetection(const std::string &blastPath,
                     Blk360G2_DataManipulationWorkflowHandle dataManipulationWorkflow,
                     std::vector<Blk360G2_ImageHandle> &images,
                     std::vector<CartesianPointRGB> &points,
                     const std::string &targetType,
                     bool isAruco = true);

// JSON만으로 점군추출 머지
static bool MergeFromSavedJson(const std::string &blastPath,
                               const std::string &targetType,
                               std::vector<CartesianPointRGB> &points);

// facemarker(aruco 0번) center 찾는 헬퍼
bool FindFaceMarkerCenterFromCenters(
    const std::vector<std::pair<std::string, cv::Vec3f>> &centers,
    cv::Point3f &outCenter);

// 타겟 센터탐지 및 타겟 포인트 머지
bool FindCenter3DPt(const std::string &blastPath,
                    const std::string &targetType,
                    std::vector<CartesianPointRGB> &points,
                    float targetSize);
