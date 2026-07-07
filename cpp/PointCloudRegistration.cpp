#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <string_view>
#include <sys/stat.h>
#include <regex>
#include <json.hpp>
#include <unordered_map>
#include "NativeConfig.h"
#include "CenterEstimation.h"
#include "PointCloudRegistration.h"
#include "ImageUtils.h"
#include "PlyWriter.h"
#include "jni/log_jni.h"

#include <optional>

using nlohmann::json;
#define LOG_TAG "PointCloud Registration"

cv::aruco::Dictionary createCustomDictionaryFromBits()
{
    int markerSize = 4;
    int maxCorrectionBits = 0;

    // 비트 정의
    cv::Mat markerBits(markerSize, markerSize, CV_8UC1);
    uchar data[16] = {
        0, 0, 1, 1,
        0, 0, 1, 1,
        1, 1, 0, 0,
        1, 1, 0, 0};
    memcpy(markerBits.data, data, 16);

    // Byte list 생성 (1 마커 → 4 회전 포함)
    cv::Mat byteList = cv::aruco::Dictionary::getByteListFromBits(markerBits);

    // 10개 ID로 복제
    int numCopies = 10;
    cv::Mat fullList(numCopies, byteList.cols, byteList.type());
    for (int i = 0; i < numCopies; ++i)
        byteList.copyTo(fullList.row(i));

    // Dictionary 설정
    cv::aruco::Dictionary dict;
    dict.bytesList = fullList;
    dict.markerSize = markerSize;
    dict.maxCorrectionBits = maxCorrectionBits;

    return dict;
}

// ArUco 마커 타겟 탐지
static std::map<int, cv::Point2f> findAllArucoMarkerCentersById(const cv::Mat &image,
                                                                int maxMarkerId)
{
    std::map<int, cv::Point2f> result;

    // 1. Dictionary 설정
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    // 2. Detector 설정
    cv::aruco::DetectorParameters params;
    //{
    //     ///---------------------------------------------------------------------------------------------------/
    //     // ────────────────────────────── [이진화/Adaptive Threshold] ──────────────────────────────
    //     // 적응형 임계값에서 사용하는 윈도우 크기 최소값(픽셀).
    //     // 작을수록 작은 마커/고해상도에서 세밀한 국소 조도 변화를 반영. 너무 작으면 노이즈 영향↑.
    //     params.adaptiveThreshWinSizeMin = 3;
    //     // 적응형 임계값에서 사용하는 윈도우 크기 최대값(픽셀).
    //     // 클수록 넓은 영역 평균을 사용 → 조명 불균일에 덜 민감(배경 변화에 강건). 너무 크면 작은 마커 감도↓.
    //     params.adaptiveThreshWinSizeMax = 23;
    //     // 윈도우 크기 증가 간격(픽셀). Min→Max까지 이 간격으로 반복 적용하여 후보를 종합 판단.
    //     // 구간(3~23)과 함께 조정. 더 촘촘히 보려면 Step ↓.
    //     params.adaptiveThreshWinSizeStep = 10;
    //     // 평균/가우시안 평균에서 빼는 상수 C. C↑ → 더 어둡게 임계(흰/검 대비 강조).
    //     // 바닥 텍스처 노이즈가 많거나 반사면이 있으면 약간 ↑하여 배경을 더 배제.
    //     params.adaptiveThreshConstant = 7;

    //     // ────────────────────── [형상 후보(컨투어→사각형) 전처리/필터링] ──────────────────────
    //     // 이미지 대각선 길이에 대한 마커 둘레의 최소 비율. 원거리/작은 마커를 잡으려면 ↓.
    //     // 너무 낮추면 점/노이즈 컨투어가 늘어 오검출↑.
    //     params.minMarkerPerimeterRate = 0.03;
    //     // 이미지 대각선 길이에 대한 마커 둘레의 최대 비율. 지나치게 큰 후보(근접 잡영) 배제.
    //     // 현장에 큰 마커가 실제로 있으면 ↑.
    //     params.maxMarkerPerimeterRate = 4.0;
    //     // 컨투어 다각형 근사 오차 비율. ↓면 꼭짓점 정확↑(정사각 추정에 유리)지만 노이즈에 취약.
    //     // 모서리가 흐릿하면 조금 ↑하여 과도한 샤프닝을 방지.
    //     params.polygonalApproxAccuracyRate = 0.03;
    //     // 검출된 코너들 간의 최소 거리(이미지 크기 대비 비율). 너무 붙어 있는 코너는 잘못된 후보로 판단.
    //     // 아주 작은 마커만 다룬다면 ↓.
    //     params.minCornerDistanceRate = 0.05;
    //     // 프레임 경계(이미지 가장자리)로부터 최소 거리(픽셀). 경계에 걸친 후보 제거.
    //     // 화면 끝에 자주 나오면 ↓하되, 잘림 마커 허용 시 오검출 주의.
    //     params.minDistanceToBorder = 3;
    //     // 서로 다른 마커 간 최소 거리(이미지 크기 대비 비율). 겹치거나 붙은 후보 동시 검출 방지.
    //     // 다중 마커가 밀집하면 ↓, 오검출이 많으면 ↑.
    //     params.minMarkerDistanceRate = 0.125;

    //     // ──────────────────────────────── [코너 정제(후처리)] ────────────────────────────────
    //     // 코너 정제 방식: NONE / SUBPIX / LINES / HARRIS 등.
    //     // 정밀도가 중요하면 SUBPIX 권장(속도↓). 현재는 성능 우선으로 NONE.
    //     params.cornerRefinementMethod = (int)cv::aruco::CORNER_REFINE_NONE;
    //     // 코너 정제 윈도우 크기(픽셀). 노이즈가 많거나 블러가 크면 ↑.
    //     // NONE일 때는 무시되지만, 나중에 SUBPIX로 바꿀 때 함께 조정.
    //     params.cornerRefinementWinSize = 5;
    //     // 이미지 크기 대비 정제 윈도우의 상대 크기(스케일 자동화). 고해상도 입력에서 유용.
    //     params.relativeCornerRefinmentWinSize = 0.3f;
    //     // 코너 정제 반복 상한. 너무 크면 시간↑, 너무 작으면 수렴 전 종료↑.
    //     params.cornerRefinementMaxIterations = 30;
    //     // 정제 수렴 허용 오차. ↓면 더 정확히 수렴(시간↑), ↑면 빠르게 종료(정확↓).
    //     params.cornerRefinementMinAccuracy = 0.1;

    //     // ─────────────────────────────── [디코딩/보정(정사영)] ───────────────────────────────
    //     // 마커 외곽 검정 보더 두께(셀 단위). 마커 생성 시의 딕셔너리 설정과 반드시 일치해야 디코딩 성공률↑.
    //     params.markerBorderBits = 1;
    //     // 셀당 픽셀 수(정사영 펼침 해상도).
    //     // ↑면 정사영 결과 해상도↑(정확↑, 속도↓). 작은 마커/원거리에서는 ↑가 유리.
    //     params.perspectiveRemovePixelPerCell = 4; // 원문의 tiveRemovePixelPerCell 오타 수정
    //     // 정사영 시 외곽 마진을 무시할 비율(셀 단위). 가장자리 아티팩트/링잉 제거에 도움.
    //     // 값이 너무 크면 실제 정보 손실 가능.
    //     params.perspectiveRemoveIgnoredMarginPerCell = 0.13;
    //     // 보더(검정 테두리)에서 허용할 에러 비율. ↑면 관대(검출↑/오검출↑), ↓면 엄격(검출↓/정확↑).
    //     params.maxErroneousBitsInBorderRate = 0.35;
    //     // Otsu 임계의 표준편차 최소값. 너무 균일(정보 부족)한 영역은 후보에서 배제.
    //     // 저대비 환경일수록 낮추면 검출↑(오검출↑ 가능).
    //     params.minOtsuStdDev = 5.0;
    //     // 비트 디코딩 에러 정정 허용 비율. ↑면 오류에 강함(하지만 타 마커와 혼동↑ 가능).
    //     params.errorCorrectionRate = 0.6;

    //     // ─────────────────────────────── [AprilTag 전용 파라미터] ───────────────────────────────
    //     // 다운샘플 비율. ↑면 입력 축소로 속도↑/정확도↓. 0=축소 없음.
    //     params.aprilTagQuadDecimate = 0.0;
    //     // 가우시안 블러 시그마. 노이즈 억제(조명 얼룩/텍스처 완화). 너무 크면 에지 소실.
    //     params.aprilTagQuadSigma = 0.0;
    //     // 에지 클러스터 최소 픽셀 수. 너무 작은 잡영을 배제.
    //     params.aprilTagMinClusterPixels = 5;
    //     // 지역 최대 개수 제한(성능 튜닝). 너무 크면 느려지고 잡영↑.
    //     params.aprilTagMaxNmaxima = 10;
    //     // 라인 결합 임계 각도(rad). 라인 병합 민감도. 값이 작으면 더 엄격(병합↓).
    //     params.aprilTagCriticalRad = (float)(10 * CV_PI / 180);
    //     // 라인 피팅 허용 MSE 상한. ↑면 거친 라인도 허용(오검출↑ 가능).
    //     params.aprilTagMaxLineFitMse = 10.0;
    //     // 흰/검 대비 최소값. 대비 부족한 후보 배제(저조도/저대비 환경에서는 ↓ 고려).
    //     params.aprilTagMinWhiteBlackDiff = 5;
    //     // 작은 글리치 제거(0/1). 1로 켜면 미세 잡영 감소(드물게 유효 신호 손실 가능).
    //     params.aprilTagDeglitch = 0;

    //     // ────────────────────────────────── [기타 옵션] ──────────────────────────────────
    //     // 반전 마커(흰 바탕/검 패턴 ↔ 검 바탕/흰 패턴)도 탐지할지.
    //     // 프로젝터/인쇄 반전 환경, 반사로 인한 음영 반전이 우려되면 true.
    //     params.detectInvertedMarker = false;
    //     // ArUco3(신규 파이프라인) 사용 여부. 일부 상황에서 성능/정확도의 트레이드오프가 다름.
    //     params.useAruco3Detection = false;
    //     // 정사영된(펼친) 마커 이미지의 최소 변 길이(픽셀). 너무 작으면 디코딩 실패↑.
    //     // 원거리/저해상도면 ↓, 정확도를 우선하면 ↑.
    //     params.minSideLengthCanonicalImg = 32;
    //     // 원본 이미지에서 마커 변 길이의 최소 비율. 아주 작은 마커 배제 필터.
    //     // 작은 마커 검출이 필요하면 0.0 유지 또는 소폭 ↑.
    //     params.minMarkerLengthRatioOriginalImg = 0.0;
    //     ///---------------------------------------------------------------------------------------------------/
    // }

    // // Adaptive Threshold 윈도우 스캔 간격(픽셀).
    // // - Min~Max 범위를 이 간격으로 훑어 여러 임계결과를 시험→최적 후보 채택.
    // // - ↓(예: 5)로 촘촘히 보면 조명 얼룩/텍스처에 덜 휘둘림(검출↑) 대신 연산량 소폭↑.
    // // - 터널/저대비/얼룩 많은 벽: 3~7 권장, 실외 균일광: 7~15도 무난.
    // params.adaptiveThreshWinSizeStep = 5;

    // // 정사영 펼침 시 셀당 픽셀 해상도(퍼셉티브 제거 샘플링).
    // // - ↑(예: 6)면 각 셀(마커 내부 격자)을 더 고해상도로 샘플→셀 패턴 뭉개짐↓, 디코딩 안정↑.
    // // - 단, 연산량/메모리↑. 원거리/기울어진(셀 픽셀 수 적은) 장면에 특히 효과적.
    // // - 일반: 4, 저대비·살짝 블러: 6~8, 매우 멀리/극저대비: 8~10 시험.
    // params.perspectiveRemovePixelPerCell = 6;

    // // 코너 정제 방식: SUBPIX (서브픽셀 정확도).
    // // - 초기 검출 코너를 서브픽셀 최적화로 재보정→포즈/크기 추정의 RMS 오차↓.
    // // - 노이즈나 블러가 약간 있어도 코너가 더 안정됨(연산량은 약간 증가).
    // // - NONE 대비 정확도↑, CONTOUR 대비 조도/텍스처 변화에 강건.
    // params.cornerRefinementMethod = (int)cv::aruco::CORNER_REFINE_SUBPIX;

    // // SUBPIX 코너 정제 윈도우 크기(픽셀).
    // // - ↑면 블러/노이즈에 덜 끌리지만, 너무 크면 잘못된 경계에 붙을 수 있음.
    // // - 5~9 범위가 실무 상 안전. 터널/저대비/살짝 아웃포커스면 7 권장.
    // // - 고해상도면 relativeCornerRefinmentWinSize(0.3~0.5)도 함께 고려.
    // params.cornerRefinementWinSize = 7;

    // // 반전 마커 허용(검↔흰 반전·노출/화이트밸런스 변동 시 대비 뒤집힘 대응).
    // // - 반사/조명 깜빡임/카메라 자동노출로 간헐적 반전처럼 보이는 프레임에서 유효.
    // // - 오검출 가능성 소폭↑. 딕셔너리/markerBorderBits가 인쇄본과 일치해야 효과적.
    // params.detectInvertedMarker = true;

    // 작은 마커 탐지

    params.minMarkerPerimeterRate = 0.01;
    params.adaptiveThreshWinSizeMax = 31;
    params.adaptiveThreshConstant = 3;
    params.cornerRefinementWinSize = 7;
    params.perspectiveRemovePixelPerCell = 8;
    params.errorCorrectionRate = 0.5;
    params.maxErroneousBitsInBorderRate = 0.25;
    params.minOtsuStdDev = 0.6;
    params.perspectiveRemoveIgnoredMarginPerCell = 0.15;

        

    // 서브픽셀 정밀도 향상
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

    cv::aruco::ArucoDetector detector(dictionary, params);

    // 3. 탐지 결과
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;

    // 4. 마커 탐지
    detector.detectMarkers(image, corners, ids);

    SENDLOGF_TAG(LOG_TAG, "Number of detected targets : %d", (int)ids.size());
    SENDLOGF_TAG(LOG_TAG, "Applying max_marker_id: %d", maxMarkerId);

    // 5. ID와 중심 매핑
    for (size_t i = 0; i < ids.size(); ++i)
    {
        const int id = ids[i];
        if (id < 0 || id > maxMarkerId)
            continue;
    
        const auto &pts = corners[i];
    
        cv::Point2f center(0.f, 0.f);
        for (const auto &pt : pts)
            center += pt;
        center *= 0.25f;
    
        result[id] = center;
    }
 
    return result;
}

// Custom ArUco 마커 타겟 탐지
static std::map<int, cv::Point2f> findCustomMarkerCenters(const cv::Mat &image)
{
    std::map<int, cv::Point2f> result;

    cv::aruco::Dictionary dict = createCustomDictionaryFromBits();

    cv::aruco::DetectorParameters params;

    // 작거나 흐린 마커 대응
    params.minMarkerPerimeterRate = 0.01; // 기본: 0.03
    // params.maxMarkerPerimeterRate = 4.0;  // 기본 유지

    // params.polygonalApproxAccuracyRate = 0.05; // 기본: 0.03 → 마커 외곽 검출 완화
    // params.minCornerDistanceRate = 0.02;       // 기본: 0.05 → 더 가까운 마커 허용

    // params.minDistanceToBorder = 1; // 경계 가까운 마커 허용
    // params.markerBorderBits = 1;    // 직접 만든 마커에 맞춰 조정

    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    // params.cornerRefinementWinSize = 5;
    // params.cornerRefinementMaxIterations = 50;
    // params.cornerRefinementMinAccuracy = 0.05;

    // // 이진화 기준 완화
    // params.minOtsuStdDev = 2.0;                // 더 민감하게 threshold
    // params.maxErroneousBitsInBorderRate = 0.5; // 외곽 오류 더 허용

    // params.errorCorrectionRate = 0.7; // dictionary 허용오차 약간 늘리기

    cv::aruco::ArucoDetector detector(dict, params);

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;

    detector.detectMarkers(image, corners, ids);

    SENDLOGF_TAG(LOG_TAG, "Number of detected targets : %d", (int)ids.size());

    for (size_t i = 0; i < ids.size(); ++i)
    {
        const auto &pts = corners[i];
        cv::Point2f center(0.f, 0.f);
        for (const auto &pt : pts)
            center += pt;
        center *= 0.25f;

        result[i] = center;
    }

    return result;
}

// 픽셀 좌표와 4x4 캘리브레이션 행렬을 사용해 방향 벡터(ray)를 계산
static cv::Vec3f pixelToRayWorld(int pixelX,
                                 int pixelY,
                                 int imageWidth,
                                 int imageHeight,
                                 const Blk360G2_ImageCalibration &calib)
{
    // 1. Projection matrix에서 fx, fy, cx, cy 추출
    const float fx = static_cast<float>(calib.projection.elements[0]); // focal length in X (relative to [-1,1] range)
    const float fy = static_cast<float>(calib.projection.elements[5]); // focal length in Y
    const float cx = static_cast<float>(calib.projection.elements[8]); // principal point X offset (relative to center)
    const float cy = static_cast<float>(calib.projection.elements[9]); // principal point Y offset (relative to center)

    // 2. NDC:Normalized Device Coordinates (픽셀 기준) → 카메라 기준 보정
    float ndcX = 2.0f * (pixelX + 0.5f) / imageWidth - 1.0f;
    float ndcY = 1.0f - 2.0f * (pixelY + 0.5f) / imageHeight;

    // 중심점 보정 (좌표계 중심 보정)
    //  cx, cy를 적용해 중심점이 실제 위치로부터 얼마나 치우쳐 있는지 보정
    //  fx, fy는 상대적인 카메라 줌 조절 효과
    //  z = -1.0f는 SDK 정의대로 카메라가 - Z 방향을 바라보고 있다는 가정 기반
    float x = (ndcX - cx) / fx;
    float y = (ndcY - cy) / fy;
    float z = -1.0f;

    cv::Vec3f rayCam(x, y, z);
    rayCam = cv::normalize(rayCam); // 카메라 좌표계에서 단위벡터로 정규화

    // 3. transformation matrix 적용
    cv::Matx44f T;
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            T(row, col) = static_cast<float>(calib.transformation.elements[col * 4 + row]);

    // 벡터(방향)은 homogeneous 좌표에서 w = 0.0으로 설정
    // 행렬을 곱해 월드 좌표계에서의 방향벡터로 변환
    cv::Vec4f rayCamHomo(rayCam[0], rayCam[1], rayCam[2], 0.0f);
    cv::Vec4f rayWorldHomo = T * rayCamHomo;
    cv::Vec3f rayWorld(rayWorldHomo[0], rayWorldHomo[1], rayWorldHomo[2]);

    return cv::normalize(rayWorld);
}

// 방법 1: 중심에서의 방향 비교 (빠름) - 현재 미사용
static CartesianPointRGB findClosestPointByDirection(const cv::Vec3f &rayDir,
                                                     const std::vector<CartesianPointRGB> &points)
{
    float maxCosine = -1.0f;       // 방향 유사도 최대값 (-1 ~ 1)
    CartesianPointRGB bestPoint{}; // 가장 유사한 점 (결과)

    for (const auto &p : points)
    {
        cv::Vec3f dir(p.x, p.y, p.z); // 점의 좌표 → 방향 벡터로 간주
        float norm = cv::norm(dir);   // 방향 벡터의 길이 계산
        if (norm < 1e-3f)
            continue; // 너무 짧은 벡터(거의 원점)는 무시

        dir /= norm; // 단위 벡터로 정규화

        float cosine = dir.dot(rayDir); // 기준 벡터와의 코사인 유사도 계산(1.0에 가까울수록 동일 방향)

        if (cosine > maxCosine) // 더 유사한 점이면 갱신
        {
            maxCosine = cosine;
            bestPoint = p;
        }
    }

    return bestPoint; // 가장 유사한 방향의 점 반환
}

// 방법 2: 거리 기반 + 각도 필터 (정밀)
static CartesianPointRGB findClosestPointOnRay(const cv::Vec3f &rayDir,
                                               const std::vector<CartesianPointRGB> &points,
                                               float maxAngleThresholdDeg,
                                               float minDepthThreshold)
{
    CartesianPointRGB closestPoint{};                  // 결과: 가장 가까운 점
    float minDist = std::numeric_limits<float>::max(); // 최소 거리 초기화

    for (const auto &p : points)
    {
        // Depth 거리
        cv::Vec3f dir(p.x, p.y, p.z); // 포인트의 벡터
        float norm = cv::norm(dir);   // 벡터 크기
        if (norm < minDepthThreshold) // minDepthThreshold 미만 벡터는 무시
            continue;

        // 각도 계산 (단위: 도)
        float angleDeg = std::acos(dir.dot(rayDir) / norm) * 180.0f / CV_PI;
        if (angleDeg > maxAngleThresholdDeg)
            continue; // 각도가 임계값보다 크면 무시

        // 방향
        float dot = rayDir.dot(dir);
        if (dot <= 0.0f)
            continue; // ray 반대 방향의 점은 무시

        // ray 방향에 수직인 거리 계산
        // dir 벡터와 rayDir 방향 성분만큼 곱한 벡터의 차 → 수직 거리
        float dist = std::abs(cv::norm(dir - rayDir * norm));

        // 가장 가까운 점이면 갱신
        if (dist < minDist)
        {
            minDist = dist;
            closestPoint = p;
        }
    }

    return closestPoint; // 결과 반환
}
// 방법 3: 거리 기반 + 각도 필터 방법으로 3차원 점군을 찾고 depth값만 활용(Ray + Depty)
static CartesianPointRGB findPointByDepthProjection(const cv::Vec3f &rayDir,
                                                    const std::vector<CartesianPointRGB> &points,
                                                    float maxAngleThresholdDeg,
                                                    float minDepthThreshold)
{
    // 기존 로직을 이용하여 가장 가까운 포인트 찾기
    CartesianPointRGB closestPoint = findClosestPointOnRay(rayDir, points, maxAngleThresholdDeg, minDepthThreshold);

    // 해당 점까지의 거리(norm)를 계산
    float closestNorm = cv::norm(cv::Vec3f(closestPoint.x, closestPoint.y, closestPoint.z));

    // 유효한 점이 없으면 {0,0,0} 반환
    if (closestNorm <= 1e-6f || std::isnan(closestNorm))
        return CartesianPointRGB{0.0f, 0.0f, 0.0f};

    // 이 거리만큼 ray 방향 벡터를 확장
    cv::Vec3f refined = rayDir * closestNorm;

    // 반환: ray 방향 + 실제 거리
    return CartesianPointRGB{refined[0], refined[1], refined[2]};
}

// 점군으로부터 이미지 픽셀 위치 계산
static cv::Point2f worldToPixel(const cv::Vec3f &worldPoint,
                                const Blk360G2_ImageCalibration &calib,
                                int imageWidth,
                                int imageHeight)
{
    // 1. 월드 기준 방향 벡터 (from origin to point)
    cv::Vec3f rayWorld(worldPoint[0], worldPoint[1], worldPoint[2]);
    rayWorld = cv::normalize(rayWorld);

    // 2. transformation matrix의 역행렬을 계산 (월드 → 카메라)
    cv::Matx44f T;
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            T(row, col) = static_cast<float>(calib.transformation.elements[col * 4 + row]);

    cv::Matx44f T_inv = T.inv(); // ← 역행렬

    // 3. 벡터(방향)는 homogeneous w = 0.0
    cv::Vec4f rayWorldHomo(rayWorld[0], rayWorld[1], rayWorld[2], 0.0f);
    cv::Vec4f camHomo = T_inv * rayWorldHomo;

    float x = camHomo[0];
    float y = camHomo[1];
    float z = camHomo[2];

    // 4. projection matrix에서 fx, fy, cx, cy 추출
    const float fx = static_cast<float>(calib.projection.elements[0]);
    const float fy = static_cast<float>(calib.projection.elements[5]);
    const float cx = static_cast<float>(calib.projection.elements[8]);
    const float cy = static_cast<float>(calib.projection.elements[9]);

    // 5. 정규화된 방향 벡터를 NDC로 역계산
    float ndcX = (x / -z) * fx + cx;
    float ndcY = (y / -z) * fy + cy;

    // 6. NDC → 픽셀 좌표로 변환
    float pixelX = (ndcX + 1.0f) * 0.5f * imageWidth - 0.5f;
    float pixelY = (1.0f - ndcY) * 0.5f * imageHeight - 0.5f;

    return cv::Point2f(pixelX, pixelY);
}

// 보조함수 - OpenCV 변환 함수
cv::Mat convertImageToCvMat(Blk360G2_ImageHandle image)
{
    auto meta = Blk360G2_Image_GetMetadata(image);
    const void *raw = Blk360G2_Image_GetData(image);
    int width = meta.imageWidth;
    int height = meta.imageHeight;

    if (!raw || width == 0 || height == 0)
        return cv::Mat();

    const uint8_t *rgbData = reinterpret_cast<const uint8_t *>(raw);
    cv::Mat rgb(height, width, CV_8UC3, const_cast<uint8_t *>(rgbData));

    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR, 0, cv::ALGO_HINT_DEFAULT);
    return bgr.clone();
}

// 보조함수 - 반경 점 필터링
static std::vector<CartesianPointRGB> findPointsInRadius(const cv::Vec3f &center,
                                                         const std::vector<CartesianPointRGB> &points,
                                                         float radius)
{
    std::vector<CartesianPointRGB> result;
    float r2 = radius * radius;

    for (const auto &p : points)
    {
        float dx = p.x - center[0];
        float dy = p.y - center[1];
        float dz = p.z - center[2];
        float dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 <= r2)
            result.push_back(p);
    }

    return result;
}

// 중복을 제거한 ICP용 기준점군(.ply) 저장 함수
static bool SaveMergedRegCloud(const std::string &outputPath,
                               const std::vector<std::vector<CartesianPointRGB>> &allGroups,
                               float mergeDistance)
{
    std::vector<CartesianPointRGB> mergedPoints;

    // lambda 함수: 두 점 사이 거리가 mergeDistance 이하면 중복으로 판단
    auto isClose = [mergeDistance](const CartesianPointRGB &a, const CartesianPointRGB &b)
    {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float dz = a.z - b.z;
        float dist2 = dx * dx + dy * dy + dz * dz;
        return dist2 < (mergeDistance * mergeDistance); // 제곱 거리 비교
    };

    // 그룹별로 모든 점을 순회하며 중복 여부 검사
    for (const auto &group : allGroups)
    {
        for (const auto &pt : group)
        {
            bool isDuplicate = false;

            // 이미 mergedPoints에 존재하는지 검사
            for (const auto &existing : mergedPoints)
            {
                if (isClose(pt, existing))
                {
                    isDuplicate = true;
                    break;
                }
            }

            // 중복이 아니면 추가
            if (!isDuplicate)
                mergedPoints.push_back(pt);
        }
    }

    // 최종 중복 제거된 점군을 ply 파일로 저장
    if (!savePointCloud(outputPath, mergedPoints))
    {
        SENDLOGF_TAG(LOG_TAG, "Failed to save PLY file: %s", outputPath.c_str());
        return false;
    }

    SENDLOGF_TAG(LOG_TAG, "Merged PLY saved after removing duplicates: %s (Number of points: %zu)", outputPath.c_str(), mergedPoints.size());
    return true;
}

// save face marker to JSON
//  facemarker(아루코) 탐지 결과 + 선택적으로 3D center 저장
static bool SaveFacemarkerJson(const std::string &jsonPath,
                               const std::vector<std::pair<int, cv::Point2f>> &detections,
                               const cv::Point3f *faceCenter3D) //  3D 센터 (옵션)
{
    try
    {
        std::ofstream ofs(jsonPath, std::ios::out | std::ios::trunc);
        if (!ofs.is_open())
        {
            SENDLOGF_TAG(LOG_TAG, "SaveFacemarkerJson - failed to open: %s",
                         jsonPath.c_str());
            return false;
        }

        ofs << "{\n";
        ofs << "  \"facemarkers\": [\n";

        ofs << std::fixed << std::setprecision(6);

        for (size_t i = 0; i < detections.size(); ++i)
        {
            const auto &entry = detections[i];
            ofs << "    {\n";
            ofs << "      \"id\": " << entry.first << ",\n";
            ofs << "      \"pixel\": [\n";
            ofs << "        " << entry.second.x << ",\n";
            ofs << "        " << entry.second.y << "\n";
            ofs << "      ]\n";
            ofs << "    }";
            if (i + 1 < detections.size())
                ofs << ",";
            ofs << "\n";
        }

        ofs << "  ]";

        // 선택적으로 facemarkers3D 섹션 추가
        if (faceCenter3D != nullptr)
        {
            ofs << ",\n";
            ofs << "  \"facemarkers3D\": [\n";
            ofs << "    {\n";
            ofs << "      \"id\": 0,\n";
            ofs << "      \"center\": [\n";
            ofs << "        " << faceCenter3D->x << ",\n";
            ofs << "        " << faceCenter3D->y << ",\n";
            ofs << "        " << faceCenter3D->z << "\n";
            ofs << "      ]\n";
            ofs << "    }\n";
            ofs << "  ]\n";
        }
        else
        {
            ofs << "\n";
        }

        ofs << "}\n";

        ofs.close();
        SENDLOGF_TAG(LOG_TAG, "SaveFacemarkerJson - saved to %s",
                     jsonPath.c_str());
        return true;
    }
    catch (const std::exception &e)
    {
        SENDLOGF_TAG(LOG_TAG,
                     "SaveFacemarkerJson - exception: %s", e.what());
        return false;
    }
}
bool LoadFaceMarkerCenter3D(const std::string &jsonPath,
                            cv::Point3f &outCenter)
{
    try
    {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open())
        {
            SENDLOGF_TAG(LOG_TAG,
                         "LoadFaceMarkerCenter3D - failed to open: %s",
                         jsonPath.c_str());
            return false;
        }

        // 파일 전체를 문자열로 읽기
        std::string content((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());

        // "facemarkers3D" 섹션 찾기
        const std::string keyFacemarker3D = "\"facemarkers3D\"";
        auto pos = content.find(keyFacemarker3D);
        if (pos == std::string::npos)
        {
            SENDLOGF_TAG(LOG_TAG,
                         "LoadFaceMarkerCenter3D - facemarkers3D not found in %s",
                         jsonPath.c_str());
            return false;
        }

        // "center" 키 위치 찾기 (facemarkers3D 이후에서)
        const std::string keyCenter = "\"center\"";
        auto cpos = content.find(keyCenter, pos);
        if (cpos == std::string::npos)
        {
            SENDLOGF_TAG(LOG_TAG,
                         "LoadFaceMarkerCenter3D - center not found in %s",
                         jsonPath.c_str());
            return false;
        }

        // center: [ x, y, z ] 부분의 대괄호 찾기
        auto lbrack = content.find('[', cpos);
        auto rbrack = content.find(']', lbrack);
        if (lbrack == std::string::npos || rbrack == std::string::npos || rbrack <= lbrack)
        {
            SENDLOGF_TAG(LOG_TAG,
                         "LoadFaceMarkerCenter3D - invalid center array in %s",
                         jsonPath.c_str());
            return false;
        }

        // 대괄호 안만 잘라서 "x, y, z" 부분만 추출
        std::string arr = content.substr(lbrack + 1, rbrack - lbrack - 1);

        // 파싱을 위해 ',' 를 공백으로 치환
        for (char &ch : arr)
        {
            if (ch == ',')
                ch = ' ';
        }

        std::stringstream ss(arr);
        float x = 0.0f, y = 0.0f, z = 0.0f;
        if (!(ss >> x >> y >> z))
        {
            SENDLOGF_TAG(LOG_TAG,
                         "LoadFaceMarkerCenter3D - failed to parse floats in %s",
                         jsonPath.c_str());
            return false;
        }

        outCenter = cv::Point3f(x, y, z);

        SENDLOGF_TAG(LOG_TAG,
                     "LoadFaceMarkerCenter3D - loaded center: (%.6f, %.6f, %.6f) from %s",
                     x, y, z, jsonPath.c_str());
        return true;
    }
    catch (const std::exception &e)
    {
        SENDLOGF_TAG(LOG_TAG,
                     "LoadFaceMarkerCenter3D - exception: %s", e.what());
        return false;
    }
}

// save id + pixel to JSON
static bool SaveDetectionJson(const std::string &jsonPath,
                              const std::vector<std::pair<int, cv::Point2f>> &markers)
{
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(jsonPath).parent_path(), ec);

    json j;
    j["markers"] = json::array();

    for (const auto &mk : markers)
    {
        j["markers"].push_back({{"id", mk.first},
                                {"pixel", {mk.second.x, mk.second.y}}});
    }

    std::ofstream ofs(jsonPath, std::ios::binary);
    if (!ofs)
        return false;
    ofs << j.dump(2);
    return true;
}

// load id + pixel from JSON
static bool LoadPixelsFromJson(const std::string &jsonPath,
                               std::vector<std::pair<int, cv::Point2f>> &outMarkers)
{
    outMarkers.clear();
    std::ifstream ifs(jsonPath, std::ios::binary);
    if (!ifs)
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

    for (const auto &m : j["markers"])
    {
        if (!m.contains("id") || !m.contains("pixel"))
            continue;
        if (!m["pixel"].is_array() || m["pixel"].size() != 2)
            continue;

        int id = m["id"].get<int>();
        float px = static_cast<float>(m["pixel"][0].get<double>());
        float py = static_cast<float>(m["pixel"][1].get<double>());

        outMarkers.emplace_back(id, cv::Point2f(px, py));
    }
    return true;
}

// 캘리브레이션 JSON 저장: width/height + projection(16) + transformation(16)
static bool SaveCalibrationJson(const std::string &jsonPath,
                                int width, int height,
                                const Blk360G2_ImageCalibration &calib)
{
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(jsonPath).parent_path(), ec);

    json j;
    j["width"] = width;
    j["height"] = height;

    j["projection"] = json::array();
    j["transformation"] = json::array();
    for (int i = 0; i < 16; ++i)
    {
        j["projection"].push_back(static_cast<double>(calib.projection.elements[i]));
        j["transformation"].push_back(static_cast<double>(calib.transformation.elements[i]));
    }

    std::ofstream ofs(jsonPath, std::ios::binary);
    if (!ofs)
        return false;
    ofs << j.dump(2);
    return true;
}

// 캘리브레이션 JSON 로드: width, height, projection[16], transformation[16]
static bool LoadCalibrationJson(const std::string &jsonPath,
                                int &width, int &height,
                                float proj[16], float xf[16])
{
    width = height = 0;
    std::fill(proj, proj + 16, 0.0);
    std::fill(xf, xf + 16, 0.0);

    std::ifstream ifs(jsonPath, std::ios::binary);
    if (!ifs)
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

    if (!j.contains("width") || !j.contains("height"))
        return false;
    if (!j.contains("projection") || !j.contains("transformation"))
        return false;
    if (!j["projection"].is_array() || !j["transformation"].is_array())
        return false;
    if (j["projection"].size() != 16 || j["transformation"].size() != 16)
        return false;

    width = j["width"].get<int>();
    height = j["height"].get<int>();
    for (int i = 0; i < 16; ++i)
    {
        proj[i] = j["projection"][i].get<double>();
        xf[i] = j["transformation"][i].get<double>();
    }
    return (width > 0 && height > 0);
}

static float ComputeCenterProximityWeight(const cv::Point2f &pixelPt,
                                          int imageWidth,
                                          int imageHeight)
{
    //중심보정
    if (imageWidth <= 1 || imageHeight <= 1)
        return 1.0f;

    const float cx = 0.5f * static_cast<float>(imageWidth - 1);
    const float cy = 0.5f * static_cast<float>(imageHeight - 1);
    const float dx = pixelPt.x - cx;
    const float dy = pixelPt.y - cy;
    const float radius = std::sqrt(dx * dx + dy * dy);
    const float maxRadius = std::sqrt(cx * cx + cy * cy);
    if (maxRadius <= 1e-6f)
        return 1.0f;

    const float normalizedRadius = std::clamp(radius / maxRadius, 0.0f, 1.0f);
    const float weight = static_cast<float>(std::exp(-3.0f * normalizedRadius * normalizedRadius));
    return std::max(0.05f, weight);
}

// centers JSON 저장
static bool SaveDetectedCentersJson(const std::string &jsonPath,
                                    const std::vector<std::pair<int, cv::Point3f>> &centers)
{
    try
    {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto &p : centers)
        {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "T%03d", p.first);
            arr.push_back({{"id", std::string(buf)},
                           {"center", {p.second.x, p.second.y, p.second.z}}});
        }
        std::ofstream ofs(jsonPath);
        if (!ofs.is_open())
            return false;
        ofs << arr.dump(2);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

// centers JSON 로드
static bool LoadDetectedCentersJson(const std::string &jsonPath,
                                    std::map<int, cv::Point3f> &outCenters)
{
    try
    {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open())
            return false;
        nlohmann::json j;
        ifs >> j;

        const nlohmann::json &arr = j.is_array() ? j : j["centers"];
        if (!arr.is_array())
            return false;

        outCenters.clear();
        for (const auto &it : arr)
        {
            // id: 문자열/정수 모두 허용, 문자열이면 숫자만 모아 int 변환
            int id = 0;
            if (it.contains("id"))
            {
                if (it["id"].is_number_integer())
                {
                    id = it["id"].get<int>();
                }
                else if (it["id"].is_string())
                {
                    const std::string s = it["id"].get<std::string>();
                    for (char c : s)
                        if (c >= '0' && c <= '9')
                            id = id * 10 + (c - '0');
                }
                else
                    return false;
            }
            else
                return false;

            const auto &c = it["center"];
            if (!c.is_array() || c.size() < 3)
                return false;
            outCenters[id] = cv::Point3f(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
        }
        return !outCenters.empty();
    }
    catch (...)
    {
        return false;
    }
}

// 페어링 및 pair.json 저장
bool PairTargetsFromJsonAndPt(const std::string &blastPath,
                              const std::string &targetNameCmp, // "compared"
                              const std::string &targetNameRef, // "reference"
                              bool overwrite)
{
    auto start = std::chrono::steady_clock::now();

    const std::string refDir = blastPath ;
    const std::string cmpDir = blastPath + "/Registration/" + targetNameCmp;

    const std::string refAdjustedCentersPath = refDir + "/reference_targets.json";
    const std::string cmpAdjustedCentersPath = cmpDir + "/targets3d_adjust.json";
    const std::string refCentersPath = refDir + "/reference_targets.json";
    const std::string cmpCentersPath = cmpDir + "/targets3d_adjust.json";

    std::map<int, cv::Point3f> refCenters, cmpCenters;
    auto loadCentersWithFallback = [](const std::string &adjustedPath,
                                      const std::string &originalPath,
                                      std::map<int, cv::Point3f> &outCenters)
    {
        if (LoadDetectedCentersJson(adjustedPath, outCenters))
        {
            SENDLOGF_TAG(LOG_TAG, "Loaded centers: %s", adjustedPath.c_str());
            return true;
        }
        if (LoadDetectedCentersJson(originalPath, outCenters))
        {
            SENDLOGF_TAG(LOG_TAG, "Loaded centers fallback: %s", originalPath.c_str());
            return true;
        }
        SENDLOGF_TAG(LOG_TAG, "Load failed: %s, %s", adjustedPath.c_str(), originalPath.c_str());
        return false;
    };

    if (!loadCentersWithFallback(refAdjustedCentersPath, refCentersPath, refCenters))
    {
        return false;
    }
    if (!loadCentersWithFallback(cmpAdjustedCentersPath, cmpCentersPath, cmpCenters))
    {
        return false;
    }

    // 교집합 id만 추출
    std::vector<int> commonIds;
    commonIds.reserve(std::min(refCenters.size(), cmpCenters.size()));
    for (const auto &kv : refCenters)
    {
        if (cmpCenters.count(kv.first))
            commonIds.push_back(kv.first);
    }
    std::sort(commonIds.begin(), commonIds.end());

    // 요청 JSON 구성
    nlohmann::json out;
    out[targetNameRef] = {
        {"type", targetNameRef},
        {"reg_ply", refDir + "/" + targetNameRef + "_reg.ply"},
        {"centers", nlohmann::json::array()}};
    out[targetNameCmp] = {
        {"type", targetNameCmp},
        {"reg_ply", cmpDir + "/" + targetNameCmp + "_reg.ply"},
        {"centers", nlohmann::json::array()}};
    out["pairs"] = nlohmann::json::array();

    for (int id : commonIds)
    {
        const auto &r = refCenters[id];
        const auto &c = cmpCenters[id];

        // pairs 항목
        nlohmann::json pair;
        pair["id"] = id;
        pair["ref"] = {r.x, r.y, r.z};
        pair["comp"] = {c.x, c.y, c.z};
        out["pairs"].push_back(pair);

        // (선택) 참고용: 각 세트별도 저장
        out["reference"]["centers"].push_back({{"id", id}, {"center", {r.x, r.y, r.z}}});
        out["compared"]["centers"].push_back({{"id", id}, {"center", {c.x, c.y, c.z}}});
    }

    // 저장 위치: Registration/pairs.json (또는 Registration/<targetTypeRef> 하위로 저장하고 싶으면 경로 바꿔도 무방)
    const std::string regRoot = blastPath + "/Registration";
    std::filesystem::create_directories(regRoot);

    // 2025-12-01 파일 넘버링으로 인한 output 파일 이름 수정
    // targetNameCmp: "compared_001" -> suffix: "001"
    std::string suffix;
    const std::string prefix = "compared_";
    if (targetNameCmp.rfind(prefix, 0) == 0 && targetNameCmp.size() > prefix.size())
    {
        // "compared_"로 시작하면 그 뒤를 잘라서 사용
        suffix = targetNameCmp.substr(prefix.size());
    }
    else
    {
        // 혹시 다른 형식이면 통째로 붙여서 파일 이름으로 사용
        // ex) targetNameCmp == "cmpA" -> pairs_cmpA.json
        suffix = targetNameCmp;
    }

    const std::string outPath = regRoot + "/pairs_" + suffix + ".json";

    if (!overwrite && std::filesystem::exists(outPath))
    {
        SENDLOGF_TAG(LOG_TAG, "pairs.json already exists. overwrite=false, skip.");
        return true;
    }

    // ADD: 교집합 ID만으로 Reference/Compared 각각 재머지
    {
        // reference
        std::vector<std::vector<CartesianPointRGB>> refGroups;
        for (int id : commonIds)
        {
            std::string plyPath = refDir + "/id_" + std::to_string(id) + ".ply";
            auto pts = loadPointCloud(plyPath);
            if (!pts.empty())
                refGroups.push_back(std::move(pts));
        }
        if (!refGroups.empty())
        {
            std::string outRefPaired = refDir + "/" + targetNameRef + "_reg_paired.ply";
            SaveMergedRegCloud(outRefPaired, refGroups);
            SENDLOGF_TAG(LOG_TAG, "Saved ref paired merge: %s", outRefPaired.c_str());
        }

        // compared
        std::vector<std::vector<CartesianPointRGB>> cmpGroups;
        for (int id : commonIds)
        {
            std::string plyPath = cmpDir + "/id_" + std::to_string(id) + ".ply";
            auto pts = loadPointCloud(plyPath);
            if (!pts.empty())
                cmpGroups.push_back(std::move(pts));
        }
        if (!cmpGroups.empty())
        {
            std::string outCmpPaired = cmpDir + "/" + targetNameCmp + "_reg_paired.ply";
            SaveMergedRegCloud(outCmpPaired, cmpGroups);
            SENDLOGF_TAG(LOG_TAG, "Saved cmp paired merge: %s", outCmpPaired.c_str());
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    SENDLOGF_TAG("ProcessTime", "Find Center And Merge PtCloud Sub - MergePtCloud time: %lld ms (%lld초)", ms, sec);

    try
    {
        std::ofstream ofs(outPath);
        ofs << out.dump(2);
        SENDLOGF_TAG(LOG_TAG, "Saved pairs: %s (common ids: %zu)", outPath.c_str(), commonIds.size());
        return true;
    }
    catch (...)
    {
        SENDLOGF_TAG(LOG_TAG, "Failed to save %s", outPath.c_str());
        return false;
    }
}

// write only id+pixel - 타겟만 검출
static bool DetectMarkersAndSaveImages(const std::string &blastPath,
                                       Blk360G2_DataManipulationWorkflowHandle dataManipulationWorkflow,
                                       const std::vector<Blk360G2_ImageHandle> &images,
                                       const std::vector<CartesianPointRGB> &pointCloud,
                                       const std::string &targetType,
                                       bool isAruco)
{
    const int maxMarkerId = GetConfigInt(blastPath, "max_marker_id", 15);
    SENDLOGF_TAG(LOG_TAG, "DetectMarkersAndSaveImages max_marker_id=%d", maxMarkerId);
    std::string detectImgFolder = blastPath + "/DetectTarget/" + targetType;
    std::filesystem::create_directories(detectImgFolder);
    std::string facemarkerpath = blastPath + "/DetectTarget/" + targetType + "/detect_facemarker.json";
    // facemarker(아루코) 전체 목록 누적용 컨테이너
    std::vector<std::pair<int, cv::Point2f>> facemarkerDetections;
    size_t totalDetectedMarkers = 0;

    for (int i = 0; i < static_cast<int>(images.size()); ++i)
    {
        Blk360G2_ImageMetadata meta = Blk360G2_Image_GetMetadata(images[i]);
        Blk360G2_ImageCalibration calib = Blk360G2_DataManipulationWorkflow_GetImageCalibration(dataManipulationWorkflow, images[i], 1);
        std::string camposStr = getCameraPositionCode(meta.cameraPosition);

        cv::Mat image = convertImageToCvMat(images[i]);
        if (image.empty())
        {
            SENDLOGF_TAG(LOG_TAG, "[%s] Image conversion failed", camposStr.c_str());
            continue;
        }

        // ArUco 마커 탐지
        std::map<int, cv::Point2f> markerMap;
        if (isAruco)
            markerMap = findAllArucoMarkerCentersById(image, maxMarkerId);
        else
            markerMap = findCustomMarkerCenters(image);

        SENDLOGF_TAG(LOG_TAG, "[%s] Number of detected markers: %zu", camposStr.c_str(), markerMap.size());
        totalDetectedMarkers += markerMap.size();

        cv::Mat visImg = image.clone();
        std::vector<std::pair<int, cv::Point2f>> toSave; // id + pixel

        for (const auto &kv : markerMap)
        {
            int markerId = kv.first;
            const cv::Point2f &pixelPt = kv.second;

            cv::drawMarker(visImg, pixelPt, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 2);
            cv::putText(visImg, std::to_string(markerId),
                        cv::Point(pixelPt.x + 10, pixelPt.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

            toSave.emplace_back(markerId, pixelPt);
            // aruco 0번을 facemarker로 사용
            if (markerId == 0)
                facemarkerDetections.emplace_back(markerId, pixelPt);
        }

        // 저장
        std::string imgPath = detectImgFolder + "/target_marked_" + std::to_string(i) + "_" + camposStr + ".jpg";
        cv::imwrite(imgPath, visImg);

        // JSON 저장 (id + pixel)
        std::string jsonPath = detectImgFolder + "/detect_" + std::to_string(i) + "_" + camposStr + ".json";
        if (!SaveDetectionJson(jsonPath, toSave))
            SENDLOGF_TAG(LOG_TAG, "[%s] Failed to save detection JSON: %s", camposStr.c_str(), jsonPath.c_str());

        // 캘리브레이션 JSON 저장 (한 이미지당 1개)
        std::string calibPath = detectImgFolder + "/calib_" + std::to_string(i) + "_" + camposStr + ".json";
        SaveCalibrationJson(calibPath, image.cols, image.rows, calib);

        SENDLOGF_TAG(LOG_TAG, "[%s] Marker image & JSON saved. markers=%zu", camposStr.c_str(), toSave.size());
    }

    // facemarker json 저장
    if (totalDetectedMarkers == 0)
    {
        SENDLOGF_TAG(LOG_TAG, "No target markers detected. Marker detection failed.");
        return false;
    }

    if (!facemarkerDetections.empty())
    {
        if (SaveFacemarkerJson(facemarkerpath, facemarkerDetections))
        {
            cv::Point3f faceCenter;
            std::vector<std::pair<int, cv::Point2f>> allMarkers2D;

            // FaceMarkerCenter -> 3D
            bool ok = ExtractFaceMarkerCenter_Standalone(
                blastPath,
                targetType,
                pointCloud,
                faceCenter,
                allMarkers2D);

            if (ok)
                SENDLOGF_TAG(LOG_TAG, "Face marker (id=0) detected");
        }
    }
    else
        SENDLOGF_TAG(LOG_TAG, "No ArUco markers detected at all. detect_facemarker.json will not be created");

    return true;
}

// JSON만으로 점군추출 머지
bool ExtractAndSaveAndMergePointClouds_Standalone(const std::string &blastPath,
                                                  const std::string &targetType,
                                                  const std::vector<CartesianPointRGB> &pointCloud)
{
    //중심보정
    struct DetectedCenterCandidate
    {
        int id = -1;
        cv::Point3f center;
        float viewWeight = 1.0f;
    };

    std::vector<DetectedCenterCandidate> detectedCenters;
    std::vector<std::pair<int, cv::Point3f>> mergedCenters;

    const std::string detectFolder = blastPath + "/DetectTarget/" + targetType;
    const std::string regFolder = blastPath + "/Registration/" + targetType;
    std::filesystem::create_directories(regFolder);

    std::vector<std::vector<CartesianPointRGB>> allTargetGroups;

    // ID → 점군 버킷 (최종 ID별 PLY 저장/머지용)
    std::unordered_map<int, std::vector<CartesianPointRGB>> idBuckets;

    for (const auto &entry : std::filesystem::directory_iterator(detectFolder))
    {
        if (!entry.is_regular_file())
            continue;
        const auto &p = entry.path();
        if (p.extension() != ".json")
            continue;

        const std::string fname = p.filename().string();
        if (fname.rfind("detect_", 0) != 0)
            continue;

        // detect_<idx>_<campos>.json
        int imgIndex = -1;
        std::string camposStr = "NA";
        {
            auto core = fname.substr(std::string("detect_").size());
            core = core.substr(0, core.size() - std::string(".json").size());
            auto us = core.find_last_of('_');
            if (us != std::string::npos)
            {
                imgIndex = std::atoi(core.substr(0, us).c_str());
                camposStr = core.substr(us + 1);
            }
        }

        // id+pixel 로드
        std::vector<std::pair<int, cv::Point2f>> markers;
        if (!LoadPixelsFromJson(p.string(), markers) || markers.empty())
        {
            SENDLOGF_TAG(LOG_TAG, "No markers or parse failed: %s", fname.c_str());
            continue;
        }

        // 캘리브레이션 로드
        std::string calibPath = (std::filesystem::path(detectFolder) / ("calib_" + std::to_string(imgIndex) + "_" + camposStr + ".json")).string();
        int width = 0, height = 0;
        float proj[16], xf[16];
        if (!LoadCalibrationJson(calibPath, width, height, proj, xf))
        {
            SENDLOGF_TAG(LOG_TAG, "Failed to load calib: %s", calibPath.c_str());
            continue;
        }

        // 임시 calib 구성
        Blk360G2_ImageCalibration calib{};
        for (int k = 0; k < 16; ++k)
        {
            calib.projection.elements[k] = proj[k];
        }
        for (int k = 0; k < 16; ++k)
        {
            calib.transformation.elements[k] = xf[k];
        }

        std::vector<CartesianPointRGB> allNearbyPoints;
        allNearbyPoints.reserve(markers.size() * 2000);

        // id를 mk.first에서 꺼내 사용 (기존 에러 fix)
        for (const auto &[id, pixelPt] : markers)
        {
            cv::Vec3f ray = pixelToRayWorld((int)pixelPt.x, (int)pixelPt.y, width, height, calib);

            CartesianPointRGB targetPos = findPointByDepthProjection(ray, pointCloud);
            const float viewWeight = ComputeCenterProximityWeight(pixelPt, width, height);
            detectedCenters.push_back({id, cv::Point3f(targetPos.x, targetPos.y, targetPos.z), viewWeight});

            cv::Vec3f w(targetPos.x, targetPos.y, targetPos.z);
            auto nearby = findPointsInRadius(w, pointCloud, 0.15f);
            allNearbyPoints.insert(allNearbyPoints.end(), nearby.begin(), nearby.end());

            //  ID 버킷에 누적 (ID 기준 머지 목적)
            auto &bucket = idBuckets[id];
            bucket.insert(bucket.end(), nearby.begin(), nearby.end());
        }

        // PLY 저장
        std::string plyPath = regFolder + "/target_" + std::to_string(imgIndex) + "_" + camposStr + ".ply";
        if (!savePointCloud(plyPath, allNearbyPoints))
        {
            SENDLOGF_TAG(LOG_TAG, "Failed to save PLY: %s", plyPath.c_str());
            return false;
        }
        allTargetGroups.push_back(std::move(allNearbyPoints));
        SENDLOGF_TAG(LOG_TAG, "[%s] Standalone PLY saved. points=%zu, file=%s",
                     camposStr.c_str(), allTargetGroups.back().size(), fname.c_str());
    }

    // 같은 id가 여러 번 나오면 반경 클러스터를 만들고,
    // 이미지 중심에 가까운 검출(viewWeight)을 더 신뢰하도록 가중 선택
    {
        const float mergeThresh_m = 0.05f; // 5 cm

        std::unordered_map<int, std::vector<DetectedCenterCandidate>> byId;
        byId.reserve(detectedCenters.size());
        for (const auto &e : detectedCenters)
            byId[e.id].push_back(e);

        std::vector<std::pair<int, cv::Point3f>> filtered;
        filtered.reserve(byId.size());

        auto dist = [](const cv::Point3f &a, const cv::Point3f &b)
        {
            float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        };

        auto supportCount = [&](const cv::Point3f &c, float radius_m) -> int
        {
            const float r2 = radius_m * radius_m;
            int cnt = 0;
            for (const auto &p : pointCloud)
            {
                float dx = p.x - c.x, dy = p.y - c.y, dz = p.z - c.z;
                if (dx * dx + dy * dy + dz * dz <= r2)
                    ++cnt;
            }
            return cnt;
        };

        for (auto &kv : byId)
        {
            const int id = kv.first;
            const auto &pts = kv.second;
            if (pts.empty())
                continue;

            struct Cluster
            {
                cv::Point3f mean;
                int count = 0;
                float weightSum = 0.0f;
            };
            std::vector<Cluster> clusters;

            for (const auto &p : pts)
            {
                int bestIdx = -1;
                float bestD = mergeThresh_m;
                for (int i = 0; i < (int)clusters.size(); ++i)
                {
                    float d = dist(p.center, clusters[i].mean);
                    if (d < bestD)
                    {
                        bestD = d;
                        bestIdx = i;
                    }
                }
                if (bestIdx < 0)
                {
                    clusters.push_back({p.center, 1, p.viewWeight});
                }
                else
                {
                    Cluster &c = clusters[bestIdx];
                    const float weight = std::max(0.001f, p.viewWeight);
                    const float prevWeight = std::max(0.001f, c.weightSum);
                    const float totalWeight = prevWeight + weight;
                    c.mean.x = (c.mean.x * prevWeight + p.center.x * weight) / totalWeight;
                    c.mean.y = (c.mean.y * prevWeight + p.center.y * weight) / totalWeight;
                    c.mean.z = (c.mean.z * prevWeight + p.center.z * weight) / totalWeight;
                    c.weightSum = totalWeight;
                    c.count += 1;
                }
            }

            int best = -1;
            int bestCnt = -1;
            float bestWeightSum = -1.0f;
            int bestSupport = -1;
            const float support_r_m = 0.04f;
            for (int i = 0; i < (int)clusters.size(); ++i)
            {
                const int support = supportCount(clusters[i].mean, support_r_m);
                const float weightSum = clusters[i].weightSum;
                if (weightSum > bestWeightSum + 1e-6f ||
                    (std::abs(weightSum - bestWeightSum) <= 1e-6f && clusters[i].count > bestCnt) ||
                    (std::abs(weightSum - bestWeightSum) <= 1e-6f && clusters[i].count == bestCnt && support > bestSupport))
                {
                    best = i;
                    bestCnt = clusters[i].count;
                    bestWeightSum = weightSum;
                    bestSupport = support;
                }
            }

            if (best >= 0)
            {
                filtered.emplace_back(id, clusters[best].mean);
                SENDLOGF_TAG(LOG_TAG,
                             "[Merge] id=%d clusters=%d bestCnt=%d bestWeight=%.3f support=%d thr=%.1fcm -> accepted",
                             id, (int)clusters.size(), bestCnt, bestWeightSum, bestSupport, mergeThresh_m * 100.0f);
            }
            else
            {
                int bestIdx = -1, bestSupport = -1;
                for (int i = 0; i < (int)pts.size(); ++i)
                {
                    int sup = supportCount(pts[i].center, support_r_m);
                    const float combinedScore = pts[i].viewWeight * static_cast<float>(sup + 1);
                    const float bestCombinedScore =
                        (bestIdx >= 0) ? (pts[bestIdx].viewWeight * static_cast<float>(bestSupport + 1)) : -1.0f;
                    if (combinedScore > bestCombinedScore + 1e-6f ||
                        (std::abs(combinedScore - bestCombinedScore) <= 1e-6f && sup > bestSupport))
                    {
                        bestSupport = sup;
                        bestIdx = i;
                    }
                }
                const cv::Point3f chosen = (bestIdx >= 0) ? pts[bestIdx].center : pts.back().center;
                filtered.emplace_back(id, chosen);
                SENDLOGF_TAG(LOG_TAG,
                             "[Merge] id=%d clusters=%d -> density pick (r=%.1fcm, support=%d)",
                             id, (int)clusters.size(), support_r_m * 100.0f, bestSupport);
            }
        }

        mergedCenters = std::move(filtered);
    }

    // centers JSON 저장 (Registration/<targetType>/targets3d.json)
    {
        std::string regFolder = blastPath + "/Registration/" + targetType;
        std::filesystem::create_directories(regFolder);
        std::string centersJson = regFolder + "/targets3d.json";
        SaveDetectedCentersJson(centersJson, mergedCenters);
        SENDLOGF_TAG(LOG_TAG, "Saved centers: %s", centersJson.c_str());

        // 추가: centers PLY도 함께 저장
        // targets3d.ply (id별 3D center만 찍은 작은 PLY)
        std::vector<CartesianPointRGB> centersPts;
        centersPts.reserve(mergedCenters.size());
        for (const auto &kv : mergedCenters)
        {
            CartesianPointRGB pt{};
            pt.x = kv.second.x;
            pt.y = kv.second.y;
            pt.z = kv.second.z;
            pt.r = 255;
            pt.g = 0;
            pt.b = 0; // 센터는 눈에 잘 띄게 빨강
            pt.intensity = 0.0f;
            centersPts.push_back(pt);
        }
        std::string centersPly = regFolder + "/targets3d.ply";
        if (savePointCloud(centersPly, centersPts))
        {
            SENDLOGF_TAG(LOG_TAG, "Saved centers PLY: %s (N=%zu)",
                         centersPly.c_str(), centersPts.size());
        }
        else
        {
            SENDLOGF_TAG(LOG_TAG, "Failed to save centers PLY: %s", centersPly.c_str());
        }
    }

    // REPLACE 머지 블록: 이미지 기반 머지 + ID 기반 머지 동시 생성
    if (!allTargetGroups.empty())
    {
        // 1) 기존: 이미지별 그룹 → 전체 머지 (호환 유지)
        {
            std::string allRefPath = regFolder + "/" + targetType + "_reg.ply";
            SaveMergedRegCloud(allRefPath, allTargetGroups);
            SENDLOGF_TAG(LOG_TAG, "Saved merged(all-images): %s", allRefPath.c_str());
        }

        // 2) 신규: ID 별 PLY 저장
        std::vector<std::vector<CartesianPointRGB>> idGroups;
        idGroups.reserve(idBuckets.size());
        for (const auto &kv : idBuckets)
        {
            int id = kv.first;
            const auto &pts = kv.second;

            // 파일명: id_<ID>.ply
            std::string idPly = regFolder + "/id_" + std::to_string(id) + ".ply";
            if (!savePointCloud(idPly, pts))
            {
                SENDLOGF_TAG(LOG_TAG, "Failed to save ID PLY: %s", idPly.c_str());
                // 계속 진행
            }
            idGroups.push_back(pts);
        }

        // 3) 신규: ID 버킷 전체를 하나로 머지 (ID 기준 통합본)
        if (!idGroups.empty())
        {
            std::string idMergedPath = regFolder + "/" + targetType + "_reg_byId.ply";
            SaveMergedRegCloud(idMergedPath, idGroups);
            SENDLOGF_TAG(LOG_TAG, "Saved merged(by-id groups): %s", idMergedPath.c_str());
        }
    }
    else
    {
        SENDLOGF_TAG(LOG_TAG, "No detection JSON found. Skip merge.");
    }
    return true;
}

// JSON + 캘리브레이션만으로 facemarker(0번)의 3D center를 추출하는 경량 함수
bool ExtractFaceMarkerCenter_Standalone(
    const std::string &blastPath,
    const std::string &targetType,
    const std::vector<CartesianPointRGB> &pointCloud,
    cv::Point3f &outFaceCenter,
    std::vector<std::pair<int, cv::Point2f>> &outMarkers2D)
{
    outMarkers2D.clear();
    std::vector<cv::Point3f> faceCandidates3D;

    const std::string detectFolder = blastPath + "/DetectTarget/" + targetType;

    if (!std::filesystem::exists(detectFolder))
    {
        SENDLOGF_TAG(LOG_TAG, "detect folder not found: %s", detectFolder.c_str());
        return false;
    }

    for (const auto &entry : std::filesystem::directory_iterator(detectFolder))
    {
        if (!entry.is_regular_file())
            continue;
        const auto &p = entry.path();
        if (p.extension() != ".json")
            continue;

        const std::string fname = p.filename().string();
        if (fname.rfind("detect_", 0) != 0)
            continue;

        // detect_<idx>_<campos>.json
        int imgIndex = -1;
        std::string camposStr = "NA";
        {
            auto core = fname.substr(std::string("detect_").size());
            core = core.substr(0, core.size() - std::string(".json").size());
            auto us = core.find_last_of('_');
            if (us != std::string::npos)
            {
                imgIndex = std::atoi(core.substr(0, us).c_str());
                camposStr = core.substr(us + 1);
            }
        }

        // id + pixel 로드
        std::vector<std::pair<int, cv::Point2f>> markers;
        if (!LoadPixelsFromJson(p.string(), markers) || markers.empty())
            continue;

        // 캘리브레이션 로드
        std::string calibPath = (std::filesystem::path(detectFolder) /
                                 ("calib_" + std::to_string(imgIndex) + "_" + camposStr + ".json"))
                                    .string();
        int width = 0, height = 0;
        float proj[16], xf[16];
        if (!LoadCalibrationJson(calibPath, width, height, proj, xf))
        {
            SENDLOGF_TAG(LOG_TAG, "Failed to load calib: %s", calibPath.c_str());
            continue;
        }

        Blk360G2_ImageCalibration calib{};
        for (int k = 0; k < 16; ++k)
            calib.projection.elements[k] = proj[k];
        for (int k = 0; k < 16; ++k)
            calib.transformation.elements[k] = xf[k];

        // 이 이미지에서 facemarker(id==0) 의 3D 후보들 계산
        for (const auto &[id, pixelPt] : markers)
        {
            if (id != 0)
                continue; // facemarker만 대상

            cv::Vec3f ray = pixelToRayWorld((int)pixelPt.x, (int)pixelPt.y,
                                            width, height, calib);

            CartesianPointRGB targetPos = findPointByDepthProjection(ray, pointCloud);
            faceCandidates3D.emplace_back(targetPos.x, targetPos.y, targetPos.z);

            outMarkers2D.emplace_back(id, pixelPt);
        }
    }

    if (faceCandidates3D.empty())
    {
        SENDLOGF_TAG(LOG_TAG,
                     "No facemarker(id=0) candidates found.");
        return false;
    }

    // 후보들 중 최종 center 선택
    outFaceCenter = faceCandidates3D.front();

    // detect_facemarker.json 업데이트 (2D + 3D)
    try
    {
        const std::string facemarkerJson =
            detectFolder + "/detect_facemarker.json";

        if (!SaveFacemarkerJson(facemarkerJson,
                                outMarkers2D,
                                &outFaceCenter))
        {
            SENDLOGF_TAG(LOG_TAG, "SaveFacemarkerJson failed: %s", facemarkerJson.c_str());
        }
        else
        {
            SENDLOGF_TAG(LOG_TAG, "detect_facemarker.json updated: %s", facemarkerJson.c_str());
        }
    }
    catch (const std::exception &e)
    {
        SENDLOGF_TAG(LOG_TAG, "Exception while saving facemarker json: %s", e.what());
    }

    return true;
}

// 이미지에서 타겟 및 센터 검출
bool Markerdetection(const std::string &blastPath,
                     Blk360G2_DataManipulationWorkflowHandle dataManipulationWorkflow,
                     std::vector<Blk360G2_ImageHandle> &images,
                     std::vector<CartesianPointRGB> &points,
                     const std::string &targetType,
                     bool isAruco)
{
    auto start = std::chrono::steady_clock::now();
    SENDLOGF_TAG(LOG_TAG, "Starting registration-related image saving, marker detection, 3D conversion, and PLY saving");

    std::filesystem::create_directories(blastPath + "/DetectTarget/" + targetType);

    // 탐지 + 이미지 + JSON(id+pixel)
    if (!DetectMarkersAndSaveImages(blastPath, dataManipulationWorkflow, images, points, targetType, isAruco))
        return false;

    if (!ExtractAndSaveAndMergePointClouds_Standalone(blastPath, targetType, points))
        return false;

    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    SENDLOGF_TAG("ProcessTime", "Point cloud download Sub - Markerdetection time: %lld ms (%lld초)", ms, sec);
    return true;
}

// JSON만으로 점군추출 머지
static bool MergeFromSavedJson(const std::string &blastPath,
                               const std::string &targetType,
                               std::vector<CartesianPointRGB> &points)

{
    auto start = std::chrono::steady_clock::now();
    SENDLOGF_TAG(LOG_TAG, "Starting registration-related image saving, marker detection, 3D conversion, and PLY saving");

    std::filesystem::create_directories(blastPath + "/Registration/" + targetType);

    // JSON 기반 재투영 + 추출 + PLY + 병합
    if (!ExtractAndSaveAndMergePointClouds_Standalone(blastPath, targetType, points))
        return false;

    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    SENDLOGF_TAG("ProcessTime", "Find Center And Merge PtCloud Sub - MergePtCloud time: %lld ms (%lld초)", ms, sec);
    return true;
}

// centersOut(문자열 id, Vec3f center) 안에서
// facemarker( id == 0 ) 의 3D 중심을 찾아 outCenter에 넣는다.
bool FindFaceMarkerCenterFromCenters(
    const std::vector<std::pair<std::string, cv::Vec3f>> &centers,
    cv::Point3f &outCenter)
{
    for (const auto &kv : centers)
    {
        const std::string &idStr = kv.first;

        // 1) 앞에 'T'가 붙어 있으면 떼고 숫자만 파싱 (T000 → 0, T005 → 5)
        const char *p = idStr.c_str();
        if (!idStr.empty() && (idStr[0] == 'T' || idStr[0] == 't'))
            p = idStr.c_str() + 1;

        int id = std::atoi(p); // "0", "000", "5", "005" 다 처리됨

        if (id == 0)
        {
            const cv::Vec3f &v = kv.second;
            outCenter = cv::Point3f(v[0], v[1], v[2]);
            SENDLOGF_TAG(LOG_TAG,
                         "[IntensityOnly] facemarker found in centersOut: id=%s -> (%.3f, %.3f, %.3f)",
                         idStr.c_str(), outCenter.x, outCenter.y, outCenter.z);
            return true;
        }
    }

    SENDLOGF_TAG(LOG_TAG,
                 "[IntensityOnly] facemarker(id=0) not found in centersOut (size=%zu)",
                 centers.size());
    return false;
}

// 타겟 센터탐지 및 타겟 포인트 머지
bool FindCenter3DPt(const std::string &blastPath,
                    const std::string &targetType,
                    std::vector<CartesianPointRGB> &points,
                    float targetSize)
{


    if (!ComputeIntensityCenters(blastPath, targetType, points,
                                  targetSize, 0.15, 50))
        return false;

    return true;
}

// target test---------------------------------------------------------------
// 이미지 경로 하나 받아서 ArUco 탐지 후, 결과 이미지만 저장
// 결과 파일명: <원본파일명>_aruco_result.jpg  (원본과 같은 폴더)
static cv::Mat convertImageToCvMatTest(const std::string &imagePath)
{ //

    if (imagePath.empty())
    {
        SENDLOGF_TAG(LOG_TAG, "convertImageToCvMat: empty path");
        return cv::Mat();
    } //

    // OpenCV는 imread(IMREAD_COLOR)로 읽으면 BGR로 반환함
    cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (bgr.empty())
    {
        SENDLOGF_TAG(LOG_TAG, "convertImageToCvMat: imread failed: %s",
                     imagePath.c_str());
        return cv::Mat();
    }

    return bgr.clone(); // (소유권 분리/안전)
}
bool DetectArucoTest(const std::string &imagePath)
{
    // 1) 이미지 로드 (BGR)
    cv::Mat image = convertImageToCvMatTest(imagePath);
    if (image.empty())
    {
        SENDLOGF_TAG(LOG_TAG, "[ArucoTest] Open image failed: %s", imagePath.c_str());
        return false;
    }

    // 2) 기존 함수 활용: id → center(px)
    std::map<int, cv::Point2f> markerMap = findAllArucoMarkerCentersById(image);
    SENDLOGF_TAG(LOG_TAG, "[ArucoTest] Detected markers: %zu", markerMap.size());

    // 3) 시각화
    cv::Mat vis = image.clone();
    for (const auto &kv : markerMap)
    {
        const int id = kv.first;
        const cv::Point2f &pt = kv.second;

        cv::drawMarker(vis, pt, cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 22, 2);
        cv::putText(vis, std::to_string(id),
                    cv::Point(pt.x + 11, pt.y - 11),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
    }

    // 4) 저장: <원본파일명>_aruco_result.jpg (원본과 동일 폴더)
    namespace fs = std::filesystem;
    fs::path p(imagePath);
    const std::string outPath = (p.parent_path() / (p.stem().string() + "_aruco_result.jpg")).string();
    if (!cv::imwrite(outPath, vis))
    {
        SENDLOGF_TAG(LOG_TAG, "[ArucoTest] Failed to save: %s", outPath.c_str());
        return false;
    }

    SENDLOGF_TAG(LOG_TAG, "[ArucoTest] Done. saved=%s", outPath.c_str());
    return true;
}
