#pragma once

#include <cmath>
#include <string>
#include "PointStructures.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

// ===================== Intensity-only Center Estimation  =====================

// [Intensity를 이용한 사각(안쪽 ≤ t_low) 기반 센터 추정/저장]
// t_low (사각 내부 임계)
//  정규화(ROI 퍼센타일 2–98% + p99 클리핑)된 intensity에서 **“확실히 내부”**로 간주할 최대값.
//  내부가 어두울수록 값이 낮음.
//  권장 시작값: 0.10 ~ 0.15 (밝은 조명, 반사 많으면 0.12~0.15 쪽 유리)

// roi_cm (ROI 시작 반경)
//  C_prev 중심으로 첫 번째 반경(cm). 부족하면 확장/축소해 2~3회 시도.
//  권장: 10 ~ 15 cm (타겟 한 변이 10 cm이므로 12→15→10 cm 순서 추천: 먼저 넉넉히 포함 후 좁혀 안정화)

// alpha (근접성 패널티 가중치)
//  alpha↑ → C_prev(기존 센터)에 더 가까운 후보를 강하게 선호(=멀면 패널티 큼)
//  alpha↓ → **면적(사각 크기)**을 더 중시, C_prev에서 조금 떨어져도 면적이 크면 채택

// 방법
// 점군의 신호강도만으로 사각 타겟(10 cm)의 정확한 3D 중심을 추정.
// 기존 센터(C_prev)를 안전 폴백으로 유지 → 이상치 방지.

// 1) 실행 위치·산출물
// 위치: 촬영 이후, ICP 직전 FindCenterAndMergePt(...) 말미(Ref/Cmp 각각).
// 산출물(분리 저장):
// Registration/<type>/targets3d_intensity.ply / .json
// Registration/<type>/targets3d_intensity_dbg.json(delta_mm 로그)

// 2) 입력·파라미터
// 입력: {blastPath}/{type}.ply, 기존 targets3d.{ply|json}의 C_prev(있으면).
// 파라미터(최종):
// t_low: 내부 임계(정규화 기반) — 사무실 0.12, 터널 0.10
// roi_cm: ROI 시작 반경 — 10 (시도 순서는 12 → 15 → 10 cm)
// alpha: 근접성 패널티 가중 — 사무실 50, 터널 40
// erosion 횟수: 사무실 2, 터널 1

// 3) 알고리즘(한 타겟당)
// ROI 추출: C_prev 기준 반경 12 → 15 → 10 cm 순서로 점군 서브셋 만들기.
// 정규화(노멀라이즈): ROI 강도 분포의 2–98% 퍼센타일을 (lo, hi)로 설정,
// p99 클리핑 후 norm = clamp((min(raw,p99)-lo)/(hi-lo))로 ROI-로컬 정규화.
// 적응 임계 t_adapt: 하위 p35를 반영해 t_adapt = min(t_low, p35n + 0.02)로 밝은 환경 보정.
// 내부점(inner) 선별: norm ≤ t_adapt(부족 시 +0.02 완화).
// 평면 추정(PCA): inner만으로 평면(u, v, n) 획득 → UV 평면 좌표계 구성.
// 래스터화 + 모폴로지: 2 mm/px로 래스터 → erosion(사무실 2회/터널 1회)로 내핵(core) 확보.
// 사각 후보 선택(게이트+근접성):
// 10 cm 정사각 게이트: 한 변 0.08–0.12 m, 종횡비 0.85–1.18
// 점수식: score = area_px - alpha * dist_px
// (C_prev의 UV-픽셀 좌표와의 거리 dist_px 사용 → 가까울수록 유리)

// 중심 산출:
// 사각 검출 성공 → minAreaRect 중심을 UV→3D 복원
// 실패 → core 센트로이드 폴백, 그래도 안 되면 inner 평균 최종 폴백
// 업데이트 가드(매우 중요):

// Δ = ‖C_new − C_prev‖
// Δ > 30 mm ⇒ 업데이트 취소(C_prev 유지)
// Δ < 1 mm ⇒ 미세 변화 무시(C_prev 유지)

// 4) 로깅·디버그
// ROI/inner 수: [IntensityOnly] ROI pts=..., inner pts=... (t_low=..., t_adapt=...)
// 래스터/사각: raster WxH=..., rect=0/1 (erosion core)
// 이동량: [IntensityOnly] id=..., updated=0/1, delta=.. mm
// targets3d_intensity_dbg.json: id별 delta_mm 기록
// (필요 시 intensity_uv_*.png 저장 옵션으로 시각 디버그)

// 5) 환경별 튜닝 가이드
// 사무실(형광등): t_low=0.12, alpha=50, erosion=2
// (상단 꼬리 영향 큼 → 근접성 가중↑, 내핵 강조)
// 터널(어두움): t_low=0.10, alpha=40, erosion=1
// (대비↑ → 임계 낮춰 포인트 확보, 근접성 가중 약간↓)

// 6) 안전성·롤백
// 본선 산출물(targets3d.*)과 분리 저장 → 언제든 원복/AB 테스트 쉬움.
// 실패·애매하면 항상 C_prev 유지(안전한 기본값).

// PCA 평면
struct PlaneFit
{
    bool ok = false;
    cv::Vec3f mean, u, v, n;
};
struct UVBox
{
    float umin, umax, vmin, vmax;
    int W, H;
};

// PLY에서 점군을 로드하고 개수를 로그로 남긴다.
// static std::vector<CartesianPointRGB> loadPc(const std::string &path);

// 기존 센터(Registration/<type>/targets3d.json)를 읽어 (id, center) 목록으로 반환한다.
static bool loadPrevCenters_JSON(const std::string &jsonPath,
                                 std::vector<std::pair<std::string, cv::Vec3f>> &out);

// 기존 센터(Registration/<type>/targets3d.ply)를 읽어 T001.. 순번 id와 좌표 목록을 만든다.
static bool loadPrevCenters_PLY(const std::string &plyPath,
                                std::vector<std::pair<std::string, cv::Vec3f>> &out);

// 기준점 c 주변 반경 r_cm 내의 점만 추출해 ROI 점 목록(좌표)으로 반환한다.
static std::vector<cv::Vec3f> gatherROI(const std::vector<CartesianPointRGB> &pc,
                                        const cv::Vec3f &c, float r_cm);

// 입력 점들로 PCA를 수행해 평면의 mean,u,v,n(법선)을 추정한다.
static PlaneFit fitPlanePCA(const std::vector<cv::Vec3f> &pts);

// 평면에 투영해 (U,V) 밀도 래스터 이미지를 만들고 UV 범위를 UVBox에 기록한다.
static void rasterizeBinary(const std::vector<cv::Vec3f> &pts,
                            const PlaneFit &pf,
                            cv::Mat &img, UVBox &box,
                            float cell_mm = 2.0f);

// 바이너리 컨투어들 중 가장 큰 정사각형의 minAreaRect 중심 픽셀을 찾는다.
static bool findRectCenter_minArea(const cv::Mat &img, cv::Point2f &pixC);

// 픽셀 중심을 UVBox 범위를 통해 (U,V)로 환산하고 3D 평면 위 좌표로 역투영한다.
static cv::Vec3f uvCenterTo3D(const PlaneFit &pf, const UVBox &b, const cv::Point2f &pc);

// (id,center) 목록을 PLY(빨간 점)와 JSON으로 dir/stem.*에 저장한다.
static bool saveCentersPLYJSON(const std::string &dir, const std::string &stem,
                               const std::vector<std::pair<std::string, cv::Vec3f>> &centers);

// minAreaRect 후보 중 10cm 정사각 크기/종횡비 게이트를 통과한 사각의 중심을 반환한다.
static bool findRectCenter_minAreaGated(const cv::Mat &img, const UVBox &box, cv::Point2f &pixC,
                                        float sideMin = 0.08f, float sideMax = 0.12f,      // 한 변(m)
                                        float aspectMin = 0.85f, float aspectMax = 1.18f); // 종횡비

// C_prev 기반 ROI에서 intensity만으로 타겟 중심을 재추정해 targets3d_intensity.*로 저장한다.
bool ComputeIntensityCenters(const std::string &blastPath,
                             const std::string &targetType,
                             std::vector<CartesianPointRGB> &points,
                             Mat3 &outFrontDir,
                             bool usefacemarker = false,
                             float targetSize = 10.0f, // 타겟(마커 흰부분) 크기
                             float t_low = 0.15f,      // 사각 밝은 픽셀 상한(터널의 경우 0.1)
                             float alpha = 50.0f);     // 근접성 패털티 가중치(터널의 경우 35~45)
