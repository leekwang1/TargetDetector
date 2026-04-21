#include <vector>
#include <string>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "CenterEstimation.h"
#include "PointCloudRegistration.h"
#include "PlyWriter.h"
#include "logging.h"

using nlohmann::json;
#define LOG_TAG "Center Estimation"

// ===================== Intensity-only Center Estimation  =====================
// PLY에서 점군을 로드하고 개수를 로그로 남긴다.
// static std::vector<CartesianPointRGB> loadPc(const std::string &path)
// {
//     std::vector<CartesianPointRGB> pc = loadPointCloud(path);
//     SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] load %s -> %zu pts", path.c_str(), pc.size());
//     return pc;
// }

// 기존 센터(Registration/<type>/targets3d.json)를 읽어 (id, center) 목록으로 반환한다.
static bool loadPrevCenters_JSON(const std::string &jsonPath,
                                 std::vector<std::pair<std::string, cv::Vec3f>> &out)
{
    try
    {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open())
            return false;
        nlohmann::json j;
        ifs >> j;
        if (!j.is_array())
            return false;
        out.clear();
        for (auto &e : j)
        {
            if (!e.contains("center"))
                continue;
            std::string id = e.value("id", "");
            auto c = e["center"];
            if (c.is_array() && c.size() == 3)
                out.emplace_back(id, cv::Vec3f((float)c[0], (float)c[1], (float)c[2]));
        }
        return !out.empty();
    }
    catch (...)
    {
        return false;
    }
}

// 기존 센터(Registration/<type>/targets3d.ply)를 읽어 T001.. 순번 id와 좌표 목록을 만든다.
static bool loadPrevCenters_PLY(const std::string &plyPath,
                                std::vector<std::pair<std::string, cv::Vec3f>> &out)
{
    try
    {
        std::vector<CartesianPointRGB> pts = loadPointCloud(plyPath);
        if (pts.empty())
            return false;
        out.clear();
        out.reserve(pts.size());
        for (size_t i = 0; i < pts.size(); ++i)
        {
            char idbuf[16];
            std::snprintf(idbuf, sizeof(idbuf), "T%03zu", i + 1);
            out.emplace_back(idbuf, cv::Vec3f((float)pts[i].x, (float)pts[i].y, (float)pts[i].z));
        }
        return !out.empty();
    }
    catch (...)
    {
        return false;
    }
}

// 기준점 c 주변 반경 r_cm 내의 점만 추출해 ROI 점 목록(좌표)으로 반환한다.
static std::vector<cv::Vec3f> gatherROI(const std::vector<CartesianPointRGB> &pc,
                                        const cv::Vec3f &c, float r_cm)
{
    const float r = r_cm / 100.f, r2 = r * r;
    std::vector<cv::Vec3f> out;
    out.reserve(2048);
    for (auto &p : pc)
    {
        cv::Vec3f q(p.x, p.y, p.z), d = q - c;
        if (d.dot(d) <= r2)
            out.push_back(q);
    }
    return out;
}

// 입력 점들로 PCA를 수행해 평면의 mean,u,v,n(법선)을 추정한다.
static PlaneFit fitPlanePCA(const std::vector<cv::Vec3f> &pts)
{
    PlaneFit pf;
    if (pts.size() < 50)
        return pf;
    cv::Mat X((int)pts.size(), 3, CV_32F);
    cv::Vec3f m(0, 0, 0);
    for (int i = 0; i < (int)pts.size(); ++i)
    {
        X.at<float>(i, 0) = pts[i][0];
        X.at<float>(i, 1) = pts[i][1];
        X.at<float>(i, 2) = pts[i][2];
        m += pts[i];
    }
    m *= (1.f / (float)pts.size());
    cv::PCA pca(X, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Vec3f pc1(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1), pca.eigenvectors.at<float>(0, 2));
    cv::Vec3f pc2(pca.eigenvectors.at<float>(1, 0), pca.eigenvectors.at<float>(1, 1), pca.eigenvectors.at<float>(1, 2));
    cv::Vec3f pc3(pca.eigenvectors.at<float>(2, 0), pca.eigenvectors.at<float>(2, 1), pca.eigenvectors.at<float>(2, 2));
    pf.ok = true;
    pf.mean = m;
    pf.u = pc1;
    pf.v = pc2;
    pf.n = pc3;
    return pf;
}

// 평면에 투영해 (U,V) 밀도 래스터 이미지를 만들고 UV 범위를 UVBox에 기록한다.
static void rasterizeBinary(const std::vector<cv::Vec3f> &pts,
                            const PlaneFit &pf,
                            cv::Mat &img, UVBox &box,
                            float cell_mm)
{
    // ROI를 평면에 투영 → binary 이미지 구성
    float umin = 1e9f, umax = -1e9f, vmin = 1e9f, vmax = -1e9f;
    std::vector<cv::Point2f> uv;
    uv.reserve(pts.size());
    for (auto &p : pts)
    {
        cv::Vec3f d = p - pf.mean;
        float U = d.dot(pf.u), V = d.dot(pf.v);
        uv.emplace_back(U, V);
        umin = std::min(umin, U);
        umax = std::max(umax, U);
        vmin = std::min(vmin, V);
        vmax = std::max(vmax, V);
    }
    // 셀크기(상대)이므로 고정 해상도로 안전히
    int W = std::clamp<int>((int)std::round((umax - umin) / (cell_mm / 1000.f)), 128, 1024);
    int H = std::clamp<int>((int)std::round((vmax - vmin) / (cell_mm / 1000.f)), 128, 1024);
    img = cv::Mat(H, W, CV_8U, cv::Scalar(0));
    for (auto &q : uv)
    {
        int x = (int)std::floor((q.x - umin) / std::max(1e-6f, (umax - umin)) * (W - 1));
        int y = (int)std::floor((q.y - vmin) / std::max(1e-6f, (vmax - vmin)) * (H - 1));
        x = std::clamp(x, 0, W - 1);
        y = std::clamp(y, 0, H - 1);
        img.at<uchar>(y, x) = 255;
    }
    // 잡음 제거
    cv::morphologyEx(img, img, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
    cv::morphologyEx(img, img, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));
    box = {umin, umax, vmin, vmax, W, H};
}

// 바이너리 컨투어들 중 가장 큰 정사각형의 minAreaRect 중심 픽셀을 찾는다.
static bool findRectCenter_minArea(const cv::Mat &img, cv::Point2f &pixC)
{
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(img, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (cs.empty())
        return false;
    float bestArea = 0;
    cv::RotatedRect best;
    for (auto &c : cs)
    {
        if (cv::contourArea(c) < 50.0)
            continue;
        cv::RotatedRect r = cv::minAreaRect(c);
        float w = std::max(r.size.width, r.size.height);
        float h = std::min(r.size.width, r.size.height);
        float ratio = (h > 1e-3f) ? (w / h) : 999.f;
        if (ratio > 1.3f)
            continue; // 정사각 근사
        float area = w * h;
        if (area > bestArea)
        {
            bestArea = area;
            best = r;
        }
    }
    if (bestArea <= 0)
        return false;
    pixC = best.center;
    return true;
}

// 픽셀 중심을 UVBox 범위를 통해 (U,V)로 환산하고 3D 평면 위 좌표로 역투영한다.
static cv::Vec3f uvCenterTo3D(const PlaneFit &pf, const UVBox &b, const cv::Point2f &pc)
{
    float U = b.umin + (pc.x / std::max(1, b.W - 1)) * (b.umax - b.umin);
    float V = b.vmin + (pc.y / std::max(1, b.H - 1)) * (b.vmax - b.vmin);
    return pf.mean + U * pf.u + V * pf.v;
}

// (id,center) 목록을 PLY(빨간 점)와 JSON으로 dir/stem.*에 저장한다.
static bool saveCentersPLYJSON(const std::string &dir, const std::string &stem,
                               const std::vector<std::pair<std::string, cv::Vec3f>> &centers)
{
    std::filesystem::create_directories(dir);
    std::vector<CartesianPointRGB> pts;
    pts.reserve(centers.size());
    for (auto &kv : centers)
    {
        CartesianPointRGB p{};
        p.x = kv.second[0];
        p.y = kv.second[1];
        p.z = kv.second[2];
        p.r = 255;
        p.g = 0;
        p.b = 0;
        p.intensity = 0.0f;
        pts.push_back(p);
    }
    const std::string ply = dir + "/" + stem + ".ply";
    const std::string json = dir + "/" + stem + ".json";
    bool okply = savePointCloud(ply, pts);
    nlohmann::json j = nlohmann::json::array();
    for (auto &kv : centers)
    {
        nlohmann::json e;
        if (!kv.first.empty())
            e["id"] = kv.first;
        e["center"] = {kv.second[0], kv.second[1], kv.second[2]};
        j.push_back(e);
    }
    std::ofstream(json) << j.dump(2);
    SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] saved %s (N=%zu), %s", ply.c_str(), centers.size(), json.c_str());
    return okply;
}

static std::string makeDebugStem(const std::string &id)
{
    std::string stem = id.empty() ? "unknown" : id;
    for (char &ch : stem)
    {
        if (!std::isalnum(static_cast<unsigned char>(ch)))
            ch = '_';
    }
    return stem;
}

static std::vector<CartesianPointRGB> toDebugCloud(const std::vector<CartesianPointRGB> &pts,
                                                   float r,
                                                   float g,
                                                   float b)
{
    std::vector<CartesianPointRGB> out = pts;
    for (auto &pt : out)
    {
        pt.r = r;
        pt.g = g;
        pt.b = b;
    }
    return out;
}

static std::vector<CartesianPointRGB> toDebugCloud(const std::vector<cv::Vec3f> &pts,
                                                   float r,
                                                   float g,
                                                   float b)
{
    std::vector<CartesianPointRGB> out;
    out.reserve(pts.size());
    for (const auto &pt : pts)
    {
        CartesianPointRGB point{};
        point.x = pt[0];
        point.y = pt[1];
        point.z = pt[2];
        point.r = r;
        point.g = g;
        point.b = b;
        point.intensity = 0.0f;
        out.push_back(point);
    }
    return out;
}

static void saveIntensityDebugArtifacts(const std::string &debugDir,
                                        const std::string &id,
                                        const std::vector<CartesianPointRGB> &roiPts,
                                        const std::vector<cv::Vec3f> &innerPts,
                                        const cv::Mat &img,
                                        const cv::Mat &core,
                                        const cv::Vec3f &prevCenter,
                                        const cv::Vec3f &estimatedCenter,
                                        float roiRadiusCm,
                                        float tLow,
                                        float tAdapt)
{
    std::filesystem::create_directories(debugDir);

    const std::string stem = makeDebugStem(id);
    const std::string roiPath = debugDir + "/" + stem + "_roi.ply";
    const std::string innerPath = debugDir + "/" + stem + "_inner.ply";
    const std::string rasterPath = debugDir + "/" + stem + "_raster.png";
    const std::string corePath = debugDir + "/" + stem + "_core.png";
    const std::string statsPath = debugDir + "/" + stem + "_stats.json";

    if (!roiPts.empty())
        savePointCloud(roiPath, toDebugCloud(roiPts, 120, 180, 255));
    if (!innerPts.empty())
        savePointCloud(innerPath, toDebugCloud(innerPts, 255, 90, 90));
    if (!img.empty())
        cv::imwrite(rasterPath, img);
    if (!core.empty())
        cv::imwrite(corePath, core);

    nlohmann::json stats;
    stats["id"] = id;
    stats["roi_radius_cm"] = roiRadiusCm;
    stats["roi_points"] = roiPts.size();
    stats["inner_points"] = innerPts.size();
    stats["t_low"] = tLow;
    stats["t_adapt"] = tAdapt;
    stats["previous_center"] = {prevCenter[0], prevCenter[1], prevCenter[2]};
    stats["estimated_center"] = {estimatedCenter[0], estimatedCenter[1], estimatedCenter[2]};

    std::ofstream ofs(statsPath);
    if (ofs.is_open())
        ofs << stats.dump(2);
}

// minAreaRect 후보 중 10cm 정사각 크기/종횡비 게이트를 통과한 사각의 중심을 반환한다.
static bool findRectCenter_minAreaGated(const cv::Mat &img, const UVBox &box, cv::Point2f &pixC,
                                        float sideMin, float sideMax,
                                        float aspectMin, float aspectMax)
{
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(img, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (cs.empty())
        return false;

    float bestScore = -1.f;
    cv::RotatedRect best;
    for (auto &c : cs)
    {
        if (cv::contourArea(c) < 50.0)
            continue;
        cv::RotatedRect r = cv::minAreaRect(c);
        float wpx = std::max(r.size.width, r.size.height); // 픽셀
        float hpx = std::min(r.size.width, r.size.height);
        if (hpx < 1.0f)
            continue;

        // 픽셀 → UV → 미터 환산
        float width_m = (box.umax - box.umin) * (wpx / std::max(1, box.W - 1));
        float height_m = (box.vmax - box.vmin) * (hpx / std::max(1, box.H - 1));

        // 종횡비/사이즈 게이트
        float aspect = width_m / std::max(1e-6f, height_m);
        if (aspect < aspectMin || aspect > aspectMax)
            continue;

        float side = 0.5f * (width_m + height_m); // 평균 변 길이
        if (side < sideMin || side > sideMax)
            continue;

        // 점수: 면적(픽셀) 우선
        float score = wpx * hpx;
        if (score > bestScore)
        {
            bestScore = score;
            best = r;
        }
    }

    if (bestScore <= 0)
        return false;
    pixC = best.center;
    return true;
}

// C_prev 기반 ROI에서 intensity만으로 타겟 중심을 재추정해 targets3d_intensity.*로 저장한다.
bool ComputeIntensityCenters(const std::string &blastPath,
                             const std::string &targetType,
                             std::vector<CartesianPointRGB> &points,
                             Mat3 &outFrontDir,
                             bool usefacemarker,
                             float targetSize,
                             float t_low,
                             float alpha)
{
    auto start = std::chrono::steady_clock::now();

    const float diag_half = 0.707f * targetSize; // 정사각 대각선/2
    const float margin_cm = 3.0f;
    const float roi_base_cm = std::max(0.75f * targetSize, diag_half + margin_cm);

    const std::string regDir = blastPath + "/Registration/" + targetType;
    std::filesystem::create_directories(regDir);
    const std::string debugDir = regDir + "/debug";
    std::filesystem::create_directories(debugDir);

    // 0) 점군 로드
    // auto pc = loadPc(blastPath + "/" + targetType + ".ply");
    if (points.size() < 200)
    {
        SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] too few points");
        return false;
    }

    // 1) 기존 센터 로드 (JSON > PLY)
    std::vector<std::pair<std::string, cv::Vec3f>> prev;
    bool hasPrev = loadPrevCenters_JSON(regDir + "/targets3d.json", prev) ||
                   loadPrevCenters_PLY(regDir + "/targets3d.ply", prev);

    // 2) ROI 추출 (CartesianPointRGB 그대로 반환)
    auto roiPoints = [&](const cv::Vec3f &c, float r_cm)
    {
        const float r = r_cm / 100.f, r2 = r * r;
        std::vector<CartesianPointRGB> out;
        out.reserve(4096);
        for (auto &p : points)
        {
            cv::Vec3f q(p.x, p.y, p.z), d = q - c;
            if (d.dot(d) <= r2)
                out.push_back(p);
        }
        return out;
    };

    // 3) ROI → 정규화 → inner(≤t_low 적응) → 평면(PCA) → 래스터(2mm, erosion) → 10cm 게이트+근접성 → 3D
    auto runOneROI = [&](const std::string &debugId,
                         const std::vector<CartesianPointRGB> &roiPts,
                         const cv::Vec3f &Cprev,
                         float roiRadiusCm) -> std::optional<cv::Vec3f>
    {
        if ((int)roiPts.size() < 80)
        {
            SENDLOGF_TAG(LOG_TAG, "[IntensityOnly][fail] ROI too small: %d", (int)roiPts.size());
            return std::nullopt;
        }

        // (a) ROI intensity 정규화 (2–98% 퍼센타일 + 상단 꼬리 p99 클리핑)
        std::vector<float> I;
        I.reserve(roiPts.size());
        for (auto &p : roiPts)
            I.push_back(p.intensity);
        std::sort(I.begin(), I.end());
        auto q = [&](double a)
        {
            size_t idx = (size_t)std::clamp<double>(a * (I.size() - 1), 0, I.size() - 1);
            return I[idx];
        };
        float lo = q(0.02), hi = q(0.98);
        if (hi <= lo)
        {
            lo = I.front();
            hi = I.back();
            if (hi <= lo)
                return std::nullopt;
        }
        float p99 = q(0.99f); // 상위 꼬리
        auto normI = [&](float raw)
        {
            float clamped = std::min(raw, p99); // 형광등 꼬리 억제
            float v = (clamped - lo) / (hi - lo);
            return std::min(1.f, std::max(0.f, v));
        };
        SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] norm lo=%.4f hi=%.4f p99=%.4f", lo, hi, p99);

        // (b) 적응 임계 t_adapt (덜 남는 타겟 보호)
        float p35 = q(0.35f);
        float p35n = std::clamp((p35 - lo) / (hi - lo), 0.f, 1.f);
        float t_adapt = std::min(t_low, p35n + 0.02f);
        SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] t_low=%.3f p35n=%.3f -> t_adapt=%.3f", t_low, p35n, t_adapt);

        // (c) inner(사각 내부) 선택
        std::vector<cv::Vec3f> inner;
        inner.reserve(roiPts.size());
        for (auto &p : roiPts)
            if (normI(p.intensity) <= t_adapt)
                inner.emplace_back(p.x, p.y, p.z);
        if ((int)inner.size() < 30)
            for (auto &p : roiPts)
                if (normI(p.intensity) <= (t_adapt + 0.02f))
                    inner.emplace_back(p.x, p.y, p.z);

        SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] ROI pts=%zu, inner pts=%zu (t_low=%.3f, t_adapt=%.3f)",
                     roiPts.size(), inner.size(), t_low, t_adapt);
        if ((int)inner.size() < 30)
        {
            SENDLOGF_TAG(LOG_TAG, "[IntensityOnly][fail] inner too few after relax: %d", (int)inner.size());
            return std::nullopt;
        }

        // (d) 평면(PCA)
        auto pf = fitPlanePCA(inner);
        if (!pf.ok)
            return std::nullopt;

        // (e) 평면 투영 → 래스터(2.0mm) + erosion(내핵)
        cv::Mat img;
        UVBox box;
        rasterizeBinary(inner, pf, img, box, 2.0f /*mm*/);
        cv::Mat core = img.clone();
        cv::erode(core, core, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}), cv::Point(-1, -1), 2); // 2회 erosion

        // prev를 UV 픽셀 좌표로 (근접성 점수에 사용)
        auto worldToPix = [&](const cv::Vec3f &W)
        {
            cv::Vec3f d = W - pf.mean;
            float U = d.dot(pf.u), V = d.dot(pf.v);
            float x = (U - box.umin) / std::max(1e-6f, (box.umax - box.umin)) * (box.W - 1);
            float y = (V - box.vmin) / std::max(1e-6f, (box.vmax - box.vmin)) * (box.H - 1);
            return cv::Point2f(x, y);
        };
        cv::Point2f pixPrev = worldToPix(Cprev);

        // (f) 컨투어 스코어 = 면적(px) - α·거리(px) + 10cm 크기 게이트
        cv::Point2f pixC;
        bool gotRect = false;
        {
            std::vector<std::vector<cv::Point>> cs;
            cv::findContours(core, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            float bestScore = -1.f;
            cv::RotatedRect best;

            for (auto &c : cs)
            {
                if (cv::contourArea(c) < 50.0)
                    continue;
                cv::RotatedRect r = cv::minAreaRect(c);
                float wpx = std::max(r.size.width, r.size.height);
                float hpx = std::min(r.size.width, r.size.height);
                if (hpx < 1.0f)
                    continue;

                // 픽셀 → 미터 환산
                float width_m = (box.umax - box.umin) * (wpx / std::max(1, box.W - 1));
                float height_m = (box.vmax - box.vmin) * (hpx / std::max(1, box.H - 1));

                // 종횡비(완화)
                float aspect = width_m / std::max(1e-6f, height_m);
                if (aspect < 0.80f || aspect > 1.25f)
                    continue;

                // 수축 보정: Aruco 검은색 영역 13cm × 0.77 ≈ 10cm(Aruco 안쪽 흰색)
                const float innerRatio = 0.77f;                              // ← 필요시 0.74~0.82로 미세조정
                const float expected_m = (targetSize * innerRatio) / 100.0f; // targetSize=13(검은 정사각 물리)
                const float tol_m = std::max(0.015f, 0.15f * expected_m);    // ±1.5cm or ±15% 완화
                const float side = 0.5f * (width_m + height_m);
                if (side < expected_m - tol_m || side > expected_m + tol_m)
                    continue;

                // 점수: 큰 면적 + C_prev에 가까울수록 유리
                cv::Point2f cand = r.center;
                float area_px = wpx * hpx;
                float dist_px = std::hypot(cand.x - pixPrev.x, cand.y - pixPrev.y);
                float score = area_px - alpha * dist_px;

                if (score > bestScore)
                {
                    bestScore = score;
                    best = r;
                }
            }
            if (bestScore > 0)
            {
                pixC = best.center;
                gotRect = true;
            }
        }

        SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] raster %dx%d, rect=%d (erosion core)",
                     core.cols, core.rows, (int)gotRect);

        // (g) 사각 실패 시: core의 센트로이드 폴백
        if (!gotRect)
        {
            cv::Moments mu = cv::moments(core, true);
            if (mu.m00 > 1e-3)
            {
                pixC = cv::Point2f((float)(mu.m10 / mu.m00), (float)(mu.m01 / mu.m00));
                gotRect = true;
            }
        }

        // (h) 최종 3D 복원 (or inner 평균 최종 폴백)
        cv::Vec3f C;
        if (gotRect)
        {
            C = uvCenterTo3D(pf, box, pixC);
        }
        else
        {
            cv::Vec3f mean(0, 0, 0);
            for (auto &v : inner)
                mean += v;
            mean *= (1.f / (float)inner.size());
            C = mean;
        }

        saveIntensityDebugArtifacts(debugDir,
                                    debugId,
                                    roiPts,
                                    inner,
                                    img,
                                    core,
                                    Cprev,
                                    C,
                                    roiRadiusCm,
                                    t_low,
                                    t_adapt);

        return std::optional<cv::Vec3f>(C);
    };

    // 4) 메인 루프: prev 있으면 ROI 기반, 없으면 전체에서 1개
    std::vector<std::pair<std::string, cv::Vec3f>> centersOut;
    nlohmann::json dbg = nlohmann::json::array();

    if (hasPrev)
    {
        for (auto &kv : prev)
        {
            const std::string &id = kv.first;
            const cv::Vec3f Cprev = kv.second;

            cv::Vec3f Cbest = Cprev;
            bool okAny = false;

            // 반경: 20 → 23 → 17 cm (먼저 넉넉히 포함 후 좁힘)
            for (float r : {roi_base_cm + 2.0f, roi_base_cm + 5.0f, roi_base_cm})
            {
                auto roi = roiPoints(Cprev, r);
                auto C = runOneROI(id, roi, Cprev, r);
                if (C.has_value())
                {
                    Cbest = *C;
                    okAny = true;
                    break;
                }
            }

            float delta_mm = cv::norm(Cbest - Cprev) * 1000.0f;

            // 이동량 캡: 30mm 이상이면 업데이트 취소
            if (okAny && delta_mm > 30.0f)
            {
                okAny = false;
                Cbest = Cprev;
                delta_mm = 0.0f;
            }

            // 보수적 업데이트: 1mm 미만 변화는 무시
            if (okAny && delta_mm < 1.0f)
            {
                okAny = false;
                Cbest = Cprev;
                delta_mm = 0.0f;
            }

            SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] id=%s updated=%d delta=%.2f mm",
                         id.c_str(), (int)okAny, delta_mm);

            centersOut.emplace_back(id, okAny ? Cbest : Cprev);
            dbg.push_back({
                {"id", id},
                {"previous_center", {Cprev[0], Cprev[1], Cprev[2]}},
                {"selected_center_before_alignment", {Cbest[0], Cbest[1], Cbest[2]}},
                {"updated", okAny},
                {"delta_mm", delta_mm},
            });
        }
    }
    else
    {
        // C_prev 없으면: 전체 점군에서 1개만 시도
        auto C = runOneROI("T001", points, /*Cprev=*/cv::Vec3f(0, 0, 0), roi_base_cm);
        if (C.has_value())
        {
            centersOut.emplace_back(std::string("T001"), *C);
            dbg.push_back({
                {"id", "T001"},
                {"previous_center", {0.0f, 0.0f, 0.0f}},
                {"selected_center_before_alignment", {(*C)[0], (*C)[1], (*C)[2]}},
                {"updated", true},
                {"delta_mm", 0.0f},
            });
        }
    }

    // 5) 저장 및 종료
    if (centersOut.empty())
    {
        SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] no centers detected (keep previous if any)");
        return false;
    }

    // === [추가] facemarker(0번) 기준으로 centers + points 축 정렬 ===
    if (usefacemarker)
    {
        cv::Point3f faceCenter{};
        bool hasFace = FindFaceMarkerCenterFromCenters(centersOut, faceCenter);

        if (hasFace)
        {
            // 0번 facemarker 를 바라보는 방향을 -Z 정면으로 맞추는 회전행렬
            outFrontDir = computeFaceAlignmentMatrix(faceCenter);

            // (1) 타겟 센터들(centersOut) 회전
            for (auto &kv : centersOut)
            {
                cv::Vec3f &C = kv.second;
                Vec3 v{C[0], C[1], C[2]};
                Vec3 vr = mat3MulVec3(outFrontDir, v);

                C[0] = static_cast<float>(vr.x);
                C[1] = static_cast<float>(vr.y);
                C[2] = static_cast<float>(vr.z);
            }
        }
        else
        {
            SENDLOGF_TAG(LOG_TAG,
                         "[IntensityOnly] facemarker(id=0) not found in centersOut "
                         "-> skip front-axis alignment.");
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    SENDLOGF_TAG("ProcessTime", "Find Center And Merge PtCloud Sub - Compute Intensity Centers time: %lld ms (%lld초)", ms, sec);

    {
        nlohmann::json dbgRoot;
        dbgRoot["target_type"] = targetType;
        dbgRoot["target_size_cm"] = targetSize;
        dbgRoot["use_face_marker"] = usefacemarker;
        dbgRoot["entries"] = dbg;

        std::ofstream ofs(regDir + "/targets3d_adjust_dbg.json");
        if (ofs.is_open())
            ofs << dbgRoot.dump(2);
    }

    return saveCentersPLYJSON(regDir, "targets3d_adjust", centersOut);
}
