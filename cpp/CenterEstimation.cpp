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
// PLY?먯꽌 ?먭뎔??濡쒕뱶?섍퀬 媛쒖닔瑜?濡쒓렇濡??④릿??
// static std::vector<CartesianPointRGB> loadPc(const std::string &path)
// {
//     std::vector<CartesianPointRGB> pc = loadPointCloud(path);
//     SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] load %s -> %zu pts", path.c_str(), pc.size());
//     return pc;
// }

// 湲곗〈 ?쇳꽣(Registration/<type>/targets3d.json)瑜??쎌뼱 (id, center) 紐⑸줉?쇰줈 諛섑솚?쒕떎.
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

// 湲곗〈 ?쇳꽣(Registration/<type>/targets3d.ply)瑜??쎌뼱 T001.. ?쒕쾲 id? 醫뚰몴 紐⑸줉??留뚮뱺??
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

// 湲곗???c 二쇰? 諛섍꼍 r_cm ?댁쓽 ?먮쭔 異붿텧??ROI ??紐⑸줉(醫뚰몴)?쇰줈 諛섑솚?쒕떎.
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

// ?낅젰 ?먮뱾濡?PCA瑜??섑뻾???됰㈃??mean,u,v,n(踰뺤꽑)??異붿젙?쒕떎.
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

// ?됰㈃???ъ쁺??(U,V) 諛???섏뒪???대?吏瑜?留뚮뱾怨?UV 踰붿쐞瑜?UVBox??湲곕줉?쒕떎.
static void rasterizeBinary(const std::vector<cv::Vec3f> &pts,
                            const PlaneFit &pf,
                            cv::Mat &img, UVBox &box,
                            float cell_mm)
{
    // ROI瑜??됰㈃???ъ쁺 ??binary ?대?吏 援ъ꽦
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
    // ??ш린(?곷?)?대?濡?怨좎젙 ?댁긽?꾨줈 ?덉쟾??
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
    // ?≪쓬 ?쒓굅
    cv::morphologyEx(img, img, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
    cv::morphologyEx(img, img, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));
    box = {umin, umax, vmin, vmax, W, H};
}

// 諛붿씠?덈━ 而⑦닾?대뱾 以?媛?????뺤궗媛곹삎??minAreaRect 以묒떖 ?쎌???李얜뒗??
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
            continue; // ?뺤궗媛?洹쇱궗
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

// ?쎌? 以묒떖??UVBox 踰붿쐞瑜??듯빐 (U,V)濡??섏궛?섍퀬 3D ?됰㈃ ??醫뚰몴濡???닾?곹븳??
static cv::Vec3f uvCenterTo3D(const PlaneFit &pf, const UVBox &b, const cv::Point2f &pc)
{
    float U = b.umin + (pc.x / std::max(1, b.W - 1)) * (b.umax - b.umin);
    float V = b.vmin + (pc.y / std::max(1, b.H - 1)) * (b.vmax - b.vmin);
    return pf.mean + U * pf.u + V * pf.v;
}

// (id,center) 紐⑸줉??PLY(鍮④컙 ??? JSON?쇰줈 dir/stem.*????ν븳??
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
    const std::string statsPath = debugDir + "/" + stem + "_stats.json";

    if (!roiPts.empty())
        savePointCloud(roiPath, toDebugCloud(roiPts, 120, 180, 255));
    if (!innerPts.empty())
        savePointCloud(innerPath, toDebugCloud(innerPts, 255, 90, 90));

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

// minAreaRect ?꾨낫 以?10cm ?뺤궗媛??ш린/醫낇슒鍮?寃뚯씠?몃? ?듦낵???ш컖??以묒떖??諛섑솚?쒕떎.
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
        float wpx = std::max(r.size.width, r.size.height); // ?쎌?
        float hpx = std::min(r.size.width, r.size.height);
        if (hpx < 1.0f)
            continue;

        // ?쎌? ??UV ??誘명꽣 ?섏궛
        float width_m = (box.umax - box.umin) * (wpx / std::max(1, box.W - 1));
        float height_m = (box.vmax - box.vmin) * (hpx / std::max(1, box.H - 1));

        // 醫낇슒鍮??ъ씠利?寃뚯씠??
        float aspect = width_m / std::max(1e-6f, height_m);
        if (aspect < aspectMin || aspect > aspectMax)
            continue;

        float side = 0.5f * (width_m + height_m); // ?됯퇏 蹂 湲몄씠
        if (side < sideMin || side > sideMax)
            continue;

        // ?먯닔: 硫댁쟻(?쎌?) ?곗꽑
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

// C_prev 湲곕컲 ROI?먯꽌 intensity留뚯쑝濡??寃?以묒떖???ъ텛?뺥빐 targets3d_intensity.*濡???ν븳??
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

    const float diag_half = 0.707f * targetSize; // ?뺤궗媛??媛곸꽑/2
    const float margin_cm = 3.0f;
    const float roi_base_cm = std::max(0.75f * targetSize, diag_half + margin_cm);

    const std::string regDir = blastPath + "/Registration/" + targetType;
    std::filesystem::create_directories(regDir);
    const std::string debugDir = regDir + "/debug";
    std::filesystem::create_directories(debugDir);

    // 0) ?먭뎔 濡쒕뱶
    // auto pc = loadPc(blastPath + "/" + targetType + ".ply");
    if (points.size() < 200)
    {
        SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] too few points");
        return false;
    }

    // 1) 湲곗〈 ?쇳꽣 濡쒕뱶 (JSON > PLY)
    std::vector<std::pair<std::string, cv::Vec3f>> prev;
    bool hasPrev = loadPrevCenters_JSON(regDir + "/targets3d.json", prev) ||
                   loadPrevCenters_PLY(regDir + "/targets3d.ply", prev);

    // 2) ROI 異붿텧 (CartesianPointRGB 洹몃?濡?諛섑솚)
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

    // 3) ROI ???뺢퇋????inner(?쨟_low ?곸쓳) ???됰㈃(PCA) ???섏뒪??2mm, erosion) ??10cm 寃뚯씠??洹쇱젒????3D
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

        // (a) ROI intensity ?뺢퇋??(2??8% ?쇱꽱???+ ?곷떒 瑗щ━ p99 ?대━??
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
        float p99 = q(0.99f); // ?곸쐞 瑗щ━
        auto normI = [&](float raw)
        {
            float clamped = std::min(raw, p99); // ?뺢킅??瑗щ━ ?듭젣
            float v = (clamped - lo) / (hi - lo);
            return std::min(1.f, std::max(0.f, v));
        };
        SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] norm lo=%.4f hi=%.4f p99=%.4f", lo, hi, p99);

        // (b) ?곸쓳 ?꾧퀎 t_adapt (???⑤뒗 ?寃?蹂댄샇)
        float p35 = q(0.35f);
        float p35n = std::clamp((p35 - lo) / (hi - lo), 0.f, 1.f);
        float t_adapt = std::min(t_low, p35n + 0.02f);
        SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] t_low=%.3f p35n=%.3f -> t_adapt=%.3f", t_low, p35n, t_adapt);

        // (c) inner(?ш컖 ?대?) ?좏깮
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

        // (d) ?됰㈃(PCA)
        auto pf = fitPlanePCA(inner);
        if (!pf.ok)
            return std::nullopt;

        // (e) ?됰㈃ ?ъ쁺 ???섏뒪??2.0mm) + erosion(?댄빑)
        cv::Mat img;
        UVBox box;
        rasterizeBinary(inner, pf, img, box, 2.0f /*mm*/);
        cv::Mat core = img.clone();
        cv::erode(core, core, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}), cv::Point(-1, -1), 2); // 2??erosion

        // prev瑜?UV ?쎌? 醫뚰몴濡?(洹쇱젒???먯닔???ъ슜)
        auto worldToPix = [&](const cv::Vec3f &W)
        {
            cv::Vec3f d = W - pf.mean;
            float U = d.dot(pf.u), V = d.dot(pf.v);
            float x = (U - box.umin) / std::max(1e-6f, (box.umax - box.umin)) * (box.W - 1);
            float y = (V - box.vmin) / std::max(1e-6f, (box.vmax - box.vmin)) * (box.H - 1);
            return cv::Point2f(x, y);
        };
        cv::Point2f pixPrev = worldToPix(Cprev);

        // (f) 而⑦닾???ㅼ퐫??= 硫댁쟻(px) - 慣쨌嫄곕━(px) + 10cm ?ш린 寃뚯씠??
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

                // ?쎌? ??誘명꽣 ?섏궛
                float width_m = (box.umax - box.umin) * (wpx / std::max(1, box.W - 1));
                float height_m = (box.vmax - box.vmin) * (hpx / std::max(1, box.H - 1));

                // 醫낇슒鍮??꾪솕)
                float aspect = width_m / std::max(1e-6f, height_m);
                if (aspect < 0.80f || aspect > 1.25f)
                    continue;

                // ?섏텞 蹂댁젙: Aruco 寃????곸뿭 13cm 횞 0.77 ??10cm(Aruco ?덉そ ?곗깋)
                const float innerRatio = 0.77f;                              // ???꾩슂??0.74~0.82濡?誘몄꽭議곗젙
                const float expected_m = (targetSize * innerRatio) / 100.0f; // targetSize=13(寃? ?뺤궗媛?臾쇰━)
                const float tol_m = std::max(0.015f, 0.15f * expected_m);    // 짹1.5cm or 짹15% ?꾪솕
                const float side = 0.5f * (width_m + height_m);
                if (side < expected_m - tol_m || side > expected_m + tol_m)
                    continue;

                // ?먯닔: ??硫댁쟻 + C_prev??媛源뚯슱?섎줉 ?좊━
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

        // (g) ?ш컖 ?ㅽ뙣 ?? core???쇳듃濡쒖씠???대갚
        if (!gotRect)
        {
            cv::Moments mu = cv::moments(core, true);
            if (mu.m00 > 1e-3)
            {
                pixC = cv::Point2f((float)(mu.m10 / mu.m00), (float)(mu.m01 / mu.m00));
                gotRect = true;
            }
        }

        // (h) 理쒖쥌 3D 蹂듭썝 (or inner ?됯퇏 理쒖쥌 ?대갚)
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

    // 4) 硫붿씤 猷⑦봽: prev ?덉쑝硫?ROI 湲곕컲, ?놁쑝硫??꾩껜?먯꽌 1媛?
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

            // 諛섍꼍: 20 ??23 ??17 cm (癒쇱? ?됰꼮???ы븿 ??醫곹옒)
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

            // ?대룞??罹? 30mm ?댁긽?대㈃ ?낅뜲?댄듃 痍⑥냼
            if (okAny && delta_mm > 30.0f)
            {
                okAny = false;
                Cbest = Cprev;
                delta_mm = 0.0f;
            }

            // 蹂댁닔???낅뜲?댄듃: 1mm 誘몃쭔 蹂?붾뒗 臾댁떆
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
        // C_prev ?놁쑝硫? ?꾩껜 ?먭뎔?먯꽌 1媛쒕쭔 ?쒕룄
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

    // 5) ???諛?醫낅즺
    if (centersOut.empty())
    {
        SENDLOGF_TAG(LOG_TAG, "[IntensityOnly] no centers detected (keep previous if any)");
        return false;
    }

    // === [異붽?] facemarker(0踰? 湲곗??쇰줈 centers + points 異??뺣젹 ===
    if (usefacemarker)
    {
        cv::Point3f faceCenter{};
        bool hasFace = FindFaceMarkerCenterFromCenters(centersOut, faceCenter);

        if (hasFace)
        {
            // 0踰?facemarker 瑜?諛붾씪蹂대뒗 諛⑺뼢??-Z ?뺣㈃?쇰줈 留욎텛???뚯쟾?됰젹
            outFrontDir = computeFaceAlignmentMatrix(faceCenter);

            // (1) ?寃??쇳꽣??centersOut) ?뚯쟾
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
    SENDLOGF_TAG("ProcessTime", "Find Center And Merge PtCloud Sub - Compute Intensity Centers time: %lld ms (%lld珥?", ms, sec);

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

