# Detection Module

`Detection` 폴더 안에서 타겟 탐지 파이프라인을 독립적으로 실행할 수 있도록 정리한 구성입니다.

파이프라인 단계:

1. `DetectTarget/<targetType>/orign_*.jpg`에서 ArUco 타겟 중심 검출
2. `calib_*.json`과 `<targetType>.ply`를 이용한 3D 라이다 좌표 변환
3. intensity 기반 중심 보정과 디버그 산출물 생성

현재 샘플셋 기준 폴더 구조:

```text
Detection/
  sample_set/
    reference.ply
    DetectTarget/reference/
      orign_3_RB.jpg
      calib_3_RB.json
      detect_3_RB.json
      target_marked_3_RB.jpg
    Registration/reference/
      targets3d.json
      targets3d_adjust.json
      debug/
```

빌드:

```powershell
cmake -S E:\SAMS-TAP\Detection -B E:\SAMS-TAP\Detection\build
cmake --build E:\SAMS-TAP\Detection\build --config Release
```

OpenCV CMake 패키지를 찾지 못하는 경우:

```powershell
cmake -S E:\SAMS-TAP\Detection -B E:\SAMS-TAP\Detection\build `
  -DOpenCV_DIR=C:\path\to\opencv\build `
  -DDETECTION_OPENCV_INCLUDE_DIRS=E:\SAMS-TAP\Detection `
  -DDETECTION_OPENCV_LIBRARIES="opencv_world4100.lib"
```

실행 예시:

```powershell
E:\SAMS-TAP\Detection\build\Release\DetectionRunner.exe all E:\SAMS-TAP\Detection\sample_set reference --use-face-marker --target-size 13
```

단계별 실행:

```powershell
DetectionRunner.exe detect  E:\SAMS-TAP\Detection\sample_set reference
DetectionRunner.exe merge   E:\SAMS-TAP\Detection\sample_set reference
DetectionRunner.exe adjust  E:\SAMS-TAP\Detection\sample_set reference --use-face-marker --target-size 13
DetectionRunner.exe summary E:\SAMS-TAP\Detection\sample_set reference
```

참고:

- OpenCV는 링크 가능한 개발 패키지가 필요합니다. 필요하면 `OpenCV_DIR`을 지정해서 CMake가 찾을 수 있게 해주세요.
- `OpenCVConfig.cmake`가 없으면 `DETECTION_OPENCV_INCLUDE_DIRS`, `DETECTION_OPENCV_LIBRARIES`로 직접 지정할 수 있습니다.
- intensity 보정 단계는 `Registration/<targetType>/debug` 아래에 raster/core 이미지와 ROI/inner PLY, 통계 JSON을 남기도록 구성했습니다.
