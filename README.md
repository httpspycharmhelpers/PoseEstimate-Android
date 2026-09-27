# PoseEstimate-Android

人体姿态估计 Android App —— 基于 [Google ML Kit Pose Detection Quickstart](https://github.com/googlesamples/mlkit)（经 [nevinbaiju/pose_estimation_android_app](https://github.com/nevinbaiju/pose_estimation_android_app) 二次开发），并整合了自研的 **YOLO11 姿态/目标检测引擎**（ONNX Runtime 与 **ncnn**）与 **跳绳计数** 功能。

## 功能特性

* **姿态检测**（传统滑动条/积分模式，ML Kit）— 实时检测人体位置，Fast / Accurate 两种模式
  * 原生模型开关：默认为 ML Kit 内置模型（`pose_39kp_lite/full.tflite`）
  * **YOLOv11 Pose 模式**：使用本项目自带的 `yolo11n-pose.onnx` 作为骨骼引擎（17 点 COCO 骨架 + 人物框）
  * **NCNN Pose 模式**：使用移植自 [nihui/ncnn-android-yolo11](https://github.com/nihui/ncnn-android-yolo11) 的 ncnn 引擎（Vulkan 加速），默认 `yolo11n-640` + GPU
* **YOLO Mode** — 用 `yolo11n.onnx` 在最新帧上异步做目标检测，叠加画框
* **跳绳 Mode** — 基于骨骼（肩/髋/踝关节）实时计数
* **报错系统** — 错误自动复制到剪贴板，顶部覆盖层 + 「终端日志」页实时查看
* **坐标平滑 / 姿态比对 / 动态着色 / 左右手对齐校验**（继承自上游 `code/` 模块）

## 目录结构

* `android/` — Android 应用工程（Java / ML Kit / ONNX Runtime / ncnn C++ JNI）
  * `android/app/src/main/jni/` — ncnn 引擎源码 + `ncnn_bridge.cpp` JNI 桥（Bitmap → 检测结果），预编译产物 `android/app/src/main/jniLibs/arm64-v8a/libncnn_yolo11.so`
  * `android/app/src/main/assets/*.ncnn.{param,bin}` — ncnn YOLO11 全系模型（n/s/m × det/seg/pose/cls/obb × 320/480/640）
* `code/` — 坐标平滑、姿态比对等算法模块与测试 notebook

## 构建

```bash
cd android
./gradlew :app:assembleDebug
```

产物：`android/app/build/outputs/apk/debug/pose-estimation-mod-debug.apk`

* **ONNX 版**（无 ncnn 模型）：普通构建，~104MB
* **NCNN 全量版**：构建时保留 `assets/*.ncnn.{param,bin}` 与 `jniLibs/arm64-v8a/libncnn_yolo11.so`，~411MB（含全部 15 组 ncnn 模型）
* 如需重新编译 `libncnn_yolo11.so`：使用 NDK r29 工具链手动编译（见 `android/app/src/main/jni/CMakeLists.txt`，依赖 `ncnn-20260526-android-vulkan` 与 `opencv-mobile-4.13.0-android`，可在 [nihui/ncnn-android-yolo11](https://github.com/nihui/ncnn-android-yolo11) 的 release 获取）

## 第三方库与模型许可（务必阅读）

| 组件 | 许可 |
| --- | --- |
| ML Kit (Google) | 免费 SDK，随 App 使用 |
| ONNX Runtime | MIT |
| ncnn / opencv-mobile（[nihui](https://github.com/nihui)） | BSD-3-Clause |
| `yolo11*.onnx` / `yolo11*.ncnn.bin` | **AGPL-3.0（Ultralytics）**。仅限个人/非商用使用；商用需向 Ultralytics 购买授权 |
| 应用代码（本仓库除模型外部分） | Apache 2.0 |

> ⚠️ 本仓库代码基于 `googlesamples/mlkit`（Copyright 2020 Google, Inc., Apache-2.0）与 `nevinbaiju/pose_estimation_android_app` 修改而来；ncnn 引擎移植自 `nihui/ncnn-android-yolo11`（BSD-3-Clause）。所有源码保留原始版权与许可声明。

## License

Copyright 2020 Google, Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

本仓库内 `android/app/src/main/assets/model/yolo11n*.onnx` 与 `android/app/src/main/assets/*.ncnn.{param,bin}` 模型权重为 Ultralytics 出品，遵循 AGPL-3.0。