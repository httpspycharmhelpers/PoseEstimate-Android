# PoseEstimate-Android

人体姿态估计 Android App —— 基于 [Google ML Kit Pose Detection Quickstart](https://github.com/googlesamples/mlkit)（经 [nevinbaiju/pose_estimation_android_app](https://github.com/nevinbaiju/pose_estimation_android_app) 二次开发），并整合了自研的 **YOLO11 姿态/目标检测引擎**（ONNX Runtime）与 **跳绳计数** 功能。

## 功能特性

* **姿态检测**（传统滑动条/积分模式，ML Kit）— 实时检测人体位置，Fast / Accurate 两种模式
  * 原生模型开关：默认为 ML Kit 内置模型（`pose_39kp_lite/full.tflite`）
  * **YOLOv11 Pose 模式**：使用本项目自带的 `yolo11n-pose.onnx` 作为骨骼引擎（17 点 COCO 骨架 + 人物框）
* **YOLO Mode** — 用 `yolo11n.onnx` 在最新帧上异步做目标检测，叠加画框
* **跳绳 Mode** — 基于骨骼（肩/髋/踝关节）实时计数
* **报错系统** — 错误自动复制到剪贴板，顶部覆盖层 + 「终端日志」页实时查看
* **坐标平滑 / 姿态比对 / 动态着色 / 左右手对齐校验**（继承自上游 `code/` 模块）

## 目录结构

* `android/` — Android 应用工程（Java / ML Kit / ONNX Runtime）
* `code/` — 坐标平滑、姿态比对等算法模块与测试 notebook

## 构建

```bash
cd android
./gradlew :app:assembleDebug
```

产物：`android/app/build/outputs/apk/debug/pose-estimation-mod-debug.apk`
（当前仅打包 `arm64-v8a` 的 ONNX Runtime 使 APK 保持在 ~104MB）

## 第三方库与模型许可（务必阅读）

| 组件 | 许可 |
| --- | --- |
| ML Kit (Google) | 免费 SDK，随 App 使用 |
| ONNX Runtime | MIT |
| `yolo11n.onnx` / `yolo11n-pose.onnx` | **AGPL-3.0（Ultralytics）**。仅限个人/非商用使用；商用需向 Ultralytics 购买授权 |
| 应用代码（本仓库除模型外部分） | Apache 2.0 |

> ⚠️ 本仓库代码基于 `googlesamples/mlkit`（Copyright 2020 Google, Inc., Apache-2.0）与 `nevinbaiju/pose_estimation_android_app` 修改而来，所有源码保留原始版权与许可声明。

## License

Copyright 2020 Google, Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

本仓库内 `android/app/src/main/assets/model/yolo11n*.onnx` 模型权重为 Ultralytics 出品，遵循 AGPL-3.0。