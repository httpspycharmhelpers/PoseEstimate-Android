/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.preference;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build.VERSION_CODES;
import android.preference.PreferenceManager;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.annotation.StringRes;
import com.google.android.gms.common.images.Size;
import com.google.common.base.Preconditions;
import com.google.mlkit.vision.demo.CameraSource;
import com.google.mlkit.vision.demo.CameraSource.SizePair;
import com.google.mlkit.vision.demo.R;
import com.google.mlkit.vision.pose.PoseDetectorOptionsBase;
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions;
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions;

/** Utility class to retrieve shared preferences. */
public class PreferenceUtils {

  private static final int POSE_DETECTOR_PERFORMANCE_MODE_FAST = 1;

  static void saveString(Context context, @StringRes int prefKeyId, @Nullable String value) {
    PreferenceManager.getDefaultSharedPreferences(context)
        .edit()
        .putString(context.getString(prefKeyId), value)
        .apply();
  }

  @Nullable
  public static SizePair getCameraPreviewSizePair(Context context, int cameraId) {
    Preconditions.checkArgument(
        cameraId == CameraSource.CAMERA_FACING_BACK
            || cameraId == CameraSource.CAMERA_FACING_FRONT);
    String previewSizePrefKey;
    String pictureSizePrefKey;
    if (cameraId == CameraSource.CAMERA_FACING_BACK) {
      previewSizePrefKey = context.getString(R.string.pref_key_rear_camera_preview_size);
      pictureSizePrefKey = context.getString(R.string.pref_key_rear_camera_picture_size);
    } else {
      previewSizePrefKey = context.getString(R.string.pref_key_front_camera_preview_size);
      pictureSizePrefKey = context.getString(R.string.pref_key_front_camera_picture_size);
    }

    try {
      SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
      return new SizePair(
          Size.parseSize(sharedPreferences.getString(previewSizePrefKey, null)),
          Size.parseSize(sharedPreferences.getString(pictureSizePrefKey, null)));
    } catch (Exception e) {
      return null;
    }
  }

  @RequiresApi(VERSION_CODES.LOLLIPOP)
  @Nullable
  public static android.util.Size getCameraXTargetResolution(Context context) {
    String prefKey = context.getString(R.string.pref_key_camerax_target_resolution);
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    try {
      return android.util.Size.parseSize(sharedPreferences.getString(prefKey, null));
    } catch (Exception e) {
      return null;
    }
  }

  public static PoseDetectorOptionsBase getPoseDetectorOptionsForLivePreview(Context context) {
    int performanceMode =
        getModeTypePreferenceValue(
            context,
            R.string.pref_key_live_preview_pose_detection_performance_mode,
            POSE_DETECTOR_PERFORMANCE_MODE_FAST);
    if (performanceMode == POSE_DETECTOR_PERFORMANCE_MODE_FAST) {
      return new PoseDetectorOptions.Builder()
        .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
        .build();
    } else {
      return new AccuratePoseDetectorOptions.Builder()
        .setDetectorMode(AccuratePoseDetectorOptions.STREAM_MODE)
        .build();
    }
  }

  /** Our modification: is the native (ML Kit) skeleton model enabled? Default true. */
  public static boolean isNativePoseModelEnabled(Context context) {
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    String prefKey = context.getString(R.string.pref_key_pose_detector_native_model);
    return sharedPreferences.getBoolean(prefKey, true);
  }

  /** Our modification (sg.md): Pose mode switch. Off => run no pose engine at all. */
  public static boolean isPoseModeEnabled(Context context) {
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    String prefKey = context.getString(R.string.pref_key_pose_mode_switch);
    return sharedPreferences.getBoolean(prefKey, true);
  }

  /** Our modification (sg.md): YOLOv11 Mode / ONNX Model. 0=YOLOv11n (det), 1=YOLOv11 Pose. */
  public static int getYolo11OnnxModel(Context context) {
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    String prefKey = context.getString(R.string.pref_key_yolo11_onnx_model);
    return Integer.parseInt(sharedPreferences.getString(prefKey, "0"));
  }

  /**
   * Our modification: which skeleton engine should we run?
   * Returns 0 = off (Pose mode disabled), 1 = native ML Kit, 2 = YOLOv11 pose,
   * 4 = ncnn pose. YOLOv11/ncnn win when the performance mode is "3"/"4" OR
   * the native model is turned off (falls back to our engine).
   */
  public static int getLivePreviewPoseEngine(Context context) {
    if (!isPoseModeEnabled(context)) {
      return 0; // sg.md: Pose mode 关闭后没有 Pose 模型
    }
    int performanceMode =
        getModeTypePreferenceValue(
            context,
            R.string.pref_key_live_preview_pose_detection_performance_mode,
            POSE_DETECTOR_PERFORMANCE_MODE_FAST);
    final int ENGINE_NATIVE = 1;
    final int ENGINE_YOLO = 2;
    final int ENGINE_NCNN = 4;
    if (performanceMode == 3) {
      return ENGINE_YOLO;
    }
    if (performanceMode == 4) {
      return ENGINE_NCNN;
    }
    if (isNativePoseModelEnabled(context)) {
      return ENGINE_NATIVE;
    }
    return ENGINE_YOLO;
  }

  public static PoseDetectorOptionsBase getPoseDetectorOptionsForStillImage(Context context) {
    int performanceMode =
        getModeTypePreferenceValue(
            context,
            R.string.pref_key_still_image_pose_detection_performance_mode,
            POSE_DETECTOR_PERFORMANCE_MODE_FAST);
    if (performanceMode == POSE_DETECTOR_PERFORMANCE_MODE_FAST) {
      return new PoseDetectorOptions.Builder()
        .setDetectorMode(PoseDetectorOptions.SINGLE_IMAGE_MODE)
        .build();
    } else {
      return new AccuratePoseDetectorOptions.Builder()
        .setDetectorMode(AccuratePoseDetectorOptions.SINGLE_IMAGE_MODE)
        .build();
    }
  }

  public static boolean shouldShowPoseDetectionInFrameLikelihoodLivePreview(Context context) {
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    String prefKey =
        context.getString(R.string.pref_key_live_preview_pose_detector_show_in_frame_likelihood);
    return sharedPreferences.getBoolean(prefKey, false);
  }

  public static boolean shouldShowPoseDetectionInFrameLikelihoodStillImage(Context context) {
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    String prefKey =
        context.getString(R.string.pref_key_still_image_pose_detector_show_in_frame_likelihood);
    return sharedPreferences.getBoolean(prefKey, false);
  }

  /**
   * Mode type preference is backed by {@link android.preference.ListPreference} which only support
   * storing its entry value as string type, so we need to retrieve as string and then convert to
   * integer.
   */
  private static int getModeTypePreferenceValue(
      Context context, @StringRes int prefKeyResId, int defaultValue) {
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    String prefKey = context.getString(prefKeyResId);
    return Integer.parseInt(sharedPreferences.getString(prefKey, String.valueOf(defaultValue)));
  }

  public static boolean isCameraLiveViewportEnabled(Context context) {
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    String prefKey = context.getString(R.string.pref_key_camera_live_viewport);
    return sharedPreferences.getBoolean(prefKey, false);
  }

  public static boolean isJumpRopeModeEnabled(Context context) {
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    String prefKey = context.getString(R.string.pref_key_jump_rope_mode);
    return sharedPreferences.getBoolean(prefKey, false);
  }

  public static boolean isYoloModeEnabled(Context context) {
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    String prefKey = context.getString(R.string.pref_key_yolo_mode);
    return sharedPreferences.getBoolean(prefKey, false);
  }

  /** Our modification: NCNN task id (0=COCO,1=SEG,2=Pose,3=CLS,4=OBB). Default Pose. */
  public static int getNcnnTask(Context context) {
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    String prefKey = context.getString(R.string.pref_key_ncnn_task);
    return Integer.parseInt(sharedPreferences.getString(prefKey, "2"));
  }

  /** Our modification: NCNN model size index (0..8 = n/s/m x 320/480/640). Default n-640. */
  public static int getNcnnSize(Context context) {
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    String prefKey = context.getString(R.string.pref_key_ncnn_size);
    return Integer.parseInt(sharedPreferences.getString(prefKey, "6"));
  }

  /** Our modification: NCNN compute (0=CPU,1=GPU,2=GPU/Turnip). Default CPU (most stable). */
  public static int getNcnnCpuGpu(Context context) {
    SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
    String prefKey = context.getString(R.string.pref_key_ncnn_cpugpu);
    return Integer.parseInt(sharedPreferences.getString(prefKey, "0"));
  }

  private PreferenceUtils() {}
}
