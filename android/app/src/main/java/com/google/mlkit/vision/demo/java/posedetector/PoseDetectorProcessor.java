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

package com.google.mlkit.vision.demo.java.posedetector;

import android.content.Context;
import androidx.annotation.NonNull;
import android.graphics.Bitmap;
import android.util.Log;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.ErrorLog;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.java.VisionProcessorBase;
import com.google.mlkit.vision.demo.java.jumprope.JumpCountGraphic;
import com.google.mlkit.vision.demo.java.jumprope.JumpRopeAdapter;
import com.google.mlkit.vision.demo.java.yolo.YoloDetector;
import com.google.mlkit.vision.demo.java.yolo.YoloGraphic;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseDetection;
import com.google.mlkit.vision.pose.PoseDetector;
import com.google.mlkit.vision.pose.PoseDetectorOptionsBase;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A processor to run pose detector.
 *
 * Added modes (our modification):
 *  - YOLO Mode: runs our own yolo11n.onnx (ONNX Runtime) asynchronously on the
 *    latest frame bitmap and draws object boxes alongside the skeleton.
 *  - 跳绳 Mode: feeds the ML Kit skeleton into our jump-rope counter.
 */
public class PoseDetectorProcessor extends VisionProcessorBase<Pose> {

  private static final String TAG = "PoseDetectorProcessor";

  private final PoseDetector detector;
  private final boolean showInFrameLikelihood;
  private final boolean yoloMode;
  private final boolean jumperMode;
  private final Context context;

  private volatile YoloDetector yolo; // lazy, loaded on the background thread
  private final java.util.concurrent.atomic.AtomicBoolean yoloLoadStarted =
      new java.util.concurrent.atomic.AtomicBoolean(false);
  private final String[] yoloLabels;
  private final AtomicReference<List<YoloDetector.Box>> yoloBoxes = new AtomicReference<>();
  private final JumpRopeAdapter ropeAdapter = new JumpRopeAdapter();
  private final ExecutorService dirtyWork = Executors.newSingleThreadExecutor();
  private boolean yoloFirstFailureShown;
  private boolean started;

  public PoseDetectorProcessor(
      Context context, PoseDetectorOptionsBase options, boolean showInFrameLikelihood,
      boolean yoloMode, boolean jumperMode) {
    super(context);
    this.context = context;
    this.showInFrameLikelihood = showInFrameLikelihood;
    this.yoloMode = yoloMode;
    this.jumperMode = jumperMode;
    detector = PoseDetection.getClient(options);
    yoloLabels = yoloMode
        ? YoloDetector.loadLabels(context, "model/coco80_zh_en.txt") : null;
  }

  /** Loads yolo11n.onnx once, on the caller's (background) thread. */
  private boolean ensureYoloLoaded() {
    if (yolo != null) return true;
    if (!yoloLoadStarted.compareAndSet(false, true)) {
      for (int i = 0; i < 100 && yolo == null; i++) {
        try { Thread.sleep(30); } catch (InterruptedException ignore) {}
      }
      return yolo != null;
    }
    try {
      ErrorLog.i(TAG, "正在加载 yolo11n.onnx(物体) ...");
      long t0 = System.currentTimeMillis();
      yolo = new YoloDetector(context, "model/yolo11n.onnx");
      ErrorLog.i(TAG, "yolo11n.onnx 加载成功 " + (System.currentTimeMillis() - t0) + "ms");
      return true;
    } catch (Throwable t) {
      ErrorLog.e(TAG, "yolo11n.onnx 加载失败", t);
      return false;
    }
  }

  @Override
  public void stop() {
    super.stop();
    dirtyWork.shutdown();
    detector.close();
    if (yolo != null) {
      try { yolo.close(); } catch (Exception ignore) {}
    }
  }

  @Override
  protected Task<Pose> detectInImage(InputImage image) {
    return detector.process(image);
  }

  @Override
  protected void onSuccess(@NonNull Pose pose, @NonNull GraphicOverlay graphicOverlay) {
    if (yoloMode) {
      if (yolo != null && !started) {
        started = true;
        Log.i(TAG, "YOLO mode enabled, model loaded");
      }
      // Kick off async detection on the latest frame if we haven't got a result yet.
      final Bitmap frame = latestFrameBitmap;
      dirtyWork.execute(() -> {
        if (frame == null) {
          // No camera bitmap available (e.g. the rare capture path race);
          // skip silently instead of spamming errors.
          return;
        }
        if (!ensureYoloLoaded()) return;
        long t0 = System.currentTimeMillis();
        try {
          List<YoloDetector.Box> boxes = yolo.detect(frame);
          yoloBoxes.set(boxes);
          long dt = System.currentTimeMillis() - t0;
          if (dt >= 100) {
            com.google.mlkit.vision.demo.ErrorLog.i(TAG,
                "物体检测: count=" + boxes.size() + " " + dt + "ms");
          }
        } catch (Throwable t) {
          if (!yoloFirstFailureShown) {
            yoloFirstFailureShown = true;
            com.google.mlkit.vision.demo.ErrorLog.e(TAG, "YOLO 推理失败", t);
          }
        }
      });
    }

    if (jumperMode) {
      int h = graphicOverlay.getImageHeight();
      int count = ropeAdapter.onPose(pose, h);
      graphicOverlay.add(new JumpCountGraphic(graphicOverlay, count));
    }

    graphicOverlay.add(new PoseGraphic(graphicOverlay, pose, showInFrameLikelihood));

    if (yoloMode && yolo != null) {
      List<YoloDetector.Box> boxes = yoloBoxes.get();
      if (boxes != null && !boxes.isEmpty()) {
        graphicOverlay.add(new YoloGraphic(graphicOverlay, boxes, yoloLabels));
      }
    }
  }

  @Override
  protected void onFailure(@NonNull Exception e) {
    com.google.mlkit.vision.demo.ErrorLog.e(TAG, "ML Kit 姿态检测失败", e);
  }
}