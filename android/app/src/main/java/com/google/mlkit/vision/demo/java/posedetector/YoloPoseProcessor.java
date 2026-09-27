package com.google.mlkit.vision.demo.java.posedetector;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.TaskCompletionSource;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.ErrorLog;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.java.VisionProcessorBase;
import com.google.mlkit.vision.demo.java.jumprope.JumpCountGraphic;
import com.google.mlkit.vision.demo.java.jumprope.JumpRopeAdapter;
import com.google.mlkit.vision.demo.java.yolo.YoloDetector;
import com.google.mlkit.vision.demo.java.yolo.YoloGraphic;
import com.google.mlkit.vision.demo.java.yolo.YoloPoseDetector;
import com.google.mlkit.vision.demo.java.yolo.YoloPoseGraphic;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Our own skeleton engine driven by yolo11n-pose.onnx (Performance mode "YOLOv11
 * Pose", or when the native ML Kit model switch is OFF). Runs pose detection
 * asynchronously on a background thread; draws the 17-point COCO skeleton and,
 * when enabled, also feeds the jump rope counter and the YOLO object boxes.
 *
 * <p>Models are loaded lazily on the background thread (NOT in the constructor,
 * which runs on the UI thread) so starting the camera never freezes the screen.
 */
public class YoloPoseProcessor extends VisionProcessorBase<YoloPoseProcessor.Result> {

  private static final String TAG = "YoloPoseProcessor";

  /** Bundle of one frame of inference output. */
  public static class Result {
    public final List<YoloPoseDetector.Pose> poses;
    public final List<YoloDetector.Box> objects;

    Result(List<YoloPoseDetector.Pose> poses, List<YoloDetector.Box> objects) {
      this.poses = poses;
      this.objects = objects;
    }
  }

  private final boolean yoloMode;
  private final boolean jumperMode;
  private final Context context;

  private volatile YoloPoseDetector poseDetector; // lazy, loaded on background thread
  private volatile YoloDetector objectDetector;   // lazy, only in object-detection mode
  private final AtomicBoolean loadStarted = new AtomicBoolean(false);
  private final AtomicBoolean busy = new AtomicBoolean(false);
  private final String[] objLabels;
  private final JumpRopeAdapter ropeAdapter = new JumpRopeAdapter();
  private final ExecutorService detectorThread = Executors.newSingleThreadExecutor();
  private volatile boolean yoloFirstFailureShown;
  private long firstFrameLoggedAt;

  private enum Engine { POSE_AND_OBJECT, POSE_ONLY }
  private volatile Engine engine = Engine.POSE_AND_OBJECT;

  private final boolean drawPose;

  public YoloPoseProcessor(Context context, boolean yoloMode, boolean jumperMode) {
    this(context, yoloMode, jumperMode, true);
  }

  public YoloPoseProcessor(Context context, boolean yoloMode, boolean jumperMode,
      boolean drawPose) {
    super(context);
    this.context = context;
    this.yoloMode = yoloMode;
    this.jumperMode = jumperMode;
    this.drawPose = drawPose;
    engine = yoloMode ? Engine.POSE_AND_OBJECT : Engine.POSE_ONLY;
    objLabels = YoloDetector.loadLabels(context, "model/coco80_zh_en.txt");
  }

  /**
   * Loads the ONNX models on the detector thread exactly once. Returns false on
   * failure so the caller can hand an empty result back.
   */
  private boolean ensureLoaded() {
    if (poseDetector != null) return true;
    if (!loadStarted.compareAndSet(false, true)) {
      // Another thread is loading; just wait a little and re-check.
      for (int i = 0; i < 100 && poseDetector == null; i++) {
        try { Thread.sleep(30); } catch (InterruptedException ignore) {}
      }
      return poseDetector != null;
    }

    try {
      if (drawPose) {
        ErrorLog.i(TAG, "正在加载 yolo11n-pose.onnx(640px) ...");
        long t0 = System.currentTimeMillis();
        YoloPoseDetector pd = new YoloPoseDetector(context, "model/yolo11n-pose.onnx");
        poseDetector = pd;
        ErrorLog.i(TAG, "yolo11n-pose.onnx 加载成功 " + (System.currentTimeMillis() - t0) + "ms");
      }

      if (engine == Engine.POSE_AND_OBJECT) {
        ErrorLog.i(TAG, "正在加载 yolo11n.onnx(物体) ...");
        long t1 = System.currentTimeMillis();
        objectDetector = new YoloDetector(context, "model/yolo11n.onnx");
        ErrorLog.i(TAG, "yolo11n.onnx 加载成功 " + (System.currentTimeMillis() - t1) + "ms");
      }
      return true;
    } catch (Throwable t) {
      ErrorLog.e(TAG, "YOLO 模型加载失败", t);
      return false;
    }
  }

  @Override
  public void stop() {
    detectorThread.shutdown();
    super.stop();
    if (poseDetector != null) {
      try { poseDetector.close(); } catch (Exception ignore) {}
    }
    if (objectDetector != null) {
      try { objectDetector.close(); } catch (Exception ignore) {}
    }
  }

  @Override
  protected Task<Result> detectInImage(InputImage image) {
    final TaskCompletionSource<Result> tcs = new TaskCompletionSource<>();
    final Bitmap frame = latestFrameBitmap;
    if (frame == null) {
      // Only reachable if no camera bitmap is available (e.g. the rare capture
      // path races); do not spam errors - just skip this frame silently.
      tcs.setResult(new Result(new ArrayList<>(), new ArrayList<>()));
      return tcs.getTask();
    }
    // Drop this frame when the previous one is still running to avoid queue
    // backlog on slow devices (keeps the preview fluid).
    if (!busy.compareAndSet(false, true)) {
      tcs.setResult(new Result(new ArrayList<>(), new ArrayList<>()));
      return tcs.getTask();
    }
    detectorThread.execute(() -> {
      try {
        if (!ensureLoaded()) {
          tcs.setResult(new Result(new ArrayList<>(), new ArrayList<>()));
          return;
        }
        long t0 = System.currentTimeMillis();
        try {
          List<YoloPoseDetector.Pose> poses = drawPose
              ? poseDetector.detect(frame) : new ArrayList<>();
          List<YoloDetector.Box> objects;
          if (engine == Engine.POSE_AND_OBJECT && objectDetector != null) {
            objects = objectDetector.detect(frame);
          } else {
            objects = new ArrayList<>();
          }
          long dt = System.currentTimeMillis() - t0;
          if (dt >= 100 || firstFrameLoggedAt == 0) {
            if (firstFrameLoggedAt == 0) {
              firstFrameLoggedAt = System.currentTimeMillis();
              ErrorLog.i(TAG, String.format("首次推理完成: poses=%d objects=%d %dms",
                  poses.size(), objects.size(), dt));
            } else if (dt >= 100) {
              ErrorLog.i(TAG, String.format("推理: poses=%d objects=%d %dms",
                  poses.size(), objects.size(), dt));
            }
          }
          tcs.setResult(new Result(poses, objects));
        } catch (Throwable t) {
          if (!yoloFirstFailureShown) {
            yoloFirstFailureShown = true;
            ErrorLog.e(TAG, "YOLO 推理失败", t);
          }
          tcs.setResult(new Result(new ArrayList<>(), new ArrayList<>()));
        }
      } finally {
        busy.set(false);
      }
    });
    return tcs.getTask();
  }

  @Override
  protected void onSuccess(@NonNull Result result, @NonNull GraphicOverlay graphicOverlay) {
    if (jumperMode && !result.poses.isEmpty()) {
      YoloPoseDetector.Pose p = result.poses.get(0);
      int count = ropeAdapter.onPoseYolo(p.x, p.y, p.vis, graphicOverlay.getImageHeight());
      graphicOverlay.add(new JumpCountGraphic(graphicOverlay, count));
    }

    // Draw pose skeleton + object boxes.
    if (drawPose) {
      graphicOverlay.add(new YoloPoseGraphic(graphicOverlay, result.poses));
    }
    if (yoloMode && result.objects != null && !result.objects.isEmpty()) {
      graphicOverlay.add(new YoloGraphic(graphicOverlay, result.objects, objLabels));
    }
  }

  @Override
  protected void onFailure(@NonNull Exception e) {
    ErrorLog.e(TAG, "YOLO pose 检测失败", e);
  }
}