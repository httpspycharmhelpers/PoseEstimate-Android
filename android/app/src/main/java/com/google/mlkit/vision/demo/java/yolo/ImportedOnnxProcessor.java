package com.google.mlkit.vision.demo.java.yolo;

import android.content.Context;
import android.graphics.Bitmap;
import androidx.annotation.NonNull;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.TaskCompletionSource;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.ErrorLog;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.java.ModelCatalog;
import com.google.mlkit.vision.demo.java.VisionProcessorBase;
import com.google.mlkit.vision.demo.java.jumprope.JumpCountGraphic;
import com.google.mlkit.vision.demo.java.jumprope.JumpRopeAdapter;

import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Runs a user-imported ONNX model (stored in files/onnx/) on each frame.
 * Det models use the generic YOLO-style box decoder; files whose name contains
 * "pose" additionally decode the 17 COCO keypoints and draw the skeleton.
 */
public class ImportedOnnxProcessor extends VisionProcessorBase<ImportedOnnxProcessor.Result> {

  private static final String TAG = "ImportedOnnx";

  /** Bundle of one frame of output. */
  public static class Result {
    public final List<YoloDetector.Box> objects;
    public final List<YoloPoseDetector.Pose> poses;
    Result(List<YoloDetector.Box> objects, List<YoloPoseDetector.Pose> poses) {
      this.objects = objects;
      this.poses = poses;
    }
  }

  private final Context context;
  private final String fileName;
  private final boolean jumperMode;
  private final boolean isPoseModel;
  private final String[] objLabels;

  private volatile YoloDetector detector;   // boxes (always attempted)
  private volatile YoloPoseDetector poseDet; // only when isPoseModel
  private final AtomicBoolean loadStarted = new AtomicBoolean(false);
  private final JumpRopeAdapter ropeAdapter = new JumpRopeAdapter();
  private final ExecutorService detectorThread = Executors.newSingleThreadExecutor();
  private volatile boolean firstFailureShown;
  private long firstFrameLoggedAt;

  public ImportedOnnxProcessor(Context context, String fileName, boolean jumperMode) {
    super(context);
    this.context = context;
    this.fileName = fileName;
    this.jumperMode = jumperMode;
    this.isPoseModel = fileName.toLowerCase().contains("pose") ||
        fileName.toLowerCase().contains("yolo11n-pose");
    objLabels = YoloDetector.loadLabels(context, "model/coco80_zh_en.txt");
  }

  private boolean ensureLoaded() {
    if (detector != null || (isPoseModel && poseDet != null)) return true;
    if (!loadStarted.compareAndSet(false, true)) {
      for (int i = 0; i < 100 && detector == null; i++) {
        try { Thread.sleep(30); } catch (InterruptedException ignore) {}
      }
      return detector != null;
    }
    try {
      File file = new File(ModelCatalog.importDir(context), fileName);
      ErrorLog.i(TAG, "正在加载导入模型 " + fileName + " ...");
      long t0 = System.currentTimeMillis();
      if (isPoseModel) {
        poseDet = new YoloPoseDetector(file);
        // Also load the generic det path so boxes draw too.
        detector = poseDet != null ? null : new YoloDetector(file);
      } else {
        detector = new YoloDetector(file);
      }
      ErrorLog.i(TAG, fileName + " 加载成功 " + (System.currentTimeMillis() - t0) + "ms");
      return true;
    } catch (Throwable t) {
      ErrorLog.e(TAG, "导入模型加载失败 " + fileName, t);
      return false;
    }
  }

  @Override
  public void stop() {
    detectorThread.shutdown();
    super.stop();
    if (detector != null) try { detector.close(); } catch (Exception ignore) {}
    if (poseDet != null) try { poseDet.close(); } catch (Exception ignore) {}
  }

  @Override
  protected Task<Result> detectInImage(InputImage image) {
    final TaskCompletionSource<Result> tcs = new TaskCompletionSource<>();
    final Bitmap frame = latestFrameBitmap;
    if (frame == null) {
      tcs.setResult(new Result(new java.util.ArrayList<>(), new java.util.ArrayList<>()));
      return tcs.getTask();
    }
    detectorThread.execute(() -> {
      if (!ensureLoaded()) {
        tcs.setResult(new Result(new java.util.ArrayList<>(), new java.util.ArrayList<>()));
        return;
      }
      long t0 = System.currentTimeMillis();
      try {
        List<YoloPoseDetector.Pose> poses = new java.util.ArrayList<>();
        if (isPoseModel && poseDet != null) poses = poseDet.detect(frame);
        List<YoloDetector.Box> objects = new java.util.ArrayList<>();
        if (detector != null) objects = detector.detect(frame);
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
        tcs.setResult(new Result(objects, poses));
      } catch (Throwable t) {
        if (!firstFailureShown) {
          firstFailureShown = true;
          ErrorLog.e(TAG, "导入模型推理失败", t);
        }
        tcs.setResult(new Result(new java.util.ArrayList<>(), new java.util.ArrayList<>()));
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
    if (!result.poses.isEmpty()) {
      graphicOverlay.add(new YoloPoseGraphic(graphicOverlay, result.poses));
    }
    if (result.objects != null && !result.objects.isEmpty()) {
      graphicOverlay.add(new YoloGraphic(graphicOverlay, result.objects, objLabels));
    }
  }

  @Override
  protected void onFailure(@NonNull Exception e) {
    ErrorLog.e(TAG, "导入模型检测失败", e);
  }
}