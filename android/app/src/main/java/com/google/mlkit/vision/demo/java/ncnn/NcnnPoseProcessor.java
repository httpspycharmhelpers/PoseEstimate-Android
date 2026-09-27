package com.google.mlkit.vision.demo.java.ncnn;

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

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * General NCNN YOLO11 engine (ported from nihui/ncnn-android-yolo11).
 * Runs any of det/seg/pose/cls/obb tasks asynchronously on a background thread
 * and draws boxes / skeletons / rotated rects. When the task is Pose and the
 * jump rope mode is on it also feeds the counter.
 *
 * <p>The ncnn model is loaded lazily on the detector thread (NOT in the
 * constructor which runs on the UI thread). The loaded model is re-created
 * automatically whenever task/size/cpugpu selections change.
 */
public class NcnnPoseProcessor extends VisionProcessorBase<NcnnPoseProcessor.Result> {

  private static final String TAG = "NcnnPoseProcessor";
  private static final String[] TASK_NAMES = {"COCO", "SEG", "Pose", "CLS", "OBB"};

  /** Bundle of one frame of ncnn inference output. */
  public static class Result {
    public final NcnnYolo11.Result[] items;
    public final int task;

    Result(NcnnYolo11.Result[] items, int task) {
      this.items = items;
      this.task = task;
    }
  }

  private final Context context;
  private final int taskId;   // 0..4
  private final int modelId;  // 0..8 -> n/s/m x 320/480/640
  private final int cpugpu;   // 0=CPU 1=GPU 2=GPU(turnip)
  private final boolean jumperMode;
  private final String[] cocoLabels;

  private final AtomicBoolean loadStarted = new AtomicBoolean(false);
  private final JumpRopeAdapter ropeAdapter = new JumpRopeAdapter();
  private final ExecutorService detectorThread = Executors.newSingleThreadExecutor();
  private volatile boolean firstFailureShown;
  private long firstFrameLoggedAt;

  /** Single shared native engine for the app lifetime; re-created on selection change. */
  private static NcnnYolo11 engine;
  private static volatile boolean engineReady;
  private static volatile int engineTask = -1, engineModelId = -1, engineCpuGpu = -1;

  public NcnnPoseProcessor(Context context, boolean jumperMode, int taskId, int modelId, int cpugpu) {
    super(context);
    this.context = context;
    this.jumperMode = jumperMode;
    this.taskId = taskId;
    this.modelId = modelId;
    this.cpugpu = cpugpu;
    cocoLabels = YoloDetector.loadLabels(context, "model/coco80_zh_en.txt");
  }

  /** Loads the ncnn model on the detector thread. Reuses engine when selections match. */
  private boolean ensureLoaded() {
    if (engineReady && engineTask == taskId && engineModelId == modelId && engineCpuGpu == cpugpu) {
      return true;
    }
    if (!loadStarted.compareAndSet(false, true)) {
      for (int i = 0; i < 200 && !engineReady; i++) {
        try { Thread.sleep(30); } catch (InterruptedException ignore) {}
      }
      return engineReady;
    }

    try {
      ErrorLog.i(TAG, "正在加载 ncnn 模型(" + TASK_NAMES[taskId]
          + " size=" + modelId + " cpugpu=" + cpugpu + ") ...");
      long t0 = System.currentTimeMillis();
      NcnnYolo11 e = new NcnnYolo11();
      if (!e.loadModel(context.getAssets(), taskId, modelId, cpugpu)) {
        ErrorLog.e(TAG, "ncnn 模型加载失败(" + TASK_NAMES[taskId] + " size=" + modelId + " cpugpu=" + cpugpu + ")");
        return false;
      }
      engine = e;
      engineReady = true;
      engineTask = taskId;
      engineModelId = modelId;
      engineCpuGpu = cpugpu;
      ErrorLog.i(TAG, "ncnn 模型加载成功 " + (System.currentTimeMillis() - t0) + "ms");
      return true;
    } catch (Throwable t) {
      ErrorLog.e(TAG, "ncnn 模型加载失败", t);
      return false;
    }
  }

  @Override
  public void stop() {
    detectorThread.shutdown();
    super.stop();
  }

  @Override
  protected Task<Result> detectInImage(InputImage image) {
    final TaskCompletionSource<Result> tcs = new TaskCompletionSource<>();
    final Bitmap frame = latestFrameBitmap;
    if (frame == null) {
      tcs.setResult(new Result(new NcnnYolo11.Result[0], taskId));
      return tcs.getTask();
    }
    final int fw = frame.getWidth();
    final int fh = frame.getHeight();
    final int[] argb = new int[fw * fh];

    detectorThread.execute(() -> {
      if (!ensureLoaded()) {
        tcs.setResult(new Result(new NcnnYolo11.Result[0], taskId));
        return;
      }
      long t0 = System.currentTimeMillis();
      try {
        frame.getPixels(argb, 0, fw, 0, 0, fw, fh);
        float[] flat = engine.detect(argb, fw, fh);
        NcnnYolo11.Result[] items = NcnnYolo11.parse(flat);
        long dt = System.currentTimeMillis() - t0;
        if (dt >= 100 || firstFrameLoggedAt == 0) {
          if (firstFrameLoggedAt == 0) {
            firstFrameLoggedAt = System.currentTimeMillis();
            ErrorLog.i(TAG, String.format("ncnn 首次推理完成: %s %d个 %dms",
                TASK_NAMES[taskId], items.length, dt));
          } else if (dt >= 100) {
            ErrorLog.i(TAG, String.format("ncnn 推理: %s %d个 %dms", TASK_NAMES[taskId], items.length, dt));
          }
        }
        tcs.setResult(new Result(items, taskId));
      } catch (Throwable t) {
        if (!firstFailureShown) {
          firstFailureShown = true;
          ErrorLog.e(TAG, "ncnn 推理失败", t);
        }
        tcs.setResult(new Result(new NcnnYolo11.Result[0], taskId));
      }
    });
    return tcs.getTask();
  }

  @Override
  protected void onSuccess(@NonNull Result result, @NonNull GraphicOverlay graphicOverlay) {
    int iw = graphicOverlay.getImageWidth();
    int ih = graphicOverlay.getImageHeight();

    // Jump rope counter only makes sense for Pose task.
    if (jumperMode && result.task == NcnnYolo11.TASK_POSE && result.items.length > 0 && ih > 0) {
      NcnnYolo11.Result p = result.items[0];
      if (p.nkpts >= 17) {
        float[] x = new float[17], y = new float[17], vis = new float[17];
        for (int k = 0; k < 17; k++) {
          x[k] = p.kpts[k * 3] / (iw > 0 ? iw : 1);
          y[k] = p.kpts[k * 3 + 1] / ih;
          vis[k] = p.kpts[k * 3 + 2];
        }
        int count = ropeAdapter.onPoseYolo(x, y, vis, ih);
        graphicOverlay.add(new JumpCountGraphic(graphicOverlay, count));
      }
    }

    String[] labels = (result.task == NcnnYolo11.TASK_CLS) ? null : cocoLabels;
    graphicOverlay.add(new NcnnGraphic(graphicOverlay, result.items, result.task, labels));
  }

  @Override
  protected void onFailure(@NonNull Exception e) {
    ErrorLog.e(TAG, "ncnn 检测失败", e);
  }
}