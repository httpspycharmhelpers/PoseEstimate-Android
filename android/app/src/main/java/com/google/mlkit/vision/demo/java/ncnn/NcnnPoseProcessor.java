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

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Pose engine driven by the ncnn YOLO11 model (self-optimized engine ported from
 * nihui/ncnn-android-yolo11). Runs the COCO pose task asynchronously on a
 * background thread and draws the 17-point skeleton plus the person box; when
 * enabled it also feeds the jump rope counter.
 *
 * <p>The ncnn model is loaded lazily on the detector thread (NOT in the
 * constructor which runs on the UI thread), mirroring YoloPoseProcessor.
 */
public class NcnnPoseProcessor extends VisionProcessorBase<NcnnPoseProcessor.Result> {

  private static final String TAG = "NcnnPoseProcessor";

  /** Convenience forwards to NcnnYolo11. @see NcnnYolo11#MODEL_N640 */
  public static final int MODEL_N640 = NcnnYolo11.MODEL_N640;
  public static final int CPUGPU_GPU = NcnnYolo11.CPUGPU_GPU;

  /** Bundle of one frame of ncnn inference output. */
  public static class Result {
    public final NcnnYolo11.Result[] items;

    Result(NcnnYolo11.Result[] items) {
      this.items = items;
    }
  }

  private final boolean jumperMode;
  private final Context context;
  private final int modelId;   // 0..8 -> n/s/m x 320/480/640
  private final int cpugpu;    // CPU / GPU / GPU(turnip)

  private volatile boolean loaded;
  private final AtomicBoolean loadStarted = new AtomicBoolean(false);
  private final JumpRopeAdapter ropeAdapter = new JumpRopeAdapter();
  private final ExecutorService detectorThread = Executors.newSingleThreadExecutor();
  private volatile boolean firstFailureShown;
  private long firstFrameLoggedAt;

  /** Single shared native engine for the app lifetime (pose task). */
  private static NcnnYolo11 engine;
  private static volatile boolean engineReady;

  public NcnnPoseProcessor(Context context, boolean jumperMode, int modelId, int cpugpu) {
    super(context);
    this.context = context;
    this.jumperMode = jumperMode;
    this.modelId = modelId;
    this.cpugpu = cpugpu;
  }

  /** Loads the ncnn pose model on the detector thread exactly once. */
  private boolean ensureLoaded() {
    if (engineReady) return true;
    if (!loadStarted.compareAndSet(false, true)) {
      for (int i = 0; i < 100 && !engineReady; i++) {
        try { Thread.sleep(30); } catch (InterruptedException ignore) {}
      }
      return engineReady;
    }

    try {
      // ncnn loadModel must be called on the thread that owns the object; keep
      // it on detectorThread (we never use it from the UI thread directly).
      ErrorLog.i(TAG, "正在加载 ncnn pose 模型(modelId=" + modelId + ", cpugpu=" + cpugpu + ") ...");
      long t0 = System.currentTimeMillis();
      NcnnYolo11 e = new NcnnYolo11();
      if (!e.loadModel(context.getAssets(), NcnnYolo11.TASK_POSE, modelId, cpugpu)) {
        ErrorLog.e(TAG, "ncnn pose 模型加载失败");
        return false;
      }
      engine = e;
      engineReady = true;
      loaded = true;
      ErrorLog.i(TAG, "ncnn pose 模型加载成功 " + (System.currentTimeMillis() - t0) + "ms");
      return true;
    } catch (Throwable t) {
      ErrorLog.e(TAG, "ncnn pose 模型加载失败", t);
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
      tcs.setResult(new Result(new NcnnYolo11.Result[0]));
      return tcs.getTask();
    }
    final int fw = frame.getWidth();
    final int fh = frame.getHeight();
    final int[] argb = new int[fw * fh];

    detectorThread.execute(() -> {
      if (!ensureLoaded()) {
        tcs.setResult(new Result(new NcnnYolo11.Result[0]));
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
            ErrorLog.i(TAG, String.format("ncnn 首次推理完成: poses=%d %dms", items.length, dt));
          } else if (dt >= 100) {
            ErrorLog.i(TAG, String.format("ncnn 推理: poses=%d %dms", items.length, dt));
          }
        }
        tcs.setResult(new Result(items));
      } catch (Throwable t) {
        if (!firstFailureShown) {
          firstFailureShown = true;
          ErrorLog.e(TAG, "ncnn 推理失败", t);
        }
        tcs.setResult(new Result(new NcnnYolo11.Result[0]));
      }
    });
    return tcs.getTask();
  }

  @Override
  protected void onSuccess(@NonNull Result result, @NonNull GraphicOverlay graphicOverlay) {
    int ih = graphicOverlay.getImageHeight();
    if (jumperMode && result.items.length > 0 && ih > 0) {
      NcnnYolo11.Result p = result.items[0];
      if (p.nkpts >= 17) {
        // Normalize kpt pixel coords -> relative for the jumper counter.
        float[] x = new float[17], y = new float[17], vis = new float[17];
        for (int k = 0; k < 17; k++) {
          x[k] = p.kpts[k * 3] / graphicOverlay.getImageWidth();
          y[k] = p.kpts[k * 3 + 1] / ih;
          vis[k] = p.kpts[k * 3 + 2];
        }
        int count = ropeAdapter.onPoseYolo(x, y, vis, ih);
        graphicOverlay.add(new JumpCountGraphic(graphicOverlay, count));
      }
    }
    graphicOverlay.add(new NcnnGraphic(graphicOverlay, result.items, NcnnYolo11.TASK_POSE, null));
  }

  @Override
  protected void onFailure(@NonNull Exception e) {
    ErrorLog.e(TAG, "ncnn pose 检测失败", e);
  }
}