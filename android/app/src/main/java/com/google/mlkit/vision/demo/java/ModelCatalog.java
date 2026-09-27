package com.google.mlkit.vision.demo.java;

import android.content.Context;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Catalog of every selectable model: ML Kit native pose, all 5 ncnn tasks
 * (COCO/SEG/Pose/CLS/OBB x n/s/m at 320/480/640), built-in ONNX models
 * (yolo11n.onnx, yolo11n-pose.onnx) and user-imported .onnx files stored in
 * {@code filesDir/onnx/}.
 *
 * <p>Value scheme used in the MultiSelectListPreference:
 * <ul>
 *   <li>{@code mlkit} - ML Kit native pose detector</li>
 *   <li>{@code ncnn:taskid:modelid} - an ncnn YOLO11 model (0..4 task, 0..8 size)</li>
 *   <li>{@code onnx:yolo11n} - built-in YOLOv11 COCO object detection</li>
 *   <li>{@code onnx:yolo11n-pose} - built-in YOLOv11 pose</li>
 *   <li>{@code onnx:import:<filename>} - a user-imported ONNX model</li>
 * </ul>
 */
public final class ModelCatalog {

  public static final String MLKIT = "mlkit";
  public static final String ONNX_YOLO11N = "onnx:yolo11n";
  public static final String ONNX_YOLO11N_POSE = "onnx:yolo11n-pose";
  public static final String ONNX_IMPORT_PREFIX = "onnx:import:";

  public static final String[] TASK_NAMES = {"COCO", "SEG", "Pose", "CLS", "OBB"};
  public static final String[] SIZE_NAMES = {
      "n-320", "s-320", "m-320", "n-480", "s-480", "m-480", "n-640", "s-640", "m-640"};

  public static class Entry {
    public final String value;
    public final String label;
    Entry(String value, String label) {
      this.value = value;
      this.label = label;
    }
  }

  private ModelCatalog() {}

  /** All model entries in canonical order (first selected one runs). */
  public static List<Entry> all(Context context) {
    List<Entry> out = new ArrayList<>();
    out.add(new Entry(MLKIT, "ML Kit Pose"));
    for (int t = 0; t < TASK_NAMES.length; t++) {
      for (int s = 0; s < SIZE_NAMES.length; s++) {
        out.add(new Entry("ncnn:" + t + ":" + s, TASK_NAMES[t] + " " + SIZE_NAMES[s]));
      }
    }
    out.add(new Entry(ONNX_YOLO11N, "ONNX COCO (yolo11n)"));
    out.add(new Entry(ONNX_YOLO11N_POSE, "ONNX Pose (yolo11n-pose)"));
    for (File f : importedOnnxFiles(context)) {
      out.add(new Entry(ONNX_IMPORT_PREFIX + f.getName(), "导入 " + f.getName()));
    }
    return out;
  }

  /** The first entry of {@link #all} whose value is in {@code selected}. */
  public static String firstSelected(Context context, Set<String> selected) {
    if (selected == null) return MLKIT;
    for (Entry e : all(context)) {
      if (selected.contains(e.value)) return e.value;
    }
    return MLKIT;
  }

  /** Human label for a value, or the raw value when unknown. */
  public static String labelFor(Context context, String value) {
    for (Entry e : all(context)) if (e.value.equals(value)) return e.label;
    return value;
  }

  /** Entries + values arrays for the MultiSelectListPreference. */
  public static Map<String, Object> buildPreferenceData(Context context) {
    List<Entry> all = all(context);
    String[] entries = new String[all.size()];
    String[] values = new String[all.size()];
    for (int i = 0; i < all.size(); i++) {
      entries[i] = all.get(i).label;
      values[i] = all.get(i).value;
    }
    LinkedHashMap<String, Object> m = new LinkedHashMap<>();
    m.put("entries", entries);
    m.put("values", values);
    return m;
  }

  /** Directory where imported .onnx files live (internal app storage). */
  public static File importDir(Context context) {
    return new File(context.getFilesDir(), "onnx");
  }

  public static List<File> importedOnnxFiles(Context context) {
    List<File> out = new ArrayList<>();
    File dir = importDir(context);
    File[] fs = dir.listFiles();
    if (fs != null) {
      for (File f : fs) if (f.getName().toLowerCase().endsWith(".onnx")) out.add(f);
    }
    java.util.Collections.sort(out);
    return out;
  }
}