package com.google.mlkit.vision.demo.java.ncnn;

import android.content.res.AssetManager;

/** JNI wrapper around the ncnn YOLO11 engine (libncnn_yolo11.so). */
public class NcnnYolo11 {

    /** taskid */
    public static final int TASK_DET = 0;
    public static final int TASK_SEG = 1;
    public static final int TASK_POSE = 2;
    public static final int TASK_CLS = 3;
    public static final int TASK_OBB = 4;

    /** modelid: 0..8 => n/s/m at 320/480/640 */
    public static final int MODEL_N320 = 0;
    public static final int MODEL_S320 = 1;
    public static final int MODEL_M320 = 2;
    public static final int MODEL_N480 = 3;
    public static final int MODEL_S480 = 4;
    public static final int MODEL_M480 = 5;
    public static final int MODEL_N640 = 6;
    public static final int MODEL_S640 = 7;
    public static final int MODEL_M640 = 8;

    /** cpugpu */
    public static final int CPUGPU_CPU = 0;
    public static final int CPUGPU_GPU = 1;
    public static final int CPUGPU_GPU_TURNIP = 2;

    static {
        System.loadLibrary("ncnn_yolo11");
    }

    public native boolean loadModel(AssetManager mgr, int taskid, int modelid, int cpugpu);

    public native float[] detect(int[] argb, int width, int height);

    public native void release();

    /**
     * Parses the flat float array from detect().
     * Layout per object: label, prob, x0, y0, x1, y1, cx, cy, w, h, angle, nkpts,
     * then nkpts*3 (x, y, prob). Coordinates are in original bitmap pixel space.
     */
    public static class Result {
        public static final int HEADER = 11;
        public float label;
        public float prob;
        public float x0, y0, x1, y1;
        public float cx, cy, w, h, angle;
        public final float[] kpts = new float[17 * 3]; // [x,y,prob] * 17
        public int nkpts;
    }

    public static Result[] parse(float[] flat) {
        if (flat == null || flat.length < 1) return new Result[0];
        int count = (int) flat[0];
        if (count <= 0) return new Result[0];
        Result[] out = new Result[count];
        int p = 1;
        for (int i = 0; i < count; i++) {
            Result r = new Result();
            r.label = flat[p++];
            r.prob = flat[p++];
            r.x0 = flat[p++];
            r.y0 = flat[p++];
            r.x1 = flat[p++];
            r.y1 = flat[p++];
            r.cx = flat[p++];
            r.cy = flat[p++];
            r.w = flat[p++];
            r.h = flat[p++];
            r.angle = flat[p++];
            r.nkpts = (int) flat[p++];
            for (int k = 0; k < r.nkpts && k < 17; k++) {
                r.kpts[k * 3] = flat[p++];
                r.kpts[k * 3 + 1] = flat[p++];
                r.kpts[k * 3 + 2] = flat[p++];
            }
            if (r.nkpts > 17) p += (r.nkpts - 17) * 3;
            out[i] = r;
        }
        return out;
    }
}