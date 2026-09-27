package com.google.mlkit.vision.demo.java.yolo;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import android.content.Context;
import android.graphics.Bitmap;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Our own YOLO11n-pose skeleton detector (ONNX Runtime), used as a full
 * replacement skeleton engine (Performance mode "YOLOv11 Pose" / native model
 * switch OFF). Same letterbox + 640 fixed input as YoloDetector; returns the
 * 17 COCO keypoints in RELATIVE (0..1) image coordinates so the graphic layer
 * maps them exactly like ML Kit landmarks.
 *
 * Ported from the JumpRopeDemo project (com.demo.jumprope.PoseDetector).
 */
public class YoloPoseDetector {

    /** 17 COCO keypoints. */
    public static final String[] KEYPOINT_NAMES_ZH = {
            "鼻", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩",
            "左肘", "右肘", "左腕", "右腕", "左髋", "右髋",
            "左膝", "右膝", "左踝", "右踝"};
    /** Skeleton connection pairs (COCO). */
    public static final int[][] CONNECTIONS = {
            {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6},
            {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12},
            {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}};

    public static class Pose {
        public float score; // person confidence
        public float boxX1, boxY1, boxX2, boxY2; // relative 0..1
        public float[] x; // 17, relative 0..1
        public float[] y;
        public float[] vis; // keypoint visibility 0..1
        public Pose(int n) {
            x = new float[n]; y = new float[n]; vis = new float[n];
            boxX1 = boxY1 = 0f; boxX2 = boxY2 = 1f;
        }
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final float confThresh = 0.35f;
    private final int inputW = 640, inputH = 640;
    private final int maxPoses = 1;
    // Latest letterbox mapping to translate model-space keypoints back to the
    // original image coordinate space (same as YoloDetector boxes).
    private volatile float lastScale = 1f;
    private volatile int lastPadX, lastPadY;
    private volatile int lastImgW = 1, lastImgH = 1;
    private volatile float[][] lastKpts; // [17][3] model-space x,y,vis (debug/back-compat)

    public YoloPoseDetector(Context ctx, String assetModel) throws IOException, OrtException {
        env = OrtEnvironment.getEnvironment();
        try (InputStream is = ctx.getAssets().open(assetModel)) {
            byte[] bytes = is.readAllBytes();
            OrtSession s;
            try {
                OrtSession.SessionOptions nn = new OrtSession.SessionOptions();
                nn.setIntraOpNumThreads(2);
                try { nn.addNnapi(); } catch (OrtException ignored) {}
                s = env.createSession(bytes, nn);
            } catch (OrtException | UnsatisfiedLinkError e) {
                OrtSession.SessionOptions cpu = new OrtSession.SessionOptions();
                cpu.setIntraOpNumThreads(2);
                s = env.createSession(bytes, cpu);
            }
            session = s;
        }
        Set<String> ins = session.getInputNames();
        inputName = ins.isEmpty() ? "images" : ins.iterator().next();
    }

    /** Loads a user-supplied ONNX pose model from an internal-storage file. */
    public YoloPoseDetector(File modelFile) throws IOException, OrtException {
        env = OrtEnvironment.getEnvironment();
        try (InputStream is = new FileInputStream(modelFile)) {
            byte[] bytes = is.readAllBytes();
            OrtSession s;
            try {
                OrtSession.SessionOptions nn = new OrtSession.SessionOptions();
                nn.setIntraOpNumThreads(2);
                try { nn.addNnapi(); } catch (OrtException ignored) {}
                s = env.createSession(bytes, nn);
            } catch (OrtException | UnsatisfiedLinkError e) {
                OrtSession.SessionOptions cpu = new OrtSession.SessionOptions();
                cpu.setIntraOpNumThreads(2);
                s = env.createSession(bytes, cpu);
            }
            session = s;
        }
        Set<String> ins = session.getInputNames();
        inputName = ins.isEmpty() ? "images" : ins.iterator().next();
    }

    public void close() {
        // NOTE: never close the OrtEnvironment here. getEnvironment() returns a
        // process-wide SHARED singleton; closing it would kill every later
        // OrtSession in the app (including the other YOLO detector), making all
        // models fail to load after the first processor stop(). Only the session
        // this instance owns is closed.
        try {
            session.close();
        } catch (Exception ignore) {}
    }

    private float[] toModelInput(Bitmap original) {
        int imgW = original.getWidth();
        int imgH = original.getHeight();
        lastImgW = imgW; lastImgH = imgH;
        float scale = Math.min((float) inputW / imgW, (float) inputH / imgH);
        lastScale = scale;
        int dw = Math.max(1, (int) (imgW * scale));
        int dh = Math.max(1, (int) (imgH * scale));
        int padX = (inputW - dw) / 2;
        int padY = (inputH - dh) / 2;
        lastPadX = padX; lastPadY = padY;
        Bitmap scaled = Bitmap.createScaledBitmap(original, dw, dh, true);
        int[] pix = new int[dw * dh];
        scaled.getPixels(pix, 0, dw, 0, 0, dw, dh);
        if (scaled != original) scaled.recycle();

        float[] rgb = new float[inputW * inputH * 3];
        int idx = 0;
        for (int dy = 0; dy < inputH; dy++) {
            for (int dx = 0; dx < inputW; dx++) {
                int sx = dx - padX;
                int sy = dy - padY;
                if (sx >= 0 && sy >= 0 && sx < dw && sy < dh) {
                    int color = pix[sy * dw + sx];
                    rgb[idx++] = (color >> 16) & 0xFF;
                    rgb[idx++] = (color >> 8) & 0xFF;
                    rgb[idx++] = color & 0xFF;
                } else {
                    rgb[idx++] = 114;
                    rgb[idx++] = 114;
                    rgb[idx++] = 114;
                }
            }
        }
        return rgb;
    }

    /** Returns up to maxPoses poses (top scores), keypoints relative 0..1 image coords. */
    public List<Pose> detect(Bitmap bmp) throws OrtException {
        float[] inRgb = toModelInput(bmp);
        float[] chw = new float[3 * inputW * inputH];
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < inputW * inputH; i++) {
                chw[c * inputW * inputH + i] = inRgb[i * 3 + c] / 255.0f;
            }
        }
        long[] shape = {1, 3, inputH, inputW};
        List<Pose> poses = new ArrayList<>();
        try (OnnxTensor tensor = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(chw, 0, chw.length), shape)) {
            OrtSession.Result out = session.run(java.util.Collections.singletonMap(inputName, tensor));
            float[][][] pred = (float[][][]) out.get(0).getValue();
            parseOutput(pred, poses, maxPoses);
            out.close();
        }
        return poses;
    }

    private void parseOutput(float[][][] pred, List<Pose> out, int maxPoses) {
        float[][][] p = normalizeDims(pred);
        if (p == null) return;
        int n = p[0].length;
        int parms = p[0][0].length;
        List<Pose> cands = new ArrayList<>();
        float scale = lastScale;
        int padX = lastPadX, padY = lastPadY;
        for (int i = 0; i < n; i++) {
            float[] row = p[0][i];
            if (parms == 56) { // classic: 4 box + 1 conf + 17*3 kpts
                float conf = row[4];
                if (conf < confThresh) continue;
                Pose pose = new Pose(17);
                pose.score = conf;
                pose.boxX1 = (row[0] - padX) / scale / lastImgW;
                pose.boxY1 = (row[1] - padY) / scale / lastImgH;
                pose.boxX2 = (row[2] - padX) / scale / lastImgW;
                pose.boxY2 = (row[3] - padY) / scale / lastImgH;
                lastKpts = new float[17][3];
                for (int k = 0; k < 17; k++) {
                    int base = 5 + k * 3;
                    lastKpts[k][0] = row[base];
                    lastKpts[k][1] = row[base + 1];
                    lastKpts[k][2] = row[base + 2];
                    pose.x[k] = (row[base] - padX) / scale / lastImgW;
                    pose.y[k] = (row[base + 1] - padY) / scale / lastImgH;
                    pose.vis[k] = row[base + 2];
                }
                cands.add(pose);
            } else if (parms >= 6 && parms < 60) {
                // end-to-end style: [x1,y1,x2,y2,conf,...kpts...]
                float conf = row[4];
                if (conf < confThresh) continue;
                Pose pose = new Pose(17);
                pose.score = conf;
                pose.boxX1 = (row[0] - padX) / scale / lastImgW;
                pose.boxY1 = (row[1] - padY) / scale / lastImgH;
                pose.boxX2 = (row[2] - padX) / scale / lastImgW;
                pose.boxY2 = (row[3] - padY) / scale / lastImgH;
                int kc = (parms - 6) / 3;
                for (int k = 0; k < 17 && k < kc; k++) {
                    int base = 6 + k * 3;
                    pose.x[k] = (row[base] - padX) / scale / lastImgW;
                    pose.y[k] = (row[base + 1] - padY) / scale / lastImgH;
                    pose.vis[k] = row[base + 2];
                }
                cands.add(pose);
            }
        }
        cands.sort((a, b) -> Float.compare(b.score, a.score));
        for (int i = 0; i < cands.size() && i < maxPoses; i++) out.add(cands.get(i));
    }

    private float[][][] normalizeDims(float[][][] raw) {
        if (raw == null || raw.length == 0 || raw[0].length == 0) return null;
        if (raw[0][0] == null || raw[0][0].length == 0) return null;
        int h1 = raw[0].length, w1 = raw[0][0].length;

        boolean wIsParm = (w1 == 56) || (w1 >= 84 && w1 <= h1 / 2)
                || (w1 < h1 && w1 >= 84);
        if (wIsParm) return raw;

        if (h1 == 56 || h1 >= 84) {
            float[][][] out = new float[1][w1][h1];
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) out[0][j][i] = raw[0][i][j];
            }
            return out;
        }
        return null;
    }
}