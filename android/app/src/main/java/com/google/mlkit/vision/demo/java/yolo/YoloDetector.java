package com.google.mlkit.vision.demo.java.yolo;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import android.content.Context;
import android.graphics.Bitmap;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Our own YOLO11n detector (ONNX Runtime), integrated as an extra mode on top of
 * the ML Kit skeleton pipeline. Runs the ultralytics yolo11n.onnx model on a
 * bitmap and returns boxes with COCO class ids in RELATIVE (0..1) coordinates so
 * the graphic layer can map them onto the overlay just like ML Kit landmarks.
 *
 * Ported from the JumpRopeDemo project (com.demo.jumprope.YoloDetector).
 */
public class YoloDetector {

    public static class Box {
        public float x1, y1, x2, y2; // relative 0..1 (letterbox space)
        public float score;
        public int cls;
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final float confThresh = 0.25f;
    private final float iouThresh = 0.45f;
    private final String inputName;
    // Model exported with FIXED 1x3x640x640 input (ultralytics default, not dynamic).
    private final int inputW = 640, inputH = 640;
    // Latest letterbox mapping (used to translate model-space boxes back to the
    // original image coordinate space so the overlay aligns them with ML Kit).
    private volatile float lastScale = 1f;
    private volatile int lastPadX, lastPadY;
    private volatile int lastImgW = 1, lastImgH = 1;

    public YoloDetector(Context ctx, String assetModel) throws IOException, OrtException {
        env = OrtEnvironment.getEnvironment();
        try (InputStream is = ctx.getAssets().open(assetModel)) {
            byte[] bytes = is.readAllBytes();
            OrtSession s;
            try {
                // Try NNAPI first (uses APU/DSP when available).
                OrtSession.SessionOptions nn = new OrtSession.SessionOptions();
                nn.setIntraOpNumThreads(2);
                try { nn.addNnapi(); } catch (OrtException ignored) {}
                s = env.createSession(bytes, nn);
            } catch (OrtException | UnsatisfiedLinkError e) {
                // NNAPI rejected the graph; fall back to CPU only.
                OrtSession.SessionOptions cpu = new OrtSession.SessionOptions();
                cpu.setIntraOpNumThreads(2);
                s = env.createSession(bytes, cpu);
            }
            session = s;
        }
        Set<String> ins = session.getInputNames();
        inputName = ins.isEmpty() ? "images" : ins.iterator().next();
    }

    /** Loads a user-supplied ONNX model from an internal-storage file. */
    public YoloDetector(File modelFile) throws IOException, OrtException {
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

    /** Letterbox a bitmap to [640x640] interleaved RGB float array, 0..255. */
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

    /**
     * Runs detection on a bitmap. Returns boxes in relative coordinates
     * (0..1, same letterbox space as the model input).
     */
    public List<Box> detect(Bitmap bmp) throws OrtException {
        float[] inRgb = toModelInput(bmp);
        float[] chw = new float[3 * inputW * inputH];
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < inputW * inputH; i++) {
                chw[c * inputW * inputH + i] = inRgb[i * 3 + c] / 255.0f;
            }
        }
        long[] shape = {1, 3, inputH, inputW};
        List<Box> boxes = new ArrayList<>();
        try (OnnxTensor tensor = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(chw, 0, chw.length), shape)) {
            OrtSession.Result out = session.run(java.util.Collections.singletonMap(inputName, tensor));
            float[][][] pred = (float[][][]) out.get(0).getValue();
            parseOutput(pred, boxes);
            out.close();
        }
        return boxes;
    }

    private void parseOutput(float[][][] pred, List<Box> out) {
        float[][][] p = permuteToNxParms(pred);
        if (p == null) return;
        int numBoxes = p[0].length;
        int parms = p[0][0].length;
        float scale = lastScale;
        int padX = lastPadX, padY = lastPadY;
        for (int i = 0; i < numBoxes; i++) {
            float[] row = p[0][i];
            if (parms == 6) { // NMS-free end-to-end: [x1,y1,x2,y2,conf,cls]
                Box b = new Box();
                b.x1 = (row[0] - padX) / scale / lastImgW;
                b.y1 = (row[1] - padY) / scale / lastImgH;
                b.x2 = (row[2] - padX) / scale / lastImgW;
                b.y2 = (row[3] - padY) / scale / lastImgH;
                b.score = row[4];
                b.cls = (int) row[5];
                if (b.score >= confThresh) out.add(b);
            } else if (parms >= 84) { // classic [4+nc]
                float maxS = -1; int maxC = -1;
                for (int c = 4; c < parms; c++) {
                    if (row[c] > maxS) { maxS = row[c]; maxC = c - 4; }
                }
                if (maxS >= confThresh) {
                    Box b = new Box();
                    float cx = (row[0] - padX) / scale / lastImgW;
                    float cy = (row[1] - padY) / scale / lastImgH;
                    float w = row[2] / scale / lastImgW;
                    float h = row[3] / scale / lastImgH;
                    b.x1 = cx - w / 2; b.y1 = cy - h / 2; b.x2 = cx + w / 2; b.y2 = cy + h / 2;
                    b.score = maxS;
                    b.cls = maxC;
                    out.add(b);
                }
            }
        }
        clampBoxes(out);
        nms(out);
    }

    private void clampBoxes(List<Box> boxes) {
        for (Box b : boxes) {
            b.x1 = Math.max(0, Math.min(1, b.x1));
            b.y1 = Math.max(0, Math.min(1, b.y1));
            b.x2 = Math.max(0, Math.min(1, b.x2));
            b.y2 = Math.max(0, Math.min(1, b.y2));
        }
    }

    private float[][][] permuteToNxParms(float[][][] raw) {
        if (raw.length != 1) return null;
        int h1 = raw[0].length, w1 = raw[0][0].length;
        if (h1 == 0 || w1 == 0) return null;

        boolean wIsParm = (w1 == 6) || (w1 >= 84 && w1 <= h1 / 2)
                || (w1 < h1 && w1 >= 84);
        if (wIsParm) return raw; // already [1,N,P]

        if (h1 == 6 || h1 >= 84) {
            float[][][] out = new float[1][w1][h1];
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    out[0][j][i] = raw[0][i][j];
                }
            }
            return out;
        }
        return raw;
    }

    private void nms(List<Box> boxes) {
        List<Box> keep = new ArrayList<>();
        float[] areas = new float[boxes.size()];
        for (int i = 0; i < boxes.size(); i++) {
            Box b = boxes.get(i);
            areas[i] = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
        }
        boolean[] removed = new boolean[boxes.size()];
        while (true) {
            int best = -1;
            for (int i = 0; i < boxes.size(); i++) {
                if (removed[i]) continue;
                if (best < 0 || boxes.get(i).score > boxes.get(best).score) best = i;
            }
            if (best < 0) break;
            Box b = boxes.get(best);
            removed[best] = true;
            keep.add(b);
            for (int j = 0; j < boxes.size(); j++) {
                if (removed[j]) continue;
                Box c = boxes.get(j);
                float ix1 = Math.max(b.x1, c.x1), iy1 = Math.max(b.y1, c.y1);
                float ix2 = Math.min(b.x2, c.x2), iy2 = Math.min(b.y2, c.y2);
                float inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
                float union = areas[best] + areas[j] - inter;
                if (union > 0 && inter / union > iouThresh) {
                    removed[j] = true;
                }
            }
        }
        boxes.clear();
        boxes.addAll(keep);
    }

    /** Reads "class_id chinese english" lines; returns [cls] -> "chinese english". */
    public static String[] loadLabels(Context ctx, String assetPath) {
        List<String> out = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(ctx.getAssets().open(assetPath)))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                // strip leading index if present
                String[] parts = line.split("\\s+");
                StringBuilder sb = new StringBuilder();
                int start = (parts.length >= 2 && parts[0].matches("\\d+")) ? 1 : 0;
                for (int i = start; i < parts.length; i++) {
                    if (sb.length() > 0) sb.append(' ');
                    sb.append(parts[i]);
                }
                out.add(sb.toString());
            }
        } catch (IOException e) {
            return new String[0];
        }
        return out.toArray(new String[0]);
    }
}