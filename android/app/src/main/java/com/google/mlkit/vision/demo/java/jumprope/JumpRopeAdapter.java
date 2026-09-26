package com.google.mlkit.vision.demo.java.jumprope;

import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseLandmark;

/**
 * Jump-rope counter driven by ML Kit pose landmarks.
 *
 * Tracks the vertical (Y) displacement of the body center (average of hip and
 * ankle landmarks, normalized by image height). A jump = body center rises
 * above its resting baseline and then falls back through it. State machine:
 *
 *   IDLE -> (rise above baseline) -> UP -> (past apex / returning) -> DOWN
 *   -> (lands below baseline for enough frames) -> COUNT++ -> IDLE
 *
 * Ported from the JumpRopeDemo self-contained engine, adapted to consume the
 * ML Kit 33-landmark skeleton.
 */
public class JumpRopeAdapter {
    public enum Phase { IDLE, UP, DOWN }

    private volatile int count = 0;
    private Phase phase = Phase.IDLE;
    private float baseline = Float.NaN;   // EMA of rest height (normalized 0..1, larger = lower)
    private int belowBaselineFrames = 0;
    private boolean wasAbove = false;

    // Tuned for ML Kit image-space normalized coords (y/height).
    private static final float RISE_FRAC = 0.006f;
    private static final float EMIT_ABOVE = 0.003f;
    private static final int MIN_FRAMES_BELOW = 2;
    private static final float BASELINE_ALPHA = 0.05f;

    private long lockUntil = 0;

    public synchronized int onPose(Pose pose, int imageHeight) {
        float cy = centroidYNorm(pose, imageHeight);
        return consumeCentroid(cy);
    }

    /**
     * YOLOv11-pose variant: 17 COCO keypoints, relative (0..1) image coords.
     * Shoulders = 5,6; hips = 11,12; ankles = 15,16.
     */
    public synchronized int onPoseYolo(float[] x, float[] y, float[] vis, int imageHeight) {
        if (x == null || y == null || vis == null || x.length < 17) return count;
        int[] ids = {5, 6, 11, 12, 15, 16};
        float sum = 0f;
        int n = 0;
        for (int id : ids) {
            if (vis[id] < 0.3f) continue;
            sum += y[id]; // already normalized 0..1
            n++;
        }
        return consumeCentroid(n == 0 ? Float.NaN : sum / n);
    }

    /** Shared state machine; cy is normalized center Y (0..1, larger = lower). */
    private synchronized int consumeCentroid(float cy) {
        if (Float.isNaN(cy)) return count;
        long now = System.currentTimeMillis();

        if (Float.isNaN(baseline)) baseline = cy;
        if (phase == Phase.IDLE) {
            baseline += (cy - baseline) * BASELINE_ALPHA;
        }

        float diff = baseline - cy;   // >0 => above baseline (jump apex)

        switch (phase) {
            case IDLE:
                if (diff > RISE_FRAC) {
                    phase = Phase.UP;
                    wasAbove = true;
                }
                break;
            case UP:
                if (diff <= EMIT_ABOVE && wasAbove) {
                    phase = Phase.DOWN;
                    belowBaselineFrames = 0;
                }
                if (diff > EMIT_ABOVE) wasAbove = true;
                break;
            case DOWN:
                if (diff < -EMIT_ABOVE) {
                    belowBaselineFrames++;
                    if (belowBaselineFrames >= MIN_FRAMES_BELOW) {
                        if (now >= lockUntil) {
                            count++;
                            lockUntil = now + 250;
                        }
                        phase = Phase.IDLE;
                        wasAbove = false;
                    }
                } else if (diff > 0) {
                    // still above baseline; keep waiting
                } else {
                    belowBaselineFrames = 0;
                }
                break;
        }
        return count;
    }

    /**
     * Normalized center-of-body Y (0..1, larger = lower on screen) from visible
     * hip/ankle/shoulder landmarks, so partial-body occlusions still track.
     */
    private float centroidYNorm(Pose pose, int imageHeight) {
        int[] ids = {
                PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
                PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
                PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE
        };
        float sum = 0f;
        int n = 0;
        if (imageHeight <= 0) imageHeight = 1;
        for (int id : ids) {
            PoseLandmark lm = pose.getPoseLandmark(id);
            if (lm == null || lm.getInFrameLikelihood() < 0.3f) continue;
            sum += lm.getPosition().y / imageHeight;
            n++;
        }
        return n == 0 ? Float.NaN : sum / n;
    }

    public synchronized int getCount() {
        return count;
    }

    public synchronized void reset() {
        count = 0;
        phase = Phase.IDLE;
        baseline = Float.NaN;
        belowBaselineFrames = 0;
        wasAbove = false;
        lockUntil = 0;
    }
}