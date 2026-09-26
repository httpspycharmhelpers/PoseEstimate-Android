package com.google.mlkit.vision.demo.java.yolo;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.GraphicOverlay.Graphic;

import java.util.List;

/**
 * Draws our YOLO11-pose skeleton (17 COCO keypoints) on the overlay as a full
 * replacement for the native ML Kit skeleton graphic. Keypoints are in relative
 * (0..1) image coordinates; we map them to the image pixel space (which the
 * overlay translates to the view) exactly like YoloGraphic does for boxes.
 */
public class YoloPoseGraphic extends Graphic {

    private static final float DOT_RADIUS = 9f;
    private static final float MIN_VIS = 0.3f;

    private final List<YoloPoseDetector.Pose> poses;
    private final Paint linePaint;
    private final Paint boxPaint;
    private final Paint dotPaint;
    private final Paint textPaint;

    public YoloPoseGraphic(GraphicOverlay overlay, List<YoloPoseDetector.Pose> poses) {
        super(overlay);
        this.poses = poses;

        linePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(10);
        linePaint.setColor(Color.CYAN);

        boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6);
        boxPaint.setColor(Color.GREEN);

        dotPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        dotPaint.setStyle(Paint.Style.FILL);
        dotPaint.setColor(Color.WHITE);

        textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setTextSize(34);
        textPaint.setColor(Color.YELLOW);
    }

    @Override
    public void draw(Canvas canvas) {
        int iw = getGraphicOverlay().getImageWidth();
        int ih = getGraphicOverlay().getImageHeight();
        if (iw <= 0 || ih <= 0) return;

        for (YoloPoseDetector.Pose pose : poses) {
            // Person bounding box from the model output.
            if (pose.boxX2 > pose.boxX1 && pose.boxY2 > pose.boxY1) {
                canvas.drawRect(
                        translateX(pose.boxX1 * iw), translateY(pose.boxY1 * ih),
                        translateX(pose.boxX2 * iw), translateY(pose.boxY2 * ih),
                        boxPaint);
            }

            // Bones.
            for (int[] c : YoloPoseDetector.CONNECTIONS) {
                float vx1 = pose.x[c[0]] * iw, vy1 = pose.y[c[0]] * ih;
                float vx2 = pose.x[c[1]] * iw, vy2 = pose.y[c[1]] * ih;
                if (pose.vis[c[0]] < MIN_VIS || pose.vis[c[1]] < MIN_VIS) continue;
                canvas.drawLine(
                        translateX(vx1), translateY(vy1), translateX(vx2), translateY(vy2), linePaint);
            }

            // Joints.
            for (int k = 0; k < 17; k++) {
                if (pose.vis[k] < MIN_VIS) continue;
                canvas.drawCircle(
                        translateX(pose.x[k] * iw), translateY(pose.y[k] * ih), DOT_RADIUS, dotPaint);
            }

            // Person confidence.
            if (pose.x[5] > 0 && pose.y[5] > 0) {
                canvas.drawText(
                        "person " + (int) (pose.score * 100) + "%",
                        translateX(pose.x[5] * iw),
                        translateY(pose.y[5] * ih) - 12,
                        textPaint);
            }
        }
    }
}