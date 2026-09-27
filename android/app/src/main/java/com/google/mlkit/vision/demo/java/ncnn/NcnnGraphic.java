package com.google.mlkit.vision.demo.java.ncnn;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.GraphicOverlay.Graphic;

/**
 * Draws ncnn YOLO11 det/pose/seg/obb results. Boxes/keypoints are already in
 * original-bitmap pixel space (same as the overlay image space), so we pass
 * them through translateX/Y directly.
 */
public class NcnnGraphic extends Graphic {

    private static final float DOT_RADIUS = 9f;
    private static final float MIN_VIS = 0.3f;

    /** COCO skeleton connection pairs (pose task). */
    private static final int[][] CONNECTIONS = {
            {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6},
            {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12},
            {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}};

    private final NcnnYolo11.Result[] results;
    private final int task;
    private final String[] labels;

    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint linePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint dotPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    public NcnnGraphic(GraphicOverlay overlay, NcnnYolo11.Result[] results, int task, String[] labels) {
        super(overlay);
        this.results = results;
        this.task = task;
        this.labels = labels;

        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6);
        boxPaint.setColor(Color.rgb(0, 200, 255));

        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(10);
        linePaint.setColor(Color.CYAN);

        dotPaint.setStyle(Paint.Style.FILL);
        dotPaint.setColor(Color.WHITE);

        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setTextSize(34);
        textPaint.setColor(Color.YELLOW);
    }

    @Override
    public void draw(Canvas canvas) {
        if (results == null || results.length == 0) return;

        for (NcnnYolo11.Result r : results) {
            // Bounding box (image pixel coords already).
            if (r.x1 > r.x0 && r.y1 > r.y0) {
                boxPaint.setColor(Color.rgb(0, 200, 255));
                canvas.drawRect(translateX(r.x0), translateY(r.y0),
                        translateX(r.x1), translateY(r.y1), boxPaint);
            }

            // Pose skeleton.
            if (task == NcnnYolo11.TASK_POSE && r.nkpts >= 17) {
                for (int[] c : CONNECTIONS) {
                    float vx1 = r.kpts[c[0] * 3], vy1 = r.kpts[c[0] * 3 + 1];
                    float vx2 = r.kpts[c[1] * 3], vy2 = r.kpts[c[1] * 3 + 1];
                    float vis1 = r.kpts[c[0] * 3 + 2], vis2 = r.kpts[c[1] * 3 + 2];
                    if (vis1 < MIN_VIS || vis2 < MIN_VIS) continue;
                    canvas.drawLine(translateX(vx1), translateY(vy1),
                            translateX(vx2), translateY(vy2), linePaint);
                }
                for (int k = 0; k < 17; k++) {
                    if (r.kpts[k * 3 + 2] < MIN_VIS) continue;
                    canvas.drawCircle(translateX(r.kpts[k * 3]), translateY(r.kpts[k * 3 + 1]),
                            DOT_RADIUS, dotPaint);
                }
            }

            // Label.
            String label;
            if (task == NcnnYolo11.TASK_CLS || task == NcnnYolo11.TASK_DET || task == NcnnYolo11.TASK_SEG) {
                int idx = (int) r.label;
                label = (labels != null && idx < labels.length && labels[idx] != null)
                        ? labels[idx] : ("cls" + idx);
            } else {
                label = task == NcnnYolo11.TASK_POSE ? "person" : "obj";
            }
            label += String.format(" %.0f%%", r.prob * 100);

            textPaint.setColor(Color.YELLOW);
            textPaint.setShadowLayer(6, 0, 0, Color.BLACK);
            float tx = task == NcnnYolo11.TASK_CLS ? 20 : r.x0;
            float ty = task == NcnnYolo11.TASK_CLS ? 44 : Math.max(r.y0 - 12, 20);
            canvas.drawText(label, translateX(tx), translateY(ty), textPaint);
            textPaint.clearShadowLayer();
        }
    }
}