package com.google.mlkit.vision.demo.java.yolo;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.GraphicOverlay.Graphic;

import java.util.List;

/**
 * Draws our YOLO detection boxes on the overlay. Boxes are in relative (0..1)
 * letterbox coordinates; we map to the ML Kit image coordinate space (which the
 * overlay already knows how to translate to the view).
 */
public class YoloGraphic extends Graphic {

    private static final int[] PALETTE = {
        Color.rgb(255, 68, 68), Color.rgb(68, 68, 255), Color.rgb(255, 180, 0),
        Color.rgb(0, 200, 120), Color.rgb(200, 0, 200), Color.rgb(0, 200, 200),
        Color.rgb(255, 110, 0), Color.rgb(90, 120, 255)
    };

    private final List<YoloDetector.Box> boxes;
    private final String[] labels;
    private final Paint boxPaint = new Paint();
    private final Paint textPaint = new Paint();

    public YoloGraphic(GraphicOverlay overlay, List<YoloDetector.Box> boxes, String[] labels) {
        super(overlay);
        this.boxes = boxes;
        this.labels = labels;
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6);
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setTextSize(34);
        textPaint.setColor(Color.WHITE);
    }

    @Override
    public void draw(Canvas canvas) {
        int iw = getGraphicOverlay().getImageWidth();
        int ih = getGraphicOverlay().getImageHeight();
        if (iw <= 0 || ih <= 0) return;
        for (int i = 0; i < boxes.size(); i++) {
            YoloDetector.Box b = boxes.get(i);
            // relative 0..1 -> image pixel space (same as ML Kit landmark coords)
            float x1 = b.x1 * iw, y1 = b.y1 * ih;
            float x2 = b.x2 * iw, y2 = b.y2 * ih;
            int color = PALETTE[b.cls % PALETTE.length];
            boxPaint.setColor(color);
            canvas.drawRect(translateX(x1), translateY(y1), translateX(x2), translateY(y2), boxPaint);

            String label = (labels != null && b.cls < labels.length && labels[b.cls] != null)
                    ? labels[b.cls] : ("cls" + b.cls);
            label += String.format(" %.2f", b.score);
            textPaint.setColor(Color.WHITE);
            textPaint.setShadowLayer(6, 0, 0, Color.BLACK);
            canvas.drawText(label, translateX(x1), translateY(y1) - 10, textPaint);
            textPaint.clearShadowLayer();
        }
    }
}