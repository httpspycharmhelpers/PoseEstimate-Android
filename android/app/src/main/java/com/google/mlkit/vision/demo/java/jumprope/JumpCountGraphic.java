package com.google.mlkit.vision.demo.java.jumprope;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;

import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.GraphicOverlay.Graphic;

/** Displays the running jump-rope count at the top of the overlay. */
public class JumpCountGraphic extends Graphic {

    private final int count;
    private final Paint textPaint = new Paint();

    public JumpCountGraphic(GraphicOverlay overlay, int count) {
        super(overlay);
        this.count = count;
        textPaint.setStyle(Paint.Style.FILL);
        textPaint.setColor(Color.rgb(255, 210, 60));
        textPaint.setTextSize(72);
        textPaint.setTypeface(Typeface.DEFAULT_BOLD);
        textPaint.setShadowLayer(8, 0, 0, Color.BLACK);
    }

    @Override
    public void draw(Canvas canvas) {
        canvas.drawText("跳绳 " + count, 40, 140, textPaint);
    }
}