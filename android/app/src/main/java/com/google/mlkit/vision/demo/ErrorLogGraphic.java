package com.google.mlkit.vision.demo;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

/** Renders the latest in-app error underneath the jump count / FPS text. */
public class ErrorLogGraphic extends GraphicOverlay.Graphic {

    private static final float TEXT_SIZE = 30.0f;
    private final String error;
    private final int errorCount;
    private final Paint textPaint;

    public ErrorLogGraphic(GraphicOverlay overlay) {
        super(overlay);
        this.error = ErrorLog.getLastError();
        this.errorCount = ErrorLog.getErrorCount();
        textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textPaint.setTextSize(TEXT_SIZE);
        textPaint.setColor(Color.RED);
    }

    @Override
    public void draw(Canvas canvas) {
        String text = error.isEmpty() ? "" : ("ERR#" + errorCount + " " + error);
        // Put it below the inference info block (y ~ 3 lines down).
        float y = TEXT_SIZE * 4.2f;
        canvas.drawText(text, TEXT_SIZE * 0.5f, y, textPaint);
    }
}