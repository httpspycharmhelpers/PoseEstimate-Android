package com.google.mlkit.vision.demo;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.os.SystemClock;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;

/**
 * In-app error/log channel shown live on the overlay and on a dedicated
 * "terminal" page. Error-level entries are automatically copied to the system
 * clipboard so the user can paste the failure into a chat for debugging.
 */
public final class ErrorLog {

    private static final int MAX_LINES = 500;
    private static final String TAG = "ErrorLog";

    private static final Object LOCK = new Object();
    private static final List<String> lines = new ArrayList<>();
    private static volatile String lastError = "";
    private static volatile int errorSeq = 0;
    private static volatile Context appContext;

    private ErrorLog() {}

    /** Must be called once at app start to enable clipboard auto-copy. */
    public static void init(Context context) {
        appContext = context.getApplicationContext();
    }

    public static void i(String tag, String msg) {
        append("I", tag, msg);
    }

    public static void w(String tag, String msg) {
        append("W", tag, msg);
    }

    public static void e(String tag, String msg) {
        append("E", tag, msg);
        Log.e(tag, msg);
    }

    public static void e(String tag, String msg, Throwable t) {
        append("E", tag, msg + (t != null ? " | " + t : ""));
        Log.e(tag, msg, t);
    }

    private static void append(String level, String tag, String msg) {
        String line = String.format("%06.2f [%s] %s: %s",
                SystemClock.uptimeMillis() / 1000f, level, tag, msg);
        synchronized (LOCK) {
            lines.add(line);
            if (lines.size() > MAX_LINES) {
                lines.remove(0);
            }
            if (level.equals("E")) {
                errorSeq++;
                lastError = line;
            }
        }
        if (level.equals("E")) {
            copyToClipboard(line);
        }
    }

    private static void copyToClipboard(String line) {
        if (appContext == null) return;
        try {
            ClipboardManager cm =
                    (ClipboardManager) appContext.getSystemService(Context.CLIPBOARD_SERVICE);
            if (cm != null) {
                cm.setPrimaryClip(ClipData.newPlainText("opencode-error", line));
            }
        } catch (Exception ignore) {}
    }

    public static String getLastError() {
        return lastError;
    }

    public static int getErrorCount() {
        return errorSeq;
    }

    /** Snapshot of the whole log (most recent last). */
    public static List<String> snapshot() {
        synchronized (LOCK) {
            return new ArrayList<>(lines);
        }
    }

    public static void clear() {
        synchronized (LOCK) {
            lines.clear();
            lastError = "";
        }
    }
}