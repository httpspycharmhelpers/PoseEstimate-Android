package com.google.mlkit.vision.demo;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.os.Bundle;
import android.text.TextUtils;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

/** Terminal-style full log viewer (pairs with {@link ErrorLog}). */
public class TerminalActivity extends AppCompatActivity {

    private TextView logView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_terminal);

        logView = findViewById(R.id.log_view);
        Button refresh = findViewById(R.id.btn_refresh);
        Button copy = findViewById(R.id.btn_copy);
        Button clear = findViewById(R.id.btn_clear);

        refresh.setOnClickListener(v -> render());

        copy.setOnClickListener(
                v -> {
                    String all = TextUtils.join("\n", ErrorLog.snapshot());
                    ClipboardManager cm =
                            (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
                    if (cm != null) {
                        cm.setPrimaryClip(ClipData.newPlainText("opencode-log", all));
                        Toast.makeText(this, "已复制 " + ErrorLog.snapshot().size() + " 行到剪贴板",
                                Toast.LENGTH_SHORT).show();
                    }
                });

        clear.setOnClickListener(
                v -> {
                    ErrorLog.clear();
                    render();
                });
    }

    @Override
    protected void onResume() {
        super.onResume();
        render();
    }

    private void render() {
        logView.setText(TextUtils.join("\n", ErrorLog.snapshot()));
    }
}