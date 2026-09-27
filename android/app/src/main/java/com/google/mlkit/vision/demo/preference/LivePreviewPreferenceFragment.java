/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.preference;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Bundle;
import android.preference.EditTextPreference;
import android.preference.ListPreference;
import android.preference.MultiSelectListPreference;
import android.preference.Preference;
import android.preference.PreferenceCategory;
import android.preference.PreferenceFragment;
import androidx.annotation.StringRes;
import android.widget.Toast;
import com.google.mlkit.vision.demo.CameraSource;
import com.google.mlkit.vision.demo.CameraSource.SizePair;
import com.google.mlkit.vision.demo.R;
import com.google.mlkit.vision.demo.java.ModelCatalog;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Configures live preview demo settings. */
public class LivePreviewPreferenceFragment extends PreferenceFragment {

  protected boolean isCameraXSetting;

  private static final int REQ_IMPORT_ONNX = 1001;

  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    addPreferencesFromResource(R.xml.preference_live_preview_quickstart);
    refreshModelListPreference();
    setUpCameraPreferences();

    Preference importPref = findPreference(getString(R.string.pref_key_import_onnx));
    if (importPref != null) {
      importPref.setOnPreferenceClickListener(pref -> {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"application/octet-stream", "*/*"});
        try {
          startActivityForResult(intent, REQ_IMPORT_ONNX);
        } catch (Exception e) {
          Toast.makeText(getActivity(), "无法打开文件选择器: " + e.getMessage(),
              Toast.LENGTH_LONG).show();
        }
        return true;
      });
    }
  }

  /** Fills the unified model MultiSelectListPreference from {@link ModelCatalog}. */
  private void refreshModelListPreference() {
    MultiSelectListPreference pref =
        (MultiSelectListPreference) findPreference(getString(R.string.pref_key_ncnn_models));
    if (pref == null) return;
    List<ModelCatalog.Entry> all = ModelCatalog.all(getActivity());
    String[] entries = new String[all.size()];
    String[] values = new String[all.size()];
    for (int i = 0; i < all.size(); i++) {
      entries[i] = all.get(i).label;
      values[i] = all.get(i).value;
    }
    pref.setEntries(entries);
    pref.setEntryValues(values);

    Set<String> selected = PreferenceUtils.getSelectedModels(getActivity());
    if (selected != null) {
      pref.setValues(new LinkedHashSet<>(selected));
    }
    pref.setSummary(getString(R.string.pref_summary_ncnn_models)
        + "（" + (selected == null ? 1 : selected.size()) + " 个已选）");
    pref.setOnPreferenceChangeListener((preference, newValue) -> {
      @SuppressWarnings("unchecked")
      Set<String> newly = (Set<String>) newValue;
      String first = ModelCatalog.firstSelected(getActivity(), newly);
      String run = (first == null) ? "未选择，不启动" : ModelCatalog.labelFor(getActivity(), first);
      preference.setSummary(getString(R.string.pref_summary_ncnn_models)
          + "（" + newly.size() + " 个已选，将运行: " + run + "）");
      return true;
    });
  }

  @Override
  public void onActivityResult(int requestCode, int resultCode, Intent data) {
    if (requestCode == REQ_IMPORT_ONNX && resultCode == Activity.RESULT_OK && data != null) {
      int imported = 0;
      int rejected = 0;
      try {
        List<Uri> uris = new ArrayList<>();
        if (data.getData() != null) {
          uris.add(data.getData());
        }
        if (data.getClipData() != null) {
          for (int i = 0; i < data.getClipData().getItemCount(); i++) {
            uris.add(data.getClipData().getItemAt(i).getUri());
          }
        }
        File dir = ModelCatalog.importDir(getActivity());
        if (!dir.exists()) dir.mkdirs();
        for (Uri uri : uris) {
          String name = queryDisplayName(uri);
          if (name == null) name = "model_" + System.currentTimeMillis() + ".onnx";
          if (!name.toLowerCase().endsWith(".onnx")) {
            Toast.makeText(getActivity(), "「" + name + "」不是 .onnx 文件，已跳过",
                Toast.LENGTH_LONG).show();
            rejected++;
            continue;
          }
          long firstByte = peekFirstByte(uri);
          if (firstByte != 0x08) {
            Toast.makeText(getActivity(), "「" + name + "」不是有效的 ONNX 模型文件，已跳过",
                Toast.LENGTH_LONG).show();
            rejected++;
            continue;
          }
          try (InputStream in = getActivity().getContentResolver().openInputStream(uri);
               OutputStream out = new FileOutputStream(new File(dir, name))) {
            byte[] buf = new byte[65536];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
          }
          imported++;
        }
      } catch (Exception e) {
        Toast.makeText(getActivity(), "导入失败: " + e.getMessage(), Toast.LENGTH_LONG).show();
      }
      if (imported > 0) {
        Toast.makeText(getActivity(), "已导入 " + imported + " 个 ONNX 模型", Toast.LENGTH_LONG).show();
        refreshModelListPreference();
      }
      if (imported == 0 && rejected > 0) {
        Toast.makeText(getActivity(), "没有导入任何文件（全部被拒绝）", Toast.LENGTH_LONG).show();
      }
      return;
    }
    super.onActivityResult(requestCode, resultCode, data);
  }

  /**
   * Returns the first byte of the URI content, or -1 when unreadable. A real
   * ONNX pb file starts with {@code 0x08} (protobuf field-1 varint), so this
   * cheap check rejects text files / renamed junk.
   */
  private long peekFirstByte(Uri uri) {
    try (InputStream in = getActivity().getContentResolver().openInputStream(uri)) {
      if (in == null) return -1;
      return in.read();
    } catch (Exception e) {
      return -1;
    }
  }

  private String queryDisplayName(Uri uri) {
    String name = null;
    try (android.database.Cursor c = getActivity().getContentResolver().query(
        uri, null, null, null, null)) {
      if (c != null && c.moveToFirst()) {
        int idx = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME);
        if (idx >= 0) name = c.getString(idx);
      }
    } catch (Exception ignore) {}
    return name;
  }

  private void setUpCameraPreferences() {
    PreferenceCategory cameraPreference =
        (PreferenceCategory) findPreference(getString(R.string.pref_category_key_camera));

    if (isCameraXSetting) {
      cameraPreference.removePreference(
          findPreference(getString(R.string.pref_key_rear_camera_preview_size)));
      cameraPreference.removePreference(
          findPreference(getString(R.string.pref_key_front_camera_preview_size)));
      setUpCameraXTargetAnalysisSizePreference();
    } else {
      cameraPreference.removePreference(
          findPreference(getString(R.string.pref_key_camerax_target_resolution)));
      setUpCameraPreviewSizePreference(
          R.string.pref_key_rear_camera_preview_size,
          R.string.pref_key_rear_camera_picture_size,
          CameraSource.CAMERA_FACING_BACK);
      setUpCameraPreviewSizePreference(
          R.string.pref_key_front_camera_preview_size,
          R.string.pref_key_front_camera_picture_size,
          CameraSource.CAMERA_FACING_FRONT);
    }
  }

  private void setUpCameraPreviewSizePreference(
      @StringRes int previewSizePrefKeyId, @StringRes int pictureSizePrefKeyId, int cameraId) {
    ListPreference previewSizePreference =
        (ListPreference) findPreference(getString(previewSizePrefKeyId));

    Camera camera = null;
    try {
      camera = Camera.open(cameraId);

      List<SizePair> previewSizeList = CameraSource.generateValidPreviewSizeList(camera);
      String[] previewSizeStringValues = new String[previewSizeList.size()];
      Map<String, String> previewToPictureSizeStringMap = new HashMap<>();
      for (int i = 0; i < previewSizeList.size(); i++) {
        SizePair sizePair = previewSizeList.get(i);
        previewSizeStringValues[i] = sizePair.preview.toString();
        if (sizePair.picture != null) {
          previewToPictureSizeStringMap.put(
              sizePair.preview.toString(), sizePair.picture.toString());
        }
      }
      previewSizePreference.setEntries(previewSizeStringValues);
      previewSizePreference.setEntryValues(previewSizeStringValues);

      if (previewSizePreference.getEntry() == null) {
        // First time of opening the Settings page.
        SizePair sizePair =
            CameraSource.selectSizePair(
                camera,
                CameraSource.DEFAULT_REQUESTED_CAMERA_PREVIEW_WIDTH,
                CameraSource.DEFAULT_REQUESTED_CAMERA_PREVIEW_HEIGHT);
        String previewSizeString = sizePair.preview.toString();
        previewSizePreference.setValue(previewSizeString);
        previewSizePreference.setSummary(previewSizeString);
        PreferenceUtils.saveString(
            getActivity(),
            pictureSizePrefKeyId,
            sizePair.picture != null ? sizePair.picture.toString() : null);
      } else {
        previewSizePreference.setSummary(previewSizePreference.getEntry());
      }

      previewSizePreference.setOnPreferenceChangeListener(
          (preference, newValue) -> {
            String newPreviewSizeStringValue = (String) newValue;
            previewSizePreference.setSummary(newPreviewSizeStringValue);
            PreferenceUtils.saveString(
                getActivity(),
                pictureSizePrefKeyId,
                previewToPictureSizeStringMap.get(newPreviewSizeStringValue));
            return true;
          });

    } catch (Exception e) {
      // If there's no camera for the given camera id, hide the corresponding preference.
      ((PreferenceCategory) findPreference(getString(R.string.pref_category_key_camera)))
          .removePreference(previewSizePreference);
    } finally {
      if (camera != null) {
        camera.release();
      }
    }
  }

  private void setUpCameraXTargetAnalysisSizePreference() {
    ListPreference pref =
        (ListPreference) findPreference(getString(R.string.pref_key_camerax_target_resolution));
    String[] entries =
        new String[] {
          "2000x2000",
          "1600x1600",
          "1200x1200",
          "1000x1000",
          "800x800",
          "600x600",
          "400x400",
          "200x200",
          "100x100",
        };
    pref.setEntries(entries);
    pref.setEntryValues(entries);
    pref.setSummary(pref.getEntry() == null ? "Default" : pref.getEntry());
    pref.setOnPreferenceChangeListener(
        (preference, newValue) -> {
          String newStringValue = (String) newValue;
          pref.setSummary(newStringValue);
          PreferenceUtils.saveString(
              getActivity(), R.string.pref_key_camerax_target_resolution, newStringValue);
          return true;
        });
  }

  private void setUpListPreference(@StringRes int listPreferenceKeyId) {
    ListPreference listPreference = (ListPreference) findPreference(getString(listPreferenceKeyId));
    listPreference.setSummary(listPreference.getEntry());
    listPreference.setOnPreferenceChangeListener(
        (preference, newValue) -> {
          int index = listPreference.findIndexOfValue((String) newValue);
          listPreference.setSummary(listPreference.getEntries()[index]);
          return true;
        });
  }
}
