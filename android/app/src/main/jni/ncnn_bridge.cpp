// JNI bridge between the PoseEstimate app and the ncnn YOLO11 engine.
// Takes an Android Bitmap (ARGB int array), runs the selected task
// (det/seg/pose/cls/obb) and returns a flat float array:
//   [ per object: label, prob, x0, y0, x1, y1, cx, cy, w, h, angle, nkpts,
//     then nkpts * 3 floats (x, y, prob) ]
// All coordinates are in ORIGINAL BITMAP pixel space (like our Java graphics).
//
// Copyright 2026, based on Tencent ncnn sample (BSD 3-Clause).

#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>

#include "yolo11.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static YOLO11* g_yolo11 = 0;
static ncnn::Mutex lock;

static YOLO11* g_det = 0;
static YOLO11* g_seg = 0;
static YOLO11* g_pose = 0;
static YOLO11* g_cls = 0;
static YOLO11* g_obb = 0;

static int g_target_size = 640;
static bool g_use_gpu = false;

static void release_detectors()
{
    delete g_det;  g_det = 0;
    delete g_seg;  g_seg = 0;
    delete g_pose; g_pose = 0;
    delete g_cls;  g_cls = 0;
    delete g_obb;  g_obb = 0;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    // NOTE: do NOT create the GPU instance here. Vulkan drivers are flaky on
    // some devices and initializing at library load time can crash the process
    // even in CPU mode. We create/destroy it only in loadModel when the user
    // actually selects GPU compute.
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    {
        ncnn::MutexLockGuard g(lock);
        release_detectors();
    }
    ncnn::destroy_gpu_instance();
}

// native boolean loadModel(AssetManager mgr, int taskid, int modelid, int cpugpu)
//   taskid: 0=det 1=seg 2=pose 3=cls 4=obb
//   modelid: 0..8 => n/s/m at 320/480/640
//   cpugpu: 0=cpu 1=gpu 2=gpu(turnip)
JNIEXPORT jboolean JNICALL
Java_com_google_mlkit_vision_demo_java_ncnn_NcnnYolo11_loadModel(
        JNIEnv* env, jobject thiz, jobject assetManager, jint taskid, jint modelid, jint cpugpu)
{
    if (taskid < 0 || taskid > 4 || modelid < 0 || modelid > 8 || cpugpu < 0 || cpugpu > 2)
        return JNI_FALSE;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    const char* tasknames[5] = {"", "_seg", "_pose", "_cls", "_obb"};
    const char* modeltypes[9] = {"n", "s", "m", "n", "s", "m", "n", "s", "m"};

    std::string parampath = std::string("yolo11") + modeltypes[(int)modelid] + tasknames[(int)taskid] + ".ncnn.param";
    std::string modelpath  = std::string("yolo11") + modeltypes[(int)modelid] + tasknames[(int)taskid] + ".ncnn.bin";

    bool use_gpu = (int)cpugpu == 1;
    bool use_turnip = (int)cpugpu == 2;

    int target_size = 320;
    if ((int)modelid >= 3) target_size = 480;
    if ((int)modelid >= 6) target_size = 640;

    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu || use_turnip)
        {
            if (ncnn::create_gpu_instance() != 0)
            {
                // GPU driver/driver init failed -> fall back to CPU instead of
                // failing the whole model. This fixes "model won't load" errors
                // on devices with flaky Vulkan drivers.
                __android_log_print(ANDROID_LOG_WARN, "ncnn",
                    "create_gpu_instance failed, falling back to CPU");
                use_gpu = false;
                use_turnip = false;
                ncnn::destroy_gpu_instance();
            }
        }
        else
        {
            ncnn::destroy_gpu_instance();
        }

        // Store the requested detector, load on first detect to keep the UI
        // thread responsive (lazy load like our YoloDetector).
        YOLO11** slot = 0;
        switch (taskid)
        {
            case 0: slot = &g_det;  break;
            case 1: slot = &g_seg;  break;
            case 2: slot = &g_pose; break;
            case 3: slot = &g_cls;  break;
            case 4: slot = &g_obb;  break;
        }

        // If the compute mode changed we must rebuild the detector (ncnn bakes
        // vulkan/cpu into its runtime options at load time).
        if (*slot && g_use_gpu != (use_gpu || use_turnip))
        {
            delete *slot;
            *slot = 0;
        }

        if (!*slot)
        {
            if (taskid == 0) *slot = new YOLO11_det;
            if (taskid == 1) *slot = new YOLO11_seg;
            if (taskid == 2) *slot = new YOLO11_pose;
            if (taskid == 3) *slot = new YOLO11_cls;
            if (taskid == 4) *slot = new YOLO11_obb;

            (*slot)->load(mgr, parampath.c_str(), modelpath.c_str(), use_gpu || use_turnip);
        }
        (*slot)->set_det_target_size(target_size);
        g_yolo11 = *slot;
        g_target_size = target_size;
        g_use_gpu = use_gpu || use_turnip;
    }

    return JNI_TRUE;
}

// native float[] detect(int[] argb, int width, int height)
// Returns the flat float array described in the header comment.
JNIEXPORT jfloatArray JNICALL
Java_com_google_mlkit_vision_demo_java_ncnn_NcnnYolo11_detect(
        JNIEnv* env, jobject thiz, jintArray argb, jint width, jint height)
{
    if (!g_yolo11)
        return 0;

    const int img_w = (int)width;
    const int img_h = (int)height;

    jint* pixels = env->GetIntArrayElements(argb, 0);
    if (!pixels)
        return 0;

    // Convert ARGB int array -> RGB cv::Mat (swap channel order).
    cv::Mat rgb(img_h, img_w, CV_8UC3);
    for (int y = 0; y < img_h; y++)
    {
        const jint* src = pixels + y * img_w;
        uchar* dst = rgb.ptr(y);
        for (int x = 0; x < img_w; x++)
        {
            jint c = src[x];
            dst[0] = (c >> 16) & 0xFF;
            dst[1] = (c >> 8)  & 0xFF;
            dst[2] = c         & 0xFF;
        }
    }
    env->ReleaseIntArrayElements(argb, pixels, JNI_ABORT);

    std::vector<Object> objects;
    {
        ncnn::MutexLockGuard g(lock);
        g_yolo11->detect(rgb, objects);
    }

    // Pack results. Keypoints count differs per model (pose=17).
    int total_kpts = 0;
    for (size_t i = 0; i < objects.size(); i++)
        total_kpts += (int)objects[i].keypoints.size();

    // Layout: header count + per object 11 header floats + kpts*3
    const int per_obj_head = 11;
    std::vector<float> out;
    out.reserve(1 + objects.size() * (per_obj_head + 0) + total_kpts * 3);
    out.push_back((float)objects.size());

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        out.push_back((float)obj.label);
        out.push_back(obj.prob);
        out.push_back(obj.rect.x);             // x0
        out.push_back(obj.rect.y);             // y0
        out.push_back(obj.rect.x + obj.rect.width);  // x1
        out.push_back(obj.rect.y + obj.rect.height); // y1
        // rotated rect (obb / unused others)
        out.push_back(obj.rrect.center.x);     // cx
        out.push_back(obj.rrect.center.y);     // cy
        out.push_back(obj.rrect.size.width);   // w
        out.push_back(obj.rrect.size.height);  // h
        out.push_back(obj.rrect.angle);        // angle
        out.push_back((float)obj.keypoints.size()); // nkpts
        for (size_t k = 0; k < obj.keypoints.size(); k++)
        {
            out.push_back(obj.keypoints[k].p.x);
            out.push_back(obj.keypoints[k].p.y);
            out.push_back(obj.keypoints[k].prob);
        }
    }

    jfloatArray ret = env->NewFloatArray((jsize)out.size());
    if (ret)
        env->SetFloatArrayRegion(ret, 0, (jsize)out.size(), out.data());
    return ret;
}

// native void release()
JNIEXPORT void JNICALL
Java_com_google_mlkit_vision_demo_java_ncnn_NcnnYolo11_release(JNIEnv* env, jobject thiz)
{
    ncnn::MutexLockGuard g(lock);
    release_detectors();
    g_yolo11 = 0;
}

} // extern "C"