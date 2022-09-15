#include <jni.h>
#include <string>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "BlobDetector.hpp"

#define TAG "NativeLib"

using namespace std;
using namespace cv;

extern "C" {
JNIEXPORT jstring JNICALL
Java_com_jante_nativeopencv_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

JNIEXPORT void JNICALL
Java_com_jante_nativeopencv_MainActivity_setHsvColorFromJNI(
        JNIEnv *env,
        jobject instance,
        jdouble v0, jdouble v1, jdouble v2) {

    // get Scalar hsv from raw address
    cv::Scalar hsv = cv::Scalar(v0, v1, v2);
    BlobDetector::Instance()->setHsvColor(hsv);
}

JNIEXPORT void JNICALL
Java_com_jante_nativeopencv_MainActivity_setMinContourAreaFromJNI(
        JNIEnv *env,
        jobject instance,
        jdouble minContourArea) {

    // get minContourArea from raw address
    BlobDetector::Instance()->setMinContourArea(minContourArea);

}

JNIEXPORT jfloatArray JNICALL
Java_com_jante_nativeopencv_MainActivity_getListBlobFromJNI(
        JNIEnv *env,
        jobject instance) {

    vector<vector<cv::Point>> contours = BlobDetector::Instance()->getContours();

    jclass clazz = (*env).FindClass("java/util/ArrayList");
    jobject obj = (*env).NewObject(clazz, (*env).GetMethodID(clazz, "<init>", "()V"));

    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    std::vector<cv::Point2f>centers( contours.size() );
    std::vector<float>radius( contours.size() );

    for (size_t i = 0; i < contours.size(); i++) {
        cv::approxPolyDP( contours[i], contours_poly[i], 3, true );
        cv::minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
    }

    jfloatArray arr = env->NewFloatArray(3 * contours.size());
    if (arr != nullptr) {
        if (float* ptr = env->GetFloatArrayElements(arr, NULL)) {
            for (size_t i = 0; i < contours.size(); i++) {
                ptr[i] = centers[i].x;
                ptr[i + 1] = centers[i].y;
                ptr[i + 2] = radius[i];
            }
            env->ReleaseFloatArrayElements(arr, ptr, JNI_COMMIT);
        }
    }
    return arr;
}

JNIEXPORT void JNICALL
Java_com_jante_nativeopencv_MainActivity_detectColorFromJNI(
        JNIEnv *env,
        jobject instance,
        jlong matAddr) {
    Mat &rgpbaImage = *(Mat *) matAddr;
    BlobDetector::Instance()->process(rgpbaImage);
}

JNIEXPORT void JNICALL
Java_com_jante_nativeopencv_MainActivity_adaptiveThresholdFromJNI(
        JNIEnv *env,
       jobject instance,
       jlong matAddr) {

    // get Mat from raw address
    Mat &mat = *(Mat *) matAddr;

    clock_t begin = clock();

    cv::adaptiveThreshold(mat, mat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 5);

    // log computation time to Android Logcat
    double totalTime = double(clock() - begin) / CLOCKS_PER_SEC;
    __android_log_print(ANDROID_LOG_INFO, TAG, "adaptiveThreshold computation time = %f seconds\n",
                        totalTime);
}


}