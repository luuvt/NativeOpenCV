//
//  BlobDetector.hpp
//
//  Created by Luu Tran on 06/09/2022.
//  Copyright © 2022 test. All rights reserved.
//

#ifndef BlobDetector_hpp
#define BlobDetector_hpp

#include <stdio.h>
//#include <thread>
#include <shared_mutex>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

class BlobDetector
{
private:
    cv::Scalar lowerBound_;
    cv::Scalar upperBound_;
    cv::Scalar colorRadius_;

    cv::Mat mask_;
    cv::Mat hsvMat_;
    cv::Mat hierarchy_;
    cv::Mat pyrDownMat_;
    cv::Mat dilatedMask_;
    std::vector<std::vector<cv::Point>> contours_;

    float minContourArea_;
    bool enabledDraw_ = false;
    std::vector<std::tuple<cv::Point, int>> blob_;
    mutable std::shared_mutex mutex_;

protected:
    BlobDetector(/* args */);
    static BlobDetector* singleton_;
public:
    BlobDetector(BlobDetector &other) = delete;
    void operator=(const BlobDetector &) = delete;
    static BlobDetector *Instance();

    void setEnableDraw(bool enabled);
    void setHsvColor(cv::Scalar hsv);
    void setMinContourArea(float area);
    void setColorRadius(cv::Scalar radius);

    void setHsvLowerColor(cv::Scalar lower);
    void setHsvUpperColor(cv::Scalar upper);

    void process(const cv::Mat &rgpbaImage);

    std::vector<std::vector<cv::Point>> getContours();

    std::vector<std::tuple<cv::Point, int>> getBlobListDetected();
    std::tuple<cv::Point, int, bool> isExistsBlobInBlobList(cv::Point p);

    cv::Mat takePicture(cv::Mat &mat);

private:
    void addBLob(cv::Point p, int radius);
    bool isInside(cv::Point p1, cv::Point p2, int radius);
};


#endif /* BlobDetector_hpp */