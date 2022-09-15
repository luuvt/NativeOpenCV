//
//  BlobDetectors.hpp
//  OpenCVSample_iOS
//
//  Created by Luu Tran on 06/09/2022.
//  Copyright © 2022 test. All rights reserved.
//

#ifndef BlobDetector_hpp
#define BlobDetector_hpp

#include <stdio.h>
//#include <thread>
#include <shared_mutex>
#include <opencv2/opencv.hpp>

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
  bool enabledDraw_ = true;
  std::vector<cv::Point> blob_;
  mutable std::shared_mutex mutex_;

protected:
    BlobDetector(/* args */);
    static BlobDetector* singleton_;
public:
    BlobDetector(BlobDetector &other) = delete;
    void operator=(const BlobDetector &) = delete;
    static BlobDetector *Instance();

    void setColorRadius(cv::Scalar radius);
    void setEnableDraw(bool enabled);
    void setHsvColor(cv::Scalar hsv);
    void setMinContourArea(float area);

    void process(const cv::Mat &rgpbaImage);

    std::vector<std::vector<cv::Point>> getContours();

    std::tuple<cv::Point, bool> isExistsBlobInBlobList(cv::Point p);

private:
    bool isInside(cv::Point p1, cv::Point p2, int radius);
};


#endif /* BlobDetector_hpp */
