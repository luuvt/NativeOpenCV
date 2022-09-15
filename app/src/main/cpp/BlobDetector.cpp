//
//  BlobDetector.cpp
//  OpenCVSample_iOS
//
//  Created by Luu Tran on 06/09/2022.
//  Copyright © 2022 test. All rights reserved.
//

#include "BlobDetector.hpp"
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;

#define MAX_SIZE_BLOB_LIST    1024 * 2
#define RADIUS_OF_CIRCLE      50
#define MINIMUM_RADIUS        10

BlobDetector *BlobDetector::singleton_ = nullptr;

BlobDetector::BlobDetector(/* args */)
{
  lowerBound_ = cv::Scalar(0, 0, 0);
  upperBound_ = cv::Scalar(0, 0, 0);
  colorRadius_ = cv::Scalar(25, 50, 50, 0);
  minContourArea_ = 0.1;
}

BlobDetector *BlobDetector::Instance() {
    if (singleton_ == nullptr) {
        singleton_ = new BlobDetector();
    }
    return singleton_;
}

void BlobDetector::setColorRadius(cv::Scalar radius) {
  colorRadius_ = radius;
}

void BlobDetector::setHsvColor(cv::Scalar hsv) {
  float minH = hsv[0] >= colorRadius_[0] ? hsv[0] - colorRadius_[0] : 0;
  float maxH = hsv[0] + colorRadius_[0] <= 255 ? hsv[0] + colorRadius_[0] : 255;

  lowerBound_[0] = minH;
  upperBound_[0] = maxH;

  lowerBound_[1] = hsv[1] - colorRadius_[1];
  upperBound_[1] = hsv[1] + colorRadius_[1];
  lowerBound_[2] = hsv[2] - colorRadius_[2];
  upperBound_[2] = hsv[2] + colorRadius_[2];
  lowerBound_[3] = 0;
  upperBound_[3] = 0;
}

void BlobDetector::setMinContourArea(float area) {
  minContourArea_ = area;
}

void BlobDetector::process(const cv::Mat &rgpbaImage) {
    if (!rgpbaImage.empty()) {
        std::shared_lock lock(mutex_);
        cv::pyrDown(rgpbaImage, pyrDownMat_);
        cv::pyrDown(pyrDownMat_, pyrDownMat_);

        cv::cvtColor(pyrDownMat_, hsvMat_, COLOR_RGB2HSV_FULL);

        cv::inRange(hsvMat_, lowerBound_, upperBound_, mask_);
        cv::Mat kernel = cv::Mat::ones(5, 5, CV_8U);
        cv::dilate(mask_, dilatedMask_, kernel);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dilatedMask_, contours, hierarchy_, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
        std::vector<cv::Point2f>centers( contours.size() );
        std::vector<float>radius( contours.size() );

        // Find max contour area;
        float maxArea = 0;
        float area;
        for (size_t i = 0; i < contours.size(); i++) {
          area = cv::contourArea(contours[i]);
          if (area > maxArea) maxArea = area;
        }

        // Filter contours by area and resize to fit the original image size
        contours_.clear();
        for (size_t i = 0; i < contours.size(); i++) {
          if (cv::contourArea(contours[i]) > minContourArea_ * maxArea) {
            cv::multiply(contours[i], Scalar(4,4), contours[i]);

            if (enabledDraw_) {
                cv::approxPolyDP( contours[i], contours_poly[i], 3, true );
                cv::minEnclosingCircle( contours_poly[i], centers[i], radius[i] );

                if (radius[i] < MINIMUM_RADIUS ) radius[i] = MINIMUM_RADIUS;
                cv::circle( rgpbaImage, centers[i], (int)radius[i], (0, 100, 100), 4 );
            }
            contours_.push_back(contours[i]);
          }
        }

        blob_.insert(std::end(blob_), std::begin(centers), std::end(centers));

        if (blob_.size() > MAX_SIZE_BLOB_LIST) blob_.clear();
    }
}

void BlobDetector::setEnableDraw(bool enabled) {
    enabledDraw_ = enabled;
}
std::vector<std::vector<cv::Point>> BlobDetector::getContours() {
    std::shared_lock lock(mutex_);
  return contours_;
}

std::tuple<cv::Point, bool> BlobDetector::isExistsBlobInBlobList(cv::Point p) {
    std::shared_lock lock(mutex_);
  
  bool isExists = false;
  cv::Point blob;
  for (size_t i = 0; i < blob_.size(); i++) {
    if (isInside(blob_[i], p, RADIUS_OF_CIRCLE)) {
      blob = blob_[i];
      isExists = true;
      break;
    }
  }
  return {blob, isExists};
}

bool BlobDetector::isInside(cv::Point p1, cv::Point p2, int radius) {
  return (p2.x - p1.x) * (p2.x - p1.x) +
                (p2.y - p1.y) * (p2.y - p1.y) <= radius * radius;
}
