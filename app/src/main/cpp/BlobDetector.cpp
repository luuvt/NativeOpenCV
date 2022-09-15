//
//  BlobDetector.cpp
//
//  Created by Luu Tran on 06/09/2022.
//  Copyright © 2022 test. All rights reserved.
//

#include "BlobDetector.hpp"
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;

#define RADIUS_OF_CIRCLE      50
#define MINIMUM_RADIUS        10
#define MAX_SIZE_BLOB_LIST    1024 * 2

int thicknessCircle = 2;
cv::Scalar red = cv::Scalar(0, 0, 255, 0);

typedef std::vector<std::tuple<cv::Point, int>> my_tuple;

BlobDetector *BlobDetector::singleton_ = nullptr;

BlobDetector::BlobDetector(/* args */)
{
    lowerBound_ = cv::Scalar(0, 0, 0);
    upperBound_ = cv::Scalar(0, 0, 0);
    colorRadius_ = cv::Scalar(20, 50, 50, 0);
    minContourArea_ = 0.4;
    // set default setHsvColor
    setHsvColor(cv::Scalar(220, 120, 110));
}

BlobDetector *BlobDetector::Instance()
{
    if (singleton_ == nullptr) {
        singleton_ = new BlobDetector();
    }
    return singleton_;
}

void BlobDetector::setColorRadius(cv::Scalar radius)
{
    colorRadius_ = radius;
}

void BlobDetector::setHsvColor(cv::Scalar hsv)
{
    float minH = hsv[0] >= colorRadius_[0] ? hsv[0] - colorRadius_[0] : 0;
    float maxH = hsv[0] + colorRadius_[0] <= 360 ? hsv[0] + colorRadius_[0] : 360;

    lowerBound_[0] = minH;
    upperBound_[0] = maxH;

    lowerBound_[1] = hsv[1] - colorRadius_[1];
    upperBound_[1] = hsv[1] + colorRadius_[1];
    lowerBound_[2] = hsv[2] - colorRadius_[2];
    upperBound_[2] = hsv[2] + colorRadius_[2];
    lowerBound_[3] = 0;
    upperBound_[3] = 0;
}

void BlobDetector::setMinContourArea(float area)
{
    minContourArea_ = area;
}

void BlobDetector::process(const cv::Mat &rgpbaImage)
{
    if (!rgpbaImage.empty()) {
        std::shared_lock lock(mutex_);

        cv::pyrDown(rgpbaImage, pyrDownMat_);
        cv::pyrDown(pyrDownMat_, pyrDownMat_);

        cv::cvtColor(pyrDownMat_, hsvMat_, COLOR_RGB2HSV_FULL);

        cv::inRange(hsvMat_, lowerBound_, upperBound_, mask_);
        cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
        cv::dilate(mask_, dilatedMask_, kernel);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dilatedMask_, contours, hierarchy_, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
        std::vector<cv::Point2f>centers( contours.size() );
        std::vector<float>radius( contours.size() );

        // Find max contour area;
        float area;
        float maxArea = 0;

        for (size_t i = 0; i < contours.size(); i++) {
            area = cv::contourArea(contours[i]);
            if (area > maxArea) maxArea = area;
        }

        // Filter contours by area and resize to fit the original image size
        contours_.clear();
        for (size_t i = 0; i < contours.size(); i++) {
            if (cv::contourArea(contours[i]) > minContourArea_ * maxArea) {
                // Calculates the per-element scaled product of two arrays.
                cv::multiply(contours[i], cv::Scalar(4,4), contours[i]);
                // Approximates a polygonal curve(s) with the specified precision.
                cv::approxPolyDP( contours[i], contours_poly[i], 3, true );
                // Find the circumcircle of an object
                cv::minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
                if (radius[i] < MINIMUM_RADIUS ) radius[i] = MINIMUM_RADIUS;

                addBLob(centers[i], radius[i]);

                if (enabledDraw_) {
                    // Draw circle if enabled
                    cv::circle( rgpbaImage, centers[i], (int)radius[i], red, thicknessCircle );
                }

                contours_.push_back(contours[i]);
            }
        }

        if (blob_.size() > MAX_SIZE_BLOB_LIST) blob_.clear();
    }
}

void BlobDetector::addBLob(cv::Point p, int radius)
{
    std::shared_lock lock(mutex_);
    for (my_tuple::const_iterator i = blob_.begin(); i != blob_.end(); ++i) {
        if (isInside(get<0>(*i), p, RADIUS_OF_CIRCLE + radius)) {
            return;
        }
    }
    blob_.push_back(std::make_tuple(p, radius));
}

void BlobDetector::setEnableDraw(bool enabled)
{
    enabledDraw_ = enabled;
}

std::vector<std::vector<cv::Point>> BlobDetector::getContours()
{
    std::shared_lock lock(mutex_);
    return contours_;
}

std::tuple<cv::Point, int, bool> BlobDetector::isExistsBlobInBlobList(cv::Point p)
{
    std::shared_lock lock(mutex_);

    bool isExists = false;
    cv::Point blob;
    int radius;
    for (size_t i = 0; i < blob_.size(); i++) {
        if (isInside(get<0>(blob_[i]), p, RADIUS_OF_CIRCLE)) {
            isExists = true;
            blob = get<0>(blob_[i]);
            radius = get<1>(blob_[i]);
            break;
        }
    }
    return {blob, radius, isExists};
}

bool BlobDetector::isInside(cv::Point p1, cv::Point p2, int radius)
{
    return (p2.x - p1.x) * (p2.x - p1.x) +
           (p2.y - p1.y) * (p2.y - p1.y) <= radius * radius;
}

std::vector<std::tuple<cv::Point, int>> BlobDetector::getBlobListDetected()
{
    std::shared_lock lock(mutex_);
    auto blobRes = blob_;
    blob_.clear();

    return blobRes;
}

void BlobDetector::setHsvLowerColor(cv::Scalar lower)
{
    lowerBound_ = lower;
}

void BlobDetector::setHsvUpperColor(cv::Scalar upper)
{
    upperBound_ = upper;
}