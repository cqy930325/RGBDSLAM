#include <iostream>
#include <vector>
#include <limits>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "constant.h"

int main(int argc, char *argv[])
{
    cv::Mat rgb1 = cv::imread("../data/rgb1.png");
    cv::Mat rgb2 = cv::imread("../data/rgb2.png");
    cv::Mat depth1 = cv::imread("../data/depth1.png");
    cv::Mat depth2 = cv::imread("../data/depth2.png");

    cv::OrbFeatureDetector _detector(500, 1.0f, 2, 10, 0, 2, 0, 10);
    cv::OrbDescriptorExtractor _descriptor;

    std::vector<cv::KeyPoint> kp1, kp2;
    _detector.detect(rgb1, kp1);
    _detector.detect(rgb2, kp2);

    //cv::Mat dispImg;
    //cv::drawKeypoints(rgb1, kp1, dispImg, cv::Scalar::all(-1.0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //cv::imshow("keypoints", dispImg);
    //cv::waitKey(0);

    cv::Mat desp1, desp2;
    _descriptor.compute(rgb1, kp1, desp1);
    _descriptor.compute(rgb2, kp2, desp2);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher  matcher; 
    matcher.match(desp1, desp2, matches);


    std::vector<cv::DMatch> good_matches;
    double min_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < matches.size(); ++i) {
        if(matches[i].distance < min_dist){
            min_dist = matches[i].distance;
        }
    }
    min_dist = min_dist*4;
    for (size_t i = 0; i < matches.size(); ++i) {
        if(matches[i].distance < min_dist){
            good_matches.push_back(matches[i]);
        }
    }
    /*
     *cv::Mat imgMatches;
     *cv::drawMatches(rgb1, kp1, rgb2, kp2, good_matches, imgMatches);
     *cv::imshow("matches", imgMatches);
     *cv::waitKey(0);
     */

    std::vector<cv::Point3f> pts_obj;
    std::vector<cv::Point2f> pts_img;

    for (size_t i = 0; i < good_matches.size(); ++i) {
        cv::Point2f p = kp1[good_matches[i].queryIdx].pt;
        ushort d = depth1.ptr(int(p.y))[int(p.x)];
        if(d == 0){
            continue;
        }
        pts_img.push_back(cv::Point2f(kp2[good_matches[i].trainIdx].pt));
        cv::Point3f pt(p.x, p.y, d);
        cv::Point3f pd;
        pd.z = double(pt.z) / camera_factor;
        pd.x = (pt.x - camera_cx)*pd.z / camera_fx;
        pd.y = (pt.y - camera_cy)*pd.z / camera_fy;
        pts_obj.push_back(pd);
    }
    double camera_matrix_data[3][3] = {
        {camera_fx, 0, camera_cx},
        {0, camera_fy, camera_cy},
        {0, 0, 1}
    };

    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );
    std::cout<<rvec<<std::endl;  
    std::cout<<tvec<<std::endl;  
    std::vector< cv::DMatch > matchesShow;
    cv::Mat imgMatches;
    for (int i=0; i<inliers.rows; i++)
    {
        matchesShow.push_back( good_matches[inliers.ptr<int>(i)[0]] );    
    }
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matchesShow, imgMatches );
    cv::imshow( "inlier matches", imgMatches );
    cv::waitKey( 0 ); 
    return 0;
}
