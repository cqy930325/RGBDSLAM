#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include "PointCloud.hpp"
#include "constant.h"
#include "FrameMatcher.hpp"

int main(int argc, char *argv[])
{
    /*
     *cv::Mat rgb, depth;
     *rgb = cv::imread("../data/rgb1.png");
     *depth = cv::imread("../data/depth1.png", -1);
     *PointCloud cloud;
     *cloud.addImage(rgb, depth);
     *pcl::visualization::PCLVisualizer viz;
     *viz.addPointCloud(cloud.getCloud());
     *viz.spin();
     */
    FrameMatcher fm;
    frame_t f1;
    frame_t f2;
    f1.rgb = cv::imread("../data/rgb1.png");
    f1.depth = cv::imread("../data/depth1.png");
    f2.rgb = cv::imread("../data/rgb2.png");
    f2.depth = cv::imread("../data/depth2.png");

    fm.updateFrame(f1, nullptr, true);

    virtual_odo_t vo;
    fm.updateFrame(f2, &vo, false);

    std::cout<<"rvec"<<vo.rvec<<std::endl;
    std::cout<<"tvec"<<vo.tvec<<std::endl;
    return 0;
}
