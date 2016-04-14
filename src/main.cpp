#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include "PointCloud.hpp"
#include "constant.h"

int main(int argc, char *argv[])
{
    cv::Mat rgb, depth;

    rgb = cv::imread("../data/rgb1.png");
    depth = cv::imread("../data/depth1.png", -1);
    //namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Display window", rgb );      
    PointCloud cloud;

    cloud.addImage(rgb, depth);

    pcl::visualization::PCLVisualizer viz;
    //cv::waitKey(0); 
    //cloud.saveCloud("../data/pc.pcd");
    viz.addPointCloud(cloud.getCloud());
    viz.spin();
    return 0;
}
