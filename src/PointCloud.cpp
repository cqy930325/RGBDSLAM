#include "PointCloud.hpp"
#include "constant.h"

PointCloud::PointCloud():pc(new point_cloud_t){}

PointCloud::~PointCloud(){
    pc->points.clear();
}

void PointCloud::addImage(cv::Mat &rgb, cv::Mat &depth){
    int i,j;
    for (i = 0; i < depth.rows; ++i) {
        for (j = 0; j < depth.cols; ++j) {
            ushort d = depth.ptr<ushort>(i)[j];
            if (d == 0) {
                continue; 
            }
            point_t p;
            p.z = double(d)/camera_factor;
            p.x = (j - camera_cx) * p.z / camera_fx;
            p.y = (i - camera_cy) * p.z / camera_fy;
            p.b = rgb.ptr<uchar>(i)[j*3];
            p.g = rgb.ptr<uchar>(i)[j*3+1];
            p.r = rgb.ptr<uchar>(i)[j*3+2];
            pc->points.push_back(p);
        } 
    }
}

void PointCloud::saveCloud(std::string path){
    pc->height = 1;
    pc->width = pc->points.size();
    pc->is_dense = false;
    pcl::io::savePCDFile(path, *pc);
}

point_cloud_t::Ptr& PointCloud::getCloud(){
    return pc;
}
