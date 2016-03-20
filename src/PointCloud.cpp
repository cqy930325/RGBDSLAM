#include "PointCloud.hpp"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

PointCloud::PointCloud():pc(new point_cloud_t){}

PointCloud::~PointCloud(){
    pc->points.clear();
}

void PointCloud::addFrame(frame_t &new_frame, tmat_t *T, bool init){
    point_cloud_t::Ptr new_pc(new point_cloud_t);
    point_cloud_t::Ptr output(new point_cloud_t);
    image2cloud(new_frame, new_pc);
    if(!init){
        pcl::transformPointCloud(*pc, *output, T->matrix());
    }
    *new_pc += *output;
    if(!init){
        static pcl::VoxelGrid<point_t> voxel;
        double gridsize = 0.01;
        voxel.setLeafSize(gridsize, gridsize, gridsize);
        voxel.setInputCloud(new_pc);
        point_cloud_t::Ptr tmp(new point_cloud_t);
        voxel.filter(*tmp);
        pc->swap(*tmp);
    }
    else{
        pc->swap(*new_pc);
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

void PointCloud::image2cloud(frame_t &new_frame, point_cloud_t::Ptr& ret){
    int i,j;
    for (i = 0; i < new_frame.depth.rows; ++i) {
        for (j = 0; j < new_frame.depth.cols; ++j) {
            ushort d = new_frame.depth.ptr<ushort>(i)[j];
            if (d == 0) {
                continue; 
            }
            //std::cout<<"d: "<<d<<std::endl;
            point_t p;
            p.z = double(d)/camera_factor;
            p.x = (j - camera_cx) * p.z / camera_fx;
            p.y = (i - camera_cy) * p.z / camera_fy;
            p.b = new_frame.rgb.ptr<uchar>(i)[j*3];
            p.g = new_frame.rgb.ptr<uchar>(i)[j*3+1];
            p.r = new_frame.rgb.ptr<uchar>(i)[j*3+2];
            ret->points.push_back(p);
        } 
    }
}
