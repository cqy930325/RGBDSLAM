#include <iostream>
#include <string>
#include <stdio.h>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "PointCloud.hpp"
#include "FrameMatcher.hpp"

std::string rgb_path = "./data/rgb_png/";
std::string depth_path = "./data/depth_png/";
int min_inliers = 5;
double max_norm = 0.3;
int min_good_match = 10;

struct state_t{
    PointCloud cloud;
    FrameMatcher fm;
    int read_count;
    boost::mutex cloud_mutex;
};

void match(state_t* state){
    while(1){
        state->cloud_mutex.lock();
        if(state->read_count > 782){
            state->cloud_mutex.unlock();
            break;
        }
        std::cout<<"enter thread"<<state->read_count<<std::endl;
        frame_t f;
        f.rgb = cv::imread(rgb_path+std::to_string(state->read_count)+".png");
        f.depth = cv::imread(depth_path+std::to_string(state->read_count)+".png", -1);
        match_result_t result;
        state->fm.matchFrame(f, &result, false);
        if(result.inliers < min_inliers){
            std::cout<<"inlier skip"<<std::endl;
            state->read_count++;
            state->cloud_mutex.unlock();
            continue;
        }
        double norm = fabs(std::min(cv::norm(result.rvec), 2*M_PI-cv::norm(result.rvec)))+ fabs(cv::norm(result.tvec));  
        //std::cout<<"norm: "<<norm<<std::endl;
        if(norm >= max_norm){
            std::cout<<"norm skip"<<std::endl;
            state->read_count++;
            state->cloud_mutex.unlock();
            continue;
        }
        tmat_t trans;
        state->fm.convertToTMat(&trans, &result);
        //std::cout<<"T = "<<trans.matrix()<<std::endl;
        state->cloud.addFrame(f, &trans, false);
        state->fm.updateFrame(f);
        state->read_count++;
        std::cout<<"leave"<<std::endl;
        state->cloud_mutex.unlock();
    }
}


int main(int argc, char *argv[])
{
    state_t state;
    frame_t f1;
    f1.rgb = cv::imread(rgb_path+"1.png");
    f1.depth = cv::imread(depth_path+"1.png", -1);
    state.fm.matchFrame(f1, nullptr, true);
    state.cloud.addFrame(f1, nullptr, true);
    state.read_count = 2;

    pcl::visualization::PCLVisualizer viz;
    viz.addPointCloud(state.cloud.getCloud(), "rgbd");

    boost::thread matchThread(match, &state);
    matchThread.detach();

    while(1){
        state.cloud_mutex.lock();
        viz.updatePointCloud(state.cloud.getCloud(), "rgbd");
        viz.spinOnce(100);
        state.cloud_mutex.unlock();
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    return 0;
}
