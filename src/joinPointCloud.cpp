/*************************************************************************
	> File Name: src/jointPointCloud.cpp
	> Author: Xiang gao
	> Mail: gaoxiang12@mails.tsinghua.edu.cn 
	> Created Time: 2015年07月22日 星期三 20时46分08秒
 ************************************************************************/

#include<iostream>
using namespace std;

#include "slamBase.h"

#include <opencv2/core/eigen.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/point_types.h>

// Eigen !
#include <Eigen/Core>
#include <Eigen/Geometry>

void
print4x4Matrix (const Eigen::Matrix4d & matrix)
{
  printf ("Rotation matrix :\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
  printf ("Translation vector :\n");
  printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}

int main( int argc, char** argv )
{
    //本节要拼合data中的两对图像
    ParameterReader pd;
    // 声明两个帧，FRAME结构请见include/slamBase.h
    FRAME frame1, frame2;
    
    //读取图像
    frame1.rgb = cv::imread("./data/xyz/rgb/1.png");
    frame2.rgb = cv::imread("./data/xyz/rgb/10.png");
    frame1.depth = cv::imread("./data/xyz/depth/1.png",-1);
    frame2.depth = cv::imread("./data/xyz/depth/10.png",-1);

    // 提取特征并计算描述子
    cout<<"extracting features"<<endl;
    string detecter = pd.getData( "detector" );
    string descriptor = pd.getData( "descriptor" );

    computeKeyPointsAndDesp( frame1, detecter, descriptor );
    computeKeyPointsAndDesp( frame2, detecter, descriptor );

    // 相机内参
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof( pd.getData( "camera.fx" ).c_str());
    camera.fy = atof( pd.getData( "camera.fy" ).c_str());
    camera.cx = atof( pd.getData( "camera.cx" ).c_str());
    camera.cy = atof( pd.getData( "camera.cy" ).c_str());
    camera.scale = atof( pd.getData( "camera.scale" ).c_str() );

    cout<<"solving pnp"<<endl;
    // 求解pnp
    RESULT_OF_PNP result = estimateMotion( frame1, frame2, camera );

    cout<<result.rvec<<endl<<result.tvec<<endl;

    // 处理result
    // 将旋转向量转化为旋转矩阵
    cv::Mat R;
    cv::Rodrigues( result.rvec, R );
    Eigen::Matrix3d r;
    cv::cv2eigen(R, r);
  
    // 将平移向量和旋转矩阵转换成变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    
    Eigen::AngleAxisd angle(r);
    cout<<"translation"<<endl;
    Eigen::Translation<double,3> trans(result.tvec.at<double>(0,0), result.tvec.at<double>(0,1), result.tvec.at<double>(0,2));
    T = angle;
    T(0,3) = result.tvec.at<double>(0,0); 
    T(1,3) = result.tvec.at<double>(0,1); 
    T(2,3) = result.tvec.at<double>(0,2);
    print4x4Matrix(T.matrix());
    Eigen::Matrix4f ATF = T.matrix().cast<float>();
    // 转换点云
    cout<<"converting image to clouds"<<endl;
    PointCloud::Ptr cloud1 = image2PointCloud( frame1.rgb, frame1.depth, camera );
    PointCloud::Ptr cloud2 = image2PointCloud( frame2.rgb, frame2.depth, camera );
    PointCloud::Ptr output1(new PointCloud());
    /*pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaximumIterations(25);
    icp.setInputSource(cloud1);
    icp.setInputTarget(cloud2);
    icp.align(*output1, ATF);
    Eigen::Matrix4d AT;
    if(icp.hasConverged()){
        std::cout<<"ICP has converged"<<std::endl;
        AT = icp.getFinalTransformation().cast<double>();
        print4x4Matrix(AT);
    }*/
    
    // 合并点云
    cout<<"combining clouds"<<endl;
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *cloud2, *output, T.inverse().matrix() );
    print4x4Matrix(T.inverse().matrix());
    //pcl::transformPointCloud( *cloud1, *output1,AT);
    *output += *cloud1;
    //*output1 += *cloud2;
    //pcl::io::savePCDFile("data/result.pcd", *output);
    //cout<<"Final result saved."<<endl;
    pcl::visualization::PCLVisualizer viz;
    int v1(0);
    int v2(1);
    viz.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    //viz.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viz.addPointCloud(output, "output", v1);
    //viz.addPointCloud(output1, "output1", v2);
    viz.spin();
    return 0;
}
