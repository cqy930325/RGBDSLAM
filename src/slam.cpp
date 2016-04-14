/*************************************************************************
  > File Name: rgbd-slam-tutorial-gx/part V/src/visualOdometry.cpp
  > Author: xiang gao
  > Mail: gaoxiang12@mails.tsinghua.edu.cn
  > Created Time: 2015年08月15日 星期六 15时35分42秒
 * add g2o slam end to visual odometry
 * add keyframe and simple loop closure
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "slamBase.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

/*#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>
#include <octomap/math/Pose6D.h>
*/
#include <ros/ros.h>
#include <ros/publisher.h>
#include <pcl_ros/publisher.h>
#include <pcl_ros/point_cloud.h>
#include <visualization_msgs/Marker.h>

// 把g2o的定义放到前面
typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
typedef g2o::LinearSolverCSparse< SlamBlockSolver::PoseMatrixType > SlamLinearSolver; 

// 给定index，读取一帧数据
FRAME readFrame( int index, ParameterReader& pd );
// 估计一个运动的大小
double normofTransform( cv::Mat rvec, cv::Mat tvec );

// 检测两个帧，结果定义
enum CHECK_RESULT {NOT_MATCHED=0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME}; 
// 函数声明
CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops=false );
// 检测近距离的回环
void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti );
// 随机检测回环
void checkRandomLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti );
bool checkPrevious( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti, int loststart);

void print4x4Matrix (const Eigen::Matrix4d & matrix)
{
	printf ("Rotation matrix :\n");
	printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
	printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
	printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
	printf ("Translation vector :\n");
	printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}

void publish_path(ros::Publisher &pub, std::vector<Eigen::Isometry3d> &poses){
	visualization_msgs::Marker marker;
    visualization_msgs::Marker line_list;
    visualization_msgs::Marker start_point;
    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    marker.header.frame_id = line_list.header.frame_id = start_point.header.frame_id = "/base_link";
    marker.header.stamp = line_list.header.stamp = start_point.header.stamp = ros::Time::now();
    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
    marker.ns = "basic_shapes";
    line_list.ns = "line_list";
    start_point.ns = "origin";
    marker.id = 0;
    line_list.id = 1;
    start_point.id = 2;
 
    marker.type = visualization_msgs::Marker::SPHERE;
    line_list.type = visualization_msgs::Marker::LINE_LIST;
    start_point.type = visualization_msgs::Marker::SPHERE;
    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = line_list.action = start_point.action = visualization_msgs::Marker::ADD;
    line_list.pose.orientation.w = 1.0;
    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header    
    Eigen::Quaterniond q(poses.back().rotation().matrix());  
    marker.pose.position.x = poses.back().translation().x();
    marker.pose.position.y = poses.back().translation().y();
    marker.pose.position.z = poses.back().translation().z();
    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();
    marker.pose.orientation.w = q.w();
    
    Eigen::Quaterniond q1(poses.front().rotation().matrix());  
    start_point.pose.position.x = poses.front().translation().x();
    start_point.pose.position.y = poses.front().translation().y();
    start_point.pose.position.z = poses.front().translation().z();
    start_point.pose.orientation.x = q1.x();
    start_point.pose.orientation.y = q1.y();
    start_point.pose.orientation.z = q1.z();
    start_point.pose.orientation.w = q1.w();
    //ROS_WARN("poses size: %d\n", poses.size());
    for(size_t j = 0; j<(poses.size()-1); ++j){
      geometry_msgs::Point p;
      geometry_msgs::Point p2;
      p.x = poses[j].translation().x();
      p.y = poses[j].translation().y();
      p.z = poses[j].translation().z();
      p2.x = poses[j+1].translation().x();
      p2.y = poses[j+1].translation().y();
      p2.z = poses[j+1].translation().z();
      line_list.points.push_back(p);
      line_list.points.push_back(p2);
    }
    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.x = 0.05;
    start_point.scale.x = 0.05;
    line_list.scale.x = 0.01;
    
    marker.scale.y = 0.05;
    start_point.scale.y = 0.05;
    line_list.scale.y = 0.01;
    
    marker.scale.z = 0.05;
    start_point.scale.z = 0.05;
    line_list.scale.z = 0.01;
    // Set the color -- be sure to set alpha to something non-zero!
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;
	
	start_point.color.r = 0.0f;
    start_point.color.g = 0.0f;
    start_point.color.b = 1.0f;
    start_point.color.a = 1.0;
    
    line_list.color.r = 1.0;
    line_list.color.a = 1.0;

    pub.publish(marker);
    pub.publish(line_list);
    //pub.publish(start_point);
}

int main( int argc, char** argv )
{
	//octomap::ColorOcTree tree( 0.01 ); //全局map
	// 前面部分和vo是一样的
	ParameterReader pd;
	int startIndex  =   atoi( pd.getData( "start_index" ).c_str() );
	int endIndex    =   atoi( pd.getData( "end_index"   ).c_str() );
	int MAXLOST = 10;
	int lostcnt = 0;
	int loststart = 0;
	bool isLost = false;
	bool lostInit = false;
    bool isloopEnd = false;

	string tf_frame = "/base_link";
	ros::init (argc, argv, "slam");
	ros::NodeHandle nh;
	ros::NodeHandle path_nh;
	ros::Publisher pub = nh.advertise<PointCloud>("slam_cloud", 1);
	ros::Publisher path_pub = path_nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);

	nh.param("frame_id", tf_frame, std::string("/base_link"));
	path_nh.param("frame_id", tf_frame, std::string("/base_link"));
    std::vector<int> lostlist;
	// 所有的关键帧都放在了这里
	vector< FRAME > keyframes; 
	// initialize
	cout<<"Initializing ..."<<endl;
	int currIndex = startIndex; // 当前索引为currIndex
	FRAME currFrame = readFrame( currIndex, pd ); // 上一帧数据

	string detector = pd.getData( "detector" );
	string descriptor = pd.getData( "descriptor" );
	CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
	computeKeyPointsAndDesp( currFrame, detector, descriptor );
	PointCloud::Ptr cloud = image2PointCloud( currFrame.rgb, currFrame.depth, camera );

	/******************************* 
	// 新增:有关g2o的初始化
	 *******************************/
	// 初始化求解器
	SlamLinearSolver* linearSolver = new SlamLinearSolver();
	linearSolver->setBlockOrdering( false );
	SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );

	g2o::SparseOptimizer globalOptimizer;  // 最后用的就是这个东东
	globalOptimizer.setAlgorithm( solver ); 
	// 不要输出调试信息
	globalOptimizer.setVerbose( false );

	// 向globalOptimizer增加第一个顶点
	float tx= atof(pd.getData("start_tx").c_str());
	float ty= atof(pd.getData("start_ty").c_str());
	float tz= atof(pd.getData("start_tz").c_str());
	float qx= atof(pd.getData("start_qx").c_str());
	float qy= atof(pd.getData("start_qy").c_str());
	float qz= atof(pd.getData("start_qz").c_str());
	float qw= atof(pd.getData("start_qw").c_str());
	ROS_WARN("x:%f y:%f, z:%f", tx, ty, tz);
	Eigen::Quaterniond q(qw, qx, qy, qz);
	Eigen::Matrix3d m= q.toRotationMatrix();
	Eigen::Isometry3d e = Eigen::Isometry3d::Identity();
	/*for(int i=0;i<3;++i){
		for(int j=0; j<3; ++j){
			e(i,j) = m(i,j);
		}
	}
	e(0,3) = tx; e(1,3) = ty; e(2,3) = tz;
	print4x4Matrix(e.matrix());
	*/
	g2o::VertexSE3* v = new g2o::VertexSE3();
	v->setId( currIndex );
	v->setEstimate(e); //估计为单位矩阵
	v->setFixed( true ); //第一个顶点固定，不用优化
	globalOptimizer.addVertex( v );

	keyframes.push_back( currFrame );

	double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );

	bool check_loop_closure = pd.getData("check_loop_closure")==string("yes");
	
	pcl::VoxelGrid<PointT> voxel; // 网格滤波器，调整地图分辨率
	pcl::PassThrough<PointT> pass; // z方向区间滤波器，由于rgbd相机的有效深度区间有限，把太远的去掉
	pass.setFilterFieldName("z");
	pass.setFilterLimits( 0.0, 4.0 ); //4m以上就不要了
	double gridsize = atof( pd.getData( "voxel_grid" ).c_str() ); //分辨图可以在parameters.txt里调
	voxel.setLeafSize( gridsize, gridsize, gridsize );
	int opti_iter = atoi( pd.getData("opti_iter").c_str());
	for ( currIndex=startIndex+1; currIndex<endIndex; currIndex++ )
	{
		PointCloud::Ptr output ( new PointCloud() ); //全局地图
		cout<<"Reading files "<<currIndex<<endl;
		FRAME currFrame = readFrame( currIndex,pd ); // 读取currFrame
		computeKeyPointsAndDesp( currFrame, detector, descriptor ); //提取特征
		CHECK_RESULT result = checkKeyframes( keyframes.back(), currFrame, globalOptimizer ); //匹配该帧与keyframes里最后一帧
		switch (result) // 根据匹配结果不同采取不同策略
		{
			case NOT_MATCHED:
				//没匹配上，直接跳过
				cout<<RED"Not enough inliers."<<endl;
				lostcnt++;
				/*if (lostcnt >= MAXLOST){
					lostlist.push_back(currFrame.frameID);
					isLost = true;
					loststart = keyframes.size();
					lostcnt = 0;
					g2o::VertexSE3 *v = new g2o::VertexSE3();
					v->setId( currFrame.frameID );
					v->setEstimate( Eigen::Isometry3d::Identity() );
					globalOptimizer.addVertex(v);
					keyframes.push_back(currFrame);
					lostInit = true;
					cout<<BLUE"a new fake keyframe"<<endl;
				}*/

				break;
			case TOO_FAR_AWAY:
				// 太近了，也直接跳
				if(isLost){
					lostcnt++;
				}
				cout<<RED"Too far away, may be an error."<<endl;
				break;
			case TOO_CLOSE:
				// 太远了，可能出错了
				cout<<RESET"Skip"<<endl;
				break;
			case KEYFRAME:
				lostcnt = 0;
				cout<<GREEN"Add a new keyframe"<<endl;
				// 检测回环
				if (check_loop_closure && !isLost)
				{
					checkNearbyLoops( keyframes, currFrame, globalOptimizer );
					checkRandomLoops( keyframes, currFrame, globalOptimizer );
				}
				/*if(lostInit){
					cout<<BLUE"try to converge"<<endl;
					if (checkPrevious(keyframes, currFrame, globalOptimizer,loststart)){
						isLost = false;
						lostcnt = 0;
						lostInit = false;
						cout<<BLUE"converged!!!!!"<<endl;
					}
					
				}*/
                keyframes.push_back( currFrame );
                break;
            default:
                break;
		}
		bool init =  (keyframes.size() == 1);
		if((keyframes.size() % 3 == 0 && result == KEYFRAME) || init){
			//every 5 frame
			PointCloud::Ptr tmp ( new PointCloud() );
			std::vector<Eigen::Isometry3d> pose_list;
			if(!init){
				ROS_WARN("Optimize");
				globalOptimizer.initializeOptimization();
				globalOptimizer.optimize( opti_iter );
				for (size_t i=0; i<keyframes.size() && nh.ok(); i++)
				{
					// 从g2o里取出一帧
					g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[i].frameID ));
					Eigen::Isometry3d pose = vertex->estimate(); //该帧优化后的位姿
					pose_list.push_back(pose);
					PointCloud::Ptr newCloud = image2PointCloud( keyframes[i].rgb, keyframes[i].depth, camera ); //转成点云
					// 以下是滤波
					voxel.setInputCloud( newCloud );
					voxel.filter( *tmp );
					pass.setInputCloud( tmp );
					pass.filter( *newCloud );
					// 把点云变换后加入全局地图中
					pcl::transformPointCloud( *newCloud, *tmp, pose.matrix());
					//print4x4Matrix (pose.matrix());	
					*output += *tmp;
					tmp->clear();
					newCloud->clear();
				}
			}
			else{
				output = image2PointCloud( keyframes[0].rgb, keyframes[0].depth, camera ); 
			}
			voxel.setInputCloud( output );
			voxel.filter( *tmp );
			output->swap( *tmp );
			
			PointCloud::Ptr msg(output);
			msg->header.frame_id = tf_frame;
			msg->header.stamp = ros::Time::now().toNSec();
			pub.publish(msg);
			if(path_nh.ok() && !init){
				publish_path(path_pub, pose_list);
			}
		}
    }
		// 优化
		//cout<<RESET"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
		//globalOptimizer.save("./data/result_before.g2o");
		
		 //可以指定优化步数
    	cout<<"Saving."<<endl;
		globalOptimizer.save( "./data/after.g2o" );
		cout<<"Saved."<<endl;

		// 拼接点云地图
		//cout<<"saving the point cloud map..."<<endl;
		

		

		
        /*for (size_t i=0; i<lostlist.size(); i++){
            cout<<"miss node: "<<lostlist[i]<<endl;
        }*/

        //ros::Rate r(40);
		
		//tree.updateInnerOccupancy();
		//tree.write( "./data/map.ot" );
		//voxel.setInputCloud( output );
		//voxel.filter( *tmp );
		//存储
		//pcl::io::savePCDFile( "./data/result.pcd", *tmp );
		/*cout<<"Final map is saved."<<endl;
		pcl::visualization::PCLVisualizer viz;
		viz.addPointCloud(tmp, "rgbd");
		viz.spin();
		*/
		//globalOptimizer.clear();

		return 0;
	}

	FRAME readFrame( int index, ParameterReader& pd )
	{
		FRAME f;
		string rgbDir   =   pd.getData("rgb_dir");
		string depthDir =   pd.getData("depth_dir");

		string rgbExt   =   pd.getData("rgb_extension");
		string depthExt =   pd.getData("depth_extension");

		stringstream ss;
		ss<<rgbDir<<index<<rgbExt;
		string filename;
		ss>>filename;
		f.rgb = cv::imread( filename );

		ss.clear();
		filename.clear();
		ss<<depthDir<<index<<depthExt;
		ss>>filename;

		f.depth = cv::imread( filename, -1 );
		f.frameID = index;
		return f;
	}

	double normofTransform( cv::Mat rvec, cv::Mat tvec )
	{
		return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
	}

	CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops)
	{
		static ParameterReader pd;
		static int min_inliers = atoi( pd.getData("min_inliers").c_str() );
		static double max_norm = atof( pd.getData("max_norm").c_str() );
		static double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
		static double max_norm_lp = atof( pd.getData("max_norm_lp").c_str() );
		static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
		static g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct( "Cauchy" );
		// 比较f1 和 f2
		RESULT_OF_PNP result = estimateMotion( f1, f2, camera );
		if ( result.inliers < min_inliers ) //inliers不够，放弃该帧
			return NOT_MATCHED;
		// 计算运动范围是否太大
		double norm = normofTransform(result.rvec, result.tvec);
		if ( is_loops == false )
		{
			if ( norm >= max_norm )
				return TOO_FAR_AWAY;   // too far away, may be error
		}
		else
		{
			if ( norm >= max_norm_lp)
				return TOO_FAR_AWAY;
		}

		if ( norm <= keyframe_threshold )
			return TOO_CLOSE;   // too adjacent frame
		// 向g2o中增加这个顶点与上一帧联系的边
		// 顶点部分
		// 顶点只需设定id即可
		if (is_loops == false)
		{
			g2o::VertexSE3 *v = new g2o::VertexSE3();
			v->setId( f2.frameID );
			v->setEstimate( Eigen::Isometry3d::Identity() );
			opti.addVertex(v);
		}
		// 边部分
		g2o::EdgeSE3* edge = new g2o::EdgeSE3();
		// 连接此边的两个顶点id
		edge->vertices() [0] = opti.vertex( f1.frameID );
		edge->vertices() [1] = opti.vertex( f2.frameID );
		edge->setRobustKernel( robustKernel );
		// 信息矩阵
		Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
		// 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
		// 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
		// 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
		information(0,0) = information(1,1) = information(2,2) = 100*result.inliers;
		information(3,3) = information(4,4) = information(5,5) = 100*result.inliers;
		// 也可以将角度设大一些，表示对角度的估计更加准确
		edge->setInformation( information );
		// 边的估计即是pnp求解之结果
		Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec );
		cout<<"frame "<<f1.frameID<<" is converged with "<<f2.frameID<<endl;
		//print4x4Matrix(T.matrix());
		edge->setMeasurement(T.inverse());
		opti.addEdge(edge);
		// 将此边加入图中
		return KEYFRAME;
	}

	void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti )
	{
		static ParameterReader pd;
		static int nearby_loops = atoi( pd.getData("nearby_loops").c_str() );

		// 就是把currFrame和 frames里末尾几个测一遍
		if ( frames.size() <= nearby_loops )
		{
			// no enough keyframes, check everyone
			for (size_t i=0; i<frames.size(); i++)
			{
				checkKeyframes( frames[i], currFrame, opti, true );
			}
		}
		else
		{
			// check the nearest ones
			for (size_t i = frames.size()-nearby_loops; i<frames.size(); i++)
			{
				checkKeyframes( frames[i], currFrame, opti, true );
			}
		}
	}

	void checkRandomLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti )
	{
		static ParameterReader pd;
		static int random_loops = atoi( pd.getData("random_loops").c_str() );
		srand( (unsigned int) time(NULL) );
		// 随机取一些帧进行检测

		if ( frames.size() <= random_loops )
		{
			// no enough keyframes, check everyone
			for (size_t i=0; i<frames.size(); i++)
			{
				checkKeyframes( frames[i], currFrame, opti, true );
			}
		}
		else
		{
			// randomly check loops
			for (int i=0; i<random_loops; i++)
			{
				int index = rand()%frames.size();
				checkKeyframes( frames[index], currFrame, opti, true );
			}
		}
	}

	bool checkPrevious( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti, int loststart)
	{
		// randomly check loops
		bool isConverge = false;
		for (int i=0; i<loststart; i++)
		{
			if(checkKeyframes( frames[i], currFrame, opti, true ) == KEYFRAME){
				isConverge =  true;
			}
		}
		for (int i=loststart; i<frames.size(); i++)
		{
			checkKeyframes( frames[i], currFrame, opti, true );
		}
		return isConverge;
	}
