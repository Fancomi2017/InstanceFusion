/*
 * This file is part of InstanceFusion.
 *
 */

#include <Core/InstanceFusion.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <cassert>

#include <map_interface/ElasticFusionInterface.h>
#include <utilities/LiveLogReader.h>
#include <utilities/RawLogReader.h>
#include <utilities/PNGLogReader.h>
#include <utilities/Types.h>

#include <gui/Gui.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


// argc是指命令行输入参数的个数(以空格符分隔)，argv存的是每个参数的内容,argv[0]指的是./InstanceFusion
int main(int argc, char *argv[])
{
	//Mask R-CNN
	const int cnn_skip_frames = 3;  // 分割间隔cnn_skip_frames帧运行一次
	const int cnn_start_frames = 80;//cnn_start_frames起始帧
	const int maskRcnnType = 1; // maskrcnntype为0时代表使用mask_benchmark，为1时代表使用mask_ori，为2时代表使用mask_paddle。

	// flann frame
	const int flann_skip_frames = 40; // flann(kdtree和三维滤波) 间隔flann_skip_frames帧运行一次
	
	const bool debugMask = false;  // 表示是否输出分割运行时掩码的中间结果
	
	const bool hasInstanceGroundTruth = false;	//e.g. "scannet/instance" scanNet中data2.txt后半段可读（为true时），否则就要自己删掉后半段

	const int instanceNum = 96; // instanceNum为96，这个和很多项都有关，尤其是ElasticFusion中OpenGL的相关代码很多，如果要修改会涉及很多OpenGL的内容。
	const int width = 640; // width和height会存在Resolution和Intrinsics中作为全局变量可在任意位置调用。
	const int height = 480;

	Resolution::getInstance(width, height);  // 创建对象用于存储初始化分辨率
	Intrinsics::getInstance(528, 528, 320, 240);

	//init Log Reader
	std::unique_ptr<LogReader> log_reader;
	std::string rawLogFile;	
	Parse::get().arg(argc, argv, "-l", rawLogFile);  // 若是命令行中有"-l"，则rawLogFile的值为dyson_lab.klg完整的地址；否则，rawLogFile的值为空
	if(rawLogFile.length())  //读取的是dyson_lab.klg
	{
		log_reader.reset( new RawLogReader(rawLogFile, false));  // RawLogReader用于加载.klg类型的数据，会解析成rgbd，rawLogFile是.klg文件的完整路径
	}
	else if (argc == 2) // 即使argc的值大于2，只要第一个if执行了，这个else if都不会执行
	{
		//data.txt + empty.txt + (hasInstanceGroundTruth?)
		log_reader.reset(new PNGLogReader(argv[1],hasInstanceGroundTruth)); //argv[1]是data.txt完整路径，argv[2]是output.txt完整路径
	} 
	else 
	{
		//kinectv1
		log_reader.reset(new LiveLogReader("./live",false));
		if (!log_reader->is_valid()) 
		{
			std::cout<<"Error initialising live device..."<<std::endl;
			return 1;
		}
	}
	std::cout<<"initialising LogReader OK"<<std::endl;


	//cudaSetDevice(0);
	//-------- pht -----------------------------------------------------------------
	// InstanceFusion对象建立
	//[final parameter]
	//0->mask_benchmark.py(maskrcnn-benchmark+fbnet)
	//1->mask_ori.py(ori maskrcnn+resnet101)
	//2->mask_paddle.py(paddle+resnet50)
	// python接口初始化中会加载maskrcnntype指定的python文件，在python文件中会加载代码指定的网络参数。随后map(ElasticFusion)和GUI初始化中会用instanceTable在自己的实例下生成一个instanceTable的备份。
	std::unique_ptr<InstanceFusion> instancefusion(new InstanceFusion(instanceNum,width,height,false,maskRcnnType));	//创建instancefusion对象,false表示不用多线程
	
	//-------- pht -----------------------------------------------------------------

	// 初始化Gui, Map。成员变量初始化中值得注意的是instanceTable在这个时候置空，但每个空位准备了96份颜色，颜色为随机生成。
	std::unique_ptr<Gui> gui(new Gui(true,instancefusion->getInstanceTable(),width,height));  // getInstanceTable()获取颜色（标签）表(96个颜色)

	std::unique_ptr<ElasticFusionInterface> map(new ElasticFusionInterface());
	std::cout<<"initialising ElasticFusionInterface OK"<<std::endl;
	if (!map->Init(instancefusion->getInstanceTable())) 
	{
		std::cout<<"ElasticFusionInterface init failure"<<std::endl;
	}


	// 主循环：更新帧率
    // 运行ElasticFusion  ProcessFrame	(如果没暂停的话)
    // 运行InstanceFusion ProcessSegmentation  (如果没暂停，且满足cnn_skip_frames或whetherDoSegmentation的话)
    // 输出Debug图像	(如果没暂停，且raw_save开启的话)
    // 更新GUI：分两种模式：通常情况下用第一种；第二种是我专门写来表现包围盒和实例提取的，速度很慢因为显示是用的cuda缺少渲染优化，bbox_type为1时以地面为基准设置AABB包围盒，bbox_type为0时以每个instance自身为基准设置OBB包围盒（没完成实现）
	// Ps: ElasticFusion ProcessFrame前后有一个通过instance执行loopClosure变形的步骤我给注释掉了，因为效果不好。
	std::cout<<"Main Loop:"<<std::endl;
	int frame_real = 0;  //正在处理的帧号
	int frame_Fusion = 0;
	int lastTimeFlann = -1;
	while(!pangolin::ShouldQuit() && log_reader->hasMore()) //Pangolin是对OpenGL进行封装的轻量级的OpenGL输入/输出和视频显示的库。可以用于3D视觉和3D导航的视觉图，可以输入各种类型的视频、并且可以保留视频和输入数据用于debug。
	{
		//====================lyc add : update GUI ====================================================
		//显示FrameCount的值
		gui->setFrameCount(std::to_string(frame_Fusion)+"/"+std::to_string(log_reader->getNumFrames()));
		
		static auto last=std::chrono::system_clock::now();  // 1秒开始时的时间
		static int lastFrame_real=frame_real;  // 1秒计时开始时处理的帧号
		// 算帧率（1秒内处理的帧数）
		if(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-last).count()>1000)
		{
				last=std::chrono::system_clock::now();
				gui->setFps(frame_real-lastFrame_real);
				//gui->setFps(20);
				lastFrame_real=frame_real;
		}
		frame_real++;
		//====================lyc add over ============================================================
		

		// Read and perform an elasticFusion update 重建、实例分割、保存RGB和Depth、Mask图像
		if (!gui->paused() || gui->step())
		{
			log_reader->getNext();
			map->setTrackingOnly(gui->tracking());  // TrackingOnly按钮表示界面上的模型是否随着重建更新
			
			instancefusion->TimeTick();
			int* instanceTableLoopClosure = new int[instancefusion->getInstanceNum()*5];  // instanceTableLoopClosure用来干嘛的？
			instancefusion->getLoopClosureInstanceTable(instanceTableLoopClosure);  // 执行后，instanceTableLoopClosure会存储一个实例的rgb值，类别ID，还有instance标号
			if (!map->ProcessFrame(log_reader->rgb, log_reader->depth,log_reader->timestamp,instanceTableLoopClosure,// 该看这里了
							(hasInstanceGroundTruth?log_reader->instanceGroundTruth:NULL)))
			{
				std::cout<<"Elastic fusion lost!"<<argv[1]<<std::endl;
				return 1;
			}
			//instancefusion->checkLoopClosure(instanceTableLoopClosure,map);
			instancefusion->TimeTock("ElasticFusion");
			
			//------------------------------------------------------------------------------------	
			instancefusion->renderProjectMap(map,gui->project_bbox());   //从模型中反投影回来的图片中画每个实例的包围框。渲染instance投影图，用于显示，包含二维包围盒

			//frame_Fusion > cnn_start_frames+1
			//if (frame_Fusion >= cnn_start_frames && gui->instance_seg() && ((frame_Fusion + 1) % cnn_skip_frames == 0))
			//实例分割,此处如果使用whetherDoSegmentation判断分割间隔，则cnn_skip_frames和cnn_start_frames不再适用
			if (frame_Fusion >= cnn_start_frames && gui->instance_seg() && instancefusion->whetherDoSegmentation(map,frame_Fusion))
			{
				std::cout<<std::endl;
				std::cout<<"==============================================="<<std::endl;
				std::cout<<"InstanceFusion Segmentation"<<std::endl;

				bool flannFlag = 0;  // flannFlag是什么？？  FLANN(Fast Library for Approximate Nearest Neighbors)是一个执行快速近似最近邻搜索的库。
				if(frame_Fusion-lastTimeFlann > flann_skip_frames)
				{
					lastTimeFlann = frame_Fusion;
					flannFlag=1;
				}

				instancefusion->ProcessSegmentation(log_reader->rgb,log_reader->depth,map,frame_Fusion,flannFlag);

				std::cout<<"ProcessFrame end"<<std::endl<<std::endl;
			}


			//-----------Save RGB and Depth and Mask------------------------------------------------------	
			
			if(gui->raw_save())
			{
				// 保存input里的rgb和depth吗？
				cv::Mat rgb_image(height,width,CV_8UC3, log_reader->rgb);
				cv::Mat bgr_image;
				cvtColor(rgb_image,bgr_image,CV_RGB2BGR);
				std::string rgb_save_dir("./RAW/");
				std::string rgb_suffix("_color.png");
				rgb_save_dir += std::to_string(frame_Fusion);
				rgb_save_dir += rgb_suffix;
				cv::imwrite(rgb_save_dir,bgr_image);  // 输出raw color到build/RAW/文件夹下

				cv::Mat depth_image(height,width,CV_16UC1, log_reader->depth);
				std::string depth_save_dir("./RAW/");
				std::string depth_suffix("_depth.png");
				depth_save_dir += std::to_string(frame_Fusion);
				depth_save_dir += depth_suffix;
				cv::imwrite(depth_save_dir,depth_image);  // 输出raw depth到build/RAW/文件夹下

				//Save Mask
				if(debugMask)   // debugMask是什么??这段代码似乎根本就没用到
				{
					float* mask_debug = (float*)malloc(height*width*4*sizeof(float)); // 4表示4个通道？？
					cudaMemcpy(mask_debug, instancefusion->getMaskColorMap_gpu(), height*width *4* sizeof(float), cudaMemcpyDeviceToHost);
					cv::Mat mask_image(height,width,CV_32FC4, mask_debug);
					cv::Mat mask_bgr,mask_dst;
					cv::cvtColor(mask_image,mask_bgr,CV_BGRA2BGR);
					mask_bgr.convertTo(mask_dst,CV_8U,255,0);

					std::string mask_save_dir("./mask/");
					std::string mask_suffix("_Mask.png");
					mask_save_dir += std::to_string(frame_Fusion);
					mask_save_dir += mask_suffix;
					cv::imwrite(mask_save_dir,mask_dst);  // 输出mask到build/mask/文件夹下
					free(mask_debug);
					
					
					float* proj_debug = (float*)malloc(height*width*4*sizeof(float));
					cudaMemcpy(proj_debug, instancefusion->getProjectColorMap_gpu(), height*width*4 * sizeof(float), cudaMemcpyDeviceToHost);
					cv::Mat proj_image(height,width,CV_32FC4, proj_debug);
					cv::Mat proj_bgr,proj_rgb,proj_dst;
					cv::cvtColor(proj_image,proj_bgr,CV_BGRA2BGR);
					cv::cvtColor(proj_bgr,proj_rgb,CV_BGR2RGB);
					proj_rgb.convertTo(proj_dst,CV_8U,255,0);

					std::string proj_save_dir("./mask/");
					std::string proj_suffix("_Proj.png");
					proj_save_dir += std::to_string(frame_Fusion);
					proj_save_dir += proj_suffix;
					cv::imwrite(proj_save_dir,proj_dst);
					free(proj_debug);
				}
				//Save GroundTruth,这段代码似乎也没用上
				if(hasInstanceGroundTruth)
				{
					cv::Mat inst_image(height,width,CV_8UC1, log_reader->instanceGroundTruth);
					std::string inst_save_dir("./RAW/");
					std::string inst_suffix("_instance.png");
					inst_save_dir += std::to_string(frame_Fusion);
					inst_save_dir += inst_suffix;
					cv::imwrite(inst_save_dir,inst_image);   // 输出groundTruth图像到build/RAW/文件夹下(如果启用了scanNet并读取了gt)
				}
				std::cout<<"  Save to RAW"<<std::endl;
			}
			//-----------Save RGB and Depth------------------------------------------------------
			
			frame_Fusion++;
			std::cout<<".";
		}

		
	
		if(gui->display_mode())	//ori in semanticFusion  在三维模型中显示包围盒
		{
			instancefusion->TimeTick();

			gui->preCall();

			gui->renderMapMethod1(map);

			gui->displayRawNetworkPredictions("pred",instancefusion->getMaskColorMap_gpu());

			gui->displayProjectColor("segmentation",instancefusion->getProjectColorMap_gpu());

			gui->displayImg("raw",map->getRawImageTexture());

			gui->postCall();

			instancefusion->TimeTock("GUI");
		}
		else	//modify in InstanceFusion(DANGER CODE!JUST USE FOR DEBUG)
		{
			gui->renderMapID(map,instancefusion->getInstanceTable());

			gui->preCall();

			instancefusion->computeMapBoundingBox(map,gui->bbox_type());  // 计算三维包围盒

			gui->renderMapMethod2(map,instancefusion,true,gui->bbox_type());  // 将包围盒还原并显示

			gui->displayRawNetworkPredictions("pred",instancefusion->getMaskColorMap_gpu());

			gui->displayProjectColor("segmentation",instancefusion->getProjectColorMap_gpu());

			gui->displayImg("raw",map->getRawImageTexture());

			gui->postCall();

			if(gui->instance_save_once())
			{
				gui->setInstanceSaveFalse();

				instancefusion->getInstancePointCloud(map,gui->bbox_type());  // 将gpu的全局点云按二级指针拷贝到cpu中

				instancefusion->saveInstancePointCloud();//not finish  将96个instance分别输出到/temp文件夹下，会按照地面坐标系(或instance坐标系)修正位置，其中地面也会单独输出。
			} // Ps:instance坐标系可以通过传统计算OBB包围盒的方法实现（通过PCA），这里没有完全写好(效果不好的结果在Step 3的cuda程序中)。
		}

		// 清理掉之前的重建结果，从当前帧开始重新重建
		if (gui->reset()) {
			map.reset(new ElasticFusionInterface());
			if (!map->Init(instancefusion->getInstanceTable())) {
				std::cout<<"ElasticFusionInterface init failure"<<std::endl;
			}
		}
		//==================== lyc ============================================================
		if(gui->save()){
			map->SavePly();
			instancefusion->saveInstanceTable();  // saveInstanceTable()函数似乎只声明了，并没有实现
			instancefusion->printHistoryInstances();  // printHistoryInstances()函数似乎只声明了，并没有实现
			std::cout<< "SavePly and InstanceTable" << std::endl;
		}
		//==================== lyc ============================================================
		//Stopwatch::getInstance().printAll();
	}

	// Debug  调用ElasticFusion保存全局点云build/文件夹下，具体函数在ElasticFusion.cpp中。ResultModel.ply是颜色为rgb的点云；ResultModel_instance.ply是颜色为instance的rgb点云；
	// ResultModel_instanceGT.ply是scanNet启用GT时利用GT颜色生成的点云；ResultModel_test.ply是不可视的，用于debug检查点其他数据是否正常用的点云。
	map->SavePly();							//rgb + instance + gt + test
	instancefusion->saveInstanceTable();	//	保存InstanceTable信息到build/temp/文件夹下，这里只写了类名和颜色
	instancefusion->printHistoryInstances();

	// 这段代码保存TimeDebug信息到build/temp/文件夹下以追加的方式，里面内容为InstanceFusion各个步骤的运行时间，具体测量位置可以根据Time_Debug第一行的名称在代码中查找，
	// 前几项单位是次数，其他时间项单位是微秒，横杠表示未曾运行。
	std::string fileName;
	if(argc > 2) 
	{
		if(rawLogFile.length())
		{
			fileName=argv[2];
			int n1 = fileName.rfind('.');
			int n2 = fileName.rfind('/',n1-1);
			fileName = fileName.substr(n2+1,n1-n2-1);
			std::cout<<"fileName:"<<fileName<<std::endl;
		}
		else
		{
			fileName=argv[1];
			int n1 = fileName.rfind('/');
			int n2 = fileName.rfind('/',n1-1);
			fileName = fileName.substr(n2+1,n1-n2-1);
			std::cout<<"fileName:"<<fileName<<std::endl;
		}
	}
	else fileName="live";
	instancefusion->saveTimeDebug(fileName);			//temp/Time_Debug.txt

	if(hasInstanceGroundTruth)instancefusion->evaluateAndSave(map,fileName);

	std::cout<<"Finished InstanceFusion"<<std::endl;

	return 0;
}
