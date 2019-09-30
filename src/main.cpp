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



int main(int argc, char *argv[])
{
	//Mask R-CNN
	const int cnn_skip_frames = 3;
	const int cnn_start_frames = 80;//80;
	const int maskRcnnType = 1;

	// flann frame
	const int flann_skip_frames = 40;
	
	const bool debugMask = false;
	
	const bool hasInstanceGroundTruth = true;	//e.g. "scannet/instance" scanNet中data.txt后半段可读（为true时），否则就要自己删掉后半段

	const int instanceNum = 96;
	const int width = 640;
	const int height = 480;

	Resolution::getInstance(width, height);
	Intrinsics::getInstance(528, 528, 320, 240);

	//init Log Reader
	std::unique_ptr<LogReader> log_reader;
	std::string rawLogFile;	
	Parse::get().arg(argc, argv, "-l", rawLogFile);
	if(rawLogFile.length())		
	{
		//dyson_lab.klg
		log_reader.reset( new RawLogReader(rawLogFile, false));
	}
	else if (argc > 2) 
	{
		//data.txt + empty.txt + (hasInstanceGroundTruth?)
		log_reader.reset(new PNGLogReader(argv[1],argv[2],hasInstanceGroundTruth));
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
	std::unique_ptr<InstanceFusion> instancefusion(new InstanceFusion(instanceNum,width,height,false,maskRcnnType));	
	
	//-------- pht -----------------------------------------------------------------

	// 初始化Gui, Map
	std::unique_ptr<Gui> gui(new Gui(true,instancefusion->getInstanceTable(),width,height));

	std::unique_ptr<ElasticFusionInterface> map(new ElasticFusionInterface());
	std::cout<<"initialising ElasticFusionInterface OK"<<std::endl;
	if (!map->Init(instancefusion->getInstanceTable())) 
	{
		std::cout<<"ElasticFusionInterface init failure"<<std::endl;
	}


	// 主循环
	std::cout<<"Main Loop:"<<std::endl;
	int frame_real = 0;
	int frame_Fusion = 0;
	int lastTimeFlann = -1;
	while(!pangolin::ShouldQuit() && log_reader->hasMore()) 
	{
		//====================lyc add : update GUI ====================================================
		//显示帧率
		//show
		gui->setFrameCount(std::to_string(frame_Fusion)+"/"+std::to_string(log_reader->getNumFrames()));
		//time
		static auto last=std::chrono::system_clock::now();
		static int lastFrame_real=frame_real;
		if(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()-last).count()>1000)
		{
				last=std::chrono::system_clock::now();
				gui->setFps(frame_real-lastFrame_real);
				//gui->setFps(20);
				lastFrame_real=frame_real;
		}
		frame_real++;
		//====================lyc add over ============================================================
		

		// Read and perform an elasticFusion update
		if (!gui->paused() || gui->step()) 
		{
			log_reader->getNext();
			map->setTrackingOnly(gui->tracking());
			
			instancefusion->TimeTick();
			int* instanceTableLoopClosure = new int[instancefusion->getInstanceNum()*5];
			instancefusion->getLoopClosureInstanceTable(instanceTableLoopClosure);
			if (!map->ProcessFrame(log_reader->rgb, log_reader->depth,log_reader->timestamp,instanceTableLoopClosure,
							(hasInstanceGroundTruth?log_reader->instanceGroundTruth:NULL))) 
			{
				std::cout<<"Elastic fusion lost!"<<argv[1]<<std::endl;
				return 1;
			}
			//instancefusion->checkLoopClosure(instanceTableLoopClosure,map);
			instancefusion->TimeTock("ElasticFusion");
			
			//------------------------------------------------------------------------------------	
			instancefusion->renderProjectMap(map,gui->project_bbox());

			//frame_Fusion > cnn_start_frames+1
			//if (frame_Fusion >= cnn_start_frames && gui->instance_seg() && ((frame_Fusion + 1) % cnn_skip_frames == 0))
			if (frame_Fusion >= cnn_start_frames && gui->instance_seg() && instancefusion->whetherDoSegmentation(map,frame_Fusion))
			{
				std::cout<<std::endl;
				std::cout<<"==============================================="<<std::endl;
				std::cout<<"InstanceFusion Segmentation"<<std::endl;

				bool flannFlag = 0;
				if(frame_Fusion-lastTimeFlann > flann_skip_frames)
				{
					lastTimeFlann = frame_Fusion;
					flannFlag=1;
				}

				instancefusion->ProcessSegmentation(log_reader->rgb,log_reader->depth,map,frame_Fusion,flannFlag);

				std::cout<<"ProcessFrame end"<<std::endl<<std::endl;
			}


			//-----------Save RGB and Depth------------------------------------------------------	
			
			if(gui->raw_save())
			{
				cv::Mat rgb_image(height,width,CV_8UC3, log_reader->rgb);
				cv::Mat bgr_image;
				cvtColor(rgb_image,bgr_image,CV_RGB2BGR);
				std::string rgb_save_dir("./RAW/");
				std::string rgb_suffix("_color.png");
				rgb_save_dir += std::to_string(frame_Fusion);
				rgb_save_dir += rgb_suffix;
				cv::imwrite(rgb_save_dir,bgr_image);

				cv::Mat depth_image(height,width,CV_16UC1, log_reader->depth);
				std::string depth_save_dir("./RAW/");
				std::string depth_suffix("_depth.png");
				depth_save_dir += std::to_string(frame_Fusion);
				depth_save_dir += depth_suffix;
				cv::imwrite(depth_save_dir,depth_image);

				//Save Mask
				if(debugMask)
				{
					float* mask_debug = (float*)malloc(height*width*4*sizeof(float));
					cudaMemcpy(mask_debug, instancefusion->getMaskColorMap_gpu(), height*width *4* sizeof(float), cudaMemcpyDeviceToHost);
					cv::Mat mask_image(height,width,CV_32FC4, mask_debug);
					cv::Mat mask_bgr,mask_dst;
					cv::cvtColor(mask_image,mask_bgr,CV_BGRA2BGR);
					mask_bgr.convertTo(mask_dst,CV_8U,255,0);

					std::string mask_save_dir("./mask/");
					std::string mask_suffix("_Mask.png");
					mask_save_dir += std::to_string(frame_Fusion);
					mask_save_dir += mask_suffix;
					cv::imwrite(mask_save_dir,mask_dst);
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
				//Save GroundTruth
				if(hasInstanceGroundTruth)
				{
					cv::Mat inst_image(height,width,CV_8UC1, log_reader->instanceGroundTruth);
					std::string inst_save_dir("./RAW/");
					std::string inst_suffix("_instance.png");
					inst_save_dir += std::to_string(frame_Fusion);
					inst_save_dir += inst_suffix;
					cv::imwrite(inst_save_dir,inst_image);
				}
				std::cout<<"  Save to RAW"<<std::endl;
			}
			//-----------Save RGB and Depth------------------------------------------------------
			
			frame_Fusion++;
			std::cout<<".";
		}

		
	
		if(gui->display_mode())	//ori in semanticFusion
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

			instancefusion->computeMapBoundingBox(map,gui->bbox_type());

			gui->renderMapMethod2(map,instancefusion,true,gui->bbox_type());

			gui->displayRawNetworkPredictions("pred",instancefusion->getMaskColorMap_gpu());

			gui->displayProjectColor("segmentation",instancefusion->getProjectColorMap_gpu());

			gui->displayImg("raw",map->getRawImageTexture());

			gui->postCall();

			if(gui->instance_save_once())
			{
				gui->setInstanceSaveFalse();

				instancefusion->getInstancePointCloud(map,gui->bbox_type());

				instancefusion->saveInstancePointCloud();//not finish
			}
		}

		if (gui->reset()) {
			map.reset(new ElasticFusionInterface());
			if (!map->Init(instancefusion->getInstanceTable())) {
				std::cout<<"ElasticFusionInterface init failure"<<std::endl;
			}
		}
		//==================== lyc ============================================================
		if(gui->save()){
			map->SavePly();
			instancefusion->saveInstanceTable();
			instancefusion->printHistoryInstances();
			std::cout<< "SavePly and InstanceTable" << std::endl;
		}
		//==================== lyc ============================================================
		//Stopwatch::getInstance().printAll();
	}

	//Debug
	map->SavePly();							//rgb + instance + gt + test
	instancefusion->saveInstanceTable();				//temp/InstanceTable_In.txt
	instancefusion->printHistoryInstances();

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
