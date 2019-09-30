/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.	The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.	By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include "PNGLogReader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

PNGLogReader::PNGLogReader(std::string file, std::string labels_file, bool hasInstanceGroundTruth)
 : LogReader(file, true)
 , lastFrameTime(-1)
 , lastGot(-1)
 , has_depth_filled(false)
 , num_labelled(0)
 , has_instance_GT(hasInstanceGroundTruth)
{

	decompressionBufferDepth = new Bytef[Resolution::getInstance().numPixels() * 2];		//byte * 2 = unsigned short
	decompressionBufferDepthFilled = new Bytef[Resolution::getInstance().numPixels() * 2];
	decompressionBufferImage = new Bytef[Resolution::getInstance().numPixels() * 3];		//byte * 3 = unsigned char * 3
	
	decompressionBufferInstanceGroundTruth = new Bytef[Resolution::getInstance().numPixels()];


	std::ifstream infile(file.c_str());
	std::string timestamp, depth_path, rgb_path, depth_id, rgb_id, instanceGT_path;

	std::map<std::string,int> depth_id_lookup;
	std::string scene_id = file.substr(file.rfind("/") + 1);
	std::string base_path = file;
	base_path.erase(base_path.rfind('/'));
	std::cout<<"Looking for RGB/Depth images in folder:"<<base_path<<std::endl;
	scene_id.erase(scene_id.length()-4);
	int id = 0;

	if(has_instance_GT)
	{
		//0 ./depth/0.png ./color/0.jpg 0 0
		//1 ./depth/1.png ./color/1.jpg 0 0
		//2 ./depth/2.png ./color/2.jpg 0 0
		std::cout<<"==================================================================================================\n"<<std::endl;
		std::cout<<"Warning: You need write instanceGT_path in data.txt if you try to evaluate it.(cancel in main.cpp)\n"<<std::endl;
		std::cout<<"==================================================================================================\n"<<std::endl;
		while (infile >> timestamp >> depth_path >> rgb_path >> depth_id >> rgb_id >>instanceGT_path) 
		{
			FrameInfo frame_info;
			std::stringstream ss(timestamp.c_str());
			ss >> frame_info.timestamp;

			frame_info.depth_path = base_path + "/" + depth_path;
			frame_info.rgb_path = base_path + "/" + rgb_path;
			frame_info.instanceGT_path = base_path + "/" + instanceGT_path;


			if (id == 0)  std::cout<<"E.g.:"<<base_path+"/"+depth_path<<std::endl;

			frame_info.depth_id = scene_id+"/"+depth_id;
			frame_info.rgb_id = scene_id+"/"+rgb_id;

			frame_info.labeled_frame = false;
			depth_id_lookup[scene_id+"/"+depth_id] = id;
			frames_.push_back(frame_info);
			id++;
		}
		infile.close();
	}
	else
	{
		//0 ./depth/0.png ./color/0.jpg 0 0
		//1 ./depth/1.png ./color/1.jpg 0 0
		//2 ./depth/2.png ./color/2.jpg 0 0
		while (infile >> timestamp >> depth_path >> rgb_path >> depth_id >> rgb_id) 
		{
			FrameInfo frame_info;
			std::stringstream ss(timestamp.c_str());
			ss >> frame_info.timestamp;
			frame_info.depth_path = base_path + "/" + depth_path;
			if (id == 0) 
			{
				std::cout<<"E.g.:"<<base_path+"/"+depth_path<<std::endl;
			}
			frame_info.depth_id = scene_id+"/"+depth_id;
			frame_info.rgb_path = base_path + "/" + rgb_path;
			frame_info.rgb_id = scene_id+"/"+rgb_id;
			frame_info.labeled_frame = false;
			depth_id_lookup[scene_id+"/"+depth_id] = id;
			frames_.push_back(frame_info);
			id++;
		}
		infile.close();
	}


	//Check if any frames are labelled frames according to the input text file
	std::ifstream inlabelfile(labels_file.c_str());
	std::string frame_id;
	while (inlabelfile >> depth_id >> rgb_id >> frame_id) 
	{
		if (depth_id_lookup.find(depth_id) != depth_id_lookup.end()) 
		{
			int found_id = depth_id_lookup[depth_id];
			frames_[found_id].labeled_frame = true;
			frames_[found_id].frame_id = frame_id;
			std::cout<<"Found:"<<frames_[found_id].depth_path<<std::endl;
			if (frames_[found_id].rgb_id != rgb_id) {
				std::cout<<"Warning, unaligned RGB and Depth frames - depth wins"<<std::endl;
			}
			num_labelled++;
		}
	}
	inlabelfile.close();
}

PNGLogReader::~PNGLogReader()
{
	delete [] decompressionBufferInstanceGroundTruth;
	delete [] decompressionBufferDepth;
	delete [] decompressionBufferDepthFilled;
	delete [] decompressionBufferImage;
}

void PNGLogReader::getNext()
{
	has_instance_GT = false;
	if ((lastGot + 1) < static_cast<int>(frames_.size())) 
	{
		lastGot++;
		FrameInfo info = frames_[lastGot];
		timestamp = info.timestamp;

		//rgb
		cv::Mat rgb_image = cv::imread(info.rgb_path,CV_LOAD_IMAGE_COLOR);
		if (flipColors) 
		{
			cv::cvtColor(rgb_image, rgb_image, CV_BGR2RGB); 
		}

		rgb = (unsigned char *)&decompressionBufferImage[0];
		int index = 0;
		for (int i = 0; i < rgb_image.rows; ++i) 
		{
			for (int j = 0; j < rgb_image.cols; ++j) 
			{
				rgb[index++] = rgb_image.at<cv::Vec3b>(i,j)[0];
				rgb[index++] = rgb_image.at<cv::Vec3b>(i,j)[1];
				rgb[index++] = rgb_image.at<cv::Vec3b>(i,j)[2];
			}
		}

		//depth
		depth = (unsigned short *)&decompressionBufferDepth[0];
		cv::Mat depth_image = cv::imread(info.depth_path,CV_LOAD_IMAGE_ANYDEPTH);
		index = 0;
		for (int i = 0; i < depth_image.rows; ++i) 
		{
			for (int j = 0; j < depth_image.cols; ++j) 
			{
				depth[index++] = depth_image.at<uint16_t>(i,j);
			}
		}

		//depthfilled
		depthfilled = (unsigned short *)&decompressionBufferDepthFilled[0];
		std::string depth_filled_str = info.depth_path;
		depth_filled_str.erase(depth_filled_str.end()-9,depth_filled_str.end());
		depth_filled_str += "depthfilled.png";
		cv::Mat depthfill_image = cv::imread(depth_filled_str,CV_LOAD_IMAGE_ANYDEPTH);
		if (depthfill_image.data) 
		{
			index = 0;
			for (int i = 0; i < depthfill_image.rows; ++i) 
			{
				for (int j = 0; j < depthfill_image.cols; ++j) 
				{
					depthfilled[index++] = depthfill_image.at<uint16_t>(i,j);
				}
			}
			has_depth_filled = true;
		} 
		else 
		{
			has_depth_filled = false;
		}

		//instanceGroundTruth
		if(has_instance_GT)
		{
			instanceGroundTruth = (unsigned char *)&decompressionBufferInstanceGroundTruth[0];
			cv::Mat instanceGT_image = cv::imread(info.instanceGT_path,0);

			//std::cout<<"instanceGT_image:"<<info.instanceGT_path<<std::endl;
			//std::cout<<"instanceGT_image_re.rows:"<<instanceGT_image.rows<<" instanceGT_image_re.cols:"<<instanceGT_image.cols<<std::endl;
		
			cv::Mat instanceGT_image_re;
			cv::resize(instanceGT_image, instanceGT_image_re, cv::Size(Resolution::getInstance().width(), Resolution::getInstance().height()),
																														 (0, 0), (0, 0), cv::INTER_NEAREST);
			

			index = 0;
			for (int i = 0; i < instanceGT_image_re.rows; ++i) 
			{
				for (int j = 0; j < instanceGT_image_re.cols; ++j) 
				{
					instanceGroundTruth[index++] = instanceGT_image_re.at<uint8_t>(i,j);
				}
			}
		}

		imageSize = Resolution::getInstance().numPixels() * 3;
		depthSize = Resolution::getInstance().numPixels() * 2;
	}
}

bool PNGLogReader::isLabeledFrame()
{
	return frames_[lastGot].labeled_frame;
}

std::string PNGLogReader::getLabelFrameId() {
	if (isLabeledFrame()) {
	return frames_[lastGot].frame_id;
	}
	return "";
}

int PNGLogReader::getNumFrames()
{
	return static_cast<int>(frames_.size());
}

bool PNGLogReader::hasMore()
{
	return (lastGot + 1) < static_cast<int>(frames_.size());
}
