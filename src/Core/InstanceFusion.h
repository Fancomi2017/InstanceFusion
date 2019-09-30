/*
 * This file is part of InstanceFusion.
 *
 */

#ifndef MASK_RCNN_INTERFACE_H_
#define MASK_RCNN_INTERFACE_H_ 

#define FLANN_USE_CUDA
#define PRINT_DEBUG_TIMING	//?
#include <flann/flann.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <map_interface/ElasticFusionInterface.h>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include <utilities/Types.h>

#include <atomic>
#include <thread>
#include <queue>
#include <algorithm>


#include <gSLICr/gSLICr_Lib/gSLICr.h>	//gpu
#include <utilities/slic.h>		//cpu

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

const int classNum = 81;
const std::string class_names[classNum] = { "BG", "person", "bicycle", 
"car", "motorcycle", "airplane", "bus", "train", "truck", 
"boat", "traffic light", "fire hydrant", "stop sign", 
"parking meter", "bench", "bird", "cat", "dog", "horse", 
"sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
"backpack", "umbrella", "handbag", "tie", "suitcase", 
"frisbee", "skis", "snowboard", "sports ball", "kite", 
"baseball bat", "baseball glove", "skateboard", "surfboard",
"tennis racket", "bottle", "wine glass", "cup", "fork", 
"knife", "spoon", "bowl", "banana", "apple", "sandwich", 
"orange", "broccoli", "carrot", "hot dog", "pizza", "donut"
, "cake", "chair", "couch", "potted plant", "bed", 
"dining table", "toilet", "tv", "laptop", "mouse", "remote",
"keyboard", "cell phone", "microwave", "oven", "toaster",
"sink", "refrigerator", "book", "clock", "vase", "scissors",
"teddy bear", "hair drier", "toothbrush" };

class InstanceFusion {
public:
	InstanceFusion(int num,int w,int h,bool useMulThread,int netType);
	~InstanceFusion();

	//======================================
	void ProcessSegmentation(const ImagePtr rgb,const DepthPtr depth,const std::unique_ptr<ElasticFusionInterface>& map,
					int frame_num,bool isflann);
	//======================================

	void printHistoryInstances();
	void saveInstanceTable();
	void saveTimeDebug(std::string fileName);

	void evaluateAndSave(const std::unique_ptr<ElasticFusionInterface>& map,std::string fileName);

	void computeMapBoundingBox(const std::unique_ptr<ElasticFusionInterface>& map, bool bboxType);

	void saveInstancePointCloud();
	void getInstancePointCloud(const std::unique_ptr<ElasticFusionInterface>& map,bool bboxType);

	std::vector<ClassColour> getInstanceTable() {  return instanceTable; }

	float* getProjectColorMap_gpu() { return projectColor_gpu; }
	
	float* getMaskColorMap_gpu() { return maskColor_gpu; }
	
	float* getGcMatrix() { return gcMatrix; }

	float* getInstcMatrix() { return instcMatrix; }

	int getSurfelSize() { return surfel_size; }

	int getInstanceNum() { return instanceNum; }

	float* getMapBoundingBox() { return map3DBBox; }

	void getLoopClosureInstanceTable(int* out_table);
	

	void renderProjectMap(const std::unique_ptr<ElasticFusionInterface>& map,bool drawBBox);
	bool whetherDoSegmentation(const std::unique_ptr<ElasticFusionInterface>& map,int frame_num);
	void checkLoopClosure(int* loopClosureInstanceTable,const std::unique_ptr<ElasticFusionInterface>& map);

	//Debug
	void  TimeTick();
	void  TimeTock(std::string name,bool isShow = false);

protected:
        void startThread();
	void runThread();
	//======================================
	void pythonInit();
	void detect();
	void getCurrentResults();
	void destoryCurrentResults();

	void extractMask();
	void extractClassIDs();

	PyObject* createArguments(cv::Mat rgbImage);
	//======================================
	void instanceInit();

	void createInstanceTable();
	void getInstanceTableClassList(int* classList);
	int  getInstanceTableFirstNotUse();
	int  getInstanceTableInUseNum();

	void getInstanceTableCleanList(int* eachInstanceMaximum, int* instanceTableCleanList,int* eachInstanceSumCount);
	void cleanInstanceTable(int* instanceTableCleanList);
	void registerInstanceTable(int emptyIndex, int maskID, int* compareMap);
	float scoreInClean(int eachInstanceMaximum, int eachInstanceSumCount);

	void processInstance(const std::unique_ptr<ElasticFusionInterface>& map,bool isflann);
	void computeCompareMap(int* MasksBBox,int* projectBBox, int* instTableClassList,int* compareMap);

	void flannKnnVoteSurfelMap(const int n,float* map_surfels,int* bestIDInEachSurfel_gpu);

	//Geometric
	void filterAreaCompute(int* filterMap, const unsigned short *depth, int x,int y, int areaFlag,float& points );
	void maskGeometricFilter(DepthPtr depthMap,unsigned char* mask, unsigned char* oriMask);

	//void connectComponentAnalysis(int* filterMap);
	//void maskGeometricFilter2(DepthPtr depthMap);
	//void maskGeometricFilter1(DepthPtr depthMap);
	
	//gSlICr
	void gSLICrInterface(int* segMask);
	void imageCV2SLIC(const cv::Mat& inimg, gSLICr::UChar4Image* outimg);
	void imageSLIC2CV(const gSLICr::UChar4Image* inimg, cv::Mat& outimg);

	void mergeSuperPixel(DepthPtr depthMap,int spNum,int* segMask,int *finalSPixel,const int frameID);
	void getSuperPixelInfo(int* segMask,unsigned short * depthMap,float* posMap,float* normalMap,int spNum,float* spInfo);
	void connectSuperPixel(int spNum,float* spInfo);
	void maskSuperPixelFilter_OverSeg(int spNum,int *finalSPixel);


	//void  cpuDebug_slic

	//Render
	void  renderRawMask();
	void  renderProjectBoundingBox( float* projectColor, float* instanceTable_color, int* projectBBox);
	
	float getDepthThreshold(int depth)
	{
		float threshold = 0.074f*depth-246.0f;			//0.06x-120(4000->120   9000->420)
		threshold = std::max(50.0f,threshold);
		threshold = std::min(420.0f,threshold);
		return threshold;
	}
private:	
	std::thread thread;
	bool useThread;
	bool initSignal;
	bool endSignal;
	bool ProcessSignal;

	//======================================
	ImagePtr inputImage;
	PyObject *pModule;
	PyObject *pExecute;
	int resultMasks_Num;
	int* resultClass_ids;
	unsigned char* resultMasks;
	unsigned char* resultMasks_gpu;
	bool* unavailableMask;

	//======================================
	//InstanceTable
	const std::string undefineName = "?";

	DepthPtr inputDepth;
	int width;
	int height;
	int instanceNum;
	int maskrcnnType;
	std::vector<ClassColour> instanceTable;
	std::vector<float> instanceTable_color;

	//InstanceFusion
	float* projectColor_gpu;
	float* maskColor_gpu;
	float* instanceTable_color_gpu;
	int*   projectBBox;
	int*   tempQueueFAC;	//filterAreaCompute
	float*   map3DBBox;

	//gSlICr
	gSLICr::objects::settings my_settings;
	gSLICr::engines::core_engine* gSLICr_engine;
	int spNum;

	//BBOX
	float gcMatrix[16];
	float* instcMatrix;
	
	float** instSurfels; 

	//======================================
	//const
	const int compareMapType = 1;			//2->mask 100HZ~3HZ   1->box 200HZ
	const int cleanNum = 20;

	const int surfel_size = 64; 			//4 * 16
	const int surfel_normal_offset = 8;	
	const int surfel_instance_offset = 16;	
	const int surfel_rgbColor_offset = 4;		//4	//rgbColor
	const int surfel_instanceColor_offset = 5;	//5	//instanceColor
	const int surfel_instanceGT_offset = 15;	//15	//instanceGT
	//const int surfel_partColor_offset = 15;	//15	//partColor

	const float compareThreshold = 0.5;
	const int depthRatio = 1186;			//DepthMap.d = Model.dis * depthRatio (other scene need try)	1000 in openGL

	//Geometric use
	//const float filterDiffThreshold = 300.0f;	//not use				0-10240?
	//const float filterCoverThreshold = 0.65f;	//not use

	//const int   filterNumThreshold = 300;		//not use

	const float totalGeoThreshold = 0.65f;
	const float eachGeoThreshold =  0.25f;

	const float ratio3DBBox = 1000000;		//cuda max/min() just handle [int] but pos of point is [float] so mul10^6
	//======================================
	//Debug
	timeval time_1,time_2;
	int frameID = -1;
	int lastSegFrameID=-1;
	int cleanTimes;					//how many times do clean
	int instanceFusionTimes;			//how many times do instanceFusion
	int flannTimes;					//how many times do flann
	int GuiTime;					//how many times do GUI
	int elasticFusionTime;				//how many times do elasticFusion == frame ID (if instanceFusion always on)
	
	const int  tickChain_num = 60;
	std::string* tickChain_name;
	long double* tickChain_time;
	int tickChain_p;
	
	//cpu_debug
	//const bool cpuDebug = true;
	//float* map_cpu;


	
};


#endif /* MASK_RCNN_INTERFACE_H_ */
