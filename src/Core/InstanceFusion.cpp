/*
 * This file is part of InstanceFusion.
 *
 */

#include "InstanceFusion.h"
#include "InstanceFusionCuda.h"

//==============================Init=======================================================

InstanceFusion::InstanceFusion(int num,int w,int h,bool useMulThread,int netType)
{
	//construct
	instanceNum = num;
	width = w;
	height = h;
	useThread = useMulThread;
	maskrcnnType = netType;

	//InstanceFusion
	instanceInit();

	//MaskRcnn
	if(!useThread){
		pythonInit();		//PlanA  synchronous
		std::cout << "useThread: false" << std::endl;
	}else{
		startThread();		//PlanB  asynchronous
		std::cout << "useThread: true" << std::endl;
	}

}

InstanceFusion::~InstanceFusion()
{
	//释放python
	if(pModule)  Py_XDECREF(pModule);
	if(pExecute) Py_XDECREF(pExecute);
	Py_Finalize();

	//释放Thread
	if(useThread)
	{
		endSignal = true;
	}

	//释放GUI相关
	if(instanceTable_color_gpu)	cudaFree(instanceTable_color_gpu);
	if(projectColor_gpu)		cudaFree(projectColor_gpu);
	if(maskColor_gpu)			cudaFree(maskColor_gpu);
	if(projectBBox) 			delete[] projectBBox;
	if(tempQueueFAC)			free(tempQueueFAC);

	//3D_BBOX
	if(map3DBBox) 				delete[] map3DBBox;
	if(instcMatrix)				free(instcMatrix);

	if(instSurfels)	
	{
		for(int i=0;i<instanceNum;i++)
		{
			if(instSurfels[i]) delete instSurfels[i];
			instSurfels[i]=0;
		}
		delete[] instSurfels;
		instSurfels = 0;
	}	

	//gSLICr
	if(gSLICr_engine)			delete gSLICr_engine;
	
	//instanceTable
	if(tickChain_name)			delete[] tickChain_name;
	if(tickChain_time)			delete[] tickChain_time;

	//cpudebug
	//if(map_cpu)				free(map_cpu);

}

//==================================Thread===========================================================
void InstanceFusion::startThread()
{
	initSignal = false;
	endSignal = false;

	std::cout << "* InstanceFusion Main thread: " << std::this_thread::get_id() << std::endl;
	if(thread.get_id() == std::thread::id())
			thread = std::thread(&InstanceFusion::runThread, this);

	while(!initSignal) usleep(20);
}

void InstanceFusion::runThread()
{
	std::cout << "* InstanceFusion run thread: " << std::this_thread::get_id()  << std::endl;
	pythonInit();
	
	//mainLoop
	while(1)
	{
		if(endSignal) break;
		usleep(20);
		if(!ProcessSignal) continue;
		detect();
		getCurrentResults();
		ProcessSignal = false;
	}
}


//====================================MaskRcnn=======================================================
// MaskRcnn初始化
void InstanceFusion::pythonInit()
{
	//初始化python
	Py_Initialize();
		
	//PyRun_SimpleString("import os");
	//PyRun_SimpleString("print(os.getcwd())");

	std::cout << "* Init MaskRcnn " << std::endl;

	if(maskrcnnType==0)		//maskrcnn-benchmark+fbnet
	{
		Py_SetProgramName((wchar_t*)L"mask_benchmark");
		wchar_t const * argv2[] = { L"mask_benchmark.py" };
		PySys_SetArgv(1, const_cast<wchar_t**>(argv2));

		// Load module
		std::cout << " * Loading module..." << std::endl;
		pModule = PyImport_ImportModule("mask_benchmark");
		if(pModule == NULL) {
			if(PyErr_Occurred()) {
				std::cout << "Python error indicator is set:" << std::endl;
				PyErr_Print();
			}
			throw std::runtime_error("Could not open MaskRCNN module.");
		}
	}
	else if(maskrcnnType==1)	//ori maskrcnn
	{
		Py_SetProgramName((wchar_t*)L"mask_ori");
		wchar_t const * argv2[] = { L"mask_ori.py" };
		PySys_SetArgv(1, const_cast<wchar_t**>(argv2));

		// Load module
		std::cout << " * Loading module..." << std::endl;
		pModule = PyImport_ImportModule("mask_ori");
		if(pModule == NULL) {
			if(PyErr_Occurred()) {
				std::cout << "Python error indicator is set:" << std::endl;
				PyErr_Print();
			}
			throw std::runtime_error("Could not open MaskRCNN module.");
		}
	}
	else if(maskrcnnType==2)	//paddle_maskrcnn
	{
		Py_SetProgramName((wchar_t*)L"mask_paddle");
		wchar_t const * argv2[] = { L"mask_paddle.py" };
		PySys_SetArgv(1, const_cast<wchar_t**>(argv2));

		// Load module
		std::cout << " * Loading module..." << std::endl;
		pModule = PyImport_ImportModule("mask_paddle");
		if(pModule == NULL) {
			if(PyErr_Occurred()) {
				std::cout << "Python error indicator is set:" << std::endl;
				PyErr_Print();
			}
			throw std::runtime_error("Could not open MaskRCNN module.");
		}
	}

	// Get function
	pExecute = PyObject_GetAttrString(pModule, "detect");
	if(pExecute == NULL || !PyCallable_Check(pExecute)) {
		if(PyErr_Occurred()) {
			std::cout << "Python error indicator is set:" << std::endl;
			PyErr_Print();
		}
		throw std::runtime_error("Could not load function 'execute' from MaskRCNN module.");
	}

	//初始化var
	resultMasks_Num = -1;
	ProcessSignal = false;
	initSignal = true;		//Finish
}

bool InstanceFusion::whetherDoSegmentation(const std::unique_ptr<ElasticFusionInterface>& map,int frame_num)
{
	const int downsample = 10;
	const int fixedL = 2;
	const int fixedH = 45;
	
	int n = map->getMapSurfelCount();
	float* map_surfels = map->getMapSurfelsGpu();
	cudaTextureObject_t index_surfelsIds = map->getSurfelIdsAfterFusionGpu();

	Eigen::Vector3f trans = map->getCurrPose().topRightCorner(3, 1);	//notuse

	//count[0]->sum inst Num	count[1]->not reconstruct num
	int* count_gpu;
	cudaMalloc((void **)&count_gpu, 2 * sizeof(int));
	cudaMemset( count_gpu, 0, 2 * sizeof(int));

	checkProjectDepthAndInstance(index_surfelsIds,n,map_surfels,width,height,surfel_size,surfel_instance_offset,downsample,count_gpu);
	
	int count[2];
	cudaMemcpy(count, count_gpu, 2*sizeof(int), cudaMemcpyDeviceToHost);
	

	bool test1 = count[0]>(width/downsample*height/downsample*0.48*30);	//>50% vote 30
	bool test2 = count[1]<(width/downsample*height/downsample*0.2);		//<20% reconstruct

	if(test1||test2) 
	{
		if(frame_num-lastSegFrameID>fixedH)
		{
			lastSegFrameID = frame_num;
			return true;
		}
		return false;
	}
	else
	{
		if(frame_num-lastSegFrameID>fixedL)
		{
			lastSegFrameID = frame_num;
			return true;
		}
		else return false;
	}
	
	cudaFree(count_gpu);
}


void InstanceFusion::ProcessSegmentation(const ImagePtr rgb,const DepthPtr depth,const std::unique_ptr<ElasticFusionInterface>& map,
											int frame_num,bool isflann)
{
	frameID = frame_num;
	inputImage = rgb;
	inputDepth = depth;

	std::cout<<"Frame: "<<frameID<<std::endl;
	//std::cout<<"useThread: "<<useThread<<std::endl;
	if(!useThread)
	{
		TimeTick();
		detect();
		TimeTock("Detect",true);

		getCurrentResults();	
		renderRawMask();
		processInstance(map,isflann);
		destoryCurrentResults();
	}
	else
	{
		ProcessSignal = true; //signal to "runThread" function
		while(ProcessSignal) usleep(20);
		renderRawMask();
		processInstance(map,isflann);
		destoryCurrentResults();
	}

}

void InstanceFusion::detect()
{
	cv::Mat rgbImage(height, width, CV_8UC3, inputImage);
	

	Py_XDECREF(PyObject_CallFunctionObjArgs(pExecute, createArguments(rgbImage), NULL));
	if(PyErr_Occurred()) 
	{
		std::cout << "Python error indicator is set:" << std::endl;
		PyErr_Print();
	}
	//PyEval_CallObject(pExecute,Py_BuildValue("(s)", "magictong"));
	//PyObject* ArgList = PyTuple_New(1);
	//Py_XDECREF(PyObject_CallFunctionObjArgs(pExecute, ArgList, NULL));
	
	//Debug
	//std::string savename = "./temp/oriImage"+std::to_string(frameID)+".png";
	//imwrite(savename,rgbImage);
}

void InstanceFusion::getCurrentResults()
{
	extractMask();
	extractClassIDs();
	
	if(resultMasks_Num)
	{
		unavailableMask = new bool[resultMasks_Num];					//use After Geometric
		memset(unavailableMask,0, resultMasks_Num * sizeof(bool));
	}
	else unavailableMask = 0;

}


void InstanceFusion::destoryCurrentResults()
{
	resultMasks_Num = -1;
	if(resultMasks_gpu) cudaFree(resultMasks_gpu);	
	if(resultMasks)     delete[] resultMasks;
	if(resultClass_ids) delete[] resultClass_ids;
	if(unavailableMask)	delete[] unavailableMask;

	resultMasks_gpu = 0;
	resultMasks = 0;
	resultClass_ids = 0;
	unavailableMask = 0;
}


void InstanceFusion::extractMask()
{
	PyObject* pMasksList = PyObject_GetAttrString(pModule, "masks");
	if(!pMasksList || pMasksList == Py_None) throw std::runtime_error(std::string("Failed to get python object: masks"));

	if(!PySequence_Check(pMasksList)) throw std::runtime_error("pMasksList is not a sequence.");

	resultMasks_Num = PySequence_Length(pMasksList);

	if(resultMasks_Num) resultMasks = new unsigned char[resultMasks_Num*width*height];
	else resultMasks = 0;

	for (int i = 0; i < resultMasks_Num; i++) 
	{	
		PyObject* o = PySequence_GetItem(pMasksList, i);
		PyArrayObject* pMaskArray = (PyArrayObject*)o;
		//error? check?
		int h = PyArray_DIM(pMaskArray,0);
		int w = PyArray_DIM(pMaskArray,1);
		//std::cout<<"h: "<<h<<" w: "<<w<<std::endl;
		if(width == w && height == h)
		{
			unsigned char* pData = (unsigned char*)PyArray_GETPTR1(pMaskArray,0);
			memcpy(resultMasks+i*width*height,pData,h*w);
		}
		Py_DECREF(o);
	}

	if(resultMasks_Num)
	{
	 	cudaMalloc((void **)&resultMasks_gpu, resultMasks_Num*width*height * sizeof(unsigned char));
		cudaMemcpy(resultMasks_gpu, resultMasks, resultMasks_Num*width*height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	}
	else resultMasks_gpu = 0;

	Py_DECREF(pMasksList);
	return;

	//PyObject* pImage = PyObject_GetAttrString(pModule, "resultMasks");
	//if(!pImage || pImage == Py_None) throw std::runtime_error(std::string("Failed to get python object: resultMasks"));

	//PyArrayObject *pImageArray = (PyArrayObject*)(pImage);

	//unsigned char* pData = (unsigned char*)PyArray_GETPTR1(pImageArray,0);
	//int h = PyArray_DIM(pImageArray,0);
	//int w = PyArray_DIM(pImageArray,1);
	//int n = PyArray_DIM(pImageArray,2);
	//assert(width == w && height == h);
	//std::cout<<"h: "<<h<<"w: "<<w<<"n: "<<n<<std::endl;
	
	//resultMasks = new unsigned char[h*w*n];
	//memcpy(resultMasks,pData,h*w*n);

	//resultMasks_Num = n;

	//Py_DECREF(pImage);
}

void InstanceFusion::extractClassIDs()
{
	assert(resultMasks_Num != -1);
   
	PyObject* pClassList = PyObject_GetAttrString(pModule, "class_ids");
	if(!pClassList || pClassList == Py_None) throw std::runtime_error(std::string("Failed to get python object: class_ids"));

	if(!PySequence_Check(pClassList)) throw std::runtime_error("pClassList is not a sequence.");

	assert(resultMasks_Num == PySequence_Length(pClassList));

	if(resultMasks_Num) resultClass_ids = new int[resultMasks_Num];
	else resultClass_ids = 0;

	for (int i = 0; i < resultMasks_Num; i++) 
	{
		PyObject* o = PySequence_GetItem(pClassList, i);
		//assert(PyLong_Check(o));	//error?
		resultClass_ids[i] = PyLong_AsLong(o);
		Py_DECREF(o);
	}
	Py_DECREF(pClassList);
}

PyObject* InstanceFusion::createArguments(cv::Mat rgbImage)
{
	assert(rgbImage.channels() == 3);
	npy_intp dims[3] = { rgbImage.rows, rgbImage.cols, 3 };
	import_array();
	return PyArray_SimpleNewFromData(3, dims, NPY_UINT8, rgbImage.data); 
}
//====================InstanceFusion==================================================

void InstanceFusion::instanceInit()
{
	//创建(仅有Color)InstanceTable
	createInstanceTable();
	cleanTimes = 0;
	instanceFusionTimes = 0;
	flannTimes = 0;
	GuiTime = 0;
	elasticFusionTime = 0;

	tickChain_name = new std::string[tickChain_num];
	tickChain_time = new long double[tickChain_num];
	tickChain_p = 0;
	memset(tickChain_time,0, tickChain_num * sizeof(long double));
	memset(tickChain_name,0, tickChain_num * sizeof(std::string));

	//cuda init
	cudaMalloc((void **)&projectColor_gpu,  4 * width * height * sizeof(float));
	cudaMalloc((void **)&maskColor_gpu,  4 * width * height * sizeof(float));
	cudaMemset(projectColor_gpu,0, 4 * width * height * sizeof(float));
	cudaMemset(maskColor_gpu,0, 4 * width * height * sizeof(float));

	//GUI
	projectBBox = NULL;

	//Geometric
	tempQueueFAC = (int*)malloc(width*height*sizeof(int));


	// gSLICr settings
	my_settings.img_size.x = width;
	my_settings.img_size.y = height;
	my_settings.no_segs = 2000;
	my_settings.spixel_size = 16;
	my_settings.coh_weight = 0.6f;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::XYZ; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step
	spNum = (my_settings.img_size.x*my_settings.img_size.y) / (my_settings.spixel_size*my_settings.spixel_size);
	gSLICr_engine = new gSLICr::engines::core_engine(my_settings);


	//flann
	//std::cout<<"FLANN_VERSION_:"<<FLANN_VERSION_<<std::endl;	//opencv_flann?

	//BBOX
	map3DBBox = 0;
	instcMatrix = (float*)malloc(16*instanceNum*sizeof(int));
	instSurfels = new float*[instanceNum];


	//cpu debug
	//map_cpu = (float*)malloc(surfel_size*3000000*sizeof(float));
	//memset(map_cpu,0, surfel_size*3000000*sizeof(float));
}

void InstanceFusion::filterAreaCompute(int* filterMap, const unsigned short *depth, int x,int y, int areaFlag,float& points )
{
	const int StepX[4] = {0,0,1,-1};
	const int StepY[4] = {1,-1,0,0};

	int front = 0;
	int tail = 0;
	tempQueueFAC[front++] = y*width+x;

	while(front>tail)
	{
		int now = tempQueueFAC[tail++];
		int nx = now%width;
		int ny = now/width;
		
		//if(filterMap[ny*width+nx] == areaFlag || filterMap[ny*width+nx] == 0) continue;
		//if(depth[ny*width+nx] == 0) continue;

		if(filterMap[ny*width+nx] != 1) continue;
		
		points++;
		filterMap[ny*width+nx] = areaFlag;
	
		//check tempQueueFAC (out of memory)
		if(front>=width*height)
		{
			std::cout<<"tempQueueFAC(out of memory)"<<std::endl;
			continue;
		}
	
		for(int i=0;i<4;i++)
		{
			int dx = nx+StepX[i];
			int dy = ny+StepY[i];
			float threshold = getDepthThreshold(depth[ny*width+nx]);	//float threshold = 300;
			if(filterMap[dy*width+dx]==1&&abs(depth[ny*width+nx]-depth[dy*width+dx])<threshold )
				tempQueueFAC[front++] = dy*width+dx;
		}
	}
}

void InstanceFusion::maskGeometricFilter(DepthPtr depthMap,unsigned char* mask, unsigned char* oriMask)
{
	int filterMap[width*height];
	for(int i=0;i<resultMasks_Num;i++)
	{
		if(unavailableMask[i]) continue;

		//Step 1 copy filterMap
		memset(filterMap,0, width*height * sizeof(int));
		float oriPoints = 0;
		for(int y = 1; y< height-1; y++)
		{
			for(int x = 1; x< width-1; x++)
			{
				if( oriMask[i*width*height+y*width+x] )	
				{
					oriPoints++;
				}
				if( mask[i*width*height+y*width+x] && depthMap[y*width+x] )
				{
					filterMap[y*width+x] = 1;
				}
			}
		}

		//Step 2 find List		(ori-find MAX)
		int areaFlag = 2;	
		//int maxArea = areaFlag;
		//int maxNum = 0;
		int list[20];
		int p = 0;
		for(int y = 1; y< height-1; y++)
		{
			for(int x = 1; x< width-1; x++)
			{
				if(filterMap[y*width+x] == 1)
				{
					float points = 0;
					filterAreaCompute(filterMap, depthMap, x, y, areaFlag, points);
					//if(points > maxNum)
					//{
					//	maxNum = points;
					//	maxArea = areaFlag;
					//}
					if(points/oriPoints>eachGeoThreshold)
					{
						if(p<20) list[p++] = areaFlag;
					}
					areaFlag++;
				}
			}
		}

		//Step 3 Clear other Area
		float finalPoints = 0;
		for(int y = 0; y< height ; y++)
		{
			for(int x = 0; x< width ; x++)
			{
				int flag = 0;
				for(int j=0;j<p;j++)
				{
					if(filterMap[y*width+x]== list[j]  )
					{	
						flag = 1;
						break;
					}
				}
				if(flag)
				{
					mask[i*width*height+y*width+x] = 255;
					finalPoints ++;
				}
				else
				{
					mask[i*width*height+y*width+x] = 0;
				}
			}
		}
		if(finalPoints/oriPoints<totalGeoThreshold) unavailableMask[i]=1;
		//std::cout<<"Mask: "<<i<<" "<<finalPoints<<" "<<oriPoints<<" unavailable="<<unavailableMask[i]<<std::endl;
	}
}

void InstanceFusion::computeCompareMap(int* MasksBBox,int* projectBBox, int* instTableClassList,int* compareMap)
{
	int minX_i,minX_m;
	int maxX_i,maxX_m;
	int minY_i,minY_m;
	int maxY_i,maxY_m;
	for(int maskID = 0;maskID<resultMasks_Num;maskID++)
	{
		minX_m = MasksBBox[maskID*4+0];
		maxX_m = MasksBBox[maskID*4+1];
		minY_m = MasksBBox[maskID*4+2];
		maxY_m = MasksBBox[maskID*4+3];
		//std::cout<<"maskID: "<<maskID<<" class: "<< class_names[resultClass_ids[maskID]] <<"\t\tmin("<<
		//											minX_m<<","<<minY_m<<") max("<<maxX_m<<","<<maxY_m<<")"<<std::endl;
		if(maxX_m<=minX_m||maxY_m<=minY_m||unavailableMask[maskID])
		{
			//std::cout<<" ...unavailableMask"<<std::endl;
			unavailableMask[maskID] = 1;
			continue;
		}
		
		int   bestInstanceID  = -1;
		float bestInstanceIOU = 0;
		for(int instanceID = 0;instanceID<instanceNum;instanceID++)
		{
			if(instTableClassList[instanceID] != -1)	//instance is exist
			{
				if(resultClass_ids[maskID] == instTableClassList[instanceID])	//InstanceClass == MaskClass
				{
					minX_i = projectBBox[instanceID*4+0];
					maxX_i = projectBBox[instanceID*4+1];
					minY_i = projectBBox[instanceID*4+2];
					maxY_i = projectBBox[instanceID*4+3];
					if(maxX_i<=minX_i||maxY_i<=minY_i)	continue;

					float Intersection_W  = (std::min(maxX_i,maxX_m) - std::max(minX_i,minX_m));
					float Intersection_H  = (std::min(maxY_i,maxY_m) - std::max(minY_i,minY_m));
					if(Intersection_W<=0||Intersection_H<=0) continue;
					
					float Intersection = Intersection_W * Intersection_H;
					float Union = ((maxX_i - minX_i) * (maxY_i - minY_i))+((maxX_m - minX_m) * (maxY_m - minY_m)) - Intersection;
					
					//Debug
					//std::cout<<"->instanceID: "<<instanceID<<"  "<<instanceTable[instanceID].name<<"\t\tmin("<<
					//														minX_i<<","<<minY_i<<") max("<<maxX_i<<","<<maxY_i<<")";
					
					if( Intersection / Union > compareThreshold)
					{
						bestInstanceID = instanceID;
						bestInstanceIOU = Intersection / Union;
					}
				}
			}
		}
		if(bestInstanceID>0 ) compareMap[bestInstanceID+maskID*instanceNum] = 1;
	}
}



void InstanceFusion::processInstance(const std::unique_ptr<ElasticFusionInterface>& map,bool isflann)
{
	int n = map->getMapSurfelCount();
	//int del = map->getMapDeletedSurfelCount();
	
	
	//================init in Each frame=========================
	if(projectBBox) delete[] projectBBox;
	projectBBox = 0;
	if(map3DBBox) delete[] map3DBBox;
	map3DBBox = 0;

	memset(instcMatrix,0, 16 * instanceNum * sizeof(float));
	memset(gcMatrix,0, 16 * sizeof(float));
	gcMatrix[0]=1;	gcMatrix[5] =-1; gcMatrix[10]=-1; gcMatrix[15]=1;
	for(int i=0;i<instanceNum;i++)
	{
		instcMatrix[16*i+0]=1;	instcMatrix[16*i+5] =-1; instcMatrix[16*i+10]=-1; instcMatrix[16*i+15]=1;
	}
	for(int i=0;i<instanceNum;i++)
	{
		if(instSurfels[i]) delete instSurfels[i];
		instSurfels[i]=0;
	}
	//===========================================================

	if( instanceNum != ( surfel_size - surfel_instance_offset ) * 2 )
	{
		std::cout<<"instanceNum is inconsistent with the surfelsInfo"<<std::endl;
		std::cout<<"step ERROR"<<std::endl<<std::endl;
		return;
	}
	if( resultMasks_Num == 0 )
	{
		std::cout<<"maskRcnn has 0 result"<<std::endl;
		std::cout<<"step end"<<std::endl<<std::endl;
		return;
	}
	if( !n )
	{
		std::cout<<"map has 0 surfels"<<std::endl;
		std::cout<<"step end"<<std::endl<<std::endl;
		return;
	}
	std::cout<<std::endl;	
	std::cout<<"n: "<<n<<std::endl;
	std::cout<<"InUse instance: "<<getInstanceTableInUseNum() <<std::endl;


	//BAK ORI MASK
	unsigned char* oriResultMasks = (unsigned char*)malloc(resultMasks_Num*width*height * sizeof(unsigned char));
	memcpy(oriResultMasks, resultMasks, resultMasks_Num*width*height * sizeof(unsigned char));

	//step 0	mask post processing
	//Step 0_0 CPU Geometric
	//TimeTick();
	//maskGeometricFilter2(inputDepth);
	//maskGeometricFilter(inputDepth,resultMasks,oriResultMasks);
	//cudaMemcpy(resultMasks_gpu, resultMasks, resultMasks_Num*height*width * sizeof(unsigned char), cudaMemcpyHostToDevice);
	//TimeTock("maskGeometricFilter");

	//Step 0_1 CUDA CleanOverlap	(sort is in python)
	maskCleanOverlap(resultMasks_gpu,resultMasks_Num, width, height);
	cudaMemcpy(resultMasks, resultMasks_gpu, resultMasks_Num*height*width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	//SuperPixels
	std::cout<<std::endl;
	//step -1_1 gSLICr
	TimeTick();
	int segMask[height*width];			//output
	gSLICrInterface(segMask);
	TimeTock("gSLICr");

	
	//Step -1_2	 merge SuperPixel by inputDepth
	//TimeTick();
	int finalSPixel[height*width];		//output
	mergeSuperPixel(inputDepth,spNum,segMask,finalSPixel,frameID);
	//TimeTock("Clustering&Merging");

	//Step -1_3 CPU Geometric
	TimeTick();
	maskSuperPixelFilter_OverSeg(spNum,finalSPixel);
	cudaMemcpy(resultMasks_gpu, resultMasks, resultMasks_Num*height*width * sizeof(unsigned char), cudaMemcpyHostToDevice);
	TimeTock("middleMask");
		
	//Debug ORI_MASK
	
	//std::cout<<std::endl<<"result Num: "<<resultMasks_Num<<std::endl;
	/*
	for(int i=0;i<resultMasks_Num;i++)
	{
		cv::Mat maskImage(height, width, CV_8UC1, oriResultMasks+i*width*height);
		std::string savename = "./mask/mask"+std::to_string(frameID) +"_"+ std::to_string(i) +"_"+std::to_string(resultClass_ids[i])
							+"_A"+std::to_string(unavailableMask[i])+".png";
		imwrite(savename.c_str(),maskImage);
	}
	//Debug Middle MASK
	//std::cout<<std::endl<<"result Num: "<<resultMasks_Num<<std::endl;
	for(int i=0;i<resultMasks_Num;i++)
	{
		cv::Mat maskImage(height, width, CV_8UC1, resultMasks+i*width*height);
		std::string savename = "./mask/mask"+std::to_string(frameID)+"_"+ std::to_string(i) +"_"+std::to_string(resultClass_ids[i])
							+"_B"+std::to_string(unavailableMask[i])+".png";
		imwrite(savename.c_str(),maskImage);
	}
	*/
	
	//std::cout<<"\nstep 0"<<std::endl;

	//step 1	get Project, A map 480*640*96, is the number of each instance.
	float* map_surfels = map->getMapSurfelsGpu();
	cudaTextureObject_t index_surfelsIds = map->getSurfelIdsAfterFusionGpu();	//input		getInstanceSurfelIdsGpu
	
	int* projectInstanceList_gpu;			//output
 	cudaMalloc((void **)&projectInstanceList_gpu, height * width * (instanceNum+1) * sizeof(int));
	cudaMemset(projectInstanceList_gpu, 0, height * width * (instanceNum+1) * sizeof(int));

	TimeTick();
	getProjectInstanceList(index_surfelsIds,n,map_surfels,projectInstanceList_gpu,width,height,surfel_size,surfel_instance_offset);
	TimeTock("ProjectInstanceImage");
	//std::cout<<"step 1"<<std::endl;

	//step 2	get CompareMap, resultMasks[i] and instances[j] is one instance
	//Step 2 (Plan A use box_iou compareMap)
	//Step 2_1
	int* MasksBBox_gpu;							//output
	cudaMalloc((void **)&MasksBBox_gpu, resultMasks_Num * 4 * sizeof(int));
	int* projectBBox_gpu;						//output
	cudaMalloc((void **)&projectBBox_gpu, instanceNum * 4 * sizeof(int));
	TimeTick();
	computeProjectBoundingBox(resultMasks_gpu, resultMasks_Num, projectInstanceList_gpu, width, height, instanceNum, MasksBBox_gpu, projectBBox_gpu);
	TimeTock("2dBoundingBox(M&I)");
	int MasksBBox[ resultMasks_Num * 4 ];		//Next Input
	cudaMemcpy( MasksBBox, MasksBBox_gpu, resultMasks_Num * 4 * sizeof(int), cudaMemcpyDeviceToHost);
	projectBBox = new int[ instanceNum * 4 ];	//Next Input
	cudaMemcpy( projectBBox, projectBBox_gpu, instanceNum * 4 * sizeof(int), cudaMemcpyDeviceToHost);
	//std::cout<<"step 2_1"<<std::endl;

	//Step 2_2
	int  instTableClassList[instanceNum];		//input
	getInstanceTableClassList(instTableClassList);
	int compareMap[resultMasks_Num * instanceNum];	//output
	memset(compareMap,0, resultMasks_Num * instanceNum * sizeof(int));
	TimeTick();
	computeCompareMap(MasksBBox,projectBBox,instTableClassList,compareMap);
	TimeTock("CompareMap");
	//std::cout<<"step 2_2"<<std::endl;

	//Step 2 (Plan B use mask_iou compareMap)
	if(compareMapType==2)									//you can undo this scpoe then use PLAN A	[ I'm not do optimization :) ]
	{
		TimeTick();
		memset(compareMap,0, resultMasks_Num * instanceNum * sizeof(int));
		//Step2_1
		int* tempI_gpu;					//temp
		cudaMalloc((void **)&tempI_gpu, resultMasks_Num * instanceNum * sizeof(int));
		cudaMemset(tempI_gpu, 0, resultMasks_Num * instanceNum * sizeof(int));
		int* tempU_gpu;					//temp
		cudaMalloc((void **)&tempU_gpu, resultMasks_Num * instanceNum * sizeof(int));
		cudaMemset(tempU_gpu, 0, resultMasks_Num * instanceNum * sizeof(int));
		
		int* resultClass_ids_gpu;	//input
		cudaMalloc((void **)&resultClass_ids_gpu, resultMasks_Num * sizeof(int));
		cudaMemcpy( resultClass_ids_gpu, resultClass_ids, resultMasks_Num * sizeof(int), cudaMemcpyHostToDevice);
		int* instTableClassList_gpu;	//input
		cudaMalloc((void **)&instTableClassList_gpu, instanceNum * sizeof(int));
		cudaMemcpy( instTableClassList_gpu, instTableClassList, instanceNum * sizeof(int), cudaMemcpyHostToDevice);
		
		maskCompareMap(resultMasks_gpu, resultMasks_Num, resultClass_ids_gpu, projectInstanceList_gpu, width, height, 
						instanceNum, instTableClassList_gpu, tempI_gpu, tempU_gpu);
	
		//Step2_2
		int tempI[resultMasks_Num * instanceNum];
		int tempU[resultMasks_Num * instanceNum];
		cudaMemcpy( tempI, tempI_gpu, resultMasks_Num * instanceNum * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy( tempU, tempU_gpu, resultMasks_Num * instanceNum * sizeof(int), cudaMemcpyDeviceToHost);

		
		for(int maskID = 0;maskID<resultMasks_Num;maskID++)
		{
			if(unavailableMask[maskID]) continue;

			int   bestInstanceID  = -1;
			float bestInstanceIOU = 0;
			for(int instanceID = 0;instanceID<instanceNum;instanceID++)
			{
				//if(instanceID<5)std::cout<<tempI[instanceID + maskID * instanceNum] <<","<< tempU[instanceID + maskID * instanceNum]<<" ";

				if(instTableClassList[instanceID] != -1)	//instance is exist
				{
					if(resultClass_ids[maskID] == instTableClassList[instanceID])	//InstanceClass == MaskClass
					{
						if( float(tempI[instanceID + maskID * instanceNum]) / tempU[instanceID + maskID * instanceNum] > compareThreshold)
						{
							if(float(tempI[instanceID + maskID * instanceNum]) / tempU[instanceID + maskID * instanceNum]> bestInstanceIOU)
							{
								bestInstanceID = instanceID;
								bestInstanceIOU = float(tempI[instanceID + maskID * instanceNum]) / tempU[instanceID + maskID * instanceNum];
							}
						}				
					}
				}
			}
			if(bestInstanceID>0 )compareMap[bestInstanceID + maskID * instanceNum] = 1;
			//std::cout<<std::endl;
		}
			
		cudaFree(tempI_gpu);
		cudaFree(tempU_gpu);

		cudaFree(resultClass_ids_gpu);
		cudaFree(instTableClassList_gpu);
		
		TimeTock("CompareMap2",true);
	}


	//step 3_0	get projectDepthMap and mask post processing(3)
	//step 3_0_1 projectDepthMap
	float* currPoseTrans_gpu;						//input
	Eigen::Vector3f trans = map->getCurrPose().topRightCorner(3, 1);
	cudaMalloc((void **)&currPoseTrans_gpu, 3 * sizeof(float));
	cudaMemcpy(currPoseTrans_gpu, &trans(0), 3 * sizeof(float), cudaMemcpyHostToDevice);

	unsigned short* projectDepthMap_gpu;			//output
 	cudaMalloc((void **)&projectDepthMap_gpu, height * width * sizeof(unsigned short));
	cudaMemset(projectDepthMap_gpu, 0, height * width  * sizeof(unsigned short));
	TimeTick();
	getProjectDepthMap(index_surfelsIds, n, map_surfels, projectDepthMap_gpu, width, height, surfel_size,currPoseTrans_gpu,depthRatio);
	TimeTock("ProjectDepthMap");
	unsigned short projectDepthMap[height * width];	//Next Input
	cudaMemcpy(projectDepthMap, projectDepthMap_gpu, height * width * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	

	//Step 3_0_4 CPU Geometric
	TimeTick();
	maskGeometricFilter(projectDepthMap,resultMasks,oriResultMasks);
	TimeTock("Geometric");
	cudaMemcpy(resultMasks_gpu, resultMasks, resultMasks_Num*height*width * sizeof(unsigned char), cudaMemcpyHostToDevice);

	
	//Debug project Depth
	/*
	cv::Mat projDepthImage(height, width, CV_16UC1, projectDepthMap);
	std::string savenameProj = "./temp/projectDepth.png";
	imwrite(savenameProj.c_str(),projDepthImage);
	*/

	//Debug MASK 
	/*
	//std::cout<<std::endl<<"result Num: "<<resultMasks_Num<<std::endl;
	for(int i=0;i<resultMasks_Num;i++)
	{
		cv::Mat maskImage(height, width, CV_8UC1, resultMasks+i*width*height);
		std::string savename = "./mask/mask"+std::to_string(frameID)+"_"+ std::to_string(i) +"_"+std::to_string(resultClass_ids[i])
							+"_C"+std::to_string(unavailableMask[i])+".png";
		imwrite(savename.c_str(),maskImage);
	}
	*/

	//step 3	for each mask, register a new instance, and update instance count
	std::cout<<std::endl;
	for(int maskID = 0; maskID < resultMasks_Num; maskID++)
	{
		//Debug
		//std::cout<<std::endl<<"mask: "<<maskID <<std::endl;

		bool instanceExist = false;
		for(int instanceID = 0; instanceID < instanceNum; instanceID++)
		{
			if(compareMap[instanceID + maskID * instanceNum] == 1)
			{
				instanceExist = true;
				break;
			}
		}
		
		//Step 3_1 register a new instance, if instanceTable is full, then clean.
		if(!instanceExist && !unavailableMask[maskID])
		{
			int emptyIndex = getInstanceTableFirstNotUse();

			if(emptyIndex == -1 )	//rarely triggered
			{
				cleanTimes++;

				std::cout<<"  clean"<<std::endl;	
				
				//Step 3_1_1 maximum and count
				int* eachInstanceMaximum_gpu;		//output InstanceFusion
 				cudaMalloc((void **)&eachInstanceMaximum_gpu, instanceNum * sizeof(int));
				cudaMemset(eachInstanceMaximum_gpu, 0, instanceNum * sizeof(int));

				int* eachInstanceSumCount_gpu;			//output
 				cudaMalloc((void **)&eachInstanceSumCount_gpu, instanceNum * sizeof(int));
				cudaMemset(eachInstanceSumCount_gpu, 0, instanceNum * sizeof(int));

				TimeTick();
				computeMaxCountInMap(n,map_surfels,surfel_size,surfel_instance_offset,eachInstanceMaximum_gpu, eachInstanceSumCount_gpu);
				TimeTock("MapCountInstance");

				//std::cout<<"step 3_1_1"<<std::endl;

				//Step 3_1_2 sort (not cuda)
				int eachInstanceMaximum[instanceNum];			// each int is 0-65535(16bits)		input
				cudaMemcpy(eachInstanceMaximum, eachInstanceMaximum_gpu, instanceNum * sizeof(int), cudaMemcpyDeviceToHost);

				int eachInstanceSumCount[instanceNum];			// each int is int					input
				cudaMemcpy(eachInstanceSumCount, eachInstanceSumCount_gpu, instanceNum * sizeof(int), cudaMemcpyDeviceToHost);

				int instanceTableCleanList[instanceNum];		// each int is (0 or 1)				output
				
				TimeTick();
				getInstanceTableCleanList(eachInstanceMaximum,instanceTableCleanList,eachInstanceSumCount);
				TimeTock("CleanList");

				//Step 3_1_3 cleanInstanceFusion
				int* instanceTableCleanList_gpu;				//input
			 	cudaMalloc((void **)&instanceTableCleanList_gpu, instanceNum * sizeof(int));
				cudaMemcpy(instanceTableCleanList_gpu, instanceTableCleanList, instanceNum * sizeof(int), cudaMemcpyHostToDevice);
				
				TimeTick();
				cleanInstanceTable(instanceTableCleanList);
				cleanInstanceTableMap(n,map_surfels,surfel_size,surfel_instance_offset,instanceTableCleanList_gpu);
				TimeTock("cleanInstance");
				//std::cout<<"step 3_1_3"<<std::endl;

				//reCompute
				TimeTick();
				cudaMemset(projectInstanceList_gpu,0, height * width * instanceNum * sizeof(int));												//reStep 1		
				getProjectInstanceList(index_surfelsIds,n,map_surfels,projectInstanceList_gpu,width,height,surfel_size,surfel_instance_offset);	//reStep 1
				
				computeProjectBoundingBox(resultMasks_gpu, resultMasks_Num, projectInstanceList_gpu, 											//reStep 2
										width, height, instanceNum, MasksBBox_gpu, projectBBox_gpu);
				cudaMemcpy( MasksBBox, MasksBBox_gpu, resultMasks_Num * 4 * sizeof(int), cudaMemcpyDeviceToHost);								//reStep 2
				cudaMemcpy( projectBBox, projectBBox_gpu, instanceNum * 4 * sizeof(int), cudaMemcpyDeviceToHost);								//reStep 2
				getInstanceTableClassList(instTableClassList);																					//reStep 2
				memset(compareMap,0, resultMasks_Num * instanceNum * sizeof(int));																//reStep 2
				computeCompareMap(MasksBBox,projectBBox,instTableClassList,compareMap);															//reStep 2
				
				emptyIndex = getInstanceTableFirstNotUse();																						//reStep 3_1	emptyIndex
				TimeTock("reCompute");
	
				//free
				cudaFree(eachInstanceMaximum_gpu);
				cudaFree(instanceTableCleanList_gpu);
				cudaFree(eachInstanceSumCount_gpu);
			}
			
			//Step 3_1_4 register
			TimeTick();
			registerInstanceTable(emptyIndex,maskID,compareMap);																				//reStep 3_1	emptyIndex
			TimeTock("register");
			
			//std::cout<<"Mask: "<<maskID<<" register: "<< emptyIndex << " class: " << class_names[resultClass_ids[maskID]]<<std::endl;
			
		}

		//Step 3_2 add up
		TimeTick();
		for(int instanceID = 0; instanceID < instanceNum; instanceID++)
		{
			if(compareMap[instanceID + maskID * instanceNum] == 1)
			{
				//std::cout<<"update: I:"<< instanceID << " M: " <<maskID<<std::endl;

				//Step 3_2_1 old Point(Project_Before)
				cudaTextureObject_t index_surfelsIdsBefore = map->getSurfelIdsAfterFusionGpu();	//input	getSurfelIdsBeforeFusionGpu
				updateSurfelMapInstance(index_surfelsIdsBefore,width,height,n,map_surfels,surfel_size,surfel_instance_offset,resultMasks_gpu,maskID,instanceID,-1);

				//Step 3_2_2 new Point(Project_After+id>delete_id)
				//cudaTextureObject_t index_surfelsIdsAfter = map->getSurfelIdsAfterFusionGpu();		//input
				//updateSurfelMapInstance(index_surfelsIdsAfter,width,height,n,map_surfels,surfel_size,surfel_instance_offset,resultMasks_gpu,maskID,instanceID,del);
			}
		}
		TimeTock("updateMap",false);
		//std::cout<<"step 3_2"<<std::endl;
		
	}
	//std::cout<<"step 3"<<std::endl;
	

		
	//Step 4 colour	(temp color)
	int* bestIDInEachSurfel_gpu;	//output
 	cudaMalloc((void **)&bestIDInEachSurfel_gpu, n * sizeof(int));

	TimeTick();
	countAndColourSurfelMap(n,map_surfels,surfel_size,surfel_instance_offset,surfel_instanceColor_offset,instanceTable_color_gpu,bestIDInEachSurfel_gpu);
	TimeTock("ColourSurfelMap");


	//std::cout<<"step 4"<<std::endl;

	//Step 5 flann-kdtree-knn-vote	(main color)
	if(isflann) flannKnnVoteSurfelMap(n,map_surfels,bestIDInEachSurfel_gpu);		//->call in ProcessFlann in main.cpp
	//std::cout<<"step 5"<<std::endl;

	//step end
	cudaFree(projectInstanceList_gpu);
	cudaFree(currPoseTrans_gpu);
	cudaFree(projectDepthMap_gpu);
	cudaFree(bestIDInEachSurfel_gpu);

	cudaFree(MasksBBox_gpu);
	cudaFree(projectBBox_gpu);

	instanceFusionTimes ++;

	std::cout<<std::endl;

}

//===================================flann===========================================================
void InstanceFusion::flannKnnVoteSurfelMap(const int n,float* map_surfels,int* bestIDInEachSurfel_gpu)
{
	std::cout<<"InstanceFusion Flann"<<std::endl;
	
	//Step 1	GET DATA
	float* data_gpu;			//output
 	cudaMalloc((void **)&data_gpu, n*4*sizeof(float));
	cudaMemset(data_gpu, 0, n*4*sizeof(float));
	getVertexFromMap(n,map_surfels,surfel_size,data_gpu);
	
	//Step 2	BUILD TREE
	flann::Matrix<float> target(data_gpu,n,3,4*sizeof(float));
	
	flann::KDTreeCuda3dIndexParams params;
	params["input_is_gpu_float4"]=true;
	params["leaf_max_size"]=64;
	
	//flann::Index<flann::L2<float> > flannindex( target, params  );
	flann::KDTreeCuda3dIndex< flann::L2<float> > flannindex(target, params);
	
	TimeTick();
	flannindex.buildIndex();
	TimeTock("flann::buildIndex");
	
	
	//Step 3	SEARCH
	
	const int knn = 10;
/*
	float* 	data;				//input
	data = (float*)malloc(n*4*sizeof(float));
	cudaMemcpy( data, data_gpu, n * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	flann::Matrix<float> source(data_gpu,n,3,4*sizeof(float));
*/
	int* indicesResults;		//output
	//indicesResults = (int*)malloc(n*knn*sizeof(int));
	//memset(indicesResults,0, n*knn*sizeof(int));
 	cudaMalloc((void **)&indicesResults, n*knn*sizeof(int));
	cudaMemset(indicesResults, 0, n*knn*sizeof(int));
	flann::Matrix<int> indices(indicesResults,n,knn);
	
	float* 	distsResults;		//output
	//distsResults = (float*)malloc(n*knn*sizeof(float));
	//memset(distsResults,0, n*knn*sizeof(float));
 	cudaMalloc((void **)&distsResults, n*knn*sizeof(float));
	cudaMemset(distsResults, 0, n*knn*sizeof(float));
	flann::Matrix<float> dists(distsResults,n,knn);
	
	
	flann::SearchParams searchParams;
	searchParams.matrices_in_gpu_ram = false;	//true (unreasonable bug?)
	searchParams.use_heap = FLANN_True;
	searchParams.sorted = false;
	
	TimeTick();
	flannindex.knnSearch(target, indices, dists, knn, searchParams);
	TimeTock("flann::knnSearch");

	

	//Step 4	Gaussian
	float* tempInstanceMap_gpu;			//temp
 	cudaMalloc((void **)&tempInstanceMap_gpu, n*instanceNum*sizeof(float));
	cudaMemset(tempInstanceMap_gpu, 0, n*instanceNum*sizeof(float));

	TimeTick();
	mapKnnVoteColour(n,map_surfels,surfel_size,surfel_instance_offset,surfel_instanceColor_offset,indicesResults,distsResults,
							knn,bestIDInEachSurfel_gpu,tempInstanceMap_gpu,instanceTable_color_gpu);
	TimeTock("flann::mapKnnGaussian");


/*//debug
	int* indicesResults_cpu = (int*)malloc(n*knn*sizeof(int));
	cudaMemcpy( indicesResults_cpu, indicesResults, n*knn* sizeof(int), cudaMemcpyDeviceToHost);
	std::cout<<"1: ";
	for(int i=0;i<knn;i++)
	{
		std::cout<<indicesResults_cpu[1*knn+i]<<" ";
	}
	std::cout<<std::endl;
	free(indicesResults_cpu);
//debug*/


	//TimeTick();
	//free(data);
	cudaFree(data_gpu);
	cudaFree(tempInstanceMap_gpu);
	cudaFree(indicesResults);
	cudaFree(distsResults);
	//TimeTock("flann::FreeSum");

	flannTimes++;
}


//==================================render===========================================================
void InstanceFusion::renderProjectBoundingBox( float* projectColor, float* instanceTable_color, int* projectBBox)
{
	for(int instanceID = 0;instanceID<instanceNum;instanceID++)
	{
		int minX_i = projectBBox[instanceID*4+0];
		int maxX_i = projectBBox[instanceID*4+1];
		int minY_i = projectBBox[instanceID*4+2];
		int maxY_i = projectBBox[instanceID*4+3];
		if(maxX_i<=minX_i||maxY_i<=minY_i)	continue;
		
		float r = float(int(instanceTable_color[instanceID]) >> 16 & 0xFF) / 255.0f;
		float g = float(int(instanceTable_color[instanceID]) >> 8 & 0xFF) / 255.0f;
		float b = float(int(instanceTable_color[instanceID]) & 0xFF) / 255.0f;
		
		//row
		for(int i = minX_i;i<maxX_i;i++)
		{
			if(i<0||i>=width) continue;
			for(int w = -1;w<=1;w++)	//width = 3
			{
				//UP
				if(minY_i+w>=0)
				{
					projectColor[(minY_i+w)*width*4+i*4+0] = r;
					projectColor[(minY_i+w)*width*4+i*4+1] = g;
					projectColor[(minY_i+w)*width*4+i*4+2] = b;
					projectColor[(minY_i+w)*width*4+i*4+3] = 1.0f;
				}
				//DOWN
				if(maxY_i+w<height)
				{
					projectColor[(maxY_i+w)*width*4+i*4+0] = r;
					projectColor[(maxY_i+w)*width*4+i*4+1] = g;
					projectColor[(maxY_i+w)*width*4+i*4+2] = b;
					projectColor[(maxY_i+w)*width*4+i*4+3] = 1.0f;
				}
			}
		}
		//col
		for(int i = minY_i;i<maxY_i;i++)
		{			
			if(i<0||i>=height) continue;
			for(int w = -1;w<=1;w++)	//width = 3
			{
				//LEFT
				if(minX_i+w>=0)
				{
					projectColor[i*width*4+(minX_i+w)*4+0] = r;
					projectColor[i*width*4+(minX_i+w)*4+1] = g;
					projectColor[i*width*4+(minX_i+w)*4+2] = b;
					projectColor[i*width*4+(minX_i+w)*4+3] = 1.0f;
				}
				//RIGHT
				if(maxX_i+w<width)
				{
					projectColor[i*width*4+(maxX_i+w)*4+0] = r;
					projectColor[i*width*4+(maxX_i+w)*4+1] = g;
					projectColor[i*width*4+(maxX_i+w)*4+2] = b;
					projectColor[i*width*4+(maxX_i+w)*4+3] = 1.0f;
				}
			}
		}
	}
}

void InstanceFusion::renderProjectMap(const std::unique_ptr<ElasticFusionInterface>& map,bool drawBBox)
{
	int n = map->getMapSurfelCount();													//input
	if(n<=0)return;

	float* map_surfels = map->getMapSurfelsGpu();										//input
	cudaTextureObject_t index_surfelsIdsAfter = map->getSurfelIdsAfterFusionGpu();		//input
	cudaMemset(projectColor_gpu,0, 4 * width * height * sizeof(float));					//output
	renderProjectFrame(index_surfelsIdsAfter,width,height,n,map_surfels,surfel_size,surfel_instance_offset,
						surfel_instanceColor_offset,projectColor_gpu,instanceTable_color_gpu);
	std::cout<<"renderProjectMap End"<<std::endl;

	if(drawBBox&&projectBBox)
	{
		float projectColor[4 * width * height];
		cudaMemcpy( projectColor, projectColor_gpu, 4 * width * height * sizeof(float), cudaMemcpyDeviceToHost);
		renderProjectBoundingBox(projectColor,instanceTable_color.data(),projectBBox);
		cudaMemcpy( projectColor_gpu, projectColor, 4 * width * height * sizeof(float), cudaMemcpyHostToDevice);
		std::cout<<"drawBBox End"<<std::endl;
	}
}

void InstanceFusion::renderRawMask()
{
	cudaMemset(maskColor_gpu,0, 4 * width * height * sizeof(float));					//output
	renderMaskFrame(resultMasks_gpu,resultMasks_Num,width,height,maskColor_gpu);
}

//=======================================================================================================================
void InstanceFusion::computeMapBoundingBox(const std::unique_ptr<ElasticFusionInterface>& map, bool bboxType)
{
	int n = map->getMapSurfelCount();						//input
	if(n<=0) return;	
	float* map_surfels = map->getMapSurfelsGpu();			//input


	//Step 1 find Normal
	//TimeTick();
	const int semCircleSegNum = 18;							//input
	const int voteBufferNum = semCircleSegNum*semCircleSegNum*2;

	int* mapGroundNormalVote_gpu;							//output
	cudaMalloc((void **)&mapGroundNormalVote_gpu, voteBufferNum*sizeof(int));
	cudaMemset(mapGroundNormalVote_gpu,0, voteBufferNum*sizeof(int));

	int* mapInstanceNormalVote_gpu;							//output
	cudaMalloc((void **)&mapInstanceNormalVote_gpu, instanceNum*voteBufferNum*sizeof(int));
	cudaMemset(mapInstanceNormalVote_gpu,0, instanceNum*voteBufferNum*sizeof(int));

	testAllSurfelNormalVote(n, map_surfels,surfel_size , surfel_instance_offset, surfel_normal_offset,
					 surfel_instanceColor_offset, instanceTable_color_gpu, semCircleSegNum, mapGroundNormalVote_gpu,mapInstanceNormalVote_gpu);

	int* mapGroundNormalVote = (int*)malloc(voteBufferNum*sizeof(int));
	cudaMemcpy( mapGroundNormalVote, mapGroundNormalVote_gpu, voteBufferNum*sizeof(int), cudaMemcpyDeviceToHost);

	//TimeTock("BBOX::findNormal");
	
	
	//Step 2 find groundNormal
	//TimeTick();
	float groundNormal[3];
	int voteMaxNum = -1;
	int voteMaxID=-1;
	for(int i=0;i<voteBufferNum;i++)
	{
		if(mapGroundNormalVote[i]>voteMaxNum)
		{
			voteMaxNum = mapGroundNormalVote[i];
			voteMaxID = i;
		}
	}
	if(voteMaxID!=-1)
	{
		int i = voteMaxID/(2*semCircleSegNum) - semCircleSegNum/2;		//latitude  ---
		int j = voteMaxID%(2*semCircleSegNum) - 1;						//longitude |||
		const float pi =3.1415926;	

		float theta = (i+0.5f) * pi / semCircleSegNum;
		float alpha = (j+0.5f) * pi / semCircleSegNum;
		float d = std::cos(theta);
		float y = std::sin(theta);
		float x = d * std::cos(alpha);
		float z = d * std::sin(alpha);

		float len = std::sqrt(x*x+y*y+z*z);
		x=x/len;
		y=y/len;
		z=z/len;

		groundNormal[0] = x;
		groundNormal[1] = y;
		groundNormal[2] = z;
	
		//std::cout<<"x:"<<x<<" y:"<<y<<" z:"<<z<<std::endl;
	}
	else
	{
		groundNormal[0] = 0;
		groundNormal[1] = -1;
		groundNormal[2] = 0;
	}
	//TimeTock("BBOX::GroundNormal");
	
	//Step 3 set Coordinate(ground+instance)
	//TimeTick();
	float* groundNormal_gpu;								//input
	cudaMalloc((void **)&groundNormal_gpu, 3*sizeof(float));
	cudaMemcpy( groundNormal_gpu, groundNormal, 3*sizeof(float), cudaMemcpyHostToDevice);

	int* mapInstanceNormalCrossVote_gpu;					//temp
	cudaMalloc((void **)&mapInstanceNormalCrossVote_gpu, instanceNum*semCircleSegNum*2*sizeof(int));
	cudaMemset(mapInstanceNormalCrossVote_gpu,0, instanceNum*semCircleSegNum*2*sizeof(int));

	float* gcMatrix_gpu;									//output
	cudaMalloc((void **)&gcMatrix_gpu, 16*instanceNum*sizeof(float));
	cudaMemset(gcMatrix_gpu,0, 16*instanceNum*sizeof(float));

	float* instcMatrix_gpu;									//output
	cudaMalloc((void **)&instcMatrix_gpu, 16*instanceNum*sizeof(float));
	cudaMemset(instcMatrix_gpu,0, 16*instanceNum*sizeof(float));

	setGroundandInstanceCoordinate(instanceNum, semCircleSegNum, groundNormal_gpu,mapInstanceNormalCrossVote_gpu,mapInstanceNormalVote_gpu,gcMatrix_gpu,instcMatrix_gpu);
	//TimeTock("BBOX::setCoordinate");

	//Step 3_2 host(cpu) -> inverse(cpu) -> device(gpu)
	//TimeTick();
	cudaMemcpy(gcMatrix, gcMatrix_gpu, 16*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(instcMatrix, instcMatrix_gpu, 16*instanceNum*sizeof(float), cudaMemcpyDeviceToHost);

	Eigen::Matrix<float,4,4> gcMatrixE;
	Eigen::Matrix<float,4,4> instcMatrixE[instanceNum];
	for(int i=0;i<16;i++)
	{
		gcMatrixE(i/4,i%4) = gcMatrix[i];
		for(int j=0;j<instanceNum;j++)
		{
			instcMatrixE[j](i/4,i%4) = instcMatrix[j*16+i];
		}
	}
	
	//std::cout<<"gcMatrixE:"<<std::endl;
	//std::cout<<gcMatrixE<<std::endl;
	//std::cout<<"instcMatrixE(0):"<<std::endl;
	//std::cout<<instcMatrixE[0]<<std::endl;
	//std::cout<<"instcMatrixE(1):"<<std::endl;
	//std::cout<<instcMatrixE[1]<<std::endl;
	
	float gcMatrixInverse[16];
	float instMatrixInverse[16*instanceNum];
	for(int i=0;i<16;i++)
	{
		gcMatrixInverse[i] = gcMatrixE.inverse()(i/4,i%4);
		for(int j=0;j<instanceNum;j++)
		{
			instMatrixInverse[j*16+i] = instcMatrixE[j].inverse()(i/4,i%4);
		}
	}
	float* gcMatrixInverse_gpu;
	cudaMalloc((void **)&gcMatrixInverse_gpu, 16*sizeof(float));
	cudaMemcpy(gcMatrixInverse_gpu, gcMatrixInverse, 16*sizeof(float), cudaMemcpyHostToDevice);
	float* instMatrixInverse_gpu;
	cudaMalloc((void **)&instMatrixInverse_gpu, 16*instanceNum*sizeof(float));
	cudaMemcpy(instMatrixInverse_gpu, instMatrixInverse, 16*instanceNum*sizeof(float), cudaMemcpyHostToDevice);

	//TimeTock("BBOX::host->inverse->device");

	//Step 4 find BBOX in Ground coordinate(Instance coordinate)
	//TimeTick();
	int map3DBBoxINT[instanceNum*6];				//input and output
	for(int i=0;i<instanceNum;i++)
	{
		map3DBBoxINT[i*6+0] =  999999999;
		map3DBBoxINT[i*6+2] =  999999999;
		map3DBBoxINT[i*6+4] =  999999999;

		map3DBBoxINT[i*6+1] = -999999999;
		map3DBBoxINT[i*6+3] = -999999999;
		map3DBBoxINT[i*6+5] = -999999999;
	}

	int* map3DBBoxINT_gpu;
	cudaMalloc((void **)&map3DBBoxINT_gpu, instanceNum*6*sizeof(int));
	cudaMemcpy(map3DBBoxINT_gpu, map3DBBoxINT, instanceNum*6*sizeof(int), cudaMemcpyHostToDevice);

	testAllSurfelFindBBox(n, map_surfels,surfel_size, surfel_instance_offset, surfel_instanceColor_offset,
								 instanceTable_color_gpu,ratio3DBBox,gcMatrixInverse_gpu,instMatrixInverse_gpu,bboxType,map3DBBoxINT_gpu);
	
	cudaMemcpy(map3DBBoxINT, map3DBBoxINT_gpu, instanceNum*6*sizeof(int), cudaMemcpyDeviceToHost);

	if(map3DBBox) delete[] map3DBBox;
	map3DBBox = new float[instanceNum*6];
	
	for(int i=0;i<instanceNum;i++)
	{
		for(int j=0;j<6;j++)
		{
			map3DBBox[i*6+j] = map3DBBoxINT[i*6+j]/ratio3DBBox;
		}
		//debug
		/*if(map3DBBoxINT[i*6+0] >= 99998) continue;
		std::cout<<"instance:"<<i<<std::endl;
		std::cout<<" minX:"<<map3DBBox[i*6+0];
		std::cout<<" maxX:"<<map3DBBox[i*6+1];
		std::cout<<" minY:"<<map3DBBox[i*6+2];
		std::cout<<" maxY:"<<map3DBBox[i*6+3];
		std::cout<<" minZ:"<<map3DBBox[i*6+4];
		std::cout<<" maxZ:"<<map3DBBox[i*6+5]<<std::endl<<std::endl;*/
		
	}
	//TimeTock("BBOX::findBBOX");

	cudaFree(groundNormal_gpu);
	cudaFree(instcMatrix_gpu);
	cudaFree(gcMatrix_gpu);

	cudaFree(gcMatrixInverse_gpu);
	cudaFree(instMatrixInverse_gpu);

	cudaFree(mapGroundNormalVote_gpu);
	cudaFree(mapInstanceNormalVote_gpu);
	cudaFree(mapInstanceNormalCrossVote_gpu);
	cudaFree(map3DBBoxINT_gpu);

	free(mapGroundNormalVote);
}


void InstanceFusion::getInstancePointCloud(const std::unique_ptr<ElasticFusionInterface>& map,bool bboxType)
{
	

	int n = map->getMapSurfelCount();
	float* map_surfels = map->getMapSurfelsGpu();

	if(!n) return;

	int* eachInstanceSumCount_gpu;		//output
 	cudaMalloc((void **)&eachInstanceSumCount_gpu, instanceNum * sizeof(int));
	cudaMemset(eachInstanceSumCount_gpu, 0, instanceNum * sizeof(int));

	mapCountInstanceByInstColor(n,map_surfels,surfel_size,surfel_instance_offset,instanceTable_color_gpu, surfel_instanceColor_offset, eachInstanceSumCount_gpu);

	int eachInstanceSumCount[instanceNum];
	cudaMemcpy(eachInstanceSumCount, eachInstanceSumCount_gpu, instanceNum * sizeof(int), cudaMemcpyDeviceToHost);
	
	//debug
	//for(int i=0;i<instanceNum;i++) {if(eachInstanceSumCount[i])std::cout<<eachInstanceSumCount[i]<<std::endl;}
	
	//instSurfels Buffer Struct
	//    all   : 0 :count   1 : class_label   		      2-n:eachSurfel 
	//eachSurfel: 0 :id		123:xyz(instance Coordinate)  456:normal       789:rgb(color) 				//10-12:rgb(partSeg)

	int  instTableClassList[instanceNum];									//input
	getInstanceTableClassList(instTableClassList);


	float** instSurfels_cpu2gpu = new float*[instanceNum];
	memset(instSurfels_cpu2gpu,0, instanceNum * sizeof(float*));
	for(int i=0;i<instanceNum;i++)
	{
		if(eachInstanceSumCount[i]>0) 
		{
			instSurfels[i] = new float[2+eachInstanceSumCount[i]*13];											//output_cpu
			memset(instSurfels[i],0, (2+eachInstanceSumCount[i]*13) * sizeof(float));
			
			//count
			instSurfels[i][0] = eachInstanceSumCount[i];
			//class label
			instSurfels[i][1] = instTableClassList[i];
			
			
 			cudaMalloc((void **)&instSurfels_cpu2gpu[i], (2+eachInstanceSumCount[i]*13) * sizeof(float));		//output_gpu
			cudaMemcpy(instSurfels_cpu2gpu[i], instSurfels[i], (2+eachInstanceSumCount[i]*13) * sizeof(float), cudaMemcpyHostToDevice);
		}
		else
		{
			instSurfels[i]=0;
			instSurfels_cpu2gpu[i]=0;
		}
	}
	float** instSurfels_gpu;
	cudaMalloc((void **)&instSurfels_gpu, instanceNum * sizeof(float*));	//output_gpu
	cudaMemcpy(instSurfels_gpu, instSurfels_cpu2gpu, instanceNum * sizeof(float*), cudaMemcpyHostToDevice);

	Eigen::Matrix<float,4,4> gcMatrixE;
	Eigen::Matrix<float,4,4> instcMatrixE[instanceNum];
	for(int i=0;i<16;i++)
	{
		gcMatrixE(i/4,i%4) = gcMatrix[i];
		for(int j=0;j<instanceNum;j++)
		{
			instcMatrixE[j](i/4,i%4) = instcMatrix[j*16+i];
		}
	}
	
	float gcMatrixInverse[16];
	float instMatrixInverse[16*instanceNum];
	for(int i=0;i<16;i++)
	{
		gcMatrixInverse[i] = gcMatrixE.inverse()(i/4,i%4);
		for(int j=0;j<instanceNum;j++)
		{
			instMatrixInverse[j*16+i] = instcMatrixE[j].inverse()(i/4,i%4);
		}
	}
	float* gcMatrixInverse_gpu;					//input
	cudaMalloc((void **)&gcMatrixInverse_gpu, 16*sizeof(float));
	cudaMemcpy(gcMatrixInverse_gpu, gcMatrixInverse, 16*sizeof(float), cudaMemcpyHostToDevice);
	float* instMatrixInverse_gpu;				//input
	cudaMalloc((void **)&instMatrixInverse_gpu, 16*instanceNum*sizeof(float));
	cudaMemcpy(instMatrixInverse_gpu, instMatrixInverse, 16*instanceNum*sizeof(float), cudaMemcpyHostToDevice);

	int* map3DBBox_gpu;							//input
	cudaMalloc((void **)&map3DBBox_gpu, instanceNum*6*sizeof(int));
	cudaMemcpy(map3DBBox_gpu, map3DBBox, instanceNum*6*sizeof(int), cudaMemcpyHostToDevice);

	int* instSurfelCountTemp_gpu;				//temp
	cudaMalloc((void **)&instSurfelCountTemp_gpu, instanceNum*sizeof(int));
	cudaMemset(instSurfelCountTemp_gpu, 0, instanceNum * sizeof(int));

	//*do it
 	getSurfelToInstanceBuffer(n, map_surfels, surfel_size, surfel_instance_offset,instanceTable_color_gpu, surfel_instanceColor_offset, surfel_normal_offset,
		surfel_rgbColor_offset, bboxType, map3DBBox_gpu,gcMatrixInverse_gpu, instMatrixInverse_gpu,instSurfelCountTemp_gpu, instSurfels_gpu);
	std::cout<<"getSurfelToInstanceBuffer End"<<std::endl;
	//*do it

	//debug
	int instSurfelCountTemp[instanceNum];
	cudaMemcpy(instSurfelCountTemp,instSurfelCountTemp_gpu, instanceNum * sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(instSurfels_cpu2gpu,instSurfels_gpu, instanceNum * sizeof(float*), cudaMemcpyDeviceToHost);
	for(int i=0;i<instanceNum;i++)
	{
		if(instSurfels_cpu2gpu[i]) 
		{
			cudaMemcpy(instSurfels[i], instSurfels_cpu2gpu[i], (2+eachInstanceSumCount[i]*13) * sizeof(float), cudaMemcpyDeviceToHost);
			//std::cout<<"instance: "<<i<<" count:"<<instSurfels[i][0]<<" class label:"<<instSurfels[i][1]<<" temp:"<<instSurfelCountTemp[i]<<std::endl;
		}
	}
	//std::cout<<"read Buffer from instSurfels_gpu End"<<std::endl;
	

	cudaFree(eachInstanceSumCount_gpu);

	cudaFree(gcMatrixInverse_gpu);
	cudaFree(instMatrixInverse_gpu);
		
	cudaFree(instSurfelCountTemp_gpu);
	
	for(int i=0;i<instanceNum;i++)
	{
		if(instSurfels_cpu2gpu[i]) cudaFree(instSurfels_cpu2gpu[i]);
	}
	delete[] instSurfels_cpu2gpu;
	cudaFree(instSurfels_gpu);
}




void InstanceFusion::saveInstancePointCloud()
{


}





