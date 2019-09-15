/*
 * This file is part of InstanceFusion.
 *
 */

#include "InstanceFusion.h"
#include "InstanceFusionCuda.h"


// 创建InstanceTable，仅有Color
void InstanceFusion::createInstanceTable()
{
	std::vector<ClassColour> colour_scheme(instanceNum);
	for(int i=0;i<instanceNum;i++)
	{
		std::string className = undefineName;
		int r = rand()*7%255;	//产生一个随机数
		int g = rand()*7%255;
		int b = rand()*7%255;
		if(r<0) r=-r;
		if(g<0) g=-g;
		if(b<0) b=-b;
		ClassColour class_colour(className,r,g,b);
		colour_scheme[i] = class_colour;
	}
	instanceTable = colour_scheme;

	
	instanceTable_color.clear();
	for (int i = 0; i < instanceNum ; i++) {
	  instanceTable_color.push_back(encode_colour(instanceTable[i]));
	}
	cudaMalloc((void **)&instanceTable_color_gpu, instanceNum * sizeof(float));
	cudaMemcpy(instanceTable_color_gpu, instanceTable_color.data(), instanceNum * sizeof(float), cudaMemcpyHostToDevice);
}

// 保存InstanceTable
void InstanceFusion::saveInstanceTable()
{
	std::string instance_path = "./temp/InstanceTable_In.txt";
	std::ofstream fout;
	fout.open(instance_path);
	std::cout<< instance_path << std::endl;

	for(int i=0;i<instanceNum;i++)
	{
		int classID = -1;
		for(int j=0;j<classNum;j++)
		{
			if(instanceTable[i].name.compare(class_names[j]) == 0)
			{
				classID = j;
				break;
			}
		}
		int r = instanceTable[i].r;
		int g = instanceTable[i].g;
		int b = instanceTable[i].b;
		std::string class_name = (classID == -1) ? "Unused" : class_names[classID];
		fout << "class:" << class_name << "\t\t\t\t\tR:"<< r << " G:"<< g <<" B:"<< b << std::endl;
	}
}

void InstanceFusion::getInstanceTableClassList(int* classList)
{
	//str Name -> int ID
	for(int i=0;i<instanceNum;i++)
	{
		int classID = -1;
		for(int j=0;j<classNum;j++)
		{
			if(instanceTable[i].name.compare(class_names[j]) == 0)
			{
				classID = j;
				break;
			}
		}
		classList[i] = classID;
	}
}

int InstanceFusion::getInstanceTableFirstNotUse()
{
	for(int i=0;i<instanceNum;i++)
	{
		if(instanceTable[i].name.compare(undefineName) == 0) return i;
	}
	return -1;
}

int InstanceFusion::getInstanceTableInUseNum()
{
	int sum = 0;
	for(int i=0;i<instanceNum;i++)
	{
		if(instanceTable[i].name.compare(undefineName) != 0) sum++;
	}
	return sum;
}

void InstanceFusion::getLoopClosureInstanceTable(int* out_table)
{
	for(int i=0;i<instanceNum;i++)
	{
		int classID = -1;
		for(int j=0;j<classNum;j++)
		{
			if(instanceTable[i].name.compare(class_names[j]) == 0)
			{
				classID = j;
				break;
			}
		}
		out_table[5*i+0] = instanceTable[i].r;	//r
		out_table[5*i+1] = instanceTable[i].g;	//g
		out_table[5*i+2] = instanceTable[i].b;	//b
		out_table[5*i+3] = classID;				//classID
		out_table[5*i+4] = i;					//match instance in loopClosure.
	}
}

void InstanceFusion::checkLoopClosure(int* loopClosureInstanceTable,const std::unique_ptr<ElasticFusionInterface>& map)
{

	float* map_surfels = map->getMapSurfelsGpu();
	int n = map->getMapSurfelCount();

	//A-check list
	int instanceTableCleanList[instanceNum];
	int matchPair[instanceNum*2];
	int matchNum = 0;
	for(int i=0;i<instanceNum;i++)
	{
		instanceTableCleanList[i] = 0;
		if(loopClosureInstanceTable[5*i+4] != i)
		{
			std::cout<<"instanceTable match:"<<i<<" "<<loopClosureInstanceTable[5*i+4]<<std::endl;

			matchPair[matchNum*2+0] = i;
			matchPair[matchNum*2+1] = loopClosureInstanceTable[5*i+4];
			matchNum++;

			instanceTableCleanList[i] = 1;
			loopClosureInstanceTable[5*i+4] = i;
		}
	}
	//B-Copy
	int* matchPair_gpu;
	cudaMalloc((void **)&matchPair_gpu, matchNum *2* sizeof(int));
	cudaMemcpy(matchPair_gpu, matchPair, matchNum *2* sizeof(int), cudaMemcpyHostToDevice);
	loopClosureCopyMatchInstance(n,map_surfels,surfel_size,surfel_instance_offset,matchPair_gpu,matchNum);
	

	//C-Step 3_1_3 cleanInstanceFusion(processInstance)
	int* instanceTableCleanList_gpu;				//input
	cudaMalloc((void **)&instanceTableCleanList_gpu, instanceNum * sizeof(int));
	cudaMemcpy(instanceTableCleanList_gpu, instanceTableCleanList, instanceNum * sizeof(int), cudaMemcpyHostToDevice);
	
	
	TimeTick();
	cleanInstanceTable(instanceTableCleanList);
	cleanInstanceTableMap(n,map_surfels,surfel_size,surfel_instance_offset,instanceTableCleanList_gpu);
	TimeTock("cleanInstance");

	
	cudaFree(matchPair_gpu);
	cudaFree(instanceTableCleanList_gpu);
			
}

float InstanceFusion::scoreInClean(int eachInstanceMaximum, int eachInstanceSumCount)
{
	//return (eachInstanceMaximum * cleanWeight + eachInstanceSumCount);
	return eachInstanceSumCount;
}

void InstanceFusion::getInstanceTableCleanList(int* eachInstanceMaximum, int* instanceTableCleanList,int* eachInstanceSumCount)
{
	int* tableClassOriOrderList = new int[instanceNum];
	for(int i=0;i<instanceNum;i++)
	{
		tableClassOriOrderList[i] = i;
	}
	//sort by score
	for(int i=0;i<instanceNum;i++)
	{
		for(int j=i+1;j<instanceNum;j++)
		{
			float scoreI = scoreInClean(eachInstanceMaximum[i],eachInstanceSumCount[i]);
			float scoreJ = scoreInClean(eachInstanceMaximum[j],eachInstanceSumCount[j]);
			if(scoreJ<scoreI)
			{
				int t = eachInstanceMaximum[j];
				eachInstanceMaximum[j] = eachInstanceMaximum[i];	
				eachInstanceMaximum[i] = t;

				t = tableClassOriOrderList[j];
				tableClassOriOrderList[j] = tableClassOriOrderList[i];
				tableClassOriOrderList[i] = t;

				t = eachInstanceSumCount[j];
				eachInstanceSumCount[j] = eachInstanceSumCount[i];	
				eachInstanceSumCount[i] = t;
			}
		}
	}
	//fill Clean_List
	for(int i=0;i<instanceNum;i++)
	{
		instanceTableCleanList[i] = 0;
	}
	for(int i=0;i<cleanNum;i++)
	{
		int oriOrder = tableClassOriOrderList[i];
		instanceTableCleanList[oriOrder] = 1;
	}
	delete[] tableClassOriOrderList;
}

void InstanceFusion::cleanInstanceTable(int* instanceTableCleanList)
{
	for(int i=0;i<instanceNum;i++)
	{
		if(instanceTableCleanList[i] == 1)
		{
			instanceTable[i].name = undefineName;
		}
	}
}

void InstanceFusion::registerInstanceTable(int emptyIndex, int maskID, int* compareMap)
{
	instanceTable[emptyIndex].name = class_names[resultClass_ids[maskID]];	//register
	compareMap[emptyIndex + maskID * instanceNum] = 1;
}

void InstanceFusion::printHistoryInstances()
{
	std::cout<< "HistoryInstances: " << getInstanceTableInUseNum()+cleanTimes*cleanNum<<std::endl;
}


void  InstanceFusion::saveTimeDebug(std::string fileName)
{
	std::string path = "./temp/Time_Debug.txt";
	std::cout<< path << std::endl;
	std::fstream ff(path,std::ios::in|std::ios::out|std::ios::app);
	
	std::string allList = "fileName,historyInstanceNum,instanceFusionTimes,flannTimes,cleanTimes,Detect,gSLICr, depthGaussian, projDepth&Normal, Reclustering, mergeSuperPixel, getFinalSuperPiexl,middleMask,ProjectInstanceImage,2dBoundingBox(M&I),CompareMap,ProjectDepthMap,Geometric,updateMap,ColourSurfelMap,register,flann::buildIndex,flann::knnSearch,flann::mapKnnGaussian,MapCountInstance,CleanList,cleanInstance,reCompute,cpu_SLIC,cpu_Reclustering,cpu_flann_buildIndex,cpu_flann_knnSearch,cpu_depthFilter,cpu_compareMap(-comparemap),cpu_instMerge,cpu_3D_Gaussian,GUI,ElasticFusion,,,,";

	bool hasFile = false;
	ff.seekp(std::ios::beg);
	if(ff.rdbuf()->sgetc()!=std::char_traits<char>::eof())
	{
		hasFile = true;
	}
	ff.seekp(std::ios::end);

	ff.setf(std::ios::fixed,std::ios::floatfield);
	ff.precision(0);

	if(!hasFile)
	{
		ff<<allList;
	}
	ff<<std::endl;
	
	std::cout<<"saveTimeDebug"<<std::endl;
	int p1=0;
	int p2= allList.find(',');
	while(p2-p1>2)
	{
		std::string term = allList.substr(p1,p2-p1);
		p1=p2+1;
		p2= allList.find(',',p1);
		
		//std::cout<<term<<std::endl;
		
		if(term.compare("fileName")==0)
		{
			ff<<fileName<<",";
		}
		else if(term.compare("historyInstanceNum")==0)
		{
			ff<<getInstanceTableInUseNum()+cleanTimes*cleanNum<<",";
		}
		else if(term.compare("instanceFusionTimes")==0)
		{
			ff<<instanceFusionTimes<<",";
		}
		else if(term.compare("flannTimes")==0)
		{
			ff<<flannTimes<<",";
		}
		else if(term.compare("cleanTimes")==0)
		{
			ff<<cleanTimes<<",";
		}
		else
		{
			int i;
			for(i=0;i<tickChain_p;i++)
			{
				if(tickChain_name[i].compare(term)==0) break;
			}
			if(i==tickChain_p)ff<<"-"<<",";
			else 
			{
				long double outValue=-1;
				if(tickChain_name[i].compare("register")==0)						outValue = tickChain_time[i]/(getInstanceTableInUseNum()+cleanTimes*cleanNum);
				else if(tickChain_name[i].compare("flann::buildIndex")==0)			outValue = tickChain_time[i]/flannTimes;
				else if(tickChain_name[i].compare("flann::knnSearch")==0)			outValue = tickChain_time[i]/flannTimes;
				else if(tickChain_name[i].compare("flann::mapKnnGaussian")==0)		outValue = tickChain_time[i]/flannTimes;
				else if(tickChain_name[i].compare("cpu_3D_Gaussian")==0)			outValue = tickChain_time[i]/flannTimes;
				else if(tickChain_name[i].compare("cpu_flann_buildIndex")==0)		outValue = tickChain_time[i]/flannTimes;
				else if(tickChain_name[i].compare("cpu_flann_knnSearch")==0)		outValue = tickChain_time[i]/flannTimes;
				else if(tickChain_name[i].compare("MapCountInstance")==0)			outValue = tickChain_time[i]/cleanTimes;
				else if(tickChain_name[i].compare("CleanList")==0)					outValue = tickChain_time[i]/cleanTimes;
				else if(tickChain_name[i].compare("cleanInstance")==0)				outValue = tickChain_time[i]/cleanTimes;
				else if(tickChain_name[i].compare("reCompute")==0)					outValue = tickChain_time[i]/cleanTimes;
				else if(tickChain_name[i].compare("GUI")==0)						outValue = tickChain_time[i]/GuiTime;
				else if(tickChain_name[i].compare("ElasticFusion")==0)				outValue = tickChain_time[i]/elasticFusionTime;
				else 	outValue = tickChain_time[i]/instanceFusionTimes;

				ff<<outValue<<",";
				//std::cout<<tickChain_name[i]<<": "<<outValue<<"       "<<tickChain_time[i]<<std::endl;
			}
		}

		
	}
	ff.close();
}


void InstanceFusion::evaluateAndSave(const std::unique_ptr<ElasticFusionInterface>& map,std::string fileName)
{
	const int groundTruthNum = 256;

	int n = map->getMapSurfelCount();				//input
	if(n<=0)return;

	float* map_surfels = map->getMapSurfelsGpu();	//input

	//Step 1 compute Precision And Recall
	int* instPointNum_gpu;			//output
 	cudaMalloc((void **)&instPointNum_gpu, instanceNum*sizeof(int));
	cudaMemset(instPointNum_gpu, 0, instanceNum*sizeof(int));

	int* gtPointNum_gpu;			//output
 	cudaMalloc((void **)&gtPointNum_gpu, groundTruthNum*sizeof(int));
	cudaMemset(gtPointNum_gpu, 0, groundTruthNum*sizeof(int));

	int* inst_gt_Map_gpu;			//output
 	cudaMalloc((void **)&inst_gt_Map_gpu, groundTruthNum*instanceNum*sizeof(int));
	cudaMemset(inst_gt_Map_gpu, 0, groundTruthNum*instanceNum*sizeof(int));
	
	computePrecisionAndRecall(n, map_surfels, surfel_size, surfel_instance_offset, surfel_instanceColor_offset,
							 surfel_instanceGT_offset, instanceTable_color_gpu,	instPointNum_gpu, gtPointNum_gpu, inst_gt_Map_gpu);

	int* instPointNum = (int*)malloc(instanceNum*sizeof(int));
	cudaMemcpy(instPointNum, instPointNum_gpu, instanceNum * sizeof(float), cudaMemcpyDeviceToHost);

	int* gtPointNum = (int*)malloc(groundTruthNum*sizeof(int));
	cudaMemcpy(gtPointNum, gtPointNum_gpu, groundTruthNum * sizeof(float), cudaMemcpyDeviceToHost);

	int* inst_gt_Map = (int*)malloc(groundTruthNum*instanceNum*sizeof(int));
	cudaMemcpy(inst_gt_Map, inst_gt_Map_gpu, groundTruthNum*instanceNum * sizeof(float), cudaMemcpyDeviceToHost);
	

	//Step 2 save To file
	std::string path = "./temp/Precision_Recall_RAW.txt";
	std::cout<< path << std::endl;
	std::fstream ff(path,std::ios::in|std::ios::out|std::ios::app);
	
	std::string allList = "fileName,class,gtPointNum,instPointNum,intersectionNum";

	bool hasFile = false;
	ff.seekp(std::ios::beg);
	if(ff.rdbuf()->sgetc()!=std::char_traits<char>::eof())
	{
		hasFile = true;
	}
	ff.seekp(std::ios::end);

	ff.setf(std::ios::fixed,std::ios::floatfield);
	ff.precision(0);

	if(!hasFile)
	{
		ff<<allList;
	}
	ff<<std::endl;
	
	std::cout<<"Save Evaluation to file"<<std::endl;


	//num
	int howManyGT=0;
	for(int gtID = 0;gtID<256;gtID++)
	{
		if(gtPointNum[gtID] <= 6000) continue;
		howManyGT++;
	}
	ff<<",,,,,,,"<<howManyGT<<","<<(getInstanceTableInUseNum()+cleanTimes*cleanNum)<<std::endl;
	
	//gt find instance
	for(int gtID = 0;gtID<256;gtID++)
	{
		if(gtPointNum[gtID] <= 150) continue;
		
		int maxInstanceInGTNum = 0;
		int maxInstanceID = -1;
		for(int instanceID = 0;instanceID<instanceNum;instanceID++)
		{
			if(inst_gt_Map[gtID*instanceNum+instanceID]>maxInstanceInGTNum)
			{
				maxInstanceInGTNum = inst_gt_Map[gtID*instanceNum+instanceID];
				maxInstanceID = instanceID;
			}
		}
		if(maxInstanceInGTNum>150)
		{
			int flag=1;
			for(int otherGT = 0;otherGT<256;otherGT++)
			{
				if(inst_gt_Map[otherGT*instanceNum+maxInstanceID]>inst_gt_Map[gtID*instanceNum+maxInstanceID])
				{
					flag = 0;
				}
			}
			if(flag)
			{
				ff<<fileName<<",";
				ff<<instanceTable[maxInstanceID].name<<",";
				ff<<gtPointNum[gtID]<<",";
				ff<<instPointNum[maxInstanceID]<<",";
				ff<<inst_gt_Map[gtID*instanceNum+maxInstanceID]<<std::endl;
			}
		}
	}
	ff.close();

	cudaFree(instPointNum_gpu);
	cudaFree(gtPointNum_gpu);
	cudaFree(inst_gt_Map_gpu);
	free(instPointNum);
	free(gtPointNum);
	free(inst_gt_Map);
}


void InstanceFusion::TimeTick()
{
	gettimeofday(&time_1,NULL);
}

void InstanceFusion::TimeTock(std::string name, bool isShow)
{
	gettimeofday(&time_2,NULL);
	float timeUse = (time_2.tv_sec-time_1.tv_sec)*1000000+(time_2.tv_usec-time_1.tv_usec);
	if(isShow) std::cout<<name<<": "<<timeUse<<" us"<<std::endl;

	//record
	int i;
	for(i=0;i<tickChain_p;i++)
	{
		if(tickChain_name[i].compare(name)==0) break;
	}
	if(i==tickChain_p)	tickChain_name[tickChain_p++] = name;

	tickChain_time[i] += timeUse;
	//std::cout<<name<<"_sum: "<<tickChain_time[i]<<" us"<<std::endl;

	if(name.compare("GUI")==0)
	{
		GuiTime++;
	}
	if(name.compare("ElasticFusion")==0)
	{
		elasticFusionTime++;
	}
}
