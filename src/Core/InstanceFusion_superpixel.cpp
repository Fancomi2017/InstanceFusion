/*
 * This file is part of InstanceFusion.
 *
 */

#include "InstanceFusion.h"
#include "InstanceFusionCuda.h"


//spInfo
//0 pixel_num   123 pos_sum   456 nor_sum 789 pos_avg   10-12 nor_avg   13 depth_avg 14 depth_avg   
//15 distance_stand_deviation 16 num_after_cluster 17 connectNum   18-28 neighbor 29 finalID
const int SPI_SIZE    = 30;
const int SPI_PNUM     = 0;

const int SPI_POS_SX     = 1;
const int SPI_POS_SY     = 2;
const int SPI_POS_SZ     = 3;
const int SPI_NOR_SX     = 4;
const int SPI_NOR_SY     = 5;
const int SPI_NOR_SZ     = 6;

const int SPI_POS_AX     = 7;
const int SPI_POS_AY     = 8;
const int SPI_POS_AZ     = 9;
const int SPI_NOR_AX     = 10;
const int SPI_NOR_AY     = 11;
const int SPI_NOR_AZ     = 12;

const int SPI_DEPTH_SUM  = 13;
const int SPI_DEPTH_AVG  = 14;

const int SPI_DIST_DEV  = 15;
const int SPI_NOR_DEV   = 16;

const int SPI_CONNECT_N = 17;
const int SPI_NP_FIRST  = 18;
const int SPI_NP_MAX    = 11;	//18+11=29

const int SPI_FINAL     = 29;

//iterations in depthMap cluster
//const int iterations = 1;

void InstanceFusion::mergeSuperPixel(DepthPtr depthMap,int spNum,int* segMask,int *finalSPixel,const int frameID)
{

	//Get Cam
	Eigen::Vector4f cam = Eigen::Vector4f(Intrinsics::getInstance().cx(),Intrinsics::getInstance().cy(),
								1.0 / Intrinsics::getInstance().fx(), 1.0 / Intrinsics::getInstance().fy());
	float* cam_gpu;						//input
	cudaMalloc((void **)&cam_gpu, 4 * sizeof(float));
	cudaMemcpy(cam_gpu, &cam(0), 4 * sizeof(float), cudaMemcpyHostToDevice);



	//Step 0	Gaussian filter
	unsigned short * oriDepthMap_gpu;		//input
	cudaMalloc((void **)&oriDepthMap_gpu, height*width * sizeof(unsigned short));
	cudaMemcpy(oriDepthMap_gpu, depthMap, height*width * sizeof(unsigned short), cudaMemcpyHostToDevice);

	unsigned short * depthMapG_gpu;			//output
	cudaMalloc((void **)&depthMapG_gpu, height*width*sizeof(unsigned short));
	cudaMemset(depthMapG_gpu, 0, height*width*sizeof(unsigned short));

	TimeTick();
	depthMapGaussianfilter(oriDepthMap_gpu,width, height,depthMapG_gpu);
	TimeTock(" depthGaussian");

	//Step 1	getPosMap
	TimeTick();
	float* posMap_gpu;					//output
	cudaMalloc((void **)&posMap_gpu, height*width*3*sizeof(float));
	cudaMemset(posMap_gpu, 0, height*width*3*sizeof(float));

	getPosMapFromDepth(depthMapG_gpu,cam_gpu,width, height,posMap_gpu);


	//Step 2	getNormalMap
	float* normalMap_gpu;				//output
	cudaMalloc((void **)&normalMap_gpu, height*width*3*sizeof(float));
	cudaMemset(normalMap_gpu, 0, height*width*3*sizeof(float));

	getNormalMapFromDepth(depthMapG_gpu,cam_gpu,width, height,normalMap_gpu);
	TimeTock(" projDepth&Normal");


	
	//Step 3
	float* spInfo_gpu;						//output
	cudaMalloc((void **)&spInfo_gpu, spNum*SPI_SIZE*sizeof(float));
	cudaMemset(spInfo_gpu, 0, spNum*SPI_SIZE*sizeof(float));
	
	int spInfoStruct[22];
	spInfoStruct[0] =  SPI_SIZE;		spInfoStruct[1] =  SPI_PNUM;
	spInfoStruct[2] =  SPI_POS_SX;		spInfoStruct[3] =  SPI_POS_SY;	spInfoStruct[4] =  SPI_POS_SZ;
	spInfoStruct[5] =  SPI_NOR_SX;		spInfoStruct[6] =  SPI_NOR_SY;	spInfoStruct[7] =  SPI_NOR_SZ;
	spInfoStruct[8] =  SPI_POS_AX;		spInfoStruct[9] =  SPI_POS_AY;	spInfoStruct[10] =  SPI_POS_AZ;
	spInfoStruct[11] =  SPI_NOR_AX;		spInfoStruct[12] =  SPI_NOR_AY;	spInfoStruct[13] =  SPI_NOR_AZ;
	spInfoStruct[14] =  SPI_DEPTH_SUM;	spInfoStruct[15] =  SPI_DEPTH_AVG;
	spInfoStruct[16] =  SPI_DIST_DEV;	spInfoStruct[17] =  SPI_NOR_DEV;	
	spInfoStruct[18] =  SPI_CONNECT_N;	spInfoStruct[19] =  SPI_NP_FIRST;	spInfoStruct[20] =  SPI_NP_MAX;
	spInfoStruct[21] =  SPI_FINAL;
	int* spInfoStruct_gpu;				//input
	cudaMalloc((void **)&spInfoStruct_gpu, 22*sizeof(int));
	cudaMemcpy(spInfoStruct_gpu, spInfoStruct, 22*sizeof(int), cudaMemcpyHostToDevice);

	int* finalSPixel_gpu;				//input   (next input & output )
	cudaMalloc((void **)&finalSPixel_gpu, height*width * sizeof(int));
	cudaMemcpy(finalSPixel_gpu, segMask, height*width * sizeof(int), cudaMemcpyHostToDevice);
	
	//*********result is not stable but code can run and fast*********//
	TimeTick();
	getSuperPixelInfoCuda(finalSPixel_gpu,depthMapG_gpu,posMap_gpu,normalMap_gpu,spNum,spInfo_gpu,spInfoStruct_gpu,width,height);	
	TimeTock(" Reclustering");

	

	
	//Step 4
	float* spInfo = (float*)malloc(spNum*SPI_SIZE*sizeof(float));	//next input
	cudaMemcpy(spInfo, spInfo_gpu, spNum*SPI_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	//Debug
	/*for(int i=0;i<spNum;i++) 
	{
		if(spInfo[i*SPI_SIZE+0]) 
		{
			std::cout<<"id: "<<i<<" num:"<<spInfo[i*SPI_SIZE + SPI_PNUM]<<" connectNum: "<<spInfo[i*SPI_SIZE + SPI_CONNECT_N];
			int n = spInfo[i*SPI_SIZE + SPI_CONNECT_N];
			std::cout<<"\n  nid: ";
			for(int j=0;j<n;j++)
			{
				std::cout<<spInfo[i*SPI_SIZE+SPI_NP_FIRST+j]<<" ";
			}
			std::cout<<std::endl;
		}
	}*/	


	TimeTick();
	connectSuperPixel(spNum,spInfo);
	TimeTock(" mergeSuperPixel");

	cudaMemcpy(spInfo_gpu, spInfo, spNum*SPI_SIZE*sizeof(float), cudaMemcpyHostToDevice);
	

	//Debug FINAL_SP
	cudaMemcpy(segMask, finalSPixel_gpu, height*width * sizeof(int), cudaMemcpyDeviceToHost);

	//Step 5

	TimeTick();
	getFinalSuperPiexl(spInfo_gpu,width, height, spInfoStruct_gpu,finalSPixel_gpu);
	TimeTock(" getFinalSuperPiexl");

	cudaMemcpy(finalSPixel, finalSPixel_gpu, height*width * sizeof(int), cudaMemcpyDeviceToHost);
	


	//Debug DEP+DEP_G
	/*
	cv::Mat depthImage1(height, width, CV_16UC1, depthMap);
	std::string savename1 = "./temp/depth1.png";
	imwrite(savename1.c_str(),depthImage1);
	cv::Mat depthImage2(height, width, CV_16UC1, depthMapG);
	std::string savename2 = "./temp/depth2.png";
	imwrite(savename2.c_str(),depthImage2);
	std::cout<<"Debug DEP+DEP_G"<<std::endl;
	*/

	//Debug  POS+NOR
	/*
	float* posMap = (float*)malloc(height*width*3*sizeof(float));
	cudaMemcpy(posMap, posMap_gpu, height*width*3*sizeof(float), cudaMemcpyDeviceToHost);
	float* normalMap = (float*)malloc(height*width*3*sizeof(float));
	cudaMemcpy(normalMap, normalMap_gpu, height*width*3*sizeof(float), cudaMemcpyDeviceToHost);
	unsigned char* posMap2 = (unsigned char*)malloc(height*width*3*sizeof(unsigned char));
	unsigned char* normalMap2 = (unsigned char*)malloc(height*width*3*sizeof(unsigned char));
	for(int x=0;x<width;x++)
	{
		for(int y=0;y<height;y++)
		{
			for(int c =0; c<3;c++)
			{
				posMap2[y*width*3+x*3+c] = posMap[y*width*3+x*3+c] * 255 + 125;
				//normalMap2[y*width*3+x*3+c] = normalMap[y*width*3+x*3+c] * 255 + 0;
				normalMap2[y*width*3+x*3+c] = spInfo[segMask[y*width+x]*SPI_SIZE+SPI_NOR_AX+c] * 255 + 0;
			}
		}
	}
	cv::Mat posImage(height, width, CV_8UC3, posMap2);
	imwrite("./temp/posImage.png",posImage);
	cv::Mat normalImage(height, width, CV_8UC3, normalMap2);
	imwrite("./temp/normalImage.png",normalImage);
	std::cout<<"Debug POS+NOR"<<std::endl;
	if(posMap)		free(posMap);
	if(posMap2)		free(posMap2);
	if(normalMap)	free(normalMap);
	if(normalMap2)	free(normalMap2);
	*/
	
	//Debug FINAL_SP
	if(false)
	{
		TimeTick();
		const int StepX[4] = {0,0,1,-1};
		const int StepY[4] = {1,-1,0,0};
		unsigned short segMask_debug[height*width];
		unsigned short finalSPixel_debug[height*width];
		for(int x=0;x<width;x++)
		{
			for(int y=0;y<height;y++)
			{
				segMask_debug[y*width+x]=segMask[y*width+x]*50;
				finalSPixel_debug[y*width+x]=segMask[y*width+x]*50;
				for(int i=0; i<4; i++)
				{
					if(x==0||x==width-1||y==0||y==height-1) continue;

					int dx = x+StepX[i];
					int dy = y+StepY[i];
					int id = finalSPixel[y*width+x];
					int nbID = finalSPixel[dy*width+dx];
					//if(nbID>=spNum||nbID<0) continue;

					if(id!=nbID) finalSPixel_debug[y*width+x]=60000;

					
					id = segMask[y*width+x];
					nbID = segMask[dy*width+dx];

					if(id!=nbID) segMask_debug[y*width+x]=60000;
					
				}
			}
		}
		cv::Mat segMaskImage(height, width, CV_16UC1, segMask_debug);
		std::string sm_save_dir("./temp/");
		std::string sm_suffix("_segMask.png");
		sm_save_dir += std::to_string(0);	//frameID
		sm_save_dir += sm_suffix;
		cv::imwrite(sm_save_dir,segMaskImage);


		cv::Mat finalSPixelImage(height, width, CV_16UC1, finalSPixel_debug);
		std::string sp_save_dir("./temp/");
		std::string sp_suffix("_SPixel.png");
		sp_save_dir += std::to_string(0);	//frameID
		sp_save_dir += sp_suffix;
		cv::imwrite(sp_save_dir,finalSPixelImage);
		TimeTock(" Debug FINAL_SP");
	}

	
	if(posMap_gpu)			cudaFree(posMap_gpu);
	if(normalMap_gpu)		cudaFree(normalMap_gpu);
	if(spInfo_gpu)			cudaFree(spInfo_gpu);
	if(spInfoStruct_gpu)	cudaFree(spInfoStruct_gpu);

	if(cam_gpu)				cudaFree(cam_gpu);
	if(oriDepthMap_gpu) 	cudaFree(oriDepthMap_gpu);
	if(depthMapG_gpu) 		cudaFree(depthMapG_gpu);
	if(finalSPixel_gpu)		cudaFree(finalSPixel_gpu);

	if(spInfo)				free(spInfo);
}


void InstanceFusion::connectSuperPixel(int spNum,float* spInfo)
{

	//spInfo
	//0 pixel_num   123 pos_sum   456 nor_sum 789 pos_avg   10-12 nor_avg   13 depth_avg 14 depth_avg   
	//15 distance_stand_deviation 16 num_after_cluster 17 connectNum   18-28 neighbor 29 finalID

	//connect test (dist term + normal term)
	for(int i=0;i<spNum;i++)
	{
		spInfo[i*SPI_SIZE + SPI_FINAL] = -1;
		int connectNum = spInfo[i*SPI_SIZE + SPI_CONNECT_N];

		for(int j=0;j<connectNum;j++)
		{
			int idA =i;
			int idB =spInfo[i*SPI_SIZE+SPI_NP_FIRST+j];
			if(idB==-1) continue;
			
			//not negative
			//if(spInfo[idA*SPI_SIZE+SPI_DEPTH_AVG]<0) std::cout<<"ERROR7: "<<idA<<std::endl;
			//if(spInfo[idA*SPI_SIZE+SPI_DIST_DEV]<0) std::cout<<"ERROR8: "<<idA<<std::endl;

			int flag = 1;	
			//D
			float vecA[3];
			vecA[0] = spInfo[idA*SPI_SIZE+SPI_NOR_AX];
			vecA[1] = spInfo[idA*SPI_SIZE+SPI_NOR_AY];
			vecA[2] = spInfo[idA*SPI_SIZE+SPI_NOR_AZ];
			float vecB[3];
			vecB[0] = spInfo[idA*SPI_SIZE + SPI_POS_AX]  - spInfo[idB*SPI_SIZE + SPI_POS_AX];
			vecB[1] = spInfo[idA*SPI_SIZE + SPI_POS_AY]  - spInfo[idB*SPI_SIZE + SPI_POS_AY];
			vecB[2] = spInfo[idA*SPI_SIZE + SPI_POS_AZ]  - spInfo[idB*SPI_SIZE + SPI_POS_AZ];
			float lenA = std::sqrt( vecA[0]*vecA[0] + vecA[1]*vecA[1] + vecA[2]*vecA[2]);
			float lenB = std::sqrt( vecB[0]*vecB[0] + vecB[1]*vecB[1] + vecB[2]*vecB[2]);
			float dotAB = vecA[0]*vecB[0] + vecA[1]*vecB[1] + vecA[2]*vecB[2];
			float distTerm = std::abs(dotAB / lenA) + 1.0*lenB;
			//D_A
			//if(distTerm>3*spInfo[idA*SPI_SIZE+SPI_DIST_DEV] || distTerm>3*spInfo[idB*SPI_SIZE+SPI_DIST_DEV]) flag = 0;
			//D_B
			//if(distTerm>0.02) flag = 0;
			//D E
			float thresholdA1 = 1*((0.026*spInfo[idA*SPI_SIZE+SPI_DEPTH_AVG]-4.0f)/1186.0f);
			float thresholdB1 = 1*((0.026*spInfo[idB*SPI_SIZE+SPI_DEPTH_AVG]-4.0f)/1186.0f);
			float thresholdA2 = 2*spInfo[idA*SPI_SIZE+SPI_DIST_DEV];			//4 (IF NO thresholdA3)
			float thresholdB2 = 2*spInfo[idB*SPI_SIZE+SPI_DIST_DEV];			//4
			//if(distTerm>1.5*thresholdA1+ || distTerm>1.5*thresholdB1) flag = 0;
			//std::cout<<distTerm<<std::endl;
			


			float thisNor[3],leftNor[3];
			thisNor[0] = spInfo[idA*SPI_SIZE+SPI_NOR_AX];
			thisNor[1] = spInfo[idA*SPI_SIZE+SPI_NOR_AY];
			thisNor[2] = spInfo[idA*SPI_SIZE+SPI_NOR_AZ];

			leftNor[0] = spInfo[idB*SPI_SIZE+SPI_NOR_AX];
			leftNor[1] = spInfo[idB*SPI_SIZE+SPI_NOR_AY];
			leftNor[2] = spInfo[idB*SPI_SIZE+SPI_NOR_AZ];
			
			float diffNor1 = std::abs(thisNor[0]-leftNor[0]);
			float diffNor2 = std::abs(thisNor[1]-leftNor[1]);
			float diffNor3 = std::abs(thisNor[2]-leftNor[2]);
			float norTerm = 0.1*std::sqrt(diffNor1*diffNor1+diffNor2*diffNor2+diffNor3*diffNor3);
			//std::cout<<norTerm<<std::endl;

			float thresholdA3 = 0*spInfo[idA*SPI_SIZE+SPI_NOR_DEV];	//0.005
			float thresholdB3 = 0*spInfo[idB*SPI_SIZE+SPI_NOR_DEV];
			
			float finTest = distTerm + norTerm;
			float ThresholdA = thresholdA1 + thresholdA2 + thresholdA3;
			float ThresholdB = thresholdB1 + thresholdB2 + thresholdB3;

			if(finTest>ThresholdA || finTest>ThresholdB) flag = 0;
			/*
			float thisVer[3],leftVer[3];
			thisVer[0] = spInfo[idA*SPI_SIZE+SPI_POS_AX];
			thisVer[1] = spInfo[idA*SPI_SIZE+SPI_POS_AY];
			thisVer[2] = spInfo[idA*SPI_SIZE+SPI_POS_AZ];

			leftVer[0] = spInfo[idB*SPI_SIZE+SPI_POS_AX];
			leftVer[1] = spInfo[idB*SPI_SIZE+SPI_POS_AY];
			leftVer[2] = spInfo[idB*SPI_SIZE+SPI_POS_AZ];

			float vecToL[3];
			vecToL[0] = leftVer[0] - thisVer[0];
			vecToL[1] = leftVer[1] - thisVer[1];
			vecToL[2] = leftVer[2] - thisVer[2];
		
			float dotPro1 = vecToL[0]*thisNor[0] + vecToL[1]*thisNor[1] + vecToL[2]*thisNor[2];
			float dotPro2 = leftNor[0]*thisNor[0] + leftNor[1]*thisNor[1] + leftNor[2]*thisNor[2];
	
			float cross2[3];
			cross2[0] = leftNor[1]*thisNor[2] - leftNor[2]*thisNor[1];
			cross2[1] = leftNor[2]*thisNor[0] - leftNor[0]*thisNor[3];
			cross2[2] = leftNor[0]*thisNor[1] - leftNor[1]*thisNor[0];
			float lenCross2 = std::sqrt( cross2[0]*cross2[0] + cross2[1]*cross2[1] + cross2[2]*cross2[2]);
			float len2L = std::sqrt( leftNor[0]*leftNor[0] + leftNor[1]*leftNor[1] + leftNor[2]*leftNor[2]);
			float len2T = std::sqrt( thisNor[0]*thisNor[0] + thisNor[1]*thisNor[1] + thisNor[2]*thisNor[2]);
			float cos2 = dotPro2 / (len2L*len2T);
			float sin2 = lenCross2 / (len2L*len2T);
			//if(dotPro1 > 0 && cos2 > 1) flag = 0;	//need test 
			//finTest += dotPro1;
			//if(finTest>ThresholdA || finTest>ThresholdB) flag = 0;
			*/
			
			if(!flag)
			{
				
				//idA close
				spInfo[idA*SPI_SIZE+SPI_NP_FIRST+j] = -1;
				
				//idB close
				int connectNumB = spInfo[idB*SPI_SIZE + SPI_CONNECT_N];
				for(int k=0;k<connectNumB;k++)
				{
					if(spInfo[idB*SPI_SIZE+SPI_NP_FIRST+k]==idA)
					{
						spInfo[idB*SPI_SIZE+SPI_NP_FIRST+k] = -1;
						break;
					}
				}
			}
		}

	}

	//finalID consistency
	int stack[spNum*10];
	int p=0;
	for(int i=0;i<spNum;i++)
	{
		int finalID;
		if(spInfo[i*SPI_SIZE + SPI_FINAL]==-1) finalID = i;
		else finalID = spInfo[i*SPI_SIZE + SPI_FINAL];

		stack[p++] = i;
		while(p>0)
		{
			int target = stack[--p];
			if(spInfo[target*SPI_SIZE + SPI_FINAL] != -1) continue;

			spInfo[target*SPI_SIZE + SPI_FINAL] = finalID;	//Set
			
			int  connectNum = spInfo[target*SPI_SIZE + SPI_CONNECT_N];
			for(int j=0;j<connectNum;j++)
			{
				int connectID = spInfo[target*SPI_SIZE + SPI_NP_FIRST + j];

				if(connectID==-1) continue;
				if(spInfo[connectID*SPI_SIZE + SPI_FINAL] != -1) continue;

				stack[p++] = connectID;	
			}
		}
	}
	
}


void InstanceFusion::getSuperPixelInfo(int* segMask,unsigned short * depthMap,float* posMap,float* normalMap,int spNum,float* spInfo)
{
	const int StepX[4] = {0,0,1,-1};
	const int StepY[4] = {1,-1,0,0};

	//spInfo
	//0 pixel_num   123 pos_sum   456 nor_sum 789 pos_avg   10-12 nor_avg   13 depth_avg 14 depth_avg   
	//15 distance_stand_deviation 16 num_after_cluster 17 connectNum   18-28 neighbor 29 finalID

	//First
	for(int x=0;x<width;x++)
	{
		for(int y=0;y<height;y++)
		{
			int id = segMask[y*width+x];
			float pnTest=0;
			pnTest += (posMap[y*width*3+x*3+0]*posMap[y*width*3+x*3+0]);
			pnTest += (posMap[y*width*3+x*3+1]*posMap[y*width*3+x*3+1]);
			pnTest += (posMap[y*width*3+x*3+2]*posMap[y*width*3+x*3+2]);

			pnTest += (normalMap[y*width*3+x*3+0]*normalMap[y*width*3+x*3+0]);
			pnTest += (normalMap[y*width*3+x*3+1]*normalMap[y*width*3+x*3+1]);
			pnTest += (normalMap[y*width*3+x*3+2]*normalMap[y*width*3+x*3+2]);

			if(pnTest<0.01||id>=spNum||id<0)
			{
				segMask[y*width+x] = -1;
				continue;
			}

			//sum
			spInfo[id*SPI_SIZE + SPI_PNUM]  += 1;

			spInfo[id*SPI_SIZE + SPI_POS_SX]  += posMap[y*width*3+x*3+0];
			spInfo[id*SPI_SIZE + SPI_POS_SY]  += posMap[y*width*3+x*3+1];
			spInfo[id*SPI_SIZE + SPI_POS_SZ]  += posMap[y*width*3+x*3+2];

			spInfo[id*SPI_SIZE + SPI_NOR_SX]  += normalMap[y*width*3+x*3+0];
			spInfo[id*SPI_SIZE + SPI_NOR_SY]  += normalMap[y*width*3+x*3+1];
			spInfo[id*SPI_SIZE + SPI_NOR_SZ]  += normalMap[y*width*3+x*3+2];
	
			spInfo[id*SPI_SIZE + SPI_DEPTH_SUM ]  += depthMap[y*width+x];
			
			//check neighbor
			for(int i=0; i<4; i++)
			{
				if(x==0||x==width-1||y==0||y==height-1) continue;

				int dx = x+StepX[i];
				int dy = y+StepY[i];
				int nbID = segMask[dy*width+dx];
				if(nbID>=spNum||nbID<0) continue; 	//-1

				if(id!=nbID)
				{
					int connectNum = spInfo[id*SPI_SIZE+SPI_CONNECT_N];			
					if(connectNum>=SPI_NP_MAX)continue;

					
					int exist = 0;
					for(int j=0;j<connectNum;j++)
					{
						if(spInfo[id*SPI_SIZE+SPI_NP_FIRST+j]==nbID)
						{
							exist = 1;
							break;
						}
					}
					if(!exist&&connectNum<SPI_NP_MAX) 
					{
						spInfo[id*SPI_SIZE+SPI_NP_FIRST+connectNum]=nbID;
						spInfo[id*SPI_SIZE+SPI_CONNECT_N]++;
					}
				}
			}
		}
	}
	
	//avg
	for(int i=0;i<spNum;i++)
	{
		//123 pos_avg   456 nor_avg   7 depth_avg
		int t = spInfo[i*SPI_SIZE+SPI_PNUM];
		if(t!=0) 
		{
			spInfo[i*SPI_SIZE+SPI_POS_AX] = spInfo[i*SPI_SIZE+SPI_POS_SX]/t;
			spInfo[i*SPI_SIZE+SPI_POS_AY] = spInfo[i*SPI_SIZE+SPI_POS_SY]/t;
			spInfo[i*SPI_SIZE+SPI_POS_AZ] = spInfo[i*SPI_SIZE+SPI_POS_SZ]/t;

			float nx = spInfo[i*SPI_SIZE+SPI_NOR_SX];
			float ny = spInfo[i*SPI_SIZE+SPI_NOR_SY];
			float nz = spInfo[i*SPI_SIZE+SPI_NOR_SZ];
			float len = std::sqrt(nx*nx+ny*ny+nz*nz);
			spInfo[i*SPI_SIZE+SPI_NOR_AX] = spInfo[i*SPI_SIZE+SPI_NOR_SX]/len;
			spInfo[i*SPI_SIZE+SPI_NOR_AY] = spInfo[i*SPI_SIZE+SPI_NOR_SY]/len;
			spInfo[i*SPI_SIZE+SPI_NOR_AZ] = spInfo[i*SPI_SIZE+SPI_NOR_SZ]/len;

			spInfo[i*SPI_SIZE+SPI_DEPTH_AVG ] = spInfo[i*SPI_SIZE+SPI_DEPTH_SUM ]/t;
		}
	}
	
	//Second
	//for(int n=0;n<iterations;n++)
	//{
		for(int x=0;x<width;x++)
		{
			for(int y=0;y<height;y++)
			{
				//depth_stand_deviation
				int id = segMask[y*width+x];
				if(id>=spNum||id<0) continue;	//-1
					
				int connectNum = spInfo[id*SPI_SIZE+SPI_CONNECT_N];
				float minDist = 999999.9f;
				float minNor  = 999999.9f;
				int minID = id;
				for(int i = 0;i<=connectNum;i++)
				{
					int idTest;
					if(i!=connectNum)idTest = spInfo[id*SPI_SIZE+SPI_NP_FIRST+i];
					else idTest = id;
				
					float vecA[3];
					vecA[0] = spInfo[idTest*SPI_SIZE+SPI_NOR_AX];
					vecA[1] = spInfo[idTest*SPI_SIZE+SPI_NOR_AY];
					vecA[2] = spInfo[idTest*SPI_SIZE+SPI_NOR_AZ];
				
					float vecB[3];
					vecB[0] = spInfo[idTest*SPI_SIZE + SPI_POS_AX]  - posMap[y*width*3+x*3+0];
					vecB[1] = spInfo[idTest*SPI_SIZE + SPI_POS_AY]  - posMap[y*width*3+x*3+1];
					vecB[2] = spInfo[idTest*SPI_SIZE + SPI_POS_AZ]  - posMap[y*width*3+x*3+2];

					float lenA = std::sqrt( vecA[0]*vecA[0] + vecA[1]*vecA[1] + vecA[2]*vecA[2]);
					float lenB = std::sqrt( vecB[0]*vecB[0] + vecB[1]*vecB[1] + vecB[2]*vecB[2]);

					float dotAB = vecA[0]*vecB[0] + vecA[1]*vecB[1] + vecA[2]*vecB[2]; 
					//float cosAB = dotAB / (lenA*lenB);
					//float dist = cosAB * lenB;
					float dist = std::abs(dotAB / lenA) + 1.0*lenB;	//****   (+lenB)
					
					float diffNor1 = std::abs(vecA[0]-normalMap[y*width*3+x*3+0]);
					float diffNor2 = std::abs(vecA[1]-normalMap[y*width*3+x*3+1]);
					float diffNor3 = std::abs(vecA[2]-normalMap[y*width*3+x*3+2]);
					float diffNor = diffNor1*diffNor1+diffNor2*diffNor2+diffNor3*diffNor3;
					if(dist<minDist)	//+0.1*diffNor			(JUST DIST MORE GOOD)
					{
						minNor = diffNor;
						minDist = dist;
						minID = idTest;
					}
				}
				float threshold = (0.026*spInfo[minID*SPI_SIZE+SPI_DEPTH_AVG]-4.0f)/1186.0f;
				if(minDist>2*threshold)	minID = -1;
				
				if(minID!=-1)
				{		
					spInfo[minID*SPI_SIZE+SPI_DIST_DEV] += (minDist * minDist);
					spInfo[minID*SPI_SIZE+SPI_NOR_DEV ] += minNor;
				}

				if(id!=minID)
				{
				
					segMask[y*width+x] = minID;
				
					spInfo[id*SPI_SIZE + SPI_PNUM]  -= 1;
					spInfo[id*SPI_SIZE + SPI_POS_SX] -= posMap[y*width*3+x*3+0];
					spInfo[id*SPI_SIZE + SPI_POS_SY] -= posMap[y*width*3+x*3+1];
					spInfo[id*SPI_SIZE + SPI_POS_SZ] -= posMap[y*width*3+x*3+2];
					spInfo[id*SPI_SIZE + SPI_NOR_SX] -= normalMap[y*width*3+x*3+0];
					spInfo[id*SPI_SIZE + SPI_NOR_SY] -= normalMap[y*width*3+x*3+1];
					spInfo[id*SPI_SIZE + SPI_NOR_SZ] -= normalMap[y*width*3+x*3+2];
					spInfo[id*SPI_SIZE + SPI_DEPTH_SUM ]  -= depthMap[y*width+x];

					if(minID!=-1)
					{
						spInfo[minID*SPI_SIZE + SPI_PNUM]  += 1;
						spInfo[minID*SPI_SIZE + SPI_POS_SX] += posMap[y*width*3+x*3+0];
						spInfo[minID*SPI_SIZE + SPI_POS_SY] += posMap[y*width*3+x*3+1];
						spInfo[minID*SPI_SIZE + SPI_POS_SZ] += posMap[y*width*3+x*3+2];
						spInfo[minID*SPI_SIZE + SPI_NOR_SX] += normalMap[y*width*3+x*3+0];
						spInfo[minID*SPI_SIZE + SPI_NOR_SY] += normalMap[y*width*3+x*3+1];
						spInfo[minID*SPI_SIZE + SPI_NOR_SZ] += normalMap[y*width*3+x*3+2];
						spInfo[minID*SPI_SIZE + SPI_DEPTH_SUM ]  += depthMap[y*width+x];
					}
				}
			}
		}

		//avg
		for(int i=0;i<spNum;i++)
		{
			int t = spInfo[i*SPI_SIZE+SPI_PNUM];
			if(t!=0) 
			{
				spInfo[i*SPI_SIZE+SPI_DIST_DEV] = std::sqrt(spInfo[i*SPI_SIZE+SPI_DIST_DEV]/t);
				spInfo[i*SPI_SIZE+SPI_NOR_DEV] = std::sqrt(spInfo[i*SPI_SIZE+SPI_NOR_DEV]/t);

				spInfo[i*SPI_SIZE+SPI_POS_AX] = spInfo[i*SPI_SIZE+SPI_POS_SX]/t;
				spInfo[i*SPI_SIZE+SPI_POS_AY] = spInfo[i*SPI_SIZE+SPI_POS_SY]/t;
				spInfo[i*SPI_SIZE+SPI_POS_AZ] = spInfo[i*SPI_SIZE+SPI_POS_SZ]/t;

				float nx = spInfo[i*SPI_SIZE+SPI_NOR_SX];
				float ny = spInfo[i*SPI_SIZE+SPI_NOR_SY];
				float nz = spInfo[i*SPI_SIZE+SPI_NOR_SZ];
				float len = std::sqrt(nx*nx+ny*ny+nz*nz);
				spInfo[i*SPI_SIZE+SPI_NOR_AX] = spInfo[i*SPI_SIZE+SPI_NOR_SX]/len;
				spInfo[i*SPI_SIZE+SPI_NOR_AY] = spInfo[i*SPI_SIZE+SPI_NOR_SY]/len;
				spInfo[i*SPI_SIZE+SPI_NOR_AZ] = spInfo[i*SPI_SIZE+SPI_NOR_SZ]/len;

				spInfo[i*SPI_SIZE+SPI_DEPTH_AVG ] = spInfo[i*SPI_SIZE+SPI_DEPTH_SUM ]/t;
			}
			
			//if(n!=iterations-1)
			//{
			//	spInfo[i*SPI_SIZE+SPI_DIST_DEV] = 0;
			//	spInfo[i*SPI_SIZE+SPI_NOR_DEV] = 0;
			//}
		}
	//}
}

void InstanceFusion::maskSuperPixelFilter_OverSeg(int spNum,int *finalSPixel)
{
	//Step 1 count pixel num
	int* NumMatrix = (int*)malloc((resultMasks_Num+1)*spNum*sizeof(int));
	memset(NumMatrix, 0, (resultMasks_Num+1)*spNum*sizeof(int));
	for(int x=0;x<width;x++)
	{
		for(int y=0;y<height;y++)
		{
			int id = finalSPixel[y*width+x];
			if(id>=spNum||id<0) continue;
				
			//last row(pointNum of each superPixel)
			NumMatrix[resultMasks_Num*spNum + id] ++;
			
			for(int i=0;i<resultMasks_Num;i++)
			{
				if( resultMasks[i*width*height+y*width+x] )
				{
					//pointNum of each Instance
					NumMatrix[i*spNum + id] ++;
				}
			}
		}
	}
	
	//Step 2
	for(int x=0;x<width;x++)
	{
		for(int y=0;y<height;y++)
		{
			int id = finalSPixel[y*width+x];
			if(id>=spNum||id<0) 
			{
				for(int i=0;i<resultMasks_Num;i++) resultMasks[i*width*height+y*width+x] = 0;
				continue;
			}
			
			for(int i=0;i<resultMasks_Num;i++)
			{
				int n = NumMatrix[resultMasks_Num*spNum + id];
				int m = NumMatrix[i*spNum+id];
				//if(n<filterNumThreshold) continue;

				float test = m*1.0f/n;
				if(test>0.75)//&&n>filterNumThreshold
				{
					resultMasks[i*width*height+y*width+x] = 255;
				}
				else 
				{
					resultMasks[i*width*height+y*width+x] = 0;
				}
			}
		}
	}


	if(NumMatrix)	free(NumMatrix);
}
//================== gSLICr Demo ==============================================================

void InstanceFusion::gSLICrInterface(int* segMask)
{

	// gSLICr takes gSLICr::UChar4Image as input and out put
	gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);

	cv::Size s(my_settings.img_size.x, my_settings.img_size.y);
	cv::Mat rgbImage(height, width, CV_8UC3, inputImage);
	cv::Mat oldFrame,frame;

	cv::cvtColor(rgbImage,oldFrame,cv::COLOR_BGR2RGB);

	cv::resize(oldFrame, frame, s);
		
	imageCV2SLIC(frame, in_img);
        
	//start
	gSLICr_engine->Process_Frame(in_img);
	//End

	//segMask = gSLICr_engine->Get_Seg_Res()->GetData(MEMORYDEVICE_CPU);
	memcpy(segMask, gSLICr_engine->Get_Seg_Res()->GetData(MEMORYDEVICE_CPU), height*width*sizeof(int));
	

	//Debug
	/*	
	unsigned short segMask_debug[height*width];
	for(int x=0;x<width;x++)
	{
		for(int y=0;y<height;y++)
		{
			segMask_debug[y*width+x]=segMask[y*width+x];
		}
	}
	cv::Mat segMaskImage(height, width, CV_16UC1, segMask_debug);
	std::string savename = "./temp/segMask.png";
	imwrite(savename.c_str(),segMaskImage);
	*/

	
	//Debug
	/*
	gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
	cv::Mat boundry_draw_frame; 
	boundry_draw_frame.create(s, CV_8UC3);
	gSLICr_engine->Draw_Segmentation_Result(out_img);
	imageSLIC2CV(out_img, boundry_draw_frame);

	char out_name[100];
	sprintf(out_name, "./temp/seg_%04i.pgm", frameID);
	gSLICr_engine->Write_Seg_Res_To_PGM(out_name);
	sprintf(out_name, "./temp/edge_%04i.png", frameID);
	imwrite(out_name, boundry_draw_frame);
	sprintf(out_name, "./temp/img_%04i.png", frameID);
	imwrite(out_name, frame);
	printf("\nsaved segmentation %04i\n", frameID);
	*/
	
	delete in_img;
}

void InstanceFusion::imageCV2SLIC(const cv::Mat& inimg, gSLICr::UChar4Image* outimg)
{
	gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < outimg->noDims.y;y++)
	{
		for (int x = 0; x < outimg->noDims.x; x++)
		{
			int idx = x + y * outimg->noDims.x;
			outimg_ptr[idx].b = inimg.at<cv::Vec3b>(y, x)[0];
			outimg_ptr[idx].g = inimg.at<cv::Vec3b>(y, x)[1];
			outimg_ptr[idx].r = inimg.at<cv::Vec3b>(y, x)[2];
		}
	}
}

void InstanceFusion::imageSLIC2CV(const gSLICr::UChar4Image* inimg, cv::Mat& outimg)
{
	const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++)
	{
		for (int x = 0; x < inimg->noDims.x; x++)
		{
			int idx = x + y * inimg->noDims.x;
			outimg.at<cv::Vec3b>(y, x)[0] = inimg_ptr[idx].b;
			outimg.at<cv::Vec3b>(y, x)[1] = inimg_ptr[idx].g;
			outimg.at<cv::Vec3b>(y, x)[2] = inimg_ptr[idx].r;
		}
	}
}

