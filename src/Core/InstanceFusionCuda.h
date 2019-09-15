/*
 * This file is part of InstanceFusion.
 *
 */

#include <cuda_runtime.h>
#include <cuda.h>


//============ Mask Post processing ===============================================

//CudaTask 0
void maskCleanOverlap(unsigned char* masks, const int masksNum,const int width, const int height);

void depthMapGaussianfilter(unsigned short *oriDepthMap,const int width, const int height,unsigned short * depthMapG);

//CudaTask SuperPiexl

void getPosMapFromDepth(unsigned short *depthMap,float* cam, const int width, const int height,float* posMap);

void getNormalMapFromDepth(unsigned short *depthMap,float* cam, const int width, const int height,float*normalMap);

void getSuperPixelInfoCuda(int* segMask,unsigned short * depthMap,float* posMap,float* normalMap,
				int spNum,float* spInfo,int* spInfoStruct,const int width, const int height);

void getFinalSuperPiexl(float* spInfo,const int width, const int height,int* spInfoStruct,int* segMask);

//============== InstancesFusion ==================================================
//CudaTask 0.0
void checkProjectDepthAndInstance(cudaTextureObject_t map_ids, const int n, float* map_surfels, const int width,
				 const int height, const int surfel_size, const int surfel_instance_offset,const int downsample, int *count);


//CudaTask 1
void getProjectInstanceList(cudaTextureObject_t map_ids,const int n,float* map_surfels,int* instanceProjectMap, 
			const int width, const int height,const int surfel_size,const int surfel_instance_offset);

//CudaTask 2(PLAN B)
void maskCompareMap(unsigned char* masks, const int masks_Num, int* resultClass_ids, int* instanceProjectMap, const int width, const int height, 
			const int instanceNum, int* instTableClassList, int* tempI, int*tempU);
//CudaTask 2_1 (PLAN A)
void computeProjectBoundingBox(unsigned char* masks, const int masksNum, int* instanceProjectMap, const int width, const int height,
			 const int instanceNum, int* MasksBBox, int* projectBBox);

//CudaTask 3_0
void getProjectDepthMap(cudaTextureObject_t map_ids,const int n,float* map_surfels,unsigned short* projectDepthMap, 
			const int width, const int height,const int surfel_size,float* trans,const int ratio);

//CudaTask 3_1_1
void computeMaxCountInMap(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, int* eachInstanceMaximum, int* eachInstanceSumCount);

//CudaTask 3_1_3
void cleanInstanceTableMap(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, int* instanceTableCleanList);

//CudaTask 3_2_1+2
void updateSurfelMapInstance(cudaTextureObject_t index_surfelsIds, const int width, const int height,const int n, float* map_surfels,
				const int surfel_size,const int surfel_instance_offset, unsigned char* masks, const int maskID,const int instanceID,const int deleteNum);

//CudaTask 4
void countAndColourSurfelMap(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset,
				const int surfel_instanceColor_offset, float* instanceTable_color, int* bestIDInEachSurfel);

//CudaTask 5_1
void getVertexFromMap(const int n, float* map_surfels,const int surfel_size,float* data);

//CudaTask 5_2
void mapKnnVoteColour(const int n,float* map_surfels,const int surfel_size,const int surfel_instance_offset,const int surfel_instanceColor_offset,
			int* indicesResults,float* distsResults,const int knn, int* bestIDInEachSurfel,float* tempInstanceMap, float* instanceTable_color);

//============ loopClosure ================================================
void loopClosureCopyMatchInstance(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, int* matchPair,const int matchNum);

//=========================================================================
void renderProjectFrame(cudaTextureObject_t index_surfelsIds, const int width, const int height,const int n, float* map_surfels,
				const int surfel_size,const int surfel_instance_offset,const int surfel_instanceColor_offset, float* projectColor, float* instanceTable_color);

void renderMaskFrame(unsigned char* masks, const int masksNum, const int width, const int height, float* maskColor);

//============== 3D BBOX ==================================================

void testAllSurfelNormalVote(const int n, float* map_surfels, const int surfel_size,const int surfel_instance_offset, const int surfel_normal_offset, 
	const int surfel_instanceColor_offset, float* instanceTable_color,const int semCircleSegNum, int* mapGroundNormalVote, int* mapInstanceNormalVote);

void setGroundandInstanceCoordinate(const int instanceNum, const int semCircleSegNum,float* groundNormal,int* mapInstanceNormalCrossVote,
						int* mapInstanceNormalVote,float* gcMatrix,float* instcMatrix);


void testAllSurfelFindBBox(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, const int surfel_instanceColor_offset, 
					float* instanceTable_color, const float ratio3DBBox,float* gcMatrixInverse, float* instMatrixInverse,bool bboxType, int *map3DBBox);


void mapCountInstanceByInstColor(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset,
					float* instanceTable_color, const int surfel_instanceColor_offset, int* eachInstanceSumCount);

void getSurfelToInstanceBuffer(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, float* instanceTable_color, 
			const int surfel_instanceColor_offset, const int surfel_normal_offset, const int surfel_rgbColor_offset,
			bool bboxType, int *map3DBBox,float* gcMatrixInverse, float* instMatrixInverse,int* instSurfelCountTemp, float** instSurfels);

//============== evaluate =================================================

void computePrecisionAndRecall(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset,const int surfel_instanceColor_offset,
							const int surfel_instanceGT_offset, float* instanceTable_color,	int* instPointNum, int* gtPointNum, int* inst_gt_Map);

//==============CPU_DEBUG==================================================

void cpuTestCopyMapIds(cudaTextureObject_t index_surfelsIds,const int width,const int height, int*  map_ids);


