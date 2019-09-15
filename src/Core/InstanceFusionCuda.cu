/*
 * This file is part of InstanceFusion.
 *
 */

#include <stdio.h>
#include <assert.h> 

#include <cuda_runtime.h>

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool
		abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n",
				cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	} 
}

__inline__ __device__
short decode1_Instance(float mapInfo) {
	short instance1 = short(int(mapInfo) >> 16 & 0xFFFF);
	return instance1;
}

__inline__ __device__
short decode2_Instance(float mapInfo) {
	short instance2 = short(int(mapInfo) & 0xFFFF);
	return instance2;
}

__inline__ __device__
float encode_Instance(short inst1,short inst2) {
	int info = inst1;
	info = (info << 16) + inst2;
	return float(info);
}

__inline__ __device__
bool checkNeighbours(unsigned short *map,const int x, const int y,const int width, const int height)
{

    if(x+1>=width)	return false;
    if(x-1<0)		return false;
    if(y+1>=height)	return false;
    if(y-1<0)		return false;
    
    if(!map[y*width+x+1])     return false;
    if(!map[y*width+x-1])     return false;
    if(!map[(y+1)*width+x])   return false;
    if(!map[(y-1)*width+x])   return false;

    if(!map[(y+1)*width+x+1])   return false;
    if(!map[(y+1)*width+x-1])   return false;
    if(!map[(y-1)*width+x+1])   return false;
    if(!map[(y-1)*width+x-1])   return false;
        
    return true;
}
//============ Math ===============================================

__inline__ __device__
void vectorCrossProduct(float* left,float* right,float* ans)
{
	ans[0] = left[1]*right[2] - left[2]*right[1];
	ans[1] = left[2]*right[0] - left[0]*right[2];
	ans[2] = left[0]*right[1] - left[1]*right[0];
}
__inline__ __device__
float vectorLen(float* vec)
{
	return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}
__inline__ __device__
float vectorDist(float* vec1,float* vec2)
{
	float dx = vec1[0] - vec2[0];
	float dy = vec1[1] - vec2[1];
	float dz = vec1[2] - vec2[2];
					
	return sqrt(dx*dx+dy*dy+dz*dz);
}
__inline__ __device__
void vectorNormalize(float* vec)
{
	float len = vectorLen(vec);
	vec[0] = vec[0]/len;
	vec[1] = vec[1]/len;
	vec[2] = vec[2]/len;
}
__inline__ __device__
void rodriguesRotation(float angle, float* V_start,float* K_axis,float* V_result)
{
		float part1[3];
		part1[0] = cos(angle)*V_start[0];
		part1[1] = cos(angle)*V_start[1];
		part1[2] = cos(angle)*V_start[2];

		float part2[3];
		part2[0] = (1-cos(angle))*(V_start[0]*K_axis[0]+V_start[1]*K_axis[1]+V_start[2]*K_axis[2]) *K_axis[0];
		part2[1] = (1-cos(angle))*(V_start[0]*K_axis[0]+V_start[1]*K_axis[1]+V_start[2]*K_axis[2]) *K_axis[1];
		part2[2] = (1-cos(angle))*(V_start[0]*K_axis[0]+V_start[1]*K_axis[1]+V_start[2]*K_axis[2]) *K_axis[2];
				
		float part3[3];
		part3[0] = sin(angle)*(V_start[1]*K_axis[2] - V_start[2]*K_axis[1]);
		part3[1] = sin(angle)*(V_start[2]*K_axis[0] - V_start[0]*K_axis[2]);
		part3[2] = sin(angle)*(V_start[0]*K_axis[1] - V_start[1]*K_axis[0]);
					
		V_result[0] = part1[0] + part2[0] + part3[0];
		V_result[1] = part1[1] + part2[1] + part3[1];
		V_result[2] = part1[2] + part2[2] + part3[2];
}
//============ Mask Post processing ===============================================

//CudaTask 0
__global__ 
void maskCleanOverlapKernel(unsigned char* masks, const int masksNum,const int width, const int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	int flag = 0;
	for(int maskID = masksNum-1;maskID>=0;maskID--)
	{
		if(flag) masks[ (maskID*width*height) + (x+y*width) ] = 0;
		if(masks[ (maskID*width*height) + (x+y*width) ]) flag = 1;
	}
}

__host__ 
void maskCleanOverlap(unsigned char* masks, const int masksNum,const int width, const int height)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);
	
	maskCleanOverlapKernel<<<dimGrid,dimBlock>>>(masks, masksNum, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}


__global__ 
void depthMapGaussianfilterKernel(unsigned short *oriDepthMap,const int width, const int height,unsigned short * depthMapG)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(checkNeighbours(oriDepthMap,x,y,width,height))
	{
		int sum=0;
		int n = 0;
		if(oriDepthMap[(y  )*width+(x  )]) {n+=4; sum += 4*oriDepthMap[(y  )*width+(x  )];}

		if(oriDepthMap[(y  )*width+(x+1)]) {n+=2; sum += 2*oriDepthMap[(y  )*width+(x+1)];}
		if(oriDepthMap[(y  )*width+(x-1)]) {n+=2; sum += 2*oriDepthMap[(y  )*width+(x-1)];}
		if(oriDepthMap[(y+1)*width+(x  )]) {n+=2; sum += 2*oriDepthMap[(y+1)*width+(x  )];}
		if(oriDepthMap[(y-1)*width+(x  )]) {n+=2; sum += 2*oriDepthMap[(y-1)*width+(x  )];}

		if(oriDepthMap[(y+1)*width+(x+1)]) {n+=1; sum +=   oriDepthMap[(y+1)*width+(x+1)];}
		if(oriDepthMap[(y+1)*width+(x-1)]) {n+=1; sum +=   oriDepthMap[(y+1)*width+(x-1)];}
		if(oriDepthMap[(y-1)*width+(x+1)]) {n+=1; sum +=   oriDepthMap[(y-1)*width+(x+1)];}
		if(oriDepthMap[(y-1)*width+(x-1)]) {n+=1; sum +=   oriDepthMap[(y-1)*width+(x-1)];}

		if(n) depthMapG[y*width+x] = sum/n;
	}
}
__host__ 
void depthMapGaussianfilter(unsigned short *oriDepthMap,const int width, const int height,unsigned short * depthMapG)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);
	
	depthMapGaussianfilterKernel<<<dimGrid,dimBlock>>>(oriDepthMap,  width, height, depthMapG);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}
//CudaTask SuperPiexl

__inline__ __device__
void getVertex(unsigned short *depth,const int x, const int y,const int width, const int height, float* cam, float* vertex)
{
	float z = float(depth[y*width+x])/1186.0f;
	vertex[0] = (x - cam[0]) * z * cam[2];
	vertex[1] = (y - cam[1]) * z * cam[3];
	vertex[2] = z ;
}

__inline__ __device__
void getNormalCross(float* left,float* right,float* up,float* down,float* ans)
{
	float del_x[3];
	float del_y[3];
	//del_x
	del_x[0] = left[0] - right[0];
	del_x[1] = left[1] - right[1];
	del_x[2] = left[2] - right[2];
	//del_y
	del_y[0] = up[0] - down[0];
	del_y[1] = up[1] - down[1];
	del_y[2] = up[2] - down[2];
	//ans
	ans[0] = del_x[1] * del_y[2] - del_x[2] * del_y[1];
	ans[1] = del_x[2] * del_y[0] - del_x[0] * del_y[2];
	ans[2] = del_x[0] * del_y[1] - del_x[1] * del_y[0];
}

__inline__ __device__
void getNormal(unsigned short *depth,const int x, const int y,const int width, const int height, float* cam, float* nor)
{
	float vPosition[3];
	getVertex( depth, x, y, width, height, cam, vPosition);

	float vPosition_xf[3];
	float vPosition_xb[3];
	//get
	getVertex( depth, x+1, y, width, height, cam, vPosition_xf);
	getVertex( depth, x-1, y, width, height, cam, vPosition_xb);
	//xb
	//vPosition_xb[0] = (vPosition_xb[0] + vPosition[0]) / 2;
	//vPosition_xb[1] = (vPosition_xb[1] + vPosition[1]) / 2;
	//vPosition_xb[2] = (vPosition_xb[2] + vPosition[2]) / 2;
	//xf
	//vPosition_xf[0] = (vPosition_xf[0] + vPosition[0]) / 2;
	//vPosition_xf[1] = (vPosition_xf[1] + vPosition[1]) / 2;
	//vPosition_xf[2] = (vPosition_xf[2] + vPosition[2]) / 2;

	float vPosition_yf[3];
	float vPosition_yb[3];
	getVertex( depth, x, y+1, width, height, cam, vPosition_yf);
	getVertex( depth, x, y-1, width, height, cam, vPosition_yb);
	//yb
	//vPosition_yb[0] = (vPosition_yb[0] + vPosition[0]) / 2;
	//vPosition_yb[1] = (vPosition_yb[1] + vPosition[1]) / 2;
	//vPosition_yb[2] = (vPosition_yb[2] + vPosition[2]) / 2;
	//yf
	//vPosition_yf[0] = (vPosition_yf[0] + vPosition[0]) / 2;
	//vPosition_yf[1] = (vPosition_yf[1] + vPosition[1]) / 2;
	//vPosition_yf[2] = (vPosition_yf[2] + vPosition[2]) / 2;

	float temp[3];
	float sum[3];
	getNormalCross(vPosition_xb,vPosition_xf,vPosition_yb,vPosition_yf,temp);
	sum[0] = temp[0]*4;
	sum[1] = temp[1]*4;
	sum[2] = temp[2]*4;

	getNormalCross(vPosition_xb,vPosition   ,vPosition_yb,vPosition   ,temp);
	sum[0] += temp[0]*2;
	sum[1] += temp[1]*2;
	sum[2] += temp[2]*2;

	getNormalCross(vPosition   ,vPosition_xf,vPosition_yb,vPosition   ,temp);
	sum[0] += temp[0]*2;
	sum[1] += temp[1]*2;
	sum[2] += temp[2]*2;

	getNormalCross(vPosition_xb,vPosition   ,vPosition   ,vPosition_yf,temp);
	sum[0] += temp[0]*2;
	sum[1] += temp[1]*2;
	sum[2] += temp[2]*2;

	getNormalCross(vPosition   ,vPosition_xf,vPosition   ,vPosition_yf,temp);
	sum[0] += temp[0]*2;
	sum[1] += temp[1]*2;
	sum[2] += temp[2]*2;

	float len = sqrt( sum[0]*sum[0] + sum[1]*sum[1] + sum[2]*sum[2]);
	nor[0] = sum[0]/len;
	nor[1] = sum[1]/len;
	nor[2] = sum[2]/len;
}

__global__ 
void getPosMapFromDepthKernel(unsigned short *depthMap,float* cam, const int width, const int height,float* posMap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
		

	if(depthMap[y*width+x])
	{
		float vPosition[3];
		getVertex( depthMap, x, y, width, height, cam, vPosition);

		posMap[y*width*3+x*3+0] = vPosition[0];
		posMap[y*width*3+x*3+1] = vPosition[1];
		posMap[y*width*3+x*3+2] = vPosition[2];
	}
}
__host__ 
void getPosMapFromDepth(unsigned short *depthMap,float* cam, const int width, const int height,float* posMap)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);
	
	getPosMapFromDepthKernel<<<dimGrid,dimBlock>>>(depthMap,cam, width, height, posMap);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

__global__ 
void getNormalMapFromDepthKernel(unsigned short *depthMap,float* cam, const int width, const int height,float*normalMap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(checkNeighbours(depthMap,x,y,width,height))
	{
		float thisNor[3];
		getNormal(depthMap,x,y,width,height,cam,thisNor);
	
		normalMap[y*width*3+x*3+0] = thisNor[0];
		normalMap[y*width*3+x*3+1] = thisNor[1];
		normalMap[y*width*3+x*3+2] = thisNor[2];
	}
}

__host__ 
void getNormalMapFromDepth(unsigned short *depthMap,float* cam, const int width, const int height,float*normalMap)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);
	
	getNormalMapFromDepthKernel<<<dimGrid,dimBlock>>>(depthMap,cam, width, height, normalMap);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}


__global__	//init spInfo
void getSuperPixelInfoCudaKernel0(int* segMask,unsigned short * depthMap,float* posMap,float* normalMap,
									int spNum,float* spInfo,int* spInfoStruct,const int width, const int height)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<spNum)
	{
		//spInfoStruct[18]=SPI_CONNECT_N	spInfoStruct[19] = SPI_NP_FIRST	spInfoStruct[20] = SPI_NP_MAX
		int connectMaxNum = spInfoStruct[20];
		for(int i=0;i<connectMaxNum;i++)
		{
			spInfo[id*spInfoStruct[0]+spInfoStruct[19]+i] = -1;
			spInfo[id*spInfoStruct[0]+spInfoStruct[18]] = connectMaxNum;
		}
	}
}

__global__ //First
void getSuperPixelInfoCudaKernelA(int* segMask,unsigned short * depthMap,float* posMap,float* normalMap,
									int spNum,float* spInfo,int* spInfoStruct,const int width, const int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
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
	}
}
__global__ //First sum
void getSuperPixelInfoCudaKernelB(int* segMask,unsigned short * depthMap,float* posMap,float* normalMap,
									int spNum,float* spInfo,int* spInfoStruct,const int width, const int height)
{
	//spInfoStruct[0]=SPI_SIZE		spInfoStruct[1] = SPI_PNUM
	//spInfoStruct[2]=SPI_POS_SX	spInfoStruct[3] = SPI_POS_SY	spInfoStruct[4] = SPI_POS_SZ
	//spInfoStruct[5]=SPI_NOR_SX	spInfoStruct[6] = SPI_NOR_SY	spInfoStruct[7] = SPI_NOR_SZ
	//spInfoStruct[14]=SPI_DEPTH_SUM	
	//spInfoStruct[18]=SPI_CONNECT_N	spInfoStruct[19] = SPI_NP_FIRST	spInfoStruct[20] = SPI_NP_MAX

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	int id = segMask[y*width+x];
	if(id>=0&&id<spNum)
	{
		atomicAdd( spInfo + id*spInfoStruct[0] + spInfoStruct[1] , 1);

		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[2] , posMap[y*width*3+x*3+0]);
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[3] , posMap[y*width*3+x*3+1]);
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[4] , posMap[y*width*3+x*3+2]);

		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[5] , normalMap[y*width*3+x*3+0]);
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[6] , normalMap[y*width*3+x*3+1]);
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[7] , normalMap[y*width*3+x*3+2]);
	
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[14], depthMap[y*width+x]);
			
		//check neighbor
		const int StepX[4] = {0,0,1,-1};
		const int StepY[4] = {1,-1,0,0};
		for(int i=0; i<4; i++)
		{
			if(x==0||x==width-1||y==0||y==height-1) continue;

			int dx = x+StepX[i];
			int dy = y+StepY[i];
			int nbID = segMask[dy*width+dx];
			if(nbID>=spNum||nbID<0) continue; 	//-1

			if(id!=nbID)
			{
				int connectMaxNum = spInfoStruct[20];
				if(connectMaxNum!=11) printf("ERROR connectMaxNum!=list_num(11) in InstanceFusionCuda line 345.Please set list_num = connectMaxNum.");

				int list[11];
				int p=0;
				for(int j=0;j<connectMaxNum;j++)
				{
					//***********unstable code************//
					int existInList = 0;
					for(int k=0;k<p;k++)
					{
						if(list[k]==spInfo[id*spInfoStruct[0]+spInfoStruct[19]+j])
						{
							existInList = 1;
							break;
						}
					}
					if(existInList||spInfo[id*spInfoStruct[0]+spInfoStruct[19]+j]==-1)
					{
						spInfo[id*spInfoStruct[0]+spInfoStruct[19]+j]=nbID;
					}
					else
					{
						list[p++] = spInfo[id*spInfoStruct[0]+spInfoStruct[19]+j];
						if(p>=11) p--;
					}
					//***********unstable code************//
				}
			}
		}


		//check neighbor
		/*
		const int StepX[4] = {0,0,1,-1};
		const int StepY[4] = {1,-1,0,0};
		bool next = true;
		while(next)
		{
			int v = atomicCAS(spLock+id,-1,y*width+x);	//lock
			if(spLock[id]==y*width+x)
			{
				//handle
				for(int i=0; i<4; i++)
				{
					if(x==0||x==width-1||y==0||y==height-1) continue;

					int dx = x+StepX[i];
					int dy = y+StepY[i];
					int nbID = segMask[dy*width+dx];
					if(nbID>=spNum||nbID<0) continue; 	//-1

					if(id!=nbID)
					{
						int connectNum = spInfo[id*spInfoStruct[0]+spInfoStruct[18]];
						if(connectNum>=spInfoStruct[20])continue;
					
						int exist = 0;
						for(int j=0;j<connectNum;j++)
						{
							if(spInfo[id*spInfoStruct[0]+spInfoStruct[19]+j]==nbID)
							{
								exist = 1;
								break;
							}
						}
						if(!exist&&connectNum<spInfoStruct[20]) 
						{
							spInfo[id*spInfoStruct[0]+spInfoStruct[19]+connectNum]=nbID;
							spInfo[id*spInfoStruct[0]+spInfoStruct[18]]++;
						}
					}
				}
				//unlock
				atomicExch(spLock+id,-1);
				next = 0;
			}
		}
		*/
		
	}
}
__global__ //First avg
void getSuperPixelInfoCudaKernelC(int* segMask,unsigned short * depthMap,float* posMap,float* normalMap,
									int spNum,float* spInfo,int* spInfoStruct,const int width, const int height)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<spNum)
	{
		//spInfoStruct[0] =  SPI_SIZE;		spInfoStruct[1] =  SPI_PNUM;
		//spInfoStruct[2] =  SPI_POS_SX;		spInfoStruct[3] =  SPI_POS_SY;	spInfoStruct[4] =  SPI_POS_SZ;
		//spInfoStruct[5] =  SPI_NOR_SX;		spInfoStruct[6] =  SPI_NOR_SY;	spInfoStruct[7] =  SPI_NOR_SZ;
		//spInfoStruct[8] =  SPI_POS_AX;		spInfoStruct[9] =  SPI_POS_AY;	spInfoStruct[10] =  SPI_POS_AZ;
		//spInfoStruct[11] =  SPI_NOR_AX;		spInfoStruct[12] =  SPI_NOR_AY;	spInfoStruct[13] =  SPI_NOR_AZ;
		//spInfoStruct[14] =  SPI_DEPTH_SUM;	spInfoStruct[15] =  SPI_DEPTH_AVG;

		int t = spInfo[id*spInfoStruct[0]+spInfoStruct[1]];
		if(t!=0) 
		{
			spInfo[id*spInfoStruct[0]+spInfoStruct[8]] = spInfo[id*spInfoStruct[0]+spInfoStruct[2]]/t;
			spInfo[id*spInfoStruct[0]+spInfoStruct[9]] = spInfo[id*spInfoStruct[0]+spInfoStruct[3]]/t;
			spInfo[id*spInfoStruct[0]+spInfoStruct[10]] = spInfo[id*spInfoStruct[0]+spInfoStruct[4]]/t;

			float nx = spInfo[id*spInfoStruct[0]+spInfoStruct[5]];
			float ny = spInfo[id*spInfoStruct[0]+spInfoStruct[6]];
			float nz = spInfo[id*spInfoStruct[0]+spInfoStruct[7]];

			float len = sqrt(nx*nx+ny*ny+nz*nz);
			spInfo[id*spInfoStruct[0]+spInfoStruct[11]] = spInfo[id*spInfoStruct[0]+spInfoStruct[5]]/len;
			spInfo[id*spInfoStruct[0]+spInfoStruct[12]] = spInfo[id*spInfoStruct[0]+spInfoStruct[6]]/len;
			spInfo[id*spInfoStruct[0]+spInfoStruct[13]] = spInfo[id*spInfoStruct[0]+spInfoStruct[7]]/len;

			spInfo[id*spInfoStruct[0]+spInfoStruct[15] ] = spInfo[id*spInfoStruct[0]+spInfoStruct[14] ]/t;
		}
	}
}
__global__ //Second 
void getSuperPixelInfoCudaKernelD(int* segMask,unsigned short * depthMap,float* posMap,float* normalMap,
									int spNum,float* spInfo,int* spInfoStruct,const int width, const int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	//depth_stand_deviation
	int id = segMask[y*width+x];
	if(id>=spNum||id<0) return;	//-1
	
	//spInfoStruct[0] =  SPI_SIZE;			spInfoStruct[1] =  SPI_PNUM;
	//spInfoStruct[2] =  SPI_POS_SX;		spInfoStruct[3] =  SPI_POS_SY;	spInfoStruct[4] =  SPI_POS_SZ;
	//spInfoStruct[5] =  SPI_NOR_SX;		spInfoStruct[6] =  SPI_NOR_SY;	spInfoStruct[7] =  SPI_NOR_SZ;
	//spInfoStruct[8] =  SPI_POS_AX;		spInfoStruct[9] =  SPI_POS_AY;	spInfoStruct[10] =  SPI_POS_AZ;
	//spInfoStruct[11] =  SPI_NOR_AX;		spInfoStruct[12] =  SPI_NOR_AY;	spInfoStruct[13] =  SPI_NOR_AZ;
	//spInfoStruct[14] =  SPI_DEPTH_SUM;	spInfoStruct[15] =  SPI_DEPTH_AVG;
	//spInfoStruct[16] =  SPI_DIST_DEV;		spInfoStruct[17] =  SPI_NOR_DEV;	
	//spInfoStruct[18] =  SPI_CONNECT_N;	spInfoStruct[19] =  SPI_NP_FIRST;	spInfoStruct[20] =  SPI_NP_MAX;
	//spInfoStruct[21] =  SPI_FINAL;
	int connectNum = spInfoStruct[20];
	float minDist = 999999.9f;
	float minNor  = 999999.9f;
	int minID = id;
	for(int i = 0;i<=connectNum;i++)
	{
		int idTest;
		if(i==connectNum) idTest = id;	//self
		else idTest = spInfo[id*spInfoStruct[0]+spInfoStruct[19]+i];
				
		float vecA[3];	//SPI_NOR_A
		vecA[0] = spInfo[idTest*spInfoStruct[0]+spInfoStruct[11]];
		vecA[1] = spInfo[idTest*spInfoStruct[0]+spInfoStruct[12]];
		vecA[2] = spInfo[idTest*spInfoStruct[0]+spInfoStruct[13]];
				
		float vecB[3];	//SPI_POS_A - P
		vecB[0] = spInfo[idTest*spInfoStruct[0] + spInfoStruct[8] ]  - posMap[y*width*3+x*3+0];
		vecB[1] = spInfo[idTest*spInfoStruct[0] + spInfoStruct[9] ]  - posMap[y*width*3+x*3+1];
		vecB[2] = spInfo[idTest*spInfoStruct[0] + spInfoStruct[10]]  - posMap[y*width*3+x*3+2];

		float lenA = sqrt( vecA[0]*vecA[0] + vecA[1]*vecA[1] + vecA[2]*vecA[2]);
		float lenB = sqrt( vecB[0]*vecB[0] + vecB[1]*vecB[1] + vecB[2]*vecB[2]);

		float dotAB = vecA[0]*vecB[0] + vecA[1]*vecB[1] + vecA[2]*vecB[2]; 
		float dist = abs(dotAB / lenA) + 1.0*lenB;
					
		float diffNor1 = abs(vecA[0]-normalMap[y*width*3+x*3+0]);
		float diffNor2 = abs(vecA[1]-normalMap[y*width*3+x*3+1]);
		float diffNor3 = abs(vecA[2]-normalMap[y*width*3+x*3+2]);
		float diffNor = diffNor1*diffNor1+diffNor2*diffNor2+diffNor3*diffNor3;
		if(dist<minDist)	
		{
			minNor = diffNor;
			minDist = dist;
			minID = idTest;
		}
	}
	
	float threshold = (0.026*spInfo[minID*spInfoStruct[0]+spInfoStruct[15]]-4.0f)/1186.0f;
	if(minDist>2*threshold)	minID = -1;
				
	if(minID!=-1)
	{		
		atomicAdd(spInfo+minID*spInfoStruct[0]+spInfoStruct[16] , (minDist * minDist));
		atomicAdd(spInfo+minID*spInfoStruct[0]+spInfoStruct[17] , minNor);
	}

	if(id!=minID)	//re-clustering
	{
		segMask[y*width+x] = minID;
	
		atomicAdd(spInfo+id*spInfoStruct[0]+spInfoStruct[1] , -1);
		atomicAdd(spInfo+id*spInfoStruct[0]+spInfoStruct[2] , -posMap[y*width*3+x*3+0]);
		atomicAdd(spInfo+id*spInfoStruct[0]+spInfoStruct[3] , -posMap[y*width*3+x*3+1]);
		atomicAdd(spInfo+id*spInfoStruct[0]+spInfoStruct[4] , -posMap[y*width*3+x*3+2]);
		atomicAdd(spInfo+id*spInfoStruct[0]+spInfoStruct[5] , -normalMap[y*width*3+x*3+0]);
		atomicAdd(spInfo+id*spInfoStruct[0]+spInfoStruct[6] , -normalMap[y*width*3+x*3+1]);
		atomicAdd(spInfo+id*spInfoStruct[0]+spInfoStruct[7] , -normalMap[y*width*3+x*3+2]);
		atomicAdd(spInfo+id*spInfoStruct[0]+spInfoStruct[14], -depthMap[y*width+x]);

		if(minID!=-1)
		{
			atomicAdd(spInfo+minID*spInfoStruct[0]+spInfoStruct[1] , 1);
			atomicAdd(spInfo+minID*spInfoStruct[0]+spInfoStruct[2] , posMap[y*width*3+x*3+0]);
			atomicAdd(spInfo+minID*spInfoStruct[0]+spInfoStruct[3] , posMap[y*width*3+x*3+1]);
			atomicAdd(spInfo+minID*spInfoStruct[0]+spInfoStruct[4] , posMap[y*width*3+x*3+2]);
			atomicAdd(spInfo+minID*spInfoStruct[0]+spInfoStruct[5] , normalMap[y*width*3+x*3+0]);
			atomicAdd(spInfo+minID*spInfoStruct[0]+spInfoStruct[6] , normalMap[y*width*3+x*3+1]);
			atomicAdd(spInfo+minID*spInfoStruct[0]+spInfoStruct[7] , normalMap[y*width*3+x*3+2]);
			atomicAdd(spInfo+minID*spInfoStruct[0]+spInfoStruct[14], depthMap[y*width+x]);
		}
	}

}
__global__ //Second avg
void getSuperPixelInfoCudaKernelE(int* segMask,unsigned short * depthMap,float* posMap,float* normalMap,
									int spNum,float* spInfo,int* spInfoStruct,const int width, const int height)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<spNum)
	{
		//spInfoStruct[0] =  SPI_SIZE;		spInfoStruct[1] =  SPI_PNUM;
		//spInfoStruct[2] =  SPI_POS_SX;		spInfoStruct[3] =  SPI_POS_SY;	spInfoStruct[4] =  SPI_POS_SZ;
		//spInfoStruct[5] =  SPI_NOR_SX;		spInfoStruct[6] =  SPI_NOR_SY;	spInfoStruct[7] =  SPI_NOR_SZ;
		//spInfoStruct[8] =  SPI_POS_AX;		spInfoStruct[9] =  SPI_POS_AY;	spInfoStruct[10] =  SPI_POS_AZ;
		//spInfoStruct[11] =  SPI_NOR_AX;		spInfoStruct[12] =  SPI_NOR_AY;	spInfoStruct[13] =  SPI_NOR_AZ;
		//spInfoStruct[14] =  SPI_DEPTH_SUM;	spInfoStruct[15] =  SPI_DEPTH_AVG;
		//spInfoStruct[16] =  SPI_DIST_DEV;		spInfoStruct[17] =  SPI_NOR_DEV;	
		int t = spInfo[id*spInfoStruct[0]+spInfoStruct[1]];
		if(t!=0) 
		{
			spInfo[id*spInfoStruct[0]+spInfoStruct[16] ] = sqrt(spInfo[id*spInfoStruct[0]+spInfoStruct[16]]/t);
			spInfo[id*spInfoStruct[0]+spInfoStruct[17]] = sqrt(spInfo[id*spInfoStruct[0]+spInfoStruct[17]]/t);

			spInfo[id*spInfoStruct[0]+spInfoStruct[8]]  = spInfo[id*spInfoStruct[0]+spInfoStruct[2]]/t;
			spInfo[id*spInfoStruct[0]+spInfoStruct[9]]  = spInfo[id*spInfoStruct[0]+spInfoStruct[3]]/t;
			spInfo[id*spInfoStruct[0]+spInfoStruct[10]] = spInfo[id*spInfoStruct[0]+spInfoStruct[4]]/t;

			float nx = spInfo[id*spInfoStruct[0]+spInfoStruct[5]];
			float ny = spInfo[id*spInfoStruct[0]+spInfoStruct[6]];
			float nz = spInfo[id*spInfoStruct[0]+spInfoStruct[7]];
			float len = sqrt(nx*nx+ny*ny+nz*nz);
			spInfo[id*spInfoStruct[0]+spInfoStruct[11]] = spInfo[id*spInfoStruct[0]+spInfoStruct[5]]/len;
			spInfo[id*spInfoStruct[0]+spInfoStruct[12]] = spInfo[id*spInfoStruct[0]+spInfoStruct[6]]/len;
			spInfo[id*spInfoStruct[0]+spInfoStruct[13]] = spInfo[id*spInfoStruct[0]+spInfoStruct[7]]/len;

			spInfo[id*spInfoStruct[0]+spInfoStruct[15]] = spInfo[id*spInfoStruct[0]+spInfoStruct[14]]/t;
		}
	}
}

__host__ 
void getSuperPixelInfoCuda(int* segMask,unsigned short * depthMap,float* posMap,float* normalMap,
									int spNum,float* spInfo,int* spInfoStruct,const int width, const int height)
{
	//width*height
	const int blocks1 = 32;
	dim3 dimGrid1(blocks1,blocks1);
	dim3 dimBlock1(width/blocks1,height/blocks1);

	//spNum
	const int threads2 = 512;
	const int blocks2 = (spNum + threads2 - 1) / threads2;
	dim3 dimGrid2(blocks2);
	dim3 dimBlock2(threads2);
	
	getSuperPixelInfoCudaKernel0<<<dimGrid2,dimBlock2>>>(segMask, depthMap, posMap, normalMap, spNum, spInfo, spInfoStruct, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());

	getSuperPixelInfoCudaKernelA<<<dimGrid1,dimBlock1>>>(segMask, depthMap, posMap, normalMap, spNum, spInfo, spInfoStruct, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());

	getSuperPixelInfoCudaKernelB<<<dimGrid1,dimBlock1>>>(segMask, depthMap, posMap, normalMap, spNum, spInfo, spInfoStruct, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());


	getSuperPixelInfoCudaKernelC<<<dimGrid2,dimBlock2>>>(segMask, depthMap, posMap, normalMap, spNum, spInfo, spInfoStruct, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());

	getSuperPixelInfoCudaKernelD<<<dimGrid1,dimBlock1>>>(segMask, depthMap, posMap, normalMap, spNum, spInfo, spInfoStruct, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());

	getSuperPixelInfoCudaKernelE<<<dimGrid2,dimBlock2>>>(segMask, depthMap, posMap, normalMap, spNum, spInfo, spInfoStruct, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}


__global__ 
void getFinalSuperPiexlKernel(float* spInfo,const int width, const int height,int* spInfoStruct,int* segMask)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int id_ori   = segMask[y*width+x];
	if(id_ori<0)
	{
		segMask[y*width+x] = id_ori;
	}
	else
	{
		//spInfoStruct[0]=SPI_SIZE 	spInfoStruct[21]=SPI_FINAL
		int id_final = spInfo[id_ori*spInfoStruct[0]+spInfoStruct[21]];	
		segMask[y*width+x] = id_final;
	}
}
__host__ 
void getFinalSuperPiexl(float* spInfo,const int width, const int height,int* spInfoStruct,int* segMask)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);
	
	getFinalSuperPiexlKernel<<<dimGrid,dimBlock>>>(spInfo, width, height,spInfoStruct, segMask);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}



//============== InstancesFusion ==================================================

__global__ 
void checkProjectDepthAndInstanceKernel(cudaTextureObject_t map_ids, const int n, float* map_surfels, const int width,
				 const int height, const int surfel_size, const int surfel_instance_offset, const int downsample, int *count)
{
	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x%downsample==0&&y%downsample==0)
	{
		int surfel_id = tex2D<int>(map_ids,x,y);
		int instanceNum_half = surfel_size - surfel_instance_offset;
		if (surfel_id > 0&&surfel_id<n) 
		{
			for(int i=0; i<instanceNum_half; i++)
			{
		
				float mapInfo = map_surfels[surfel_id * surfel_size + surfel_instance_offset + i];
				//decode
				atomicAdd(count+0, decode1_Instance(mapInfo));
				atomicAdd(count+0, decode2_Instance(mapInfo));
			}
		}
		else atomicAdd(count+1,1);
	}
}


__host__ 
void checkProjectDepthAndInstance(cudaTextureObject_t map_ids, const int n, float* map_surfels, const int width,
				 const int height, const int surfel_size, const int surfel_instance_offset, const int downsample, int *count)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);
	
	checkProjectDepthAndInstanceKernel<<<dimGrid,dimBlock>>>(map_ids, n, map_surfels, width, height, 
									surfel_size, surfel_instance_offset, downsample, count);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
	
}



//CudaTask 1
__global__ 
void getProjectInstanceListKernel(cudaTextureObject_t map_ids,const int n,float* map_surfels,int* instanceProjectMap, 
				const int width, const int height,const int surfel_size,const int surfel_instance_offset)
{
	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	int surfel_id = tex2D<int>(map_ids,x,y);
	int instanceNum_half = surfel_size - surfel_instance_offset;
	
	for(int i=0; i<instanceNum_half; i++)
	{
		if (surfel_id > 0&&surfel_id<n) 
		{
			float mapInfo = map_surfels[surfel_id * surfel_size + surfel_instance_offset + i];
			//decode
			instanceProjectMap[ y*width + x + (2*i)*width*height ] = decode1_Instance(mapInfo);
			instanceProjectMap[ y*width + x + (2*i+1)*width*height ] = decode2_Instance(mapInfo);
		}
		else
		{
			instanceProjectMap[ y*width + x + (2*i)*width*height ] = -1;			//check?
			instanceProjectMap[ y*width + x + (2*i+1)*width*height ] = -1;
		}
	}
	
}


__host__ 
void getProjectInstanceList(cudaTextureObject_t map_ids,const int n,float* map_surfels,int* instanceProjectMap, 
			const int width, const int height,const int surfel_size,const int surfel_instance_offset)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);
	
	getProjectInstanceListKernel<<<dimGrid,dimBlock>>>(map_ids,n,map_surfels,instanceProjectMap,width,height,surfel_size,surfel_instance_offset);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
	
}


//CudaTask 2(PLAN B)
__global__ 
void maskCompareMapKernel(unsigned char* masks, const int masks_Num, int* resultClass_ids, int* instanceProjectMap, const int width, const int height, 
							const int instanceNum, int* instTableClassList, int* tempI, int*tempU)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	for(int maskID=0;maskID<masks_Num;maskID++)
	{
		unsigned char* mask = masks + maskID * width * height;
		for(int instanceID=0;instanceID<instanceNum;instanceID++)
		{
			int* instance = instanceProjectMap + instanceID*width*height;
			if(resultClass_ids[maskID] != instTableClassList[instanceID]) 	continue;		//***
			
			int flagM = 0;
			int falgI = 0;
			
			int dx,dy;
			for(int i=-1;i<=1;i++)
			{
				for(int j=-1;j<=1;j++)
				{
					dx=x+i;
					dy=y+j;
					if(dx>=width||dx<0||dy>=height||dy<0)	continue;

					if(mask[dy*width+dx] > 0) flagM=1;
					if(instance[dy*width+dx] > 0) falgI=1;
					
				}
			}
			if(flagM&&falgI) atomicAdd(tempI+instanceID + maskID * instanceNum,1);
			if(flagM||falgI) atomicAdd(tempU+instanceID + maskID * instanceNum,1);
			
		}
		
	}

}


__host__ 
void maskCompareMap(unsigned char* masks, const int masks_Num, int* resultClass_ids, int* instanceProjectMap, const int width, const int height, 
							const int instanceNum, int* instTableClassList, int* tempI, int*tempU)
{	
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);

	maskCompareMapKernel<<<dimGrid,dimBlock>>>(masks, masks_Num, resultClass_ids, instanceProjectMap, width, height,
							 instanceNum, instTableClassList, tempI, tempU);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());

}

//CudaTask 2_1(PLAN A)
__global__ 
void initMapsBoundingBoxKernel(const int masksNum, const int instanceNum, int* MasksBBox, int* projectBBox, const int width, const int height)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	int id = index/4;
	int bboxType = index%4;
	bool mapType = id<instanceNum?0:1;
	if(!mapType)
	{
		if(bboxType==0)			projectBBox[index] = width+1;
		else if(bboxType==1)	projectBBox[index] = -1;
		else if(bboxType==2)	projectBBox[index] = height+1;
		else if(bboxType==3)	projectBBox[index] = -1;
		
	}
	else
	{
		id-=instanceNum;
		if(bboxType==0)			MasksBBox[id*4+bboxType] = width+1;
		else if(bboxType==1)	MasksBBox[id*4+bboxType] = -1;
		else if(bboxType==2)	MasksBBox[id*4+bboxType] = height+1;
		else if(bboxType==3)	MasksBBox[id*4+bboxType] = -1;
	}
}

__global__ 
void computeProjectBoundingBoxKernel(unsigned char* masks, const int masksNum, int* instanceProjectMap, const int width, const int height,
								 const int instanceNum, int* MasksBBox, int* projectBBox)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(instanceProjectMap[ y*width + x]!=-1)
	{
		//Instance
		int maxNum = 0;
		int maxID  = -1;
		for(int instanceID=0;instanceID<instanceNum;instanceID++)
		{
			int* instance = instanceProjectMap + instanceID*width*height;
			if(instance[ y*width + x]>maxNum)
			{
				maxNum = instance[ y*width + x];
				maxID  = instanceID;
			}
		}
		if(maxID!=-1)
		{
			atomicMin(projectBBox+4*maxID+0,x);
			atomicMax(projectBBox+4*maxID+1,x);
			atomicMin(projectBBox+4*maxID+2,y);
			atomicMax(projectBBox+4*maxID+3,y);
		}
		
		//Mask
		for(int maskID=0;maskID<masksNum;maskID++)
		{
			unsigned char* mask = masks + maskID * width * height;
			if(mask[ y*width + x] > 0)
			{
				atomicMin(MasksBBox+4*maskID+0,x);
				atomicMax(MasksBBox+4*maskID+1,x);
				atomicMin(MasksBBox+4*maskID+2,y);
				atomicMax(MasksBBox+4*maskID+3,y);
			}
		}
	}
	
	
}

__host__ 
void computeProjectBoundingBox(unsigned char* masks, const int masksNum, int* instanceProjectMap, const int width, const int height,
						 const int instanceNum, int* MasksBBox, int* projectBBox)
{
	const int threads1 = 512;
	const int blocks1 = ((instanceNum + masksNum)*4 + threads1 - 1) / threads1;
	dim3 dimGrid1(blocks1);
	dim3 dimBlock1(threads1);

	initMapsBoundingBoxKernel<<<dimGrid1,dimBlock1>>>( masksNum, instanceNum, MasksBBox, projectBBox, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());

	const int blocks2 = 32;
	dim3 dimGrid2(blocks2,blocks2);
	dim3 dimBlock2(width/blocks2,height/blocks2);
	
	computeProjectBoundingBoxKernel<<<dimGrid2,dimBlock2>>>(masks, masksNum, instanceProjectMap, width, height, instanceNum, MasksBBox, projectBBox);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

//CudaTask 3_0
__global__ 
void getProjectDepthMapKernel(cudaTextureObject_t index_surfelsIds,const int n,float* map_surfels,unsigned short* projectDepthMap, 
								const int width, const int height,const int surfel_size,float* trans,const int ratio)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	int surfel_id = tex2D<int>(index_surfelsIds,x,y);
	if(surfel_id > 0&&surfel_id < n)
	{
		float mx = map_surfels[surfel_id * surfel_size + 0];
		float my = map_surfels[surfel_id * surfel_size + 1];
		float mz = map_surfels[surfel_id * surfel_size + 2];

		float dx = trans[0]-mx;
		float dy = trans[1]-my;
		float dz = trans[2]-mz;
		
		projectDepthMap[x+y*width] = (unsigned short)(sqrt(dx*dx+dy*dy+dz*dz)*ratio);
	}
}

__host__ 
void getProjectDepthMap(cudaTextureObject_t index_surfelsIds,const int n,float* map_surfels,unsigned short* projectDepthMap, 
								const int width, const int height,const int surfel_size,float* trans,const int ratio)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);
	
	getProjectDepthMapKernel<<<dimGrid,dimBlock>>>(index_surfelsIds,n, map_surfels, projectDepthMap, width, height, surfel_size, trans, ratio);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

//CudaTask 3_1_1
__global__ 
void computeMaxCountInMapKernel(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, int* eachInstanceMaximum, int* eachInstanceSumCount)
{

	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	int instanceNum_half = surfel_size - surfel_instance_offset;
	if (id < n) {
		for(int i=0; i< instanceNum_half; i++)
		{
			float mapInfo = map_surfels[id * surfel_size + surfel_instance_offset + i];
			//decode
			int instanceI_1 = decode1_Instance(mapInfo);
			int instanceI_2 = decode2_Instance(mapInfo);
			//In InstanceTable, each instance has a maxCount.
			atomicMax(eachInstanceMaximum+2*i  ,instanceI_1);
			atomicMax(eachInstanceMaximum+2*i+1,instanceI_2);
			//In Map, each instance has many surfels.
			//if(instanceI_1) atomicAdd(eachInstanceSumCount+2*i,1);
			//if(instanceI_2) atomicAdd(eachInstanceSumCount+2*i+1,1);
			if(instanceI_1) atomicAdd(eachInstanceSumCount+2*i,instanceI_1);
			if(instanceI_2) atomicAdd(eachInstanceSumCount+2*i+1,instanceI_2);
		}
	}

}


__host__ 
void computeMaxCountInMap(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, int* eachInstanceMaximum, int* eachInstanceSumCount)
{
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	
	computeMaxCountInMapKernel<<<dimGrid,dimBlock>>>(n, map_surfels, surfel_size, surfel_instance_offset, eachInstanceMaximum, eachInstanceSumCount);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}




//CudaTask 3_1_3

__global__ 
void cleanInstanceTableMapKernel(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, int* instanceTableCleanList)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	int instanceNum_half = surfel_size - surfel_instance_offset;
	if (id < n) {
		for(int i=0; i< instanceNum_half; i++)
		{
			int instanceI_1_needClean = instanceTableCleanList[2*i];
			int instanceI_2_needClean = instanceTableCleanList[2*i+1];
			if(!instanceI_1_needClean&&!instanceI_2_needClean) continue;

			float mapInfo = map_surfels[id * surfel_size + surfel_instance_offset + i];
			//decode
			int instanceI_1 = decode1_Instance(mapInfo);
			int instanceI_2 = decode2_Instance(mapInfo);
			
			if(instanceI_1_needClean) instanceI_1 = 0;
			if(instanceI_2_needClean) instanceI_2 = 0;

			//encode
			mapInfo = encode_Instance(instanceI_1,instanceI_2);

			//update
			map_surfels[id * surfel_size + surfel_instance_offset + i] = mapInfo;
		}
	}
}

__host__ 
void cleanInstanceTableMap(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, int* instanceTableCleanList)
{
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	
	cleanInstanceTableMapKernel<<<dimGrid,dimBlock>>>(n, map_surfels, surfel_size, surfel_instance_offset, instanceTableCleanList);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

//CudaTask 3_2_1+2
__global__ 
void updateSurfelMapInstanceKernel(cudaTextureObject_t index_surfelsIds, const int width, const int height,const int n, float* map_surfels,
										const int surfel_size,const int surfel_instance_offset, unsigned char* masks, int maskID, int instanceID,int deleteNum)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	unsigned char* mask = masks + maskID*width*height;
	if(mask[ y * width + x] > 0)
	{
		int surfel_id = tex2D<int>(index_surfelsIds,x,y);
		if(surfel_id > 0 && surfel_id < n)
		{
			// OLD or NEW ( Specified by the last parameter)
			if(deleteNum == -1 || (deleteNum != -1&&surfel_id>deleteNum))
			{

				int i = instanceID / 2;
				int p = instanceID % 2;
				float mapInfo = map_surfels[surfel_id * surfel_size + surfel_instance_offset + i];
	
				//decode
				int instanceI_1 = decode1_Instance(mapInfo);	//short
				int instanceI_2 = decode2_Instance(mapInfo);

				if(p==0) instanceI_1 += (maskID+1);		//++
				if(p==1) instanceI_2 += (maskID+1);
				
				if(instanceI_1>=65535)instanceI_1=65535;
				if(instanceI_2>=65535)instanceI_2=65535;
		
				//encode
				mapInfo = encode_Instance(instanceI_1,instanceI_2);
				//update
				map_surfels[surfel_id * surfel_size + surfel_instance_offset + i] = mapInfo;
			}
		}
	}
	
}

__host__ 
void updateSurfelMapInstance(cudaTextureObject_t index_surfelsIds, const int width, const int height,const int n, float* map_surfels,
								const int surfel_size,const int surfel_instance_offset, unsigned char* masks, int maskID, int instanceID,int deleteNum)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);
	
	updateSurfelMapInstanceKernel<<<dimGrid,dimBlock>>>(index_surfelsIds, width, height, n, map_surfels, 
															surfel_size, surfel_instance_offset, masks, maskID, instanceID,deleteNum);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}


//CudaTask 4

__global__ 
void countAndColourSurfelMapKernel(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset,
									const int surfel_instanceColor_offset, float* instanceTable_color, int* bestIDInEachSurfel)
{
	const float defaultColor = 7434609;//0x717171			//1.67772e+07;//0xFFFFFF
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		int BestInstance = -1;
		int BestCount = 0;
		int instanceNum_half = surfel_size - surfel_instance_offset;
		
		for(int i=0; i< instanceNum_half; i++)
		{
			float mapInfo = map_surfels[id * surfel_size + surfel_instance_offset + i];
			//decode
			int instanceI_1 = decode1_Instance(mapInfo);
			int instanceI_2 = decode2_Instance(mapInfo);
			if(BestCount < instanceI_1)
			{
				BestCount = instanceI_1;
				BestInstance = 2 * i;
			}
			if(BestCount < instanceI_2)
			{
				BestCount = instanceI_2;
				BestInstance = 2 * i + 1;
			}
		}
		bestIDInEachSurfel[id] = BestInstance;
		if(map_surfels[id*surfel_size+surfel_instanceColor_offset]==0||map_surfels[id*surfel_size+surfel_instanceColor_offset]==defaultColor)
		{
			if(BestInstance != -1)
			{
				map_surfels[id * surfel_size + surfel_instanceColor_offset] = instanceTable_color[BestInstance];
			}
			else
			{
				map_surfels[id * surfel_size + surfel_instanceColor_offset] = defaultColor;
			}
		}

	}
}

__host__ 
void countAndColourSurfelMap(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset,
									const int surfel_instanceColor_offset, float* instanceTable_color, int* bestIDInEachSurfel)
{
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	
	countAndColourSurfelMapKernel<<<dimGrid,dimBlock>>>(n, map_surfels, surfel_size, surfel_instance_offset, surfel_instanceColor_offset, instanceTable_color,bestIDInEachSurfel);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

//CudaTask 5_1
__global__ 
void getVertexFromMapKernel(const int n, float* map_surfels,const int surfel_size,float* data)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		data[id*4+0] = map_surfels[id * surfel_size + 0];
		data[id*4+1] = map_surfels[id * surfel_size + 1];
		data[id*4+2] = map_surfels[id * surfel_size + 2];
		data[id*4+3] = 0;
	}
}
__host__ 
void getVertexFromMap(const int n, float* map_surfels,const int surfel_size,float* data)
{
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	
	getVertexFromMapKernel<<<dimGrid,dimBlock>>>(n, map_surfels, surfel_size, data);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

//CudaTask 5_2
__global__ 
void mapKnnVoteColourKernel(const int n,float* map_surfels,const int surfel_size,const int surfel_instance_offset,const int surfel_instanceColor_offset,
					int* indicesResults,float* distsResults,const int knn, int* bestIDInEachSurfel,float* tempInstanceMap, float* instanceTable_color)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	const int instanceNum_half = surfel_size - surfel_instance_offset;
	const int instanceNum      = instanceNum_half*2;
	
/*//debug
	if(id==20000)
	{
		for(int i=0;i<instanceNum;i++) tempInstanceMap[id*instanceNum+i] = 0;
		tempInstanceMap[id*instanceNum+0] = 6000;
		for(int i=0;i<knn;i++)
		{
			int knnID = indicesResults[id*knn+i];
			tempInstanceMap[knnID*instanceNum+1] = 6000;
		}
	}
	return;
//debug*/
	if (id < n) {
		for(int i=0;i<knn;i++)
		{
			int knnID = indicesResults[id*knn+i];
			if (knnID>=n||knnID<0) continue;
			
			//PLAN C simple InstanceVote
			//int maxNumI=-1;
			//int maxIDI =-1;
			/*
			for(int i=0; i< instanceNum_half; i++)
			{
				float mapInfoI = map_surfels[knnID * surfel_size + surfel_instance_offset + i];
				//decode
				int instanceI_1 = decode1_Instance(mapInfoI);
				int instanceI_2 = decode2_Instance(mapInfoI);
				//PLAN_C1 count
				tempInstanceMap[id*instanceNum + 2*i    ] += instanceI_1;
				tempInstanceMap[id*instanceNum + 2*i + 1] += instanceI_2;
				//PLAN_C2 maxNum
				//if(maxNumI<instanceI_1)
				//{
				//	maxNumI = instanceI_1;
				//	maxIDI = 2*i;
				//}
				//if(maxNumI<instanceI_2)
				//{
				//	maxNumI = instanceI_2;
				//	maxIDI = 2*i+1;
				//}
			}*/
			//if(maxNumI>0) tempInstanceMap[id*instanceNum+maxIDI] ++; //PLAN_C2
			//PLAN_C3
			if(bestIDInEachSurfel[knnID]>=0)	tempInstanceMap[id*instanceNum+bestIDInEachSurfel[knnID]]++;

			/*
			//PLAN B colorVote
			float i_color = map_surfels[knnID*surfel_size+surfel_instanceColor_offset];
			int i_finalID=-1;
			for(int j=0; j< instanceNum; j++)
			{
				if(i_color == instanceTable_color[j])
				{
					i_finalID = j;
					break;
				}
			}
			if(i_finalID>0)	tempInstanceMap[id*instanceNum + i_finalID]++;
			*/

			/*
			//PLAN A GaussianInstanceVote
			const float pi = 3.1415926f;
			const float sigma = 0.05;
			float dx = distsResults[id*knn+i];
			float temp = -(dx*dx)/(2*sigma*sigma);
			float fx = expf(temp)/(sqrt(2*pi)*sigma);
			*/
	
			/*
			for(int j=0; j< instanceNum_half; j++)
			{
				float mapInfo = map_surfels[knnID * surfel_size + surfel_instance_offset + j];
				//decode
				int instanceI_1 = decode1_Instance(mapInfo);
				int instanceI_2 = decode2_Instance(mapInfo);
				tempInstanceMap[id*instanceNum+2*j] += fx*instanceI_1;
				tempInstanceMap[id*instanceNum+2*j+1] += fx*instanceI_2;
			}
			*/
		}
		int maxNum=-1;
		int maxID =-1;
		for(int i=0;i<instanceNum;i++)
		{
			if(tempInstanceMap[id*instanceNum+i]>maxNum)
			{
				maxID = i;
				maxNum = tempInstanceMap[id*instanceNum+i];
			}
		}
		if(maxNum>0)
		{
			map_surfels[id*surfel_size+surfel_instanceColor_offset] = instanceTable_color[maxID];
		}
	}
}

__host__ 
void mapKnnVoteColour(const int n,float* map_surfels,const int surfel_size,const int surfel_instance_offset,const int surfel_instanceColor_offset,
					int* indicesResults,float* distsResults,const int knn, int* bestIDInEachSurfel,float* tempInstanceMap, float* instanceTable_color)
{
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	
	mapKnnVoteColourKernel<<<dimGrid,dimBlock>>>(n, map_surfels, surfel_size,surfel_instance_offset, surfel_instanceColor_offset, 
												indicesResults, distsResults, knn, bestIDInEachSurfel, tempInstanceMap, instanceTable_color);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());

}

//============ loopClosure ================================================


__global__ 
void loopClosureCopyMatchInstanceKernel(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, int* matchPair,const int matchNum)
{
	const int surfel_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (surfel_id < n) {
		for(int i=0; i< matchNum; i++)
		{
			int matchFromID = matchPair[2*i+0];
			int  matchToID  = matchPair[2*i+1];


			int instanceFromI = matchFromID/2;
			int instanceFromP = matchFromID%2;
			int  instanceToI  =  matchToID/2 ;
			int  instanceToP  =  matchToID%2 ;


			//decode
			float mapInfoFrom = map_surfels[surfel_id * surfel_size + surfel_instance_offset + instanceFromI];
			int instanceFrom;
			if(instanceFromP==0) 	  instanceFrom = decode1_Instance(mapInfoFrom);
			else if(instanceFromP==1) instanceFrom = decode2_Instance(mapInfoFrom);
			else break;

			
			//decode
			float mapInfoTo = map_surfels[surfel_id * surfel_size + surfel_instance_offset + instanceToI];
			int instanceTo1 = decode1_Instance(mapInfoTo);
			int instanceTo2 = decode2_Instance(mapInfoTo);
			
			if(instanceToP==0) instanceTo1 += instanceFrom;		//++
			if(instanceToP==1) instanceTo2 += instanceFrom;


			if(instanceTo1>=65535)instanceTo1=65535;
			if(instanceTo2>=65535)instanceTo2=65535;

			//encode
			mapInfoTo = encode_Instance(instanceTo1,instanceTo2);
			//update
			map_surfels[surfel_id * surfel_size + surfel_instance_offset + instanceToI] = mapInfoTo;

		}
	}
}

__host__ 
void loopClosureCopyMatchInstance(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, int* matchPair,const int matchNum)
{
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	
	loopClosureCopyMatchInstanceKernel<<<dimGrid,dimBlock>>>(n, map_surfels, surfel_size, surfel_instance_offset, matchPair,matchNum);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}



//=========================================================================
__global__ 
void renderProjectFrameKernel(cudaTextureObject_t index_surfelsIds, const int width, const int height,const int n, float* map_surfels,
						const int surfel_size,const int surfel_instance_offset,const int surfel_instanceColor_offset, float* projectColor, float* instanceTable_color)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	int surfel_id = tex2D<int>(index_surfelsIds,x,y);
	//int BestInstance = -1;
	//int BestCount = 0;
	//int instanceNum_half = surfel_size - surfel_instance_offset;
	if(surfel_id > 0 && surfel_id<n)
	{
		projectColor[y*width*4+x*4+0] = float(int(map_surfels[surfel_id*surfel_size+surfel_instanceColor_offset]) >> 16 & 0xFF) / 255.0f;
		projectColor[y*width*4+x*4+1] = float(int(map_surfels[surfel_id*surfel_size+surfel_instanceColor_offset]) >> 8 & 0xFF) / 255.0f;
		projectColor[y*width*4+x*4+2] = float(int(map_surfels[surfel_id*surfel_size+surfel_instanceColor_offset]) & 0xFF) / 255.0f;
		projectColor[y*width*4+x*4+3] = 1.0f;

		//debug
		//projectColor[y*width*4+x*4+0] = float(int(map_surfels[surfel_id*surfel_size+4]) >> 16 & 0xFF) / 255.0f;
		//projectColor[y*width*4+x*4+1] = float(int(map_surfels[surfel_id*surfel_size+4]) >> 8 & 0xFF) / 255.0f;
		//projectColor[y*width*4+x*4+2] = float(int(map_surfels[surfel_id*surfel_size+4]) & 0xFF) / 255.0f;
		
		
		/*		
		for(int i=0; i< instanceNum_half; i++)
		{
			float mapInfo = map_surfels[surfel_id * surfel_size + surfel_instance_offset + i];
			//decode
			int instanceI_1 = decode1_Instance(mapInfo);
			int instanceI_2 = decode2_Instance(mapInfo);
			if(BestCount < instanceI_1)
			{
				BestCount = instanceI_1;
				BestInstance = 2 * i;
			}
			if(BestCount < instanceI_2)
			{
				BestCount = instanceI_2;
				BestInstance = 2 * i + 1;
			}
		}
		if(BestInstance == -1)
		{
			projectColor[y*width*4+x*4+0] = 1.0f;
			projectColor[y*width*4+x*4+1] = 1.0f;
			projectColor[y*width*4+x*4+2] = 1.0f;
			projectColor[y*width*4+x*4+3] = 1.0f;
		}
		else
		{
			//decode
			projectColor[y*width*4+x*4+0] = float(int(instanceTable_color[BestInstance]) >> 16 & 0xFF) / 255.0f;
			projectColor[y*width*4+x*4+1] = float(int(instanceTable_color[BestInstance]) >> 8 & 0xFF) / 255.0f;
			projectColor[y*width*4+x*4+2] = float(int(instanceTable_color[BestInstance]) & 0xFF) / 255.0f;
			projectColor[y*width*4+x*4+3] = 1.0f;
		}*/
	}
	else
	{
		projectColor[y*width*4+x*4+0] = 0.0f;
		projectColor[y*width*4+x*4+1] = 0.0f;
		projectColor[y*width*4+x*4+2] = 0.0f;
		projectColor[y*width*4+x*4+3] = 1.0f;
	}
	
}

__host__ 
void renderProjectFrame(cudaTextureObject_t index_surfelsIds, const int width, const int height,const int n, float* map_surfels,
						const int surfel_size,const int surfel_instance_offset,const int surfel_instanceColor_offset, float* projectColor, float* instanceTable_color)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);
	
	renderProjectFrameKernel<<<dimGrid,dimBlock>>>(index_surfelsIds, width, height, n, map_surfels, 
													surfel_size, surfel_instance_offset,surfel_instanceColor_offset, projectColor, instanceTable_color);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

__global__ 
void renderMaskFrameKernel(unsigned char* masks, const int masksNum, const int width, const int height, float* maskColor)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	int flag = 0;
	for(int maskID = 0; maskID<masksNum; maskID++)
	{
		if(masks[ (maskID*width*height) + (x+y*width) ]) flag++;
	}
	if(flag*30>255)
	{
		maskColor[y*width*4+x*4+0] = 1.0f;
		maskColor[y*width*4+x*4+1] = 1.0f;
		maskColor[y*width*4+x*4+2] = 1.0f;
		maskColor[y*width*4+x*4+3] = 1.0f;
	}
	else
	{
		maskColor[y*width*4+x*4+0] = flag*30/255.0f;
		maskColor[y*width*4+x*4+1] = flag*30/255.0f;
		maskColor[y*width*4+x*4+2] = flag*30/255.0f;
		maskColor[y*width*4+x*4+3] = 1.0f;
	}
}

__host__ 
void renderMaskFrame(unsigned char* masks, const int masksNum, const int width, const int height, float* maskColor)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);
	
	renderMaskFrameKernel<<<dimGrid,dimBlock>>>(masks,masksNum, width, height, maskColor);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}


//============== 3D BBOX ==================================================


__global__ 
void testAllSurfelNormalVoteKernel(const int n, float* map_surfels, const int surfel_size,const int surfel_instance_offset, const int surfel_normal_offset, 
								const int surfel_instanceColor_offset, float* instanceTable_color,const int semCircleSegNum, int* mapGroundNormalVote, int* mapInstanceNormalVote)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	const int instanceNum_half = surfel_size - surfel_instance_offset;
	const int instanceNum      = instanceNum_half*2;
	if(id<n)
	{
			int instance = -1;
			for(int i=0;i<instanceNum;i++)
			{
				if(map_surfels[id*surfel_size+surfel_instanceColor_offset] == instanceTable_color[i])
				{
					instance = i;
					break;
				}
			}

			//* surfels normal in (Map) is inconsistent with (World)
			float norX = map_surfels[id * surfel_size + surfel_normal_offset + 0];
			float norY = -map_surfels[id * surfel_size + surfel_normal_offset + 1];
			float norZ = -map_surfels[id * surfel_size + surfel_normal_offset + 2];
			float normalLen = sqrt(norX*norX+norY*norY+norZ*norZ);

			norX = norX/normalLen;
			norY = norY/normalLen;
			norZ = norZ/normalLen;

			const float pi = 3.1415926;
			//vertex
			float y_v, d_v, x_v, z_v;
			float theta_v, alpha_v;
			//middle
			float y_m, d_m, x_m, z_m;
			float theta_m, alpha_m;

			int p = 0;
			//longitude 36 * latitude 18, sphere
			for (int i = -semCircleSegNum/2; i < semCircleSegNum/2; i++)	//latitude 18
			{
				//vertex of each part(surface)
				//middle of each part(surface)
				theta_v = i * pi / semCircleSegNum;
				d_v = cos(theta_v);
				y_v = sin(theta_v);
		
				theta_m = (i+0.5f) * pi / semCircleSegNum;
				d_m = cos(theta_m);
				y_m = sin(theta_m);

				for (int j = 0; j < 2*semCircleSegNum; j++)					//longitude 36
				{
					alpha_v = j * pi / semCircleSegNum;
					x_v = d_v * cos(alpha_v);
					z_v = d_v * sin(alpha_v);

					alpha_m = (j+0.5f) * pi / semCircleSegNum;
					x_m = d_m * cos(alpha_m);
					z_m = d_m * sin(alpha_m);

					float dx = x_v - x_m;
					float dy = y_v - y_m;
					float dz = z_v - z_m;
					float R = sqrt(dx*dx+dy*dy+dz*dz);
	
					float dxMap = norX - x_m;
					float dyMap = norY - y_m;
					float dzMap = norZ - z_m;
					float norDist = sqrt(dxMap*dxMap+dyMap*dyMap+dzMap*dzMap);
					
					if(norDist<R)	
					{
						atomicAdd(mapGroundNormalVote+p,1);
						if(instance!=-1)	atomicAdd(mapInstanceNormalVote+(semCircleSegNum*semCircleSegNum*2)*instance+p,1);
					}
					
					p++;
				}
			}
		
	}

}
__host__ 
void testAllSurfelNormalVote(const int n, float* map_surfels, const int surfel_size,const int surfel_instance_offset, const int surfel_normal_offset, 
								const int surfel_instanceColor_offset, float* instanceTable_color,const int semCircleSegNum, int* mapGroundNormalVote, int* mapInstanceNormalVote)
{
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	
	testAllSurfelNormalVoteKernel<<<dimGrid,dimBlock>>>(n, map_surfels,surfel_size,surfel_instance_offset, surfel_normal_offset,
													surfel_instanceColor_offset, instanceTable_color,semCircleSegNum, mapGroundNormalVote,mapInstanceNormalVote);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}


__inline__ __device__
void matrixSetCoordinate(float* baseX,float* baseY,float* baseZ,float* shift,float* matrix)
{
	if(shift==0)
	{
		matrix[0]  = baseX[0]; matrix[1]  = baseY[0]; matrix[2]  = baseZ[0]; matrix[3]  = 0;
		matrix[4]  = baseX[1]; matrix[5]  = baseY[1]; matrix[6]  = baseZ[1]; matrix[7]  = 0;
		matrix[8]  = baseX[2]; matrix[9]  = baseY[2]; matrix[10] = baseZ[2]; matrix[11] = 0;
		matrix[12] =    0    ; matrix[13] =    0    ; matrix[14] =    0    ; matrix[15] = 1;
	}
	else
	{
		matrix[0]  = baseX[0]; matrix[1]  = baseY[0]; matrix[2]  = baseZ[0]; matrix[3]  = 0;
		matrix[4]  = baseX[1]; matrix[5]  = baseY[1]; matrix[6]  = baseZ[1]; matrix[7]  = 0;
		matrix[8]  = baseX[2]; matrix[9]  = baseY[2]; matrix[10] = baseZ[2]; matrix[11] = 0;
		matrix[12] = shift[0]; matrix[13] = shift[1]; matrix[14] = shift[2]; matrix[15] = 1;
	}
}


__global__
void setGroundandInstanceCoordinateKernel(const int instanceNum, const int semCircleSegNum,float* groundNormal,
												int* mapInstanceNormalCrossVote/*TEMP*/,int* mapInstanceNormalVote,float* gcMatrix,float* instcMatrix)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	const int voteBufferNum = semCircleSegNum*semCircleSegNum*2;
	const float pi =3.1415926;	
	
	if(id<instanceNum+1)
	{
		float baseX[3];
		float baseY[3];
		float baseZ[3];
		
		baseY[0] = groundNormal[0];	
		baseY[1] = groundNormal[1];	
		baseY[2] = groundNormal[2];
		vectorNormalize(baseY);
		if(id==instanceNum)//Ground Coordinate Matrix
		{
			float setZ[3];

			setZ[0] = 0;
			setZ[1] = 0;
			setZ[2] = 1;
			
			vectorCrossProduct(setZ,baseY,baseX);
			vectorNormalize(baseX);
	
			vectorCrossProduct(baseY,baseX,baseZ);
			vectorNormalize(baseZ);

			matrixSetCoordinate(baseX,baseY,baseZ,0,gcMatrix);
		}
		else //Instance Coordinate Matrix
		{
			int* normalVoteBuffer = mapInstanceNormalVote+id*voteBufferNum;
			int* crossVoteBuffer = mapInstanceNormalCrossVote+id*semCircleSegNum*2;

			//FILL CROSSVOTEBUFFER
			float angleStep = pi/semCircleSegNum;	// (2PI/36)
							
			float oriZ[3],oriX[3];
			oriZ[0] = 0;
			oriZ[1] = 0;
			oriZ[2] = -1;

			vectorCrossProduct(baseY,oriZ,oriX);
			vectorNormalize(oriX);
	
			for(int i=0;i<voteBufferNum;i++)
			{
				if(normalVoteBuffer[i]==0) continue;

				//Step 1 get normal form VoteBuffer
				int a = i/(2*semCircleSegNum) - semCircleSegNum/2;		//latitude  ---
				int b = i%(2*semCircleSegNum) - 1;						//longitude |||
				
				float theta = (a+0.5f) * pi / semCircleSegNum;
				float alpha = (b+0.5f) * pi / semCircleSegNum;
				float d = std::cos(theta);
				float y = std::sin(theta);
				float x = d * std::cos(alpha);
				float z = d * std::sin(alpha);
				
				//Step 2 get TEST vector and Original vector
				float tempZ[3];//,tempX[3];
				tempZ[0] = x;
				tempZ[1] = y;
				tempZ[2] = z;
				vectorNormalize(tempZ);

					//ignore vector(tempZ) close to (baseY) or (-baseY).			cos30=0.866  cos45=0.525
				float cos_TZ_BY  = baseY[0]*tempZ[0]+baseY[1]*tempZ[1]+baseY[2]*tempZ[2];
				if(cos_TZ_BY>0.525||-cos_TZ_BY>0.525) continue;
				
				//vectorCrossProduct(baseY,tempZ,tempX);
				//vectorNormalize(tempX);

				//Step 3 classify to crossVoteBuffer (saveTempZ)
				float minDist   = 999999.9f;
				int minDistID = -1;
				for(int j=0;j<semCircleSegNum*2;j++)
				{
					float angle = j*angleStep;
					
					//Rodrigues Rotation		theta = angle, v=oriX, k=baseY
					float resultVec[3];
					rodriguesRotation( angle, oriX, baseY, resultVec);
					vectorNormalize(resultVec);
				
					//find min dist (between resultVec and tempZ)	//tempX
					float dist = vectorDist(resultVec,tempZ);
					if(dist<minDist)
					{
						minDistID = j;
						minDist = dist;
					}
				}
				crossVoteBuffer[minDistID] += normalVoteBuffer[i];
			}
			
			//GET instcMatrix			
			int voteMaxNum = 0;
			int voteMaxID=0;	//-1
			for(int i=0;i<semCircleSegNum*2;i++)
			{
				if(crossVoteBuffer[i]>voteMaxNum)
				{
					voteMaxNum = crossVoteBuffer[i];
					voteMaxID = i;
				}
			}
			
			float angle = voteMaxID*angleStep;
			
			//tempZ					
			float resultVec[3];
			rodriguesRotation( angle, oriX, baseY, resultVec);
			vectorNormalize(resultVec);

			vectorCrossProduct(resultVec,baseY,baseX);
			vectorNormalize(baseX);
	
			vectorCrossProduct(baseY,baseX,baseZ);
			vectorNormalize(baseZ);

			//tempX
			//vectorCrossProduct(baseY,resultVec,baseZ);
			//vectorNormalize(baseZ);
				
			//vectorCrossProduct(baseZ,baseY,baseX);
			//vectorNormalize(baseX);
				
			matrixSetCoordinate(baseX,baseY,baseZ,0,instcMatrix+16*id);
				
			
		}
	}	
}
__host__
void setGroundandInstanceCoordinate(const int instanceNum, const int semCircleSegNum,float* groundNormal,
														int* mapInstanceNormalCrossVote,int* mapInstanceNormalVote,float* gcMatrix,float* instcMatrix)
{
	const int threads = 8;
	const int blocks = ((instanceNum+1) + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	
	setGroundandInstanceCoordinateKernel<<<dimGrid,dimBlock>>>(instanceNum, semCircleSegNum, groundNormal,
																		mapInstanceNormalCrossVote,mapInstanceNormalVote, gcMatrix, instcMatrix);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}


__global__ 
void testAllSurfelFindBBoxKernel(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, 
			const int surfel_instanceColor_offset, float* instanceTable_color, const float ratio3DBBox,float* gcInverseM, float* instInverseM,bool bboxType, int *map3DBBox)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	const int instanceNum_half = surfel_size - surfel_instance_offset;
	const int instanceNum      = instanceNum_half*2;
	if(id<n)
	{

		float worldX = map_surfels[id * surfel_size + 0];
		float worldY = map_surfels[id * surfel_size + 1];
		float worldZ = map_surfels[id * surfel_size + 2];

		int instance = -1;
		for(int i=0;i<instanceNum;i++)
		{
			if(map_surfels[id*surfel_size+surfel_instanceColor_offset] ==instanceTable_color[i])
			{
				instance = i;
				break;
			}
		}
		if(instance!=-1)
		{
			float groundX,groundY,groundZ;
			
			if(bboxType)
			{
				//Ground
				groundX = gcInverseM[0]*worldX + gcInverseM[1]*worldY+ gcInverseM[ 2]*worldZ + gcInverseM[ 3]*1.0f;
				groundY = gcInverseM[4]*worldX + gcInverseM[5]*worldY+ gcInverseM[ 6]*worldZ + gcInverseM[ 7]*1.0f;
				groundZ = gcInverseM[8]*worldX + gcInverseM[9]*worldY+ gcInverseM[10]*worldZ + gcInverseM[11]*1.0f;
			}
			else
			{
				//Instance
				groundX = instInverseM[16*instance+0]*worldX + instInverseM[16*instance+1]*worldY+ instInverseM[16*instance+2]*worldZ + instInverseM[16*instance+3]*1.0f;
				groundY = instInverseM[16*instance+4]*worldX + instInverseM[16*instance+5]*worldY+ instInverseM[16*instance+6]*worldZ + instInverseM[16*instance+7]*1.0f;
				groundZ = instInverseM[16*instance+8]*worldX + instInverseM[16*instance+9]*worldY+ instInverseM[16*instance+10]*worldZ + instInverseM[16*instance+11]*1.0f;
			}

			int verX = groundX*ratio3DBBox;
			int verY = groundY*ratio3DBBox;
			int verZ = groundZ*ratio3DBBox;

			int minX = verX;
			int minY = verY;
			int minZ = verZ;
			int maxX = verX;
			int maxY = verY;
			int maxZ = verZ;

			//for int(atomicMin can't handle float)
			if(minX>map_surfels[id * surfel_size + 0]*ratio3DBBox) minX--;
			if(minY>map_surfels[id * surfel_size + 1]*ratio3DBBox) minY--;
			if(minZ>map_surfels[id * surfel_size + 2]*ratio3DBBox) minZ--;
			if(maxX<map_surfels[id * surfel_size + 0]*ratio3DBBox) maxX++;
			if(maxY<map_surfels[id * surfel_size + 1]*ratio3DBBox) maxY++;
			if(maxZ<map_surfels[id * surfel_size + 2]*ratio3DBBox) maxZ++;
			
			atomicMin(map3DBBox+6*instance+0,minX);
			atomicMin(map3DBBox+6*instance+2,minY);
			atomicMin(map3DBBox+6*instance+4,minZ);

			atomicMax(map3DBBox+6*instance+1,maxX);
			atomicMax(map3DBBox+6*instance+3,maxY);
			atomicMax(map3DBBox+6*instance+5,maxZ);
		}

	}
}
__host__ 
void testAllSurfelFindBBox(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, const int surfel_instanceColor_offset,
									 float* instanceTable_color, const float ratio3DBBox,float* gcInverseM, float* instInverseM,bool bboxType, int *map3DBBox)
{
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	
	testAllSurfelFindBBoxKernel<<<dimGrid,dimBlock>>>(n, map_surfels,surfel_size,surfel_instance_offset, surfel_instanceColor_offset, 
																			instanceTable_color,ratio3DBBox,gcInverseM,instInverseM,bboxType, map3DBBox);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}



__global__ 
void mapCountInstanceByInstColorKernel(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset,
								float* instanceTable_color, const int surfel_instanceColor_offset, int* eachInstanceSumCount)
{

	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	const int instanceNum_half = surfel_size - surfel_instance_offset;
	const int instanceNum      = instanceNum_half*2;
	if (id < n) {
		
		int instance = -1;
		for(int i=0;i<instanceNum;i++)
		{
			if(map_surfels[id*surfel_size+surfel_instanceColor_offset] ==instanceTable_color[i])
			{
				instance = i;
				break;
			}
		}
		if(instance!=-1) atomicAdd(eachInstanceSumCount+instance,1);
	}

}


__host__ 
void mapCountInstanceByInstColor(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset,
								float* instanceTable_color, const int surfel_instanceColor_offset, int* eachInstanceSumCount)
{
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	
	mapCountInstanceByInstColorKernel<<<dimGrid,dimBlock>>>(n, map_surfels, surfel_size, surfel_instance_offset,
															instanceTable_color, surfel_instanceColor_offset, eachInstanceSumCount);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}



__global__ 
void getSurfelToInstanceBufferKernel(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, float* instanceTable_color,
					const int surfel_instanceColor_offset, const int surfel_normal_offset, const int surfel_rgbColor_offset,
					bool bboxType, int *map3DBBox,float* gcInverseM, float* instInverseM, int* instSurfelCountTemp, float** instSurfels)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	const int instanceNum_half = surfel_size - surfel_instance_offset;
	const int instanceNum      = instanceNum_half*2;
	if(id<n)
	{
		int instance = -1;
		for(int i=0;i<instanceNum;i++)
		{
			if(map_surfels[id*surfel_size+surfel_instanceColor_offset] ==instanceTable_color[i])
			{
				instance = i;
				break;
			}
		}
		if(instance!=-1)
		{
			//eachSurfel: 0 :id		123:xyz(instance Coordinate)  456:normal       789:rgb(color) 		//10-12:rgb(partSeg)
			//Info 1 xyz
			float worldX = map_surfels[id * surfel_size + 0];
			float worldY = map_surfels[id * surfel_size + 1];
			float worldZ = map_surfels[id * surfel_size + 2];

			float instX,instY,instZ;
			
			if(bboxType)
			{
				//Ground
				instX = gcInverseM[0]*worldX + gcInverseM[1]*worldY+ gcInverseM[ 2]*worldZ + gcInverseM[ 3]*1.0f;
				instY = gcInverseM[4]*worldX + gcInverseM[5]*worldY+ gcInverseM[ 6]*worldZ + gcInverseM[ 7]*1.0f;
				instZ = gcInverseM[8]*worldX + gcInverseM[9]*worldY+ gcInverseM[10]*worldZ + gcInverseM[11]*1.0f;
			}
			else
			{
				//Instance
				instX = instInverseM[16*instance+0]*worldX + instInverseM[16*instance+1]*worldY+ instInverseM[16*instance+2]*worldZ + instInverseM[16*instance+3]*1.0f;
				instY = instInverseM[16*instance+4]*worldX + instInverseM[16*instance+5]*worldY+ instInverseM[16*instance+6]*worldZ + instInverseM[16*instance+7]*1.0f;
				instZ = instInverseM[16*instance+8]*worldX + instInverseM[16*instance+9]*worldY+ instInverseM[16*instance+10]*worldZ + instInverseM[16*instance+11]*1.0f;
			}

			//Info 2 normal
			float norX = map_surfels[id * surfel_size + surfel_normal_offset + 0];
			float norY = -map_surfels[id * surfel_size + surfel_normal_offset + 1];
			float norZ = -map_surfels[id * surfel_size + surfel_normal_offset + 2];
			float normalLen = sqrt(norX*norX+norY*norY+norZ*norZ);

			norX = norX/normalLen;
			norY = norY/normalLen;
			norZ = norZ/normalLen;
			
			float instNorX,instNorY,instNorZ;
			if(bboxType)
			{
				//Ground
				instNorX = gcInverseM[0]*norX + gcInverseM[1]*norY+ gcInverseM[ 2]*norZ + gcInverseM[ 3]*1.0f;
				instNorY = gcInverseM[4]*norX + gcInverseM[5]*norY+ gcInverseM[ 6]*norZ + gcInverseM[ 7]*1.0f;
				instNorZ = gcInverseM[8]*norX + gcInverseM[9]*norY+ gcInverseM[10]*norZ + gcInverseM[11]*1.0f;
			}
			else
			{
				//Instance
				instNorX = instInverseM[16*instance+0]*norX + instInverseM[16*instance+1]*norY+ instInverseM[16*instance+2]*norZ + instInverseM[16*instance+3]*1.0f;
				instNorY = instInverseM[16*instance+4]*norX + instInverseM[16*instance+5]*norY+ instInverseM[16*instance+6]*norZ + instInverseM[16*instance+7]*1.0f;
				instNorZ = instInverseM[16*instance+8]*norX + instInverseM[16*instance+9]*norY+ instInverseM[16*instance+10]*norZ + instInverseM[16*instance+11]*1.0f;
			}

			//Info 3 color
			float r = float(int(map_surfels[id*surfel_size+surfel_rgbColor_offset]) >> 16 & 0xFF) / 255.0f;
			float g = float(int(map_surfels[id*surfel_size+surfel_rgbColor_offset]) >> 8 & 0xFF) / 255.0f;
			float b = float(int(map_surfels[id*surfel_size+surfel_rgbColor_offset]) & 0xFF) / 255.0f;
			
			//Info 4 partSeg
			//float partR = float(int(map_surfels[id*surfel_size+surfel_partColor_offset]) >> 16 & 0xFF) / 255.0f;
			//float partG = float(int(map_surfels[id*surfel_size+surfel_partColor_offset]) >> 8 & 0xFF) / 255.0f;
			//float partB = float(int(map_surfels[id*surfel_size+surfel_partColor_offset]) & 0xFF) / 255.0f;

			//Fill Buffer
			int idInBuffer = atomicAdd(instSurfelCountTemp+instance,1);
			
			instSurfels[instance][2+13*idInBuffer+0] = id;
			
			instSurfels[instance][2+13*idInBuffer+1]  = instX;
			instSurfels[instance][2+13*idInBuffer+2]  = instY;
			instSurfels[instance][2+13*idInBuffer+3]  = instZ;

			instSurfels[instance][2+13*idInBuffer+4]  = instNorX;
			instSurfels[instance][2+13*idInBuffer+5]  = instNorY;
			instSurfels[instance][2+13*idInBuffer+6]  = instNorZ;

			instSurfels[instance][2+13*idInBuffer+7]  = r;
			instSurfels[instance][2+13*idInBuffer+8]  = g;
			instSurfels[instance][2+13*idInBuffer+9]  = b;

			//instSurfels[instance][2+13*idInBuffer+10] = partR;
			//instSurfels[instance][2+13*idInBuffer+11] = partG;
			//instSurfels[instance][2+13*idInBuffer+12] = partB;
			
		}
	}
}

__host__ 
void getSurfelToInstanceBuffer(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset, float* instanceTable_color,
					const int surfel_instanceColor_offset, const int surfel_normal_offset, const int surfel_rgbColor_offset,
					bool bboxType, int *map3DBBox,float* gcInverseM, float* instInverseM, int* instSurfelCountTemp, float** instSurfels)
{
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	
	getSurfelToInstanceBufferKernel<<<dimGrid,dimBlock>>>(n, map_surfels,surfel_size,surfel_instance_offset, instanceTable_color,
												surfel_instanceColor_offset, surfel_normal_offset, surfel_rgbColor_offset,
												bboxType, map3DBBox, gcInverseM, instInverseM, instSurfelCountTemp, instSurfels);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

//============== evaluate =================================================
__global__ 
void computePrecisionAndRecallKernel(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset,const int surfel_instanceColor_offset,
										const int surfel_instanceGT_offset, float* instanceTable_color,	int* instPointNum, int* gtPointNum, int* inst_gt_Map)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	const int instanceNum_half = surfel_size - surfel_instance_offset;
	const int instanceNum      = instanceNum_half*2;
	if(id<n)
	{
		//instPointNum
		int instanceID = -1;
		for(int i=0;i<instanceNum;i++)
		{
			if(map_surfels[id*surfel_size+surfel_instanceColor_offset] ==instanceTable_color[i])
			{
				instanceID = i;
				break;
			}
		}
		if(instanceID>=0) atomicAdd(instPointNum+instanceID,1);

		//gtPointNum
		int gtID = map_surfels[id*surfel_size+surfel_instanceGT_offset];
		if(gtID>=0) atomicAdd(gtPointNum+gtID,1);

		//inst_gt_Map
		if(instanceID>=0&&gtID>=0) atomicAdd(inst_gt_Map+gtID*instanceNum+instanceID,1);
	}
}

__host__ 
void computePrecisionAndRecall(const int n, float* map_surfels,const int surfel_size,const int surfel_instance_offset,const int surfel_instanceColor_offset,
									const int surfel_instanceGT_offset, float* instanceTable_color,	int* instPointNum, int* gtPointNum, int* inst_gt_Map)
{
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	
	computePrecisionAndRecallKernel<<<dimGrid,dimBlock>>>(n, map_surfels, surfel_size, surfel_instance_offset, surfel_instanceColor_offset,
												            surfel_instanceGT_offset, instanceTable_color, instPointNum, gtPointNum, inst_gt_Map);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}



//==============CPU_DEBUG==================================================
__global__ 
void cpuTestCopyMapIdsKernel(cudaTextureObject_t index_surfelsIds,const int width,const int height, int*  map_ids)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	int surfel_id = tex2D<int>(index_surfelsIds,x,y);
	map_ids[width*y+x] = surfel_id;
}

__host__ 
void cpuTestCopyMapIds(cudaTextureObject_t index_surfelsIds,const int width,const int height, int*  map_ids)
{
	
	dim3 dimGrid(32,32);
	dim3 dimBlock(width/32,height/32);
	cpuTestCopyMapIdsKernel<<<dimGrid,dimBlock>>>(index_surfelsIds,width,height,map_ids);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());		
}
