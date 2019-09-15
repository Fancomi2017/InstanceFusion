/*
 * This file is part of SemanticFusion.
 *
 * Copyright (C) 2017 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is SemanticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/semantic-fusion/semantic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
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

__global__ 
void colouredArgMaxKernel(int n, float const* probabilities,  const int num_classes, float const* color_lookup, float* colour, float const* map_max, const int map_size,cudaTextureObject_t ids, const float threshold,const int width,const int height)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		const int y = id / width;
		const int x = id - (y * width);
		const int start_windowx = (x - 1) > 0 ? (x - 1) : 0;
		const int start_windowy = (y - 1) > 0 ? (y - 1) : 0;
		const int end_windowx = (x + 1) < width ? (x + 1) : (width-1);
		const int end_windowy = (y + 1) < height ? (y + 1) : (height-1);

		int max_class_id = -1;
		float max_class_prob = threshold;
		for (int sx = start_windowx; sx <= end_windowx; ++sx) {
			for (int sy = start_windowy; sy <= end_windowy; ++sy) {
				const int surfel_id = tex2D<int>(ids,sx,sy);
				if (surfel_id > 0) {
					float const* id_probabilities = map_max + surfel_id;
					if (id_probabilities[map_size] > max_class_prob) {
						max_class_id = static_cast<int>(id_probabilities[0]);
						max_class_prob = id_probabilities[map_size];
					}
				}
			}
		}

		float* local_colour = colour + (id * 4);
		if (max_class_id >= 0) {
			local_colour[0] = color_lookup[max_class_id * 3 + 0];
			local_colour[1] = color_lookup[max_class_id * 3 + 1];
			local_colour[2] = color_lookup[max_class_id * 3 + 2];
			local_colour[3] = 1.0f;
		} else {
			local_colour[0] = 0.0;
			local_colour[1] = 0.0;
			local_colour[2] = 0.0;
			local_colour[3] = 1.0f;
		}
	}
}

__host__
void colouredArgMax(int n, float const * probabilities,  const int num_classes, float const* color_lookup, float* colour, float const * map, const int map_size,cudaTextureObject_t ids, const float threshold,const int width,const int height)
{
	const int threads = 512;
	const int blocks = (n + threads - 1) / threads;
	dim3 dimGrid(blocks);
	dim3 dimBlock(threads);
	colouredArgMaxKernel<<<dimGrid,dimBlock>>>(n,probabilities,num_classes,color_lookup,colour,map,map_size,ids,threshold,width,height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}






__global__
void getRGBDformMapIDsKernel(cudaTextureObject_t surfelsIds,const int n,float* map_surfels,const int surfel_size,
								const int width,const int height,float* camPoseTrans,float* rgbdInfo)
{
	const int x0 = blockIdx.x * blockDim.x + threadIdx.x;
	const int y0 = blockIdx.y * blockDim.y + threadIdx.y;
	for(int q=0;q<25;q++)
	{
		int dx = q%5;
		int dy = q/5;

		int x=x0*5+dx;
		int y=y0*5+dy;

		int surfel_id = tex2D<int>(surfelsIds,x,y);
		if(surfel_id > 0&&surfel_id < n)
		{
			float mx = map_surfels[surfel_id * surfel_size + 0];
			float my = map_surfels[surfel_id * surfel_size + 1];
			float mz = map_surfels[surfel_id * surfel_size + 2];

			float dx = camPoseTrans[0]-mx;
			float dy = camPoseTrans[1]-my;
			float dz = camPoseTrans[2]-mz;
		
			float dist = sqrt(dx*dx+dy*dy+dz*dz);
		
			float r = float(int(map_surfels[surfel_id*surfel_size+4]) >> 16 & 0xFF) / 255.0f;
			float g = float(int(map_surfels[surfel_id*surfel_size+4]) >> 8 & 0xFF) / 255.0f;
			float b = float(int(map_surfels[surfel_id*surfel_size+4]) & 0xFF) / 255.0f;
		
			rgbdInfo[y*width*4+x*4+0] = r;
			rgbdInfo[y*width*4+x*4+1] = g;
			rgbdInfo[y*width*4+x*4+2] = b;
			rgbdInfo[y*width*4+x*4+3] = dist;
		
		}
		else
		{
			rgbdInfo[y*width*4+x*4+0] = 1.0f;
			rgbdInfo[y*width*4+x*4+1] = 1.0f;
			rgbdInfo[y*width*4+x*4+2] = 1.0f;
			rgbdInfo[y*width*4+x*4+3] = 99999.9f;
		}
	}
}

__host__
void getRGBDformMapIDs(cudaTextureObject_t surfelsIds,const int n,float* map_surfels, const int surfel_size,
						const int width,const int height,float* camPoseTrans,float* rgbdInfo)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(640/blocks,480/blocks);
	
	if(640*5!=width)	printf("ERROR WIDTH IN [getRGBDformMapIDs]");
	
	getRGBDformMapIDsKernel<<<dimGrid,dimBlock>>>(surfelsIds, n, map_surfels, surfel_size, width, height,camPoseTrans, rgbdInfo);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

__global__
void renderInfoToDisplayKernel(float* rgbdInfo,const int width,const int height,float* mapDisplay)
{
	const int x0 = blockIdx.x * blockDim.x + threadIdx.x;
	const int y0 = blockIdx.y * blockDim.y + threadIdx.y;
	for(int q=0;q<25;q++)
	{
		int dx = q%5;
		int dy = q/5;

		int x1=x0*5+dx;
		int y1=y0*5+dy;

		int x2 = x1;
		int y2 = height - y1 - 1;
		
		if(x1<0||x1>=width||y1<0||y1>=height) continue;
		if(x2<0||x2>=width||y2<0||y2>=height) continue;

		//show rgb
		mapDisplay[y2*width*4+x2*4+0] =  rgbdInfo[y1*width*4+x1*4+0];
		mapDisplay[y2*width*4+x2*4+1] =  rgbdInfo[y1*width*4+x1*4+1];
		mapDisplay[y2*width*4+x2*4+2] =  rgbdInfo[y1*width*4+x1*4+2];
		mapDisplay[y2*width*4+x2*4+3] =  1.0f;

		//show depth
		//mapDisplay[y2*width*4+x2*4+0] =  rgbdInfo[y1*width*4+x1*4+3]/10;
		//mapDisplay[y2*width*4+x2*4+1] =  rgbdInfo[y1*width*4+x1*4+3]/10;
		//mapDisplay[y2*width*4+x2*4+2] =  rgbdInfo[y1*width*4+x1*4+3]/10;
		//mapDisplay[y2*width*4+x2*4+3] =  1.0f;
	}
}
__host__
void renderInfoToDisplay(float* rgbdInfo,const int width,const int height,float* mapDisplay)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(640/blocks,480/blocks);
	
	if(640*5!=width)	printf("ERROR WIDTH IN [renderInfoToDisplay]");
	
	renderInfoToDisplayKernel<<<dimGrid,dimBlock>>>(rgbdInfo, width, height, mapDisplay);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}


