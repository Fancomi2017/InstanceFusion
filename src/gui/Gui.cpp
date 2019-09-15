/*
 * This file is part of SemanticFusion.
 *
 * Copyright (C) 2017 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is SemanticFusion is permitted for 
 * non-commercial purposes only.	The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/semantic-fusion/semantic-fusion-license/> 
 * unless explicitly stated.	By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include "Gui.h"
#include "GuiCuda.h"
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

struct ClassIdInput
{
	ClassIdInput()
		: class_id_(0) {}
	ClassIdInput(int class_id)
		: class_id_(class_id) {}
	int class_id_;
};

std::ostream& operator<< (std::ostream& os, const ClassIdInput& o){
	os << o.class_id_;
	return os;
}

std::istream& operator>> (std::istream& is, ClassIdInput& o){
	is >> o.class_id_;
	return is;
}


Gui::Gui(bool live_capture, std::vector<ClassColour> class_colour_lookup, const int segmentation_width, const int segmentation_height) 
	: width_(1280)
	, height_(980)
	, segmentation_width_(segmentation_width)
	, segmentation_height_(segmentation_height)
	, class_colour_lookup_(class_colour_lookup)
	, panel_(205)
{
	width_ += panel_;
	pangolin::Params window_params;
	window_params.Set("SAMPLE_BUFFERS", 0);
	window_params.Set("SAMPLES", 0);
	pangolin::CreateWindowAndBind("InstanceFusion", width_, height_, window_params);

	//Main Display
	

	//original display of global map(origin semanticfusion)
	//GlRenderBuffer
	render_buffer_ = new pangolin::GlRenderBuffer(mainWidth, mainHeight);
	//GPUTexture
	color_texture_ = new GPUTexture(mainWidth, mainHeight, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, true);
	//GlFramebuffer
	color_frame_buffer_ = new pangolin::GlFramebuffer;
	color_frame_buffer_->AttachColour(*color_texture_->texture);
	color_frame_buffer_->AttachDepth(*render_buffer_);

	//mapID use for display(InstanceFusion)
	//GlRenderBuffer 
	mapIdGlobalRenderBuffer= new pangolin::GlRenderBuffer(mainWidth, mainHeight);
	//GPUTexture
	mapIdGlobalTexture= new GPUTexture(mainWidth, mainHeight, GL_LUMINANCE32I_EXT, GL_LUMINANCE_INTEGER_EXT, GL_INT, false, true);
	//GlFramebuffer
	mapIdGlobalFrameBuffer = new pangolin::GlFramebuffer;
	mapIdGlobalFrameBuffer->AttachColour(*mapIdGlobalTexture->texture);
	mapIdGlobalFrameBuffer->AttachDepth(*mapIdGlobalRenderBuffer);
	//ClearBuffer
	mapIdGlobalFrameBuffer->Bind();
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0, 0, mainWidth, mainHeight);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	mapIdGlobalFrameBuffer->Unbind();

	//temp for renderMapMethod2
	cudaMalloc((void **)&camPoseTrans_gpu, 3 * sizeof(float));
	cudaMalloc((void **)&rgbdInfo_gpu,  4 * mainWidth * mainHeight * sizeof(float));
	cudaMalloc((void **)&mapDisplay_gpu,  4 * mainWidth * mainHeight * sizeof(float));
	tempMap = (float*)malloc(4 * mainWidth * mainHeight * sizeof(float));

	//ProjectionMatrix(int w, int h, GLprecision fu, GLprecision fv, GLprecision u0, GLprecision v0, GLprecision zNear, GLprecision zFar )
	s_cam_ = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
										pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY));
	pangolin::Display("cam").SetBounds(0, 1.0f, 0, 1.0f, -640 / 480.0)
				.SetHandler(new pangolin::Handler3D(s_cam_));
	// Small views along the bottom
	pangolin::Display("raw").SetAspect(640.0f/480.0f);
	//lyc add
	pangolin::Display("depth").SetAspect(640.0f/480.0f);
	//lyc add over
	pangolin::Display("pred").SetAspect(640.0f/480.0f);
	pangolin::Display("segmentation").SetAspect(640.0f/480.0f);
	pangolin::Display("multi").SetBounds(pangolin::Attach::Pix(0),1/4.0f,pangolin::Attach::Pix(180),1.0).SetLayout(pangolin::LayoutEqualHorizontal)
		.AddDisplay(pangolin::Display("pred"))
		.AddDisplay(pangolin::Display("segmentation"))
		.AddDisplay(pangolin::Display("raw"))
		.AddDisplay(pangolin::Display("depth"));

	// Vertical view along the side
	pangolin::Display("legend").SetAspect(640.0f/480.0f);
	pangolin::Display("vert").SetBounds(pangolin::Attach::Pix(0),1/4.0f,pangolin::Attach::Pix(180),1.0).SetLayout(pangolin::LayoutEqualVertical)
		.AddDisplay(pangolin::Display("legend"));

	// The control panel
	pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(panel_));
	pause_.reset(new pangolin::Var<bool>("ui.Pause", false, true));
	step_.reset(new pangolin::Var<bool>("ui.Step", false, false));
	reset_.reset(new pangolin::Var<bool>("ui.Reset", false, false));
	//lyc add
	save_.reset(new pangolin::Var<bool>("ui.Save",false,false));
	frameCount.reset(new pangolin::Var<std::string>("ui.FrameCount","0/0"));
	fps.reset(new pangolin::Var<int>("ui.FPS",0));
	//lyc add over
	tracking_.reset(new pangolin::Var<bool>("ui.Tracking Only", false, false));
	instance_.reset(new pangolin::Var<bool>("ui.Instance Segmentation", true, false));	//TRUE

	color_view_.reset(new pangolin::Var<bool>("ui.Draw Colors", true, false));
	instance_view_.reset(new pangolin::Var<bool>("ui.Draw Instances", false, false));
	time_view_.reset(new pangolin::Var<bool>("ui.Draw Time", false, false));
	projectBoudingBox_.reset(new pangolin::Var<bool>("ui.Draw ProjectMap BBox", false, false));

	displayMode_.reset(new pangolin::Var<bool>("ui.Display Mode", true, false));
	bboxType_.reset(new pangolin::Var<bool>("ui.3D BBOX Type", true, false));
	rawSave_.reset(new pangolin::Var<bool>("ui.Raw Save", false, false));
	instanceSaveOnce_.reset(new pangolin::Var<bool>("ui.Instance Save(Once)", false, false));

	class_choice_.reset(new pangolin::Var<ClassIdInput>("ui.Show class probs", ClassIdInput(0)));	//No Use


	//1600	1200
	//fxBBOX.reset(new pangolin::Var<float>("ui.fxBBOX", 1440.0, -mainWidth , mainWidth));
	//fyBBOX.reset(new pangolin::Var<float>("ui.fyBBOX", 1080.0, -mainHeight, mainHeight));
	//cxBBOX.reset(new pangolin::Var<float>("ui.cxBBOX", 0.0, -mainWidth , mainWidth));
	//cyBBOX.reset(new pangolin::Var<float>("ui.cyBBOX", 0.0, -mainHeight, mainHeight));


	//PHT Delete
	//probability_texture_array_.reset(new pangolin::GlTextureCudaArray(640,480,GL_LUMINANCE32F_ARB));
	probability_texture_array_.reset(new pangolin::GlTextureCudaArray(segmentation_width_,segmentation_height_,GL_RGBA32F));
	rendered_segmentation_texture_array_.reset(new pangolin::GlTextureCudaArray(segmentation_width_,segmentation_height_,GL_RGBA32F));

	//PHT+
	mainDisplay_texture_array_.reset(new pangolin::GlTextureCudaArray(render_buffer_->width,render_buffer_->height,GL_RGBA32F));
	boundingBox_texture_array_.reset(new pangolin::GlTextureCudaArray(render_buffer_->width,render_buffer_->height,GL_RGBA32F));

	// The gpu colour lookup
	std::vector<float> class_colour_lookup_rgb;
	for (unsigned int class_id = 0; class_id < class_colour_lookup_.size(); ++class_id) {
		class_colour_lookup_rgb.push_back(static_cast<float>(class_colour_lookup_[class_id].r)/255.0f);
		class_colour_lookup_rgb.push_back(static_cast<float>(class_colour_lookup_[class_id].g)/255.0f);
		class_colour_lookup_rgb.push_back(static_cast<float>(class_colour_lookup_[class_id].b)/255.0f);
	}
	cudaMalloc((void **)&class_colour_lookup_gpu_, class_colour_lookup_rgb.size() * sizeof(float));
	cudaMemcpy(class_colour_lookup_gpu_, class_colour_lookup_rgb.data(), class_colour_lookup_rgb.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&segmentation_rendering_gpu_,	4 * segmentation_width_ * segmentation_height_ * sizeof(float));

}

Gui::~Gui() 
{ 
	cudaFree(class_colour_lookup_gpu_);
	cudaFree(segmentation_rendering_gpu_);

	cudaFree(camPoseTrans_gpu);
	cudaFree(rgbdInfo_gpu);
	cudaFree(mapDisplay_gpu);
	free(tempMap);
}

void Gui::renderMapID(const std::unique_ptr<ElasticFusionInterface>& map,std::vector<ClassColour> class_colour_lookup) 
{
	class_colour_lookup_ = class_colour_lookup;

	mapIdGlobalFrameBuffer->Bind();
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0, 0, mainWidth, mainHeight);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	map->RenderMapIDToGUIBuffer(s_cam_,instance_colours(),time_colours(),surfel_colorus());

	mapIdGlobalFrameBuffer->Unbind();
	glPopAttrib();
    glPointSize(1);
    glFinish();
}

void Gui::preCall()
{
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LESS);
	glClearColor(1.0,1.0,1.0, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	width_ = pangolin::DisplayBase().v.w;
	height_ = pangolin::DisplayBase().v.h;
	pangolin::Display("cam").Activate(s_cam_);
}

void Gui::renderMapMethod1(const std::unique_ptr<ElasticFusionInterface>& map) 
{
	map->RenderMapToBoundGlBuffer(s_cam_,instance_colours(),time_colours(),surfel_colorus());	
}


//Debug Pack
timeval time_1,time_2;
void TimeTick()
{
	gettimeofday(&time_1,NULL);
}
void TimeTock(std::string name)
{
	gettimeofday(&time_2,NULL);
	float timeUse = (time_2.tv_sec-time_1.tv_sec)*1000000+(time_2.tv_usec-time_1.tv_usec);
	std::cout<<name<<": "<<timeUse<<" us"<<std::endl;
}


void Gui::renderMapMethod2(const std::unique_ptr<ElasticFusionInterface>& map,const std::unique_ptr<InstanceFusion>& instanceFusion,bool drawBox,bool bboxType) 
{
	//map->RenderMapToBoundGlBuffer(s_cam_,instance_colours(),time_colours(),surfel_colorus());
		
	//Step 1 get Info form MapID
	const int surfel_size = instanceFusion->getSurfelSize();
	cudaTextureObject_t mapGlobalsurfelsIds = mapIdGlobalTexture->texObj;
	float* map_surfels = map->getMapSurfelsGpu();
	int n = map->getMapSurfelCount();

	//use projectionmodelview matrix to compute BBOX position in screen
	pangolin::OpenGlMatrix mvp = s_cam_.GetProjectionModelViewMatrix();
	Eigen::Matrix<float,4,4> matMVP;
    for(int r=0; r<4; ++r ) {
        for(int c=0; c<4; ++c ) {
            matMVP(r,c) = (float)mvp.m[c*4+r];
        }
    }

	//std::cout<<"Projection:"<<std::endl;
	//std::cout<<s_cam_.GetProjectionMatrix()<<std::endl;
	//std::cout<<"ModelView:"<<std::endl;
	//std::cout<<s_cam_.GetModelViewMatrix()<<std::endl;
	//std::cout<<"ModelView Inverse:"<<std::endl;
	//std::cout<<s_cam_.GetModelViewMatrix().Inverse()<<std::endl;
	//std::cout<<"Matrix MVP"<<std::endl;
	//std::cout<<mvp<<std::endl;

	//use modelview inverse matrix to get camera position -> use for depth
	pangolin::OpenGlMatrix mvI = s_cam_.GetModelViewMatrix().Inverse();
	Eigen::Matrix<float,4,4> matMVI;
    for(int r=0; r<4; ++r ) {
        for(int c=0; c<4; ++c ) {
            matMVI(r,c) = (float)mvI.m[c*4+r];
        }
    }

	Eigen::Vector3f trans = matMVI.topRightCorner(3, 1);	//Eigen::Vector3f trans = map->getCurrPose().topRightCorner(3, 1);
	cudaMemcpy(camPoseTrans_gpu, &trans(0), 3 * sizeof(float), cudaMemcpyHostToDevice);	//input
	cudaMemset(rgbdInfo_gpu,0, 4 * mainWidth * mainHeight * sizeof(float));	//output

	//std::cout<<"trans: "<<trans(0)<<" "<<trans(1)<<" "<<trans(2)<<std::endl;
	getRGBDformMapIDs(mapGlobalsurfelsIds,n,map_surfels,surfel_size,mainWidth,mainHeight,camPoseTrans_gpu,rgbdInfo_gpu);

	//Step 2 draw 3DBBox in Info
	if(drawBox&&n>0)
	{
		float* map3DBBox =  instanceFusion->getMapBoundingBox();
		int instanceNum = instanceFusion->getInstanceNum();

		float* gcMatrix =  instanceFusion->getGcMatrix();
		float* instcMatrix =  instanceFusion->getInstcMatrix();
		
		cudaMemcpy(tempMap, rgbdInfo_gpu, 4 * mainWidth * mainHeight * sizeof(float), cudaMemcpyDeviceToHost);	//input & output
		
		int showBoxID=0;
		for(int i=0;i<instanceNum;i++)
		{
			if(map3DBBox==0)break;

			//std::cout<<"instanceID: "<<i<<std::endl;
			
			float minX = map3DBBox[i*6+0];	float maxX = map3DBBox[i*6+1];
			float minY = map3DBBox[i*6+2];	float maxY = map3DBBox[i*6+3];
			float minZ = map3DBBox[i*6+4];	float maxZ = map3DBBox[i*6+5];
			float v = (maxX-minX)*(maxY-minY)*(maxZ-minZ);
			if(minX>=maxX||minY>=maxY||minZ>=maxZ/*||v>0.01f*/)	continue;

			//for scene19 scene21
			//if(i!=1&&i!=11&&i<12)continue;
			
			//show measureBBOX(bad code)	
			//char measureInfoBuf[30];
			//std::sprintf(measureInfoBuf,"%4.2f/%4.2f/%4.2f",maxX-minX,maxY-minY,maxZ-minZ);

			//if(class_colour_lookup_[i].name.c_str()[0]=='p')measureBBOX[showBoxID].reset(new pangolin::Var<std::string>("ui.item",measureInfoBuf));
			//else if(class_colour_lookup_[i].name.c_str()[0]=='d')measureBBOX[showBoxID].reset(new pangolin::Var<std::string>("ui.table",measureInfoBuf));
			//else if(class_colour_lookup_[i].name.c_str()[0]=='t')continue;
			//else 
			//measureBBOX[showBoxID].reset(new pangolin::Var<std::string>("ui."+class_colour_lookup_[i].name+std::to_string(i),measureInfoBuf));
			
			//if(showBoxID<12)showBoxID++;
			

			//SKIP BY CROSS
			/*
			int flagContinue = 0;
			for(int j=0;j<i;j++)
			{
				int count=0;
				float minX2 = map3DBBox[j*6+0];	float maxX2 = map3DBBox[j*6+1];
				float minY2 = map3DBBox[j*6+2];	float maxY2 = map3DBBox[j*6+3];
				float minZ2 = map3DBBox[j*6+4];	float maxZ2 = map3DBBox[j*6+5];
				if(minX2>minX&&minX2<maxX)count++;
				if(minX2<minX&&minX<maxX2)count++;

				if(minY2>minY&&minY2<maxY)count++;
				if(minY2<minY&&minY<maxY2)count++;

				if(minZ2>minZ&&minZ2<maxZ)count++;
				if(minZ2<minZ&&minZ<maxZ2)count++;
				
				if(count>=3)flagContinue=1;
			}
			if(flagContinue) continue;
			*/
			
			//Debug
			//std::cout<<"instance: "<<i<<" minX:"<<minX<<" maxX:"<<maxX<<std::endl;
			//std::cout<<"instance: "<<i<<" minY:"<<minY<<" maxY:"<<maxY<<std::endl;
			//std::cout<<"instance: "<<i<<" minZ:"<<minZ<<" maxZ:"<<maxZ<<std::endl;
		
			//3d point
			float boxPoint3d_ground[8][3];
			int p=0;
			for(int a=0;a<=1;a++)
			{
				for(int b=0;b<=1;b++)
				{
					for(int c=0;c<=1;c++)
					{
						boxPoint3d_ground[p][0] = a==0?minX:maxX;
						boxPoint3d_ground[p][1] = b==0?minY:maxY;
						boxPoint3d_ground[p][2] = c==0?minZ:maxZ;

						//Debug
						//boxPoint3d_ground[p][0] = maxX;
						//boxPoint3d_ground[p][1] = maxY;
						//boxPoint3d_ground[p][2] = maxZ;
						p++;
					}
				}
			}

			//ground->world
			float boxPoint3d_world[8][3];
			for(int j=0;j<8;j++)
			{
				if(bboxType)	////Ground
				{
					boxPoint3d_world[j][0] = gcMatrix[0]*boxPoint3d_ground[j][0] + gcMatrix[1]*boxPoint3d_ground[j][1]+ gcMatrix[ 2]*boxPoint3d_ground[j][2] + gcMatrix[ 3]*1.0f;
					boxPoint3d_world[j][1] = gcMatrix[4]*boxPoint3d_ground[j][0] + gcMatrix[5]*boxPoint3d_ground[j][1]+ gcMatrix[ 6]*boxPoint3d_ground[j][2] + gcMatrix[ 7]*1.0f;
					boxPoint3d_world[j][2] = gcMatrix[8]*boxPoint3d_ground[j][0] + gcMatrix[9]*boxPoint3d_ground[j][1]+ gcMatrix[10]*boxPoint3d_ground[j][2] + gcMatrix[11]*1.0f;
				}
				else
				{
					float bpgX = boxPoint3d_ground[j][0];
					float bpgY = boxPoint3d_ground[j][1];
					float bpgZ = boxPoint3d_ground[j][2];

					boxPoint3d_world[j][0] = instcMatrix[i*16+0]*bpgX + instcMatrix[i*16+1]*bpgY+ instcMatrix[i*16+ 2]*bpgZ + instcMatrix[i*16+ 3]*1.0f;
					boxPoint3d_world[j][1] = instcMatrix[i*16+4]*bpgX + instcMatrix[i*16+5]*bpgY+ instcMatrix[i*16+ 6]*bpgZ + instcMatrix[i*16+ 7]*1.0f;
					boxPoint3d_world[j][2] = instcMatrix[i*16+8]*bpgX + instcMatrix[i*16+9]*bpgY+ instcMatrix[i*16+10]*bpgZ + instcMatrix[i*16+11]*1.0f;
				}
				//std::cout<<"x:"<<boxPoint3d_world[j][0]<<" y:"<<boxPoint3d_world[j][1]<<" z:"<<boxPoint3d_world[j][2]<<std::endl;
			}
			//std::cout<<std::endl;
			//std::cout<<std::endl;

			
			//2d point
			float boxPoint2d[8][3];
			for(int j=0;j<8;j++)
			{
				float vPosHome[3];
				vPosHome[0] = matMVP(0,0)*boxPoint3d_world[j][0] + matMVP(0,1)*boxPoint3d_world[j][1]+ matMVP(0,2)*boxPoint3d_world[j][2] + matMVP(0,3)*1.0f;
				vPosHome[1] = matMVP(1,0)*boxPoint3d_world[j][0] + matMVP(1,1)*boxPoint3d_world[j][1]+ matMVP(1,2)*boxPoint3d_world[j][2] + matMVP(1,3)*1.0f;
				vPosHome[2] = matMVP(2,0)*boxPoint3d_world[j][0] + matMVP(2,1)*boxPoint3d_world[j][1]+ matMVP(2,2)*boxPoint3d_world[j][2] + matMVP(2,3)*1.0f;
	
				//float fx = *fxBBOX.get();
				//float fy = *fyBBOX.get();
				//float cx = *cxBBOX.get();
				//float cy = *cyBBOX.get();
				
				float xloc = ((1440 * vPosHome[0]) / vPosHome[2]) + mainWidth *0.5;
				float yloc = ((1080 * vPosHome[1]) / vPosHome[2]) + mainHeight*0.5;
				
				float distX = trans(0)-boxPoint3d_world[j][0];
				float distY = trans(1)-boxPoint3d_world[j][1];
				float distZ = trans(2)-boxPoint3d_world[j][2];
				float zloc = std::sqrt(distX*distX+distY*distY+distZ*distZ);
				//float zloc = vPosHome[2] / 20.0f;
		
				boxPoint2d[j][0] = xloc;
				boxPoint2d[j][1] = yloc;
				boxPoint2d[j][2] = zloc;

				//std::cout<<"Screen_x:"<<xloc<<" Screen_y:"<<yloc<<std::endl;
			}
			
			
			//draw line
			for(int j=0;j<8;j++)
			{
				for(int k=j+1;k<8;k++)
				{
					int flag = 0;
					if(boxPoint2d[j][2]<=0||boxPoint2d[k][2]<=0)	continue;	//out of screen

					if(boxPoint3d_ground[j][0]==boxPoint3d_ground[k][0])flag++;
					if(boxPoint3d_ground[j][1]==boxPoint3d_ground[k][1])flag++;
					if(boxPoint3d_ground[j][2]==boxPoint3d_ground[k][2])flag++;
					if(flag==2)
					{
						float lineX = std::abs(boxPoint2d[j][0]-boxPoint2d[k][0]);
						float lineY = std::abs(boxPoint2d[j][1]-boxPoint2d[k][1]);
						float lineZ = std::abs(boxPoint2d[j][2]-boxPoint2d[k][2]);
						float stepX,stepY,stepZ;
						int needStep;
						if(lineX>lineY)
						{
							stepX = 1;
							stepY = (1.0f*lineY)/(1.0f*lineX);
							stepZ = (1.0f*lineZ)/(1.0f*lineX);
							needStep = lineX;
						}
						else if(lineX<lineY)
						{
							stepX = (1.0f*lineX)/(1.0f*lineY);
							stepY = 1;
							stepZ = (1.0f*lineZ)/(1.0f*lineY);
							needStep = lineY;
						}
						//j to k
						if(boxPoint2d[k][0]<boxPoint2d[j][0])	stepX = -stepX;
						if(boxPoint2d[k][1]<boxPoint2d[j][1])	stepY = -stepY;
						if(boxPoint2d[k][2]<boxPoint2d[j][2])	stepZ = -stepZ;
						
						const int lineW = 7;
						float nowX,nowY,nowZ;
						nowX = boxPoint2d[j][0];
						nowY = boxPoint2d[j][1];
						nowZ = boxPoint2d[j][2];
						for(int l=0;l<needStep;l++)
						{
							int Pixel_x = nowX;
							int Pixel_y = nowY;
							for(int m=0;m<lineW*lineW;m++)
							{
								int dy = (Pixel_y+m/lineW);
								int dx = (Pixel_x+m%lineW);
								if(dx>1&&dx<mainWidth-1&&dy>1&&dy<mainHeight-1)
								{
									if(nowZ<tempMap[dy*mainWidth*4+dx*4+3])
									{
										
										//if(j%2==0&&k%2==0)//(j<4&&k<4)
											tempMap[dy*mainWidth*4+dx*4+0] = 1.0f;
										//else
										//	tempMap[dy*mainWidth*4+dx*4+0] = 0.5f;

										tempMap[dy*mainWidth*4+dx*4+1] = std::min(1.0f, class_colour_lookup_[i].g*1.5f/255.0f);
										tempMap[dy*mainWidth*4+dx*4+2] = std::min(1.0f, class_colour_lookup_[i].b*1.5f/255.0f);
										tempMap[dy*mainWidth*4+dx*4+3] = nowZ;
									}
									//else
									//{
									//	tempMap[dy*mainWidth*4+dx*4+0] = tempMap[dy*mainWidth*4+dx*4+0]*0.95f+1.0f*0.05f;
									//	tempMap[dy*mainWidth*4+dx*4+1] = tempMap[dy*mainWidth*4+dx*4+1]*0.95f+std::min(1.0f, class_colour_lookup_[i].g*1.5f/255.0f)*0.05f;
									//	tempMap[dy*mainWidth*4+dx*4+2] = tempMap[dy*mainWidth*4+dx*4+2]*0.95f+std::min(1.0f, class_colour_lookup_[i].b*1.5f/255.0f)*0.05f;
									//}
								}
							}
							nowX+=stepX;
							nowY+=stepY;
							nowZ+=stepZ;
						}
					}
				}
			}
		
		}
		
		cudaMemcpy(rgbdInfo_gpu, tempMap, 4 * mainWidth * mainHeight * sizeof(float), cudaMemcpyHostToDevice);	
		
	}

	//Step 3 render Info To Display
	renderInfoToDisplay(rgbdInfo_gpu,mainWidth,mainHeight,mapDisplay_gpu);

	//RenderToViewport
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaGetLastError());
	pangolin::CudaScopedMappedArray arr_tex(*mainDisplay_texture_array_.get());
	cudaMemcpyToArray(*arr_tex, 0, 0, (void*)mapDisplay_gpu, sizeof(float) * 4 * mainWidth * mainHeight, cudaMemcpyDeviceToDevice);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaGetLastError());
	glDisable(GL_DEPTH_TEST);
	mainDisplay_texture_array_->RenderToViewport(true);
	glEnable(GL_DEPTH_TEST);
}

void Gui::postCall() 
{
	pangolin::FinishFrame();
	glFinish();
}

void Gui::displayProjectColor(const std::string & id, float* segmentation_rendering_gpu_) 
{
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaGetLastError());
	pangolin::CudaScopedMappedArray arr_tex(*rendered_segmentation_texture_array_.get());
	cudaMemcpyToArray(*arr_tex, 0, 0, (void*)segmentation_rendering_gpu_, sizeof(float) * 4 * segmentation_width_ * segmentation_height_, cudaMemcpyDeviceToDevice);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaGetLastError());
	glDisable(GL_DEPTH_TEST);
	pangolin::Display(id).Activate();
	rendered_segmentation_texture_array_->RenderToViewport(true);
	glEnable(GL_DEPTH_TEST);
}

void Gui::displayRawNetworkPredictions(const std::string & id, float* device_ptr) 
{
	pangolin::CudaScopedMappedArray arr_tex(*probability_texture_array_.get());
	gpuErrChk(cudaGetLastError());
	cudaMemcpyToArray(*arr_tex, 0, 0, (void*)device_ptr, sizeof(float) * 4 * segmentation_width_ * segmentation_height_, cudaMemcpyDeviceToDevice);
	gpuErrChk(cudaGetLastError());
	glDisable(GL_DEPTH_TEST);
	pangolin::Display(id).Activate();
	probability_texture_array_->RenderToViewport(true);
	glEnable(GL_DEPTH_TEST);
}

void Gui::displayImg(const std::string & id, GPUTexture * img) {
	glDisable(GL_DEPTH_TEST);
	pangolin::Display(id).Activate();
	img->texture->RenderToViewport(true);
	glEnable(GL_DEPTH_TEST);
}
