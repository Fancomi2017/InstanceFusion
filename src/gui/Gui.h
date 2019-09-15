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

#ifndef GUI_H_
#define GUI_H_
#include <iostream>
#include <memory>

#include <cuda.h>
#include <Core/InstanceFusion.h>

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/gldraw.h>

#include <map_interface/ElasticFusionInterface.h>
#include <utilities/Types.h>

struct ClassIdInput;

class Gui {
public:
	//enum SelectProbabilityMap {Books,Chairs,Floor};
	Gui(bool live_capture,std::vector<ClassColour> class_colour_lookup, const int segmentation_width, const int segmentation_height);
	virtual ~Gui();

	void renderMapID(const std::unique_ptr<ElasticFusionInterface>& map,std::vector<ClassColour> class_colour_lookup);
	void preCall();
	void renderMapMethod1(const std::unique_ptr<ElasticFusionInterface>& map);
	void renderMapMethod2(const std::unique_ptr<ElasticFusionInterface>& map,const std::unique_ptr<InstanceFusion>& instanceFusion,bool drawBox,bool bboxType);
	void postCall();
	void displayProjectColor(const std::string & id, float* segmentation_rendering_gpu_);
	void displayRawNetworkPredictions(const std::string & id, float* device_ptr);
	void displayImg(const std::string & id, GPUTexture * img);

	bool reset() const { return pangolin::Pushed(*reset_.get()); }
//lyc add
	bool save() const { return pangolin::Pushed(*save_.get()); }
	void setFps(int f) { fps->Ref().Set(f);}
	void setFrameCount(std::string fC) { frameCount->Ref().Set(fC); }
//lyc add over
	void setInstanceSaveFalse() { instanceSaveOnce_->Ref().Set(false); }

	bool paused() const { return *pause_.get(); }
	bool step() const { return pangolin::Pushed(*step_.get()); }
	bool tracking() const { return *tracking_.get(); }
	bool instance_seg() const { return *instance_.get(); }

	bool project_bbox() const { return *projectBoudingBox_.get(); }
	bool surfel_colorus() const { return *color_view_.get(); }
	bool instance_colours() const { return *instance_view_.get(); }
	bool time_colours() const { return *time_view_.get(); }
	bool display_mode() const { return *displayMode_.get(); }
	bool bbox_type() const { return *bboxType_.get(); }
	bool raw_save() const { return *rawSave_.get(); }
	bool instance_save_once() const { return *instanceSaveOnce_.get(); }


private:
	const int mainWidth  = 3200;
	const int mainHeight = 2400;
	
	int width_;
	int height_;
	const int segmentation_width_;
	const int segmentation_height_;
	int panel_;
	std::vector<ClassColour> class_colour_lookup_;
	float* class_colour_lookup_gpu_;
	float* segmentation_rendering_gpu_;

	std::unique_ptr<pangolin::Var<bool>> reset_;
//lyc add
	std::unique_ptr<pangolin::Var<bool>> save_;
	std::unique_ptr<pangolin::Var<std::string>> frameCount;
	std::unique_ptr<pangolin::Var<int>> fps;
//lyc add over
	std::unique_ptr<pangolin::Var<bool>> pause_;
	std::unique_ptr<pangolin::Var<bool>> step_;
	std::unique_ptr<pangolin::Var<bool>> tracking_;
	std::unique_ptr<pangolin::Var<bool>> instance_;

	std::unique_ptr<pangolin::Var<bool>> projectBoudingBox_;
	std::unique_ptr<pangolin::Var<bool>> color_view_;
	std::unique_ptr<pangolin::Var<bool>> instance_view_;
	std::unique_ptr<pangolin::Var<bool>> time_view_;
	std::unique_ptr<pangolin::Var<bool>> displayMode_;
	std::unique_ptr<pangolin::Var<bool>> bboxType_;
	std::unique_ptr<pangolin::Var<bool>> rawSave_;
	std::unique_ptr<pangolin::Var<bool>> instanceSaveOnce_;
	std::unique_ptr<pangolin::Var<ClassIdInput>> class_choice_;
	std::unique_ptr<pangolin::GlTextureCudaArray> mainDisplay_texture_array_;
	std::unique_ptr<pangolin::GlTextureCudaArray> boundingBox_texture_array_;
	std::unique_ptr<pangolin::GlTextureCudaArray> probability_texture_array_;
	std::unique_ptr<pangolin::GlTextureCudaArray> rendered_segmentation_texture_array_;

	//test
	//std::unique_ptr<pangolin::Var<float>> fxBBOX;
	//std::unique_ptr<pangolin::Var<float>> fyBBOX;
	//std::unique_ptr<pangolin::Var<float>> cxBBOX;
	//std::unique_ptr<pangolin::Var<float>> cyBBOX;
	
	std::unique_ptr<pangolin::Var<std::string>> measureBBOX[12];

	pangolin::GlRenderBuffer* render_buffer_;
	pangolin::GlFramebuffer* color_frame_buffer_;
	GPUTexture* color_texture_;

	//mapId
	pangolin::GlFramebuffer* mapIdGlobalFrameBuffer;
	pangolin::GlRenderBuffer* mapIdGlobalRenderBuffer;
	GPUTexture* mapIdGlobalTexture;

	//temp for renderMapMethod2
	float* camPoseTrans_gpu;	//input
	float* rgbdInfo_gpu;	//OUTPUT
	float* mapDisplay_gpu;	
	float* tempMap;

	pangolin::OpenGlRenderState s_cam_;
};


#endif /* GUI_H_ */
