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

#include "ElasticFusionInterface.h"
#include "ElasticFusionCuda.h" 
#include <utilities/Types.h>
#include <random>

#include <GL/glx.h>

float encode_colour(ClassColour rgb_colour) {
  int colour = rgb_colour.r;
  colour = (colour << 8) + rgb_colour.g;
  colour = (colour << 8) + rgb_colour.b;
  return float(colour);
}

bool ElasticFusionInterface::Init(std::vector<ClassColour> class_colour_lookup) {
  GLXContext context = glXGetCurrentContext();
  if (context == nullptr) {
    return false;
  } 
  // These are the regular elastic fusion parameters. The depthCut has been
  // increase to 8m. Also to prevent the white border in the NYU the ICP weight 	//12
  // can be turned to 100, as in the commented line below. However, the code
  // in elasticfusion has also been modified to ignore a 16 pixel border of the
  // RGB residual, which allows some RGB tracking while also ignoring the border.
  elastic_fusion_.reset(new ElasticFusion(200, 35000, 5e-05, 1e-05, true,
                                          false,false,115,10,12,10,false,0.3095,true,false,"./ResultModel"));	//8->12
                                          //false,false,115,10,8,100));	
  const int surfel_render_size = Resolution::getInstance().width() * 
                                  Resolution::getInstance().height();
  surfel_ids_.resize(surfel_render_size);
  initialised_ = true;
  class_color_lookup_.clear();
  for (unsigned int class_id = 0; class_id < class_colour_lookup.size(); ++class_id) {
    class_color_lookup_.push_back(encode_colour(class_colour_lookup[class_id]));
  }
  cudaMalloc((void **)&class_color_lookup_gpu_, class_color_lookup_.size() * sizeof(float));
  cudaMemcpy(class_color_lookup_gpu_, class_color_lookup_.data(), class_color_lookup_.size() * sizeof(float), cudaMemcpyHostToDevice);
  return true;
}

ElasticFusionInterface::~ElasticFusionInterface() {
  cudaFree(class_color_lookup_gpu_);
}

bool ElasticFusionInterface::ProcessFrame(const ImagePtr rgb, const DepthPtr depth, const int64_t timestamp, 
						int* smallInstanceTable, const unsigned char * instanceGT) {
  if (elastic_fusion_) {
    elastic_fusion_->processFrame(rgb,depth,timestamp, smallInstanceTable, instanceGT);
    if (elastic_fusion_->getLost()) {
      return false;
    }
    return true;
  }
  return false;
}

//semanticFusion code? useless
void ElasticFusionInterface::UpdateSurfelClass(const int surfel_id, const int class_id) {
  elastic_fusion_->getGlobalModel().updateSurfelClass(surfel_id,class_color_lookup_[class_id]);
}

void ElasticFusionInterface::UpdateSurfelClassGpu(const int n, const float* surfelclasses, const float* surfelprobs, const float threshold) {
  if (elastic_fusion_) {
    float* map_surfels = elastic_fusion_->getGlobalModel().getMapSurfelsGpu();
    updateSurfelClasses(n, map_surfels, surfelclasses, surfelprobs, class_color_lookup_gpu_, threshold);
  }
}

void ElasticFusionInterface::RenderMapToBoundGlBuffer(const pangolin::OpenGlRenderState& camera,const bool drawInstances,const bool drawTimes, const bool drawColors) {
  elastic_fusion_->getGlobalModel().renderPointCloud(camera.GetProjectionModelViewMatrix(),
                                                           elastic_fusion_->getConfidenceThreshold(),
                                                           false,
                                                           false,
                                                           drawColors,
                                                           false,
                                                           false,
                                                           drawTimes,
                                                           drawInstances,
                                                           elastic_fusion_->getTick(),
                                                           elastic_fusion_->getTimeDelta());
}

void ElasticFusionInterface::RenderMapIDToGUIBuffer(const pangolin::OpenGlRenderState& camera,const bool drawInstances,const bool drawTimes, const bool drawColors) {
  elastic_fusion_->getGlobalModel().renderGlobalID(camera.GetProjectionModelViewMatrix(),
                                                           elastic_fusion_->getConfidenceThreshold(),
                                                           false,
                                                           false,
                                                           drawColors,
                                                           false,
                                                           false,
                                                           drawTimes,
                                                           drawInstances,
                                                           elastic_fusion_->getTick(),
                                                           elastic_fusion_->getTimeDelta());
}



