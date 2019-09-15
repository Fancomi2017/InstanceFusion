/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#ifndef INDEXMAP_H_
#define INDEXMAP_H_

#include "Shaders/Shaders.h"
#include "Shaders/Uniform.h"
#include "Shaders/Vertex.h"
#include "GPUTexture.h"
#include "Utils/Resolution.h"
#include "Utils/Intrinsics.h"
#include <pangolin/gl/gl.h>
#include <Eigen/LU>

#include "Defines.h"

class IndexMap
{
    public:
        IndexMap();
        virtual ~IndexMap();

        enum Prediction
        {
            ACTIVE,
            INACTIVE
        };
	
	//===========Instance==================
	enum Occasion
        {
            INSTANCECOMPARE,
            GENERAL_BEFORE,
            GENERAL_AFTER,
        };
	//=====================================

        void predictIndices(const Eigen::Matrix4f & pose,
                            const int & time,
                            const std::pair<GLuint, GLuint> & model,
                            const float depthCutoff,
                            const int timeDelta);

        EFUSION_API void renderDepth(const float depthCutoff);

        EFUSION_API void renderSurfelIds(const Eigen::Matrix4f & pose,
                                         const int & time,
                                         const std::pair<GLuint, GLuint> & model,
                                         const float threshold,
                                         const float depthCutoff,
                                         const int timeDelta,
                                         IndexMap::Occasion occasionType);


        void combinedPredict(const Eigen::Matrix4f & pose,
                             const std::pair<GLuint, GLuint> & model,
                             const float depthCutoff,
                             const float confThreshold,
                             const int time,
                             const int maxTime,
                             const int timeDelta,
                             IndexMap::Prediction predictionType);

        void synthesizeInfo(const Eigen::Matrix4f & pose,
                            const std::pair<GLuint, GLuint> & model,
                            const float depthCutoff,
                            const float confThreshold);

        void synthesizeDepth(const Eigen::Matrix4f & pose,
                             const std::pair<GLuint, GLuint> & model,
                             const float depthCutoff,
                             const float confThreshold,
                             const int time,
                             const int maxTime,
                             const int timeDelta);

        GPUTexture * indexTex()
        {
            return &indexTexture;
        }

        GPUTexture * vertConfTex()
        {
            return &vertConfTexture;
        }

        GPUTexture * colorTimeTex()
        {
            return &colorTimeTexture;
        }

        GPUTexture * normalRadTex()
        {
            return &normalRadTexture;
        }

        GPUTexture * drawTex()
        {
            return &drawTexture;
        }

        GPUTexture * depthTex()
        {
            return &depthTexture;
        }

        GPUTexture * imageTex()
        {
            return &imageTexture;
        }

        GPUTexture * vertexTex()
        {
            return &vertexTexture;
        }

        GPUTexture * normalTex()
        {
            return &normalTexture;
        }

        GPUTexture * timeTex()
        {
            return &timeTexture;
        }

	GPUTexture * instTex()
	{
	    return &instanceTexture;
	}

//===============================================================


        GPUTexture * oldImageTex()
        {
            return &oldImageTexture;
        }

        GPUTexture * oldVertexTex()
        {
            return &oldVertexTexture;
        }

        GPUTexture * oldNormalTex()
        {
            return &oldNormalTexture;
        }

        GPUTexture * oldTimeTex()
        {
            return &oldTimeTexture;
        }

	GPUTexture * oldInstTex()
	{
	    return &oldInstanceTexture;
	}

//===============================================================


        GPUTexture * colorInfoTex()
        {
            return &colorInfoTexture;
        }

        GPUTexture * vertexInfoTex()
        {
            return &vertexInfoTexture;
        }

        GPUTexture * normalInfoTex()
        {
            return &normalInfoTexture;
        }

        GPUTexture * surfelIdTex_before()
        {
            return &surfelIdTexture_before;
        }

        GPUTexture * surfelIdTex_after()
        {
            return &surfelIdTexture_after;
        }

	GPUTexture * instanceSurfelIdTex()
        {
            return &instanceSurfelIdTexture;
        }

        static const int FACTOR;

    private:
        std::shared_ptr<Shader> indexProgram;
        pangolin::GlFramebuffer indexFrameBuffer;
        pangolin::GlRenderBuffer indexRenderBuffer;
        GPUTexture indexTexture;
        GPUTexture vertConfTexture;
        GPUTexture colorTimeTexture;
        GPUTexture normalRadTexture;

	//==========For Instance=======================================
        std::shared_ptr<Shader> surfelIdProgram;
	//GENERAL_AFTER
        pangolin::GlFramebuffer surfelIdFrameBuffer_after;
        pangolin::GlRenderBuffer surfelIdRenderBuffer_after;
        GPUTexture surfelIdTexture_after;
	//GENERAL_BEFORE
        pangolin::GlFramebuffer surfelIdFrameBuffer_before;
        pangolin::GlRenderBuffer surfelIdRenderBuffer_before;
        GPUTexture surfelIdTexture_before;
	//INSTANCECOMPARE
        std::shared_ptr<Shader> instanceSurfelIdProgram;
        pangolin::GlFramebuffer instanceSurfelIdFrameBuffer;
        pangolin::GlRenderBuffer instanceSurfelIdRenderBuffer;
        GPUTexture instanceSurfelIdTexture;
	//=============================================================

        std::shared_ptr<Shader> drawDepthProgram;
        pangolin::GlFramebuffer drawFrameBuffer;
        pangolin::GlRenderBuffer drawRenderBuffer;
        GPUTexture drawTexture;

        std::shared_ptr<Shader> depthProgram;
        pangolin::GlFramebuffer depthFrameBuffer;
        pangolin::GlRenderBuffer depthRenderBuffer;
        GPUTexture depthTexture;

        std::shared_ptr<Shader> combinedProgram;
        pangolin::GlFramebuffer combinedFrameBuffer;
        pangolin::GlRenderBuffer combinedRenderBuffer;
        GPUTexture imageTexture;
        GPUTexture vertexTexture;
        GPUTexture normalTexture;
        GPUTexture timeTexture;
        GPUTexture instanceTexture;	//instanceFusion+

        pangolin::GlFramebuffer oldFrameBuffer;
        pangolin::GlRenderBuffer oldRenderBuffer;
        GPUTexture oldImageTexture;
        GPUTexture oldVertexTexture;
        GPUTexture oldNormalTexture;
        GPUTexture oldTimeTexture;
        GPUTexture oldInstanceTexture;	//instanceFusion+

        pangolin::GlFramebuffer infoFrameBuffer;
        pangolin::GlRenderBuffer infoRenderBuffer;
        GPUTexture colorInfoTexture;
        GPUTexture vertexInfoTexture;
        GPUTexture normalInfoTexture;
};

#endif /* INDEXMAP_H_ */
