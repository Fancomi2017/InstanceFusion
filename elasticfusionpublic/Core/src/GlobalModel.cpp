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

#include "GlobalModel.h"
#include <GL/glext.h>

const int GlobalModel::TEXTURE_DIMENSION = 1536;		//3072
const int GlobalModel::MAX_VERTICES = GlobalModel::TEXTURE_DIMENSION * GlobalModel::TEXTURE_DIMENSION;
const int GlobalModel::NODE_TEXTURE_DIMENSION = 16384;
const int GlobalModel::MAX_NODES = GlobalModel::NODE_TEXTURE_DIMENSION / 16; //16 floats per node

GlobalModel::GlobalModel()
 : target(0),
   renderSource(1),
   bufferSize(MAX_VERTICES * Vertex::SIZE),
   count(0),
   deleted_count(0),
   initProgram(loadProgramFromFile("init_unstable.vert")),
   drawProgram(loadProgramFromFile("draw_feedback.vert", "draw_feedback.frag")),
   drawSurfelProgram(loadProgramFromFile("draw_global_surface.vert", "draw_global_surface.frag", "draw_global_surface.geom")),
   drawGlobalIDProgram(loadProgramFromFile("draw_global_ID.vert", "draw_global_ID.frag", "draw_global_ID.geom")),
   dataProgram(loadProgramFromFile("data.vert", "data.frag", "data.geom")),
   updateProgram(loadProgramFromFile("update.vert")),
   unstableProgram(loadProgramGeomFromFile("copy_unstable.vert", "copy_unstable.geom")),
   renderBuffer(TEXTURE_DIMENSION, TEXTURE_DIMENSION),
   updateMapVertsConfs(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
   updateMapColorsTime(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
   updateMapNormsRadii(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
   deformationNodes(NODE_TEXTURE_DIMENSION, 1, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT)
{
	vbos = new std::pair<GLuint, GLuint>[2];

	float * vertices = new float[bufferSize];

	memset(&vertices[0], 0, bufferSize);

	//std::cout<<"Vertex::SIZE"<<std::endl;		//3->16
	//std::cout<<Vertex::SIZE<<std::endl;

	glGenTransformFeedbacks(1, &vbos[0].second);
	glGenBuffers(1, &vbos[0].first);
	glBindBuffer(GL_ARRAY_BUFFER, vbos[0].first);
	glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// This allows recolouring of the map surfels
	size_t num_bytes;
	cudaGraphicsGLRegisterBuffer(&mapCudaRes, vbos[0].first, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsMapResources(1, &mapCudaRes, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&cuda_map_ptr, &num_bytes, mapCudaRes);
	cudaGraphicsUnmapResources(1, &mapCudaRes, 0);
	// ==========================================

	glGenTransformFeedbacks(1, &vbos[1].second);
	glGenBuffers(1, &vbos[1].first);
	glBindBuffer(GL_ARRAY_BUFFER, vbos[1].first);
	glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	delete [] vertices;

	// This is to allow us to keep track of deleted surfels
	int * indices = new int[MAX_VERTICES];
	glGenTransformFeedbacks(1, &deleted_surfel_buffer.second);
	glGenBuffers(1, &deleted_surfel_buffer.first);
	glBindBuffer(GL_ARRAY_BUFFER, deleted_surfel_buffer.first);
	glBufferData(GL_ARRAY_BUFFER, MAX_VERTICES * sizeof(int), &indices[0], GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&deletedSurfelCudaRes, deleted_surfel_buffer.first, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsMapResources(1, &deletedSurfelCudaRes, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&cuda_deleted_surfel_ptr, &num_bytes, deletedSurfelCudaRes);
	cudaGraphicsUnmapResources(1, &deletedSurfelCudaRes, 0);
	delete [] indices;
	// =====================================================

	vertices = new float[Resolution::getInstance().numPixels() * Vertex::SIZE];

	memset(&vertices[0], 0, Resolution::getInstance().numPixels() * Vertex::SIZE);

	glGenTransformFeedbacks(1, &newUnstableFid);
	glGenBuffers(1, &newUnstableVbo);
	glBindBuffer(GL_ARRAY_BUFFER, newUnstableVbo);
	glBufferData(GL_ARRAY_BUFFER, Resolution::getInstance().numPixels() * Vertex::SIZE, &vertices[0], GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	delete [] vertices;

	std::vector<Eigen::Vector2f> uv;

	for(int i = 0; i < Resolution::getInstance().width(); i++)
	{
		for(int j = 0; j < Resolution::getInstance().height(); j++)
		{
			uv.push_back(Eigen::Vector2f(((float)i / (float)Resolution::getInstance().width()) + 1.0 / (2 * (float)Resolution::getInstance().width()),
								   ((float)j / (float)Resolution::getInstance().height()) + 1.0 / (2 * (float)Resolution::getInstance().height())));
		}
	}

	uvSize = uv.size();

	glGenBuffers(1, &uvo);
	glBindBuffer(GL_ARRAY_BUFFER, uvo);
	glBufferData(GL_ARRAY_BUFFER, uvSize * sizeof(Eigen::Vector2f), &uv[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	frameBuffer.AttachColour(*updateMapVertsConfs.texture);
	frameBuffer.AttachColour(*updateMapColorsTime.texture);
	frameBuffer.AttachColour(*updateMapNormsRadii.texture);
	frameBuffer.AttachDepth(renderBuffer);

	// pht change ====================================================================================
	// Setup transform feedbacks
	static const char* varying_names16[16] =
	{
	  "vPosition0",
	  "vColor0",
	  "vNormRad0",
	  "vImgCorr0",
	  "vInstInfoA0",
	  "vInstInfoB0",
	  "vInstInfoC0",
	  "vInstInfoD0",
	  "vInstInfoE0",
	  "vInstInfoF0",
	  "vInstInfoG0",
	  "vInstInfoH0",
	  "vInstInfoI0",
	  "vInstInfoJ0",
	  "vInstInfoK0",
	  "vInstInfoL0",
	};
	glTransformFeedbackVaryings(updateProgram->programId(), 16, varying_names16, GL_INTERLEAVED_ATTRIBS);
	updateProgram->Link();
	glTransformFeedbackVaryings(dataProgram->programId(), 16, varying_names16, GL_INTERLEAVED_ATTRIBS);
	dataProgram->Link();
	glTransformFeedbackVaryings(initProgram->programId(), 16, varying_names16, GL_INTERLEAVED_ATTRIBS);
	initProgram->Link();

	// Setup dual buffer for vertex id out
	static const char* varying_names18[18] =
	{
	  "vPosition0",
	  "vColor0",
	  "vNormRad0",
	  "vImgCorr0",
	  "vInstInfoA0",
	  "vInstInfoB0",
	  "vInstInfoC0",
	  "vInstInfoD0",
	  "vInstInfoE0",
	  "vInstInfoF0",
	  "vInstInfoG0",
	  "vInstInfoH0",
	  "vInstInfoI0",
	  "vInstInfoJ0",
	  "vInstInfoK0",
	  "vInstInfoL0",
	  "gl_NextBuffer",
	  "deleted_id",
	};
	glTransformFeedbackVaryings(unstableProgram->programId(),18,varying_names18,GL_INTERLEAVED_ATTRIBS);
	unstableProgram->Link();
	// pht change ====================================================================================

	glGenQueries(1, &countQuery);
	glGenQueries(1, &deleteQuery);

	//Empty both transform feedbacks
	initProgram->Bind();
	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[0].second);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[0].first);

	//============================WORK===================================
	glBeginTransformFeedback(GL_POINTS);

	glDrawArrays(GL_POINTS, 0, 0);

	glEndTransformFeedback();
	//============================WORK===================================

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[1].second);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[1].first);

	//============================WORK===================================
	glBeginTransformFeedback(GL_POINTS);

	glDrawArrays(GL_POINTS, 0, 0);

	glEndTransformFeedback();
	//============================WORK===================================

	// Also clear out the deleted surfel buffer
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, deleted_surfel_buffer.second);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, deleted_surfel_buffer.first);

	glBeginTransformFeedback(GL_POINTS);

	glDrawArrays(GL_POINTS, 0, 0);

	glEndTransformFeedback();
	//END Transform feedback

	glDisable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	initProgram->Unbind();
}

GlobalModel::~GlobalModel()
{
	glDeleteBuffers(1, &vbos[0].first);
	glDeleteTransformFeedbacks(1, &vbos[0].second);

	glDeleteBuffers(1, &deleted_surfel_buffer.first);
	glDeleteTransformFeedbacks(1, &deleted_surfel_buffer.second);

	glDeleteBuffers(1, &vbos[1].first);
	glDeleteTransformFeedbacks(1, &vbos[1].second);

	glDeleteQueries(1, &countQuery);
	glDeleteQueries(1, &deleteQuery);

	glDeleteBuffers(1, &uvo);

	glDeleteTransformFeedbacks(1, &newUnstableFid);
	glDeleteBuffers(1, &newUnstableVbo);

	delete [] vbos;

	cudaGraphicsUnregisterResource(mapCudaRes);
	cudaGraphicsUnregisterResource(deletedSurfelCudaRes);
}

void GlobalModel::initialise(const FeedbackBuffer & rawFeedback,
							 const FeedbackBuffer & filteredFeedback)
{
	initProgram->Bind();

	glBindBuffer(GL_ARRAY_BUFFER, rawFeedback.vbo);
	//============================================== show ==========================================================
	//testFuction("1 rawFeedback.vbo");
	//============================================== show ==========================================================

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

	glBindBuffer(GL_ARRAY_BUFFER, filteredFeedback.vbo);

	//============================================== show ==========================================================
	//testFuction("2 filteredFeedback.vbo");
	//============================================== show ==========================================================

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	//glEnableVertexAttribArray(3);
	//glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 3));

	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[target].second);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[target].first);

	//============================WORK===================================
	glBeginTransformFeedback(GL_POINTS);

	glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

	//It's ok to use either fid because both raw and filtered have the same amount of vertices
	glDrawTransformFeedback(GL_POINTS, rawFeedback.fid);		//DRAW

	glEndTransformFeedback();

	glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
	//============================WORK===================================

	glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);

	glDisable(GL_RASTERIZER_DISCARD);

	//============================================== show ==========================================================
	//glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);
	//testFuction("3 vbos[target].first");
	//============================================== show ==========================================================


	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	//glDisableVertexAttribArray(3);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	initProgram->Unbind();

	glFinish();
}

void GlobalModel::renderPointCloud(pangolin::OpenGlMatrix mvp,
								   const float threshold,
								   const bool drawUnstable,
								   const bool drawNormals,
								   const bool drawColors,
								   const bool drawPoints,
								   const bool drawWindow,
								   const bool drawTimes,
								   const bool drawClasses,
								   const int time,
								   const int timeDelta)
{
	std::shared_ptr<Shader> program = drawPoints ? drawProgram : drawSurfelProgram;

	program->Bind();

	program->setUniform(Uniform("MVP", mvp));

	program->setUniform(Uniform("threshold", threshold));

	program->setUniform(Uniform("colorType", (drawClasses ? 4 : drawNormals ? 1 : drawColors ? 2 : drawTimes ? 3 : 0)));

	program->setUniform(Uniform("unstable", drawUnstable));

	program->setUniform(Uniform("drawWindow", drawWindow));

	program->setUniform(Uniform("time", time));

	program->setUniform(Uniform("timeDelta", timeDelta));

	Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
		//This is for the point shader
	program->setUniform(Uniform("pose", pose));

	glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

	//============================================== show ==========================================================
	//testFuction("4 vbos[target].first");
	//============================================== show ==========================================================

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 3));

	//============================WORK===================================
	glDrawTransformFeedback(GL_POINTS, vbos[target].second);		//DRAW
	//============================WORK===================================

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	program->Unbind();
}

//copy form "renderPointCloud". You can simplify this two code
void GlobalModel::renderGlobalID(pangolin::OpenGlMatrix mvp,
								   const float threshold,
								   const bool drawUnstable,
								   const bool drawNormals,
								   const bool drawColors,
								   const bool drawPoints,
								   const bool drawWindow,
								   const bool drawTimes,
								   const bool drawClasses,
								   const int time,
								   const int timeDelta)
{
	std::shared_ptr<Shader> program = drawGlobalIDProgram;

	program->Bind();

	program->setUniform(Uniform("MVP", mvp));

	program->setUniform(Uniform("threshold", threshold));

	program->setUniform(Uniform("colorType", (drawClasses ? 4 : drawNormals ? 1 : drawColors ? 2 : drawTimes ? 3 : 0)));

	program->setUniform(Uniform("unstable", drawUnstable));

	program->setUniform(Uniform("drawWindow", drawWindow));

	program->setUniform(Uniform("time", time));

	program->setUniform(Uniform("timeDelta", timeDelta));

	Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();

	program->setUniform(Uniform("pose", pose));

	glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 3));

	glDrawTransformFeedback(GL_POINTS, vbos[target].second);		//DRAW

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	program->Unbind();
}


const std::pair<GLuint, GLuint> & GlobalModel::model()
{
	return vbos[target];
}

void GlobalModel::fuse(const Eigen::Matrix4f & pose,
					   const int & time,
					   GPUTexture * rgb,
					   GPUTexture * depthRaw,
					   GPUTexture * depthFiltered,
                  			   GPUTexture * instanceGroundTruth,
					   GPUTexture * indexMap,
					   GPUTexture * vertConfMap,
					   GPUTexture * colorTimeMap,
					   GPUTexture * normRadMap,
					   const float depthCutoff,
					   const float confThreshold,
					   const float weighting)
{
	TICK("Fuse::Data");
	//This first part does data association and computes the vertex to merge with, storing
	//in an array that sets which vertices to update by index
	frameBuffer.Bind();

	glPushAttrib(GL_VIEWPORT_BIT);

	glViewport(0, 0, renderBuffer.width, renderBuffer.height);

	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	dataProgram->Bind();

	dataProgram->setUniform(Uniform("cSampler", 0));
	dataProgram->setUniform(Uniform("drSampler", 1));
	dataProgram->setUniform(Uniform("drfSampler", 2));
	dataProgram->setUniform(Uniform("indexSampler", 3));
	dataProgram->setUniform(Uniform("vertConfSampler", 4));
	dataProgram->setUniform(Uniform("colorTimeSampler", 5));
	dataProgram->setUniform(Uniform("normRadSampler", 6));
	dataProgram->setUniform(Uniform("instgtSampler", 7));

	dataProgram->setUniform(Uniform("time", (float)time));
	dataProgram->setUniform(Uniform("weighting", weighting));

	dataProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(),
													 Intrinsics::getInstance().cy(),
													 1.0 / Intrinsics::getInstance().fx(),
													 1.0 / Intrinsics::getInstance().fy())));
	dataProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
	dataProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));
	dataProgram->setUniform(Uniform("scale", (float)IndexMap::FACTOR));
	dataProgram->setUniform(Uniform("texDim", (float)TEXTURE_DIMENSION));
	dataProgram->setUniform(Uniform("pose", pose));
	dataProgram->setUniform(Uniform("maxDepth", depthCutoff));
	
	dataProgram->setUniform(Uniform("frameID", frameID));

	if(instanceGroundTruth) dataProgram->setUniform(Uniform("hasInstanceGroundTruth", true));
	else                    dataProgram->setUniform(Uniform("hasInstanceGroundTruth", false));


	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, uvo);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, newUnstableFid);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, newUnstableVbo);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, rgb->texture->tid);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, depthRaw->texture->tid);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, depthFiltered->texture->tid);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, indexMap->texture->tid);

	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, vertConfMap->texture->tid);

	glActiveTexture(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D, colorTimeMap->texture->tid);

	glActiveTexture(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_2D, normRadMap->texture->tid);

	if(instanceGroundTruth) 
	{
		glActiveTexture(GL_TEXTURE7);
		glBindTexture(GL_TEXTURE_2D, instanceGroundTruth->texture->tid);
	}

	//============================WORK===================================
	glBeginTransformFeedback(GL_POINTS);

	glDrawArrays(GL_POINTS, 0, uvSize); //DRAW

	glEndTransformFeedback();
	//============================WORK===================================

	frameBuffer.Unbind();

	glBindTexture(GL_TEXTURE_2D, 0);

	glActiveTexture(GL_TEXTURE0);


	//============================================== show ==========================================================
	//glBindBuffer(GL_ARRAY_BUFFER, newUnstableVbo);
	//testFuction("5 newUnstableVbo");
	//============================================== show ==========================================================


	glDisableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	dataProgram->Unbind();

	glPopAttrib();

	glFinish();
	TOCK("Fuse::Data");

	TICK("Fuse::Update");
	//Next we update the vertices at the indexes stored in the update textures
	//Using a transform feedback conditional on a texture sample
	updateProgram->Bind();

	updateProgram->setUniform(Uniform("vertSamp", 0));
	updateProgram->setUniform(Uniform("colorSamp", 1));
	updateProgram->setUniform(Uniform("normSamp", 2));
	updateProgram->setUniform(Uniform("texDim", (float)TEXTURE_DIMENSION));
	updateProgram->setUniform(Uniform("time", time));

	glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);
	//============================================== show ==========================================================
	//testFuction("6 vbos[target].first");
	//============================================== show ==========================================================

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 3));

	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 4));
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 5));
	glEnableVertexAttribArray(6);
	glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 6));
	glEnableVertexAttribArray(7);
	glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 7));
	glEnableVertexAttribArray(8);
	glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 8));
	glEnableVertexAttribArray(9);
	glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 9));
	glEnableVertexAttribArray(10);
	glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 10));
	glEnableVertexAttribArray(11);
	glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 11));
	glEnableVertexAttribArray(12);
	glVertexAttribPointer(12, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 12));
	glEnableVertexAttribArray(13);
	glVertexAttribPointer(13, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 13));
	glEnableVertexAttribArray(14);
	glVertexAttribPointer(14, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 14));
	glEnableVertexAttribArray(15);
	glVertexAttribPointer(15, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 15));

	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].second);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].first);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, deleted_surfel_buffer.first);

	//============================WORK===================================
	glBeginTransformFeedback(GL_POINTS);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, updateMapVertsConfs.texture->tid);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, updateMapColorsTime.texture->tid);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, updateMapNormsRadii.texture->tid);

	glDrawTransformFeedback(GL_POINTS, vbos[target].second);	//DRAW

	glEndTransformFeedback();
	//============================WORK===================================

	glDisable(GL_RASTERIZER_DISCARD);

	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);


	//============================================== show ==========================================================
	//glBindBuffer(GL_ARRAY_BUFFER, vbos[renderSource].first);
	//testFuction("7 vbos[renderSource].first");
	//============================================== show ==========================================================


	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
	glDisableVertexAttribArray(4);
	glDisableVertexAttribArray(5);
	glDisableVertexAttribArray(6);
	glDisableVertexAttribArray(7);
	glDisableVertexAttribArray(8);
	glDisableVertexAttribArray(9);
	glDisableVertexAttribArray(10);
	glDisableVertexAttribArray(11);
	glDisableVertexAttribArray(12);
	glDisableVertexAttribArray(13);
	glDisableVertexAttribArray(14);
	glDisableVertexAttribArray(15);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	updateProgram->Unbind();

	std::swap(target, renderSource);

	glFinish();
	TOCK("Fuse::Update");
}

void GlobalModel::clean(const Eigen::Matrix4f & pose,
						const int & time,
						GPUTexture * indexMap,
						GPUTexture * vertConfMap,
						GPUTexture * colorTimeMap,
						GPUTexture * normRadMap,
						GPUTexture * depthMap,
						const float confThreshold,
						std::vector<float> & graph,
						const int timeDelta,
						const float maxDepth,
						const bool isFern)
{
	assert(graph.size() / 16 < MAX_NODES);

	if(graph.size() > 0)
	{
		//Can be optimised by only uploading new nodes with offset
		glBindTexture(GL_TEXTURE_2D, deformationNodes.texture->tid);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, graph.size(), 1, GL_LUMINANCE, GL_FLOAT, graph.data());
	}

	TICK("Fuse::Copy");
	//Next we copy the new unstable vertices from the newUnstableFid transform feedback into the global map
	unstableProgram->Bind();
	unstableProgram->setUniform(Uniform("time", time));
	unstableProgram->setUniform(Uniform("confThreshold", confThreshold));
	unstableProgram->setUniform(Uniform("scale", (float)IndexMap::FACTOR));
	unstableProgram->setUniform(Uniform("indexSampler", 0));
	unstableProgram->setUniform(Uniform("vertConfSampler", 1));
	unstableProgram->setUniform(Uniform("colorTimeSampler", 2));
	unstableProgram->setUniform(Uniform("normRadSampler", 3));
	unstableProgram->setUniform(Uniform("nodeSampler", 4));
	unstableProgram->setUniform(Uniform("depthSampler", 5));
	unstableProgram->setUniform(Uniform("nodes", (float)(graph.size() / 16)));
	unstableProgram->setUniform(Uniform("nodeCols", (float)NODE_TEXTURE_DIMENSION));
	unstableProgram->setUniform(Uniform("timeDelta", timeDelta));
	unstableProgram->setUniform(Uniform("maxDepth", maxDepth));
	unstableProgram->setUniform(Uniform("isFern", (int)isFern));

	Eigen::Matrix4f t_inv = pose.inverse();
	unstableProgram->setUniform(Uniform("t_inv", t_inv));

	unstableProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(),
														 Intrinsics::getInstance().cy(),
														 Intrinsics::getInstance().fx(),
														 Intrinsics::getInstance().fy())));
	unstableProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
	unstableProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));

	glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

	//============================================== show ==========================================================
	//testFuction("8 vbos[target].first");
	//============================================== show ==========================================================

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 3));


	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 4));
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 5));
	glEnableVertexAttribArray(6);
	glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 6));
	glEnableVertexAttribArray(7);
	glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 7));
	glEnableVertexAttribArray(8);
	glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 8));
	glEnableVertexAttribArray(9);
	glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 9));
	glEnableVertexAttribArray(10);
	glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 10));
	glEnableVertexAttribArray(11);
	glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 11));
	glEnableVertexAttribArray(12);
	glVertexAttribPointer(12, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 12));
	glEnableVertexAttribArray(13);
	glVertexAttribPointer(13, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 13));
	glEnableVertexAttribArray(14);
	glVertexAttribPointer(14, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 14));
	glEnableVertexAttribArray(15);
	glVertexAttribPointer(15, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 15));
	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].second);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].first);	//绑定点0 -> rgb pos nor cor

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, deleted_surfel_buffer.first); //绑定点1 -> deleteID

	//============================WORK===================================
	glBeginTransformFeedback(GL_POINTS);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, indexMap->texture->tid);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, vertConfMap->texture->tid);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, colorTimeMap->texture->tid);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, normRadMap->texture->tid);

	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, deformationNodes.texture->tid);

	glActiveTexture(GL_TEXTURE5);
	glBindTexture(GL_TEXTURE_2D, depthMap->texture->tid);

	glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

	glDrawTransformFeedback(GL_POINTS, vbos[target].second);	//DRAW

	//This outputs ids of surfels still there
	glFinish();
	unstableProgram->setUniform(Uniform("isNew", 0));
	glBeginQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, 1, deleteQuery);  //只取1流
	glDrawTransformFeedback(GL_POINTS, vbos[target].second);	//DRAW
	glFinish();
	glEndQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, 1);
	glGetQueryObjectuiv(deleteQuery, GL_QUERY_RESULT, &deleted_count);
	unstableProgram->setUniform(Uniform("isNew", 1));


	//============================================== show ==========================================================
	//glBindBuffer(GL_ARRAY_BUFFER, vbos[renderSource].first);
	//testFuction("9 vbos[renderSource].first");
	//============================================== show ==========================================================

	glBindBuffer(GL_ARRAY_BUFFER, newUnstableVbo);

	//============================================== show ==========================================================
	//testFuction("10 newUnstableVbo");
	//============================================== show ==========================================================

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 3));


	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 4));
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 5));
	glEnableVertexAttribArray(6);
	glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 6));
	glEnableVertexAttribArray(7);
	glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 7));
	glEnableVertexAttribArray(8);
	glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 8));
	glEnableVertexAttribArray(9);
	glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 9));
	glEnableVertexAttribArray(10);
	glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 10));
	glEnableVertexAttribArray(11);
	glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 11));
	glEnableVertexAttribArray(12);
	glVertexAttribPointer(12, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 12));
	glEnableVertexAttribArray(13);
	glVertexAttribPointer(13, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 13));
	glEnableVertexAttribArray(14);
	glVertexAttribPointer(14, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 14));
	glEnableVertexAttribArray(15);
	glVertexAttribPointer(15, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 15));

	glDrawTransformFeedback(GL_POINTS, newUnstableFid);	 //DRAW

	glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

	glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);

	glEndTransformFeedback();
	//============================WORK===================================

	glDisable(GL_RASTERIZER_DISCARD);

	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
	glDisableVertexAttribArray(4);
	glDisableVertexAttribArray(5);
	glDisableVertexAttribArray(6);
	glDisableVertexAttribArray(7);
	glDisableVertexAttribArray(8);
	glDisableVertexAttribArray(9);
	glDisableVertexAttribArray(10);
	glDisableVertexAttribArray(11);
	glDisableVertexAttribArray(12);
	glDisableVertexAttribArray(13);
	glDisableVertexAttribArray(14);
	glDisableVertexAttribArray(15);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	unstableProgram->Unbind();

	std::swap(target, renderSource);

	glFinish();
	TOCK("Fuse::Copy");
}

unsigned int GlobalModel::lastCount()
{
	return count;
}

unsigned int GlobalModel::deletedCount()
{
	return deleted_count;
}

float* GlobalModel::getMapSurfelsGpu()
{
	return cuda_map_ptr;
}

int* GlobalModel::getDeletedSurfelsGpu()
{
	return cuda_deleted_surfel_ptr;
}

Eigen::Vector4f * GlobalModel::downloadMap()
{
	glFinish();

	Eigen::Vector4f * vertices = new Eigen::Vector4f[count * 16];

	memset(&vertices[0], 0, count * Vertex::SIZE);

	GLuint downloadVbo;

	glGenBuffers(1, &downloadVbo);
	glBindBuffer(GL_ARRAY_BUFFER, downloadVbo);
	glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_COPY_READ_BUFFER, vbos[renderSource].first);
	glBindBuffer(GL_COPY_WRITE_BUFFER, downloadVbo);

	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, count * Vertex::SIZE);
	glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, count * Vertex::SIZE, vertices);

	glBindBuffer(GL_COPY_READ_BUFFER, 0);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	glDeleteBuffers(1, &downloadVbo);

	glFinish();

	return vertices;
}

//use CPU 
void GlobalModel::updateSurfelClass(const int surfelId, const float color)
{
	glFinish();
	glBindBuffer(GL_ARRAY_BUFFER, vbos[0].first);
	float val = color;
	glBufferSubData(GL_ARRAY_BUFFER,surfelId * Vertex::SIZE + sizeof(Eigen::Vector4f) + sizeof(float),sizeof(float),&val);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glFinish();
}


void GlobalModel::updateFrameID(const int tick)
{
	frameID = tick;
}


void GlobalModel::testFuction(const std::string name)
{
	if( frameID < 33 )
	{
		std::cout<<frameID<<" "<<name<<std::endl;
		int memtttt = 9 * sizeof(Eigen::Vector4f);
		Eigen::Vector4f * verticesss = new Eigen::Vector4f[memtttt];
		glGetBufferSubData(GL_ARRAY_BUFFER, 0, memtttt, verticesss);
		for(size_t i = 0; i < 9; i++)
		{
			/*if(i==1||i==5)
			{
				unsigned char r = int(verticesss[i](0)) >> 16 & 0xFF;
				unsigned char g = int(verticesss[i](0)) >> 8 & 0xFF;
				unsigned char b = int(verticesss[i](0)) & 0xFF;


				unsigned char r2 = int(verticesss[i](1)) >> 16 & 0xFF;
				unsigned char g2 = int(verticesss[i](1)) >> 8 & 0xFF;
				unsigned char b2 = int(verticesss[i](1)) & 0xFF;

				std::cout<<"\t"<<(float)r <<"\t"<<(float)g <<"\t"<<(float)b <<std::endl;
				std::cout<<"\t"<<(float)r2<<"\t"<<(float)g2<<"\t"<<(float)b2<<std::endl;
				std::cout<<"\t"<<(float)verticesss[i](2)<<"\t"<<(float)verticesss[i](3)<<std::endl;
			}*/
			//else
			//{
				std::cout<<"\t"<<(float)verticesss[i](0)<<"\t"<<(float)verticesss[i](1)<<"\t"<<(float)verticesss[i](2)<<"\t"<<(float)verticesss[i](3)<<std::endl;
			//}
		}
		std::cout<<std::endl;
	}
}
