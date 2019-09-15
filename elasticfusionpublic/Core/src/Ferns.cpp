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

#include "Ferns.h"

Ferns::Ferns(int n, int maxDepth, const float photoThresh)
 : num(n),
   factor(8),
   width(Resolution::getInstance().width() / factor),
   height(Resolution::getInstance().height() / factor),
   maxDepth(maxDepth),
   photoThresh(photoThresh),
   widthDist(0, width - 1),
   heightDist(0, height - 1),
   rgbDist(0, 255),
   dDist(400, maxDepth),
   lastClosest(-1),
   badCode(255),
   rgbd(Resolution::getInstance().width() / factor,
		Resolution::getInstance().height() / factor,
		Intrinsics::getInstance().cx() / factor,
		Intrinsics::getInstance().cy() / factor,
		Intrinsics::getInstance().fx() / factor,
		Intrinsics::getInstance().fy() / factor),
   vertFern(width, height, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
   vertCurrent(width, height, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
   normFern(width, height, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
   normCurrent(width, height, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
   colorFern(width, height, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, false, true),
   colorCurrent(width, height, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, false, true),
   instFern(width, height, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, false, true),
   instCurrent(width, height, GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, false, true),
   resize(Resolution::getInstance().width(), Resolution::getInstance().height(), width, height),
   imageBuff(width, height),
   vertBuff(width, height),
   normBuff(width, height)
{
	random.seed(time(0));
	generateFerns();
}

Ferns::~Ferns()
{
	for(size_t i = 0; i < frames.size(); i++)
	{
		delete frames.at(i);
	}
}

void Ferns::generateFerns()
{
	for(int i = 0; i < num; i++)
	{
		Fern f;

		f.pos(0) = widthDist(random);
		f.pos(1) = heightDist(random);

		f.rgbd(0) = rgbDist(random);
		f.rgbd(1) = rgbDist(random);
		f.rgbd(2) = rgbDist(random);
		f.rgbd(3) = dDist(random);

		conservatory.push_back(f);
	}
}

bool Ferns::addFrame(GPUTexture * imageTexture, GPUTexture * vertexTexture, GPUTexture * normalTexture, GPUTexture * instTexture,
						 const Eigen::Matrix4f & pose, int srcTime, const float threshold)
{
	Img<Eigen::Matrix<unsigned char, 3, 1>> img(height, width);
	Img<Eigen::Vector4f> verts(height, width);
	Img<Eigen::Vector4f> norms(height, width);
	Img<Eigen::Matrix<unsigned char, 3, 1>> inst(height, width);	//instanceFusion +

	resize.image(imageTexture, img);
	resize.vertex(vertexTexture, verts);
	resize.vertex(normalTexture, norms);
	resize.image(instTexture, inst);

	Frame * frame = new Frame(num,
							  frames.size(),
							  pose,
							  srcTime,
							  width * height,
							  (unsigned char *)img.data,
							  (Eigen::Vector4f *)verts.data,
							  (Eigen::Vector4f *)norms.data,
							  (unsigned char *)inst.data);

	int * coOccurrences = new int[frames.size()];

	memset(coOccurrences, 0, sizeof(int) * frames.size());

	for(int i = 0; i < num; i++)
	{
		unsigned char code = badCode;

		if(verts.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(2) > 0)
		{
			const Eigen::Matrix<unsigned char, 3, 1> & pix = img.at<Eigen::Matrix<unsigned char, 3, 1>>(conservatory.at(i).pos(1), conservatory.at(i).pos(0));

			code = (pix(0) > conservatory.at(i).rgbd(0)) << 3 |
				   (pix(1) > conservatory.at(i).rgbd(1)) << 2 |
				   (pix(2) > conservatory.at(i).rgbd(2)) << 1 |
				   (int(verts.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(2) * 1000.0f) > conservatory.at(i).rgbd(3));

			frame->goodCodes++;

			for(size_t j = 0; j < conservatory.at(i).ids[code].size(); j++)
			{
				coOccurrences[conservatory.at(i).ids[code].at(j)]++;
			}
		}

		frame->codes[i] = code;
	}

	float minimum = std::numeric_limits<float>::max();

	if(frame->goodCodes > 0)
	{
		for(size_t i = 0; i < frames.size(); i++)
		{
			float maxCo = std::min(frame->goodCodes, frames.at(i)->goodCodes);

			float dissim = (float)(maxCo - coOccurrences[i]) / (float)maxCo;

			if(dissim < minimum)
			{
				minimum = dissim;
			}
		}
	}

	delete [] coOccurrences;

	if((minimum > threshold || frames.size() == 0) && frame->goodCodes > 0)
	{
		for(int i = 0; i < num; i++)
		{
			if(frame->codes[i] != badCode)
			{
				conservatory.at(i).ids[frame->codes[i]].push_back(frame->id);
			}
		}

		frames.push_back(frame);

		return true;
	}
	else
	{
		delete frame;

		return false;
	}
}

Eigen::Matrix4f Ferns::findFrame(std::vector<SurfaceConstraint> & constraints,
								 const Eigen::Matrix4f & currPose,
								 GPUTexture * vertexTexture,
								 GPUTexture * normalTexture,
								 GPUTexture * imageTexture,
								 GPUTexture * instTexture,
								 int* smallInstanceTable,
								 const int time,
								 const bool lost)
{
	lastClosest = -1;

	Img<Eigen::Matrix<unsigned char, 3, 1>> imgSmall(height, width);
	Img<Eigen::Vector4f> vertSmall(height, width);
	Img<Eigen::Vector4f> normSmall(height, width);
	Img<Eigen::Matrix<unsigned char, 3, 1>> instSmall(height, width);

	resize.image(imageTexture, imgSmall);
	resize.vertex(vertexTexture, vertSmall);
	resize.vertex(normalTexture, normSmall);	//?
	resize.image(instTexture, instSmall);		//instanceFusion +

	Frame * frame = new Frame(num, 0, Eigen::Matrix4f::Identity(), 0, width * height);

	int * coOccurrences = new int[frames.size()];

	memset(coOccurrences, 0, sizeof(int) * frames.size());

	for(int i = 0; i < num; i++)
	{
		unsigned char code = badCode;

		if(vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(2) > 0)
		{
			const Eigen::Matrix<unsigned char, 3, 1> & pix = imgSmall.at<Eigen::Matrix<unsigned char, 3, 1>>(conservatory.at(i).pos(1), conservatory.at(i).pos(0));

			code = (pix(0) > conservatory.at(i).rgbd(0)) << 3 |
				   (pix(1) > conservatory.at(i).rgbd(1)) << 2 |
				   (pix(2) > conservatory.at(i).rgbd(2)) << 1 |
				   (int(vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(2) * 1000.0f) > conservatory.at(i).rgbd(3));

			frame->goodCodes++;

			for(size_t j = 0; j < conservatory.at(i).ids[code].size(); j++)
			{
				coOccurrences[conservatory.at(i).ids[code].at(j)]++;
			}
		}

		frame->codes[i] = code;
	}

	float minimum = std::numeric_limits<float>::max();
	int minId = -1;

	for(size_t i = 0; i < frames.size(); i++)
	{
		float maxCo = std::min(frame->goodCodes, frames.at(i)->goodCodes);

		float dissim = (float)(maxCo - coOccurrences[i]) / (float)maxCo;

		if(dissim < minimum && time - frames.at(i)->srcTime > 300)
		{
			minimum = dissim;
			minId = i;
		}
	}

	delete [] coOccurrences;

	Eigen::Matrix4f estPose = Eigen::Matrix4f::Identity();

	if(minId != -1 && blockHDAware(frame, frames.at(minId)) > 0.3)
	{
		Eigen::Matrix4f fernPose = frames.at(minId)->pose;

//=====================================instanceFusion=====================================
		int frameMinIDnum = frames.at(minId)->num;
		Eigen::Vector4f* vertFernTemp = new Eigen::Vector4f[frameMinIDnum];
		Eigen::Vector4f* vertCurrTemp = new Eigen::Vector4f[frameMinIDnum];
		Eigen::Vector4f* normFernTemp = new Eigen::Vector4f[frameMinIDnum];
		Eigen::Vector4f* normCurrTemp = new Eigen::Vector4f[frameMinIDnum];

		memcpy(vertFernTemp, frames.at(minId)->initVerts, frameMinIDnum * sizeof(Eigen::Vector4f));
		memcpy(vertCurrTemp, vertSmall.data				, frameMinIDnum * sizeof(Eigen::Vector4f));
		memcpy(normFernTemp, frames.at(minId)->initNorms, frameMinIDnum * sizeof(Eigen::Vector4f));
		memcpy(normCurrTemp, normSmall.data				, frameMinIDnum * sizeof(Eigen::Vector4f));

		
		Eigen::Vector4f* vertFernTemp2 = NULL;
		Eigen::Vector4f* normFernTemp2 = NULL;
		Eigen::Vector4f* vertCurrTemp2 = NULL;
		Eigen::Vector4f* normCurrTemp2 = NULL;
		
		bool instICP = false;
		int afterNumF = frameMinIDnum;
		int afterNumC = frameMinIDnum;
		
		int instListF[96];
		int instListC[96];
		int instBBoxF[96*4];
		int instBBoxC[96*4];
		int instNumF = 0;
		int instNumC = 0;

		float matchPair[96*3];
		int matchNum = 0;

		//true -> instICP ON
		if(false)	
		{
			//debug color
			//colorFern.texture->Upload(frames.at(minId)->initRgb, GL_RGB, GL_UNSIGNED_BYTE);
			//colorCurrent.texture->Upload(imgSmall.data, GL_RGB, GL_UNSIGNED_BYTE);
			//unsigned char * image;	
				//cudaMalloc((void **)&image, width*height*4 * sizeof(unsigned char));
			//colorFern.texture->Download(image, GL_RGB, GL_UNSIGNED_BYTE);

		

			//inst
			instFern.texture->Upload(frames.at(minId)->initInst, GL_RGB, GL_UNSIGNED_BYTE);
			instCurrent.texture->Upload(instSmall.data, GL_RGB, GL_UNSIGNED_BYTE);

			//debug

			unsigned char * image;							
			unsigned char * imInst_Fern = new unsigned char[width*height*3];	
			unsigned char * imInst_Current = new unsigned char[width*height*3];
			cudaMalloc((void **)&image, width*height*3 * sizeof(unsigned char));

			cudaArray * textPtr;
			cudaGraphicsMapResources(1, &instFern.cudaRes);									//A
			cudaGraphicsSubResourceGetMappedArray(&textPtr, instFern.cudaRes, 0, 0);		//B
			imgBRGtoRGBCVFormat(textPtr, image, width, height);								//C	
			cudaGraphicsUnmapResources(1, &instFern.cudaRes);								//D
			cudaMemcpy(imInst_Fern, image, width*height*3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

			cudaGraphicsMapResources(1, &instCurrent.cudaRes);								//A
			cudaGraphicsSubResourceGetMappedArray(&textPtr, instCurrent.cudaRes, 0, 0);		//B
			imgBRGtoRGBCVFormat(textPtr, image, width, height);								//C	
			cudaGraphicsUnmapResources(1, &instCurrent.cudaRes);							//D
			cudaMemcpy(imInst_Current, image, width*height*3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

			//=================================
			int Intersection=0 , Union=0;
			
			for(int i=0;i<96;i++)
			{
				instBBoxF[i*4+0]= width+1 ;	//minX
				instBBoxF[i*4+1]= -1 ;		//maxX
				instBBoxF[i*4+2]= height+1 ;	//minY
				instBBoxF[i*4+3]= -1 ;		//maxY

				instBBoxC[i*4+0]= width+1 ;	//minX
				instBBoxC[i*4+1]= -1 ;		//maxX
				instBBoxC[i*4+2]= height+1 ;	//minY
				instBBoxC[i*4+3]= -1 ;		//maxY
			}
			//1. get list
			//2. get bbox
			for(int x=1;x<width-1;x++)
			{
				for(int y=1;y<height-1;y++)
				{
					for(int inst=0;inst<96;inst++)
					{
						for(int dx=x-1;dx<=x+1;dx++)
						{
							for(int dy=y-1;dy<=y+1;dy++)
							{
								if(imInst_Fern[dy*width*3+dx*3+0]==smallInstanceTable[5*inst+0]&&
								   imInst_Fern[dy*width*3+dx*3+1]==smallInstanceTable[5*inst+1]&&
								   imInst_Fern[dy*width*3+dx*3+2]==smallInstanceTable[5*inst+2])
								{
									//Match Fern(x,y)-> inst
									int listID = -1;
									for(int i=0;i<instNumF;i++)	//find listID
									{
										if(instListF[i]==inst)
										{
											listID = inst;
											break;
										}
									}
									if(listID==-1)//register
									{
										listID = instNumF;
										instListF[instNumF++] = inst;
									}
									instBBoxF[listID*4+0] = std::min(instBBoxF[listID*4+0],x);
									instBBoxF[listID*4+1] = std::max(instBBoxF[listID*4+1],x);
									instBBoxF[listID*4+2] = std::min(instBBoxF[listID*4+2],y);
									instBBoxF[listID*4+3] = std::max(instBBoxF[listID*4+3],y);
								}
								if(imInst_Current[dy*width*3+dx*3+0]==smallInstanceTable[5*inst+0]&&
								   imInst_Current[dy*width*3+dx*3+1]==smallInstanceTable[5*inst+1]&&
								   imInst_Current[dy*width*3+dx*3+2]==smallInstanceTable[5*inst+2])
								{
									//Match Current(x,y)-> inst
									int listID = -1;
									for(int i=0;i<instNumC;i++)	//find listID
									{
										if(instListC[i]==inst)
										{
											listID = inst;
											break;
										}
									}
									if(listID==-1)//register
									{
										listID = instNumC;
										instListC[instNumC++] = inst;
									}
									instBBoxC[listID*4+0] = std::min(instBBoxC[listID*4+0],x);
									instBBoxC[listID*4+1] = std::max(instBBoxC[listID*4+1],x);
									instBBoxC[listID*4+2] = std::min(instBBoxC[listID*4+2],y);
									instBBoxC[listID*4+3] = std::max(instBBoxC[listID*4+3],y);
								}
							}
						}
					}


				}
			}

			//3. IOU

			int minX_F,minX_C;
			int maxX_F,maxX_C;
			int minY_F,minY_C;
			int maxY_F,maxY_C;
			for(int i=0;i<instNumC;i++)
			{
				minX_C = instBBoxC[i*4+0];
				maxX_C = instBBoxC[i*4+1];
				minY_C = instBBoxC[i*4+2];
				maxY_C = instBBoxC[i*4+3];
				if(maxX_C<=minX_C||maxY_C<=minY_C)	continue;

				int bestChoose = -1;
				float bestIOU = 0;
				for(int j=0;j<instNumF;j++)
				{
					if(smallInstanceTable[5*instListC[i]+3] != smallInstanceTable[5*instListF[j]+3]) continue;

					minX_F = instBBoxF[j*4+0];
					maxX_F = instBBoxF[j*4+1];
					minY_F = instBBoxF[j*4+2];
					maxY_F = instBBoxF[j*4+3];
					if(maxX_F<=minX_F||maxY_F<=minY_F)	continue;

					
					float Intersection_W  = (std::min(maxX_F,maxX_C) - std::max(minX_F,minX_C));
					float Intersection_H  = (std::min(maxY_F,maxY_C) - std::max(minY_F,minY_C));
					if(Intersection_W<=0||Intersection_H<=0) continue;
					
					float Intersection = Intersection_W * Intersection_H;
					float Union = ((maxX_F - minX_F) * (maxY_F - minY_F))+((maxX_C - minX_C) * (maxY_C - minY_C)) - Intersection;
					
					if( Intersection / Union > 0.80 && Union > 100 && Intersection / Union > bestIOU)
					{
						bestIOU = Intersection / Union;
						bestChoose = j;
					}
				}
				if(bestChoose!=-1)
				{
					matchPair[matchNum*3+0] = bestChoose;			//fern
					matchPair[matchNum*3+1] = i;				//Current
					matchPair[matchNum*3+2] = bestIOU;
					matchNum++;
				}
			}
			if(matchNum)instICP=true;	//as least 1 match

			
			//debug
			/*std::cout<<"instICP:"<<(instICP?"OK":"nope")<<std::endl;
			for(int i=0;i<matchNum;i++)
			{
				std::cout<<"matchPair:("<<matchPair[i*3+0]<<","<<matchPair[i*3+1]<<")  iou:"<<matchPair[i*3+2]<<
							" instance:("<<instListF[(int)matchPair[i*3+0]]<<","<<instListC[(int)matchPair[i*3+1]]<<")"<<std::endl;			
			}*/


			//4.set 0
			if(instICP)
			{
				for(int x=0;x<width;x++)
				{
					for(int y=0;y<height;y++)
					{
						//Fern
						bool matchF = false;
						for(int i=0;i<matchNum;i++)
						{
							if(imInst_Fern[y*width*3+x*3+0]==smallInstanceTable[5*instListF[(int)matchPair[i*3+0]]+0]&&
							    imInst_Fern[y*width*3+x*3+1]==smallInstanceTable[5*instListF[(int)matchPair[i*3+0]]+1]&&
							    imInst_Fern[y*width*3+x*3+2]==smallInstanceTable[5*instListF[(int)matchPair[i*3+0]]+2])
							{
								matchF = true;
								break;
							}
						}
						if(!matchF)
						{
							afterNumF--;
							vertFernTemp[y*width+x] = Eigen::Vector4f(0,0,0,0);
							normFernTemp[y*width+x] = Eigen::Vector4f(0,0,0,0);
						}
						
						
						//Current
						bool matchC = false;
						for(int i=0;i<matchNum;i++)
						{
							if(imInst_Current[y*width*3+x*3+0]==smallInstanceTable[5*instListC[(int)matchPair[i*3+1]]+0]&&
							    imInst_Current[y*width*3+x*3+1]==smallInstanceTable[5*instListC[(int)matchPair[i*3+1]]+1]&&
							    imInst_Current[y*width*3+x*3+2]==smallInstanceTable[5*instListC[(int)matchPair[i*3+1]]+2])
							{
								matchC = true;
								break;
							}
						}
						if(!matchC)
						{
							afterNumC--;
							vertCurrTemp[y*width+x] = Eigen::Vector4f(0,0,0,0);
							normCurrTemp[y*width+x] = Eigen::Vector4f(0,0,0,0);
						}
					}
				}
				
				//5. copy good area to new buffer
				vertFernTemp2 = new Eigen::Vector4f[afterNumF];
				normFernTemp2 = new Eigen::Vector4f[afterNumF];
				vertCurrTemp2 = new Eigen::Vector4f[afterNumC];
				normCurrTemp2 = new Eigen::Vector4f[afterNumC];
				int fernP = 0;
				int CurrP = 0;
				for(int i=0;i<frameMinIDnum;i++)
				{
					if(vertFernTemp[i](0)!=0)
					{
						vertFernTemp2[fernP] = vertFernTemp[i];
						normFernTemp2[fernP] = normFernTemp[i];
						fernP++;
					}
					if(vertCurrTemp[i](0)!=0)
					{
						vertCurrTemp2[CurrP] = vertCurrTemp[i];
						normCurrTemp2[CurrP] = normCurrTemp[i];
						CurrP++;
					}
				}
				/*
				delete[] vertFernTemp; vertFernTemp = vertFernTemp2;
				delete[] vertCurrTemp; vertCurrTemp = vertCurrTemp2;
				delete[] normFernTemp; normFernTemp = normFernTemp2;
				delete[] normCurrTemp; normCurrTemp = normCurrTemp2;
				*/
			}
			
			//inst color debug
			//cv::Mat rgb_image2(height,width,CV_8UC3, imInst_Current);
			//cv::imwrite("instCurrent.png",rgb_image2);
			//cv::Mat rgb_image(height,width,CV_8UC3, imInst_Fern);
			//cv::imwrite("instFern.png",rgb_image);
			
			cudaFree(image);
			free(imInst_Fern);
			free(imInst_Current);
			
		}


		//downloadVertex("ferns",afterNumF,vertFernTemp);
		//downloadVertex("current",afterNumC,vertCurrTemp);



		vertFern.texture->Upload(vertFernTemp, GL_RGBA, GL_FLOAT);
		vertCurrent.texture->Upload(vertCurrTemp, GL_RGBA, GL_FLOAT);

		normFern.texture->Upload(normFernTemp, GL_RGBA, GL_FLOAT);
		normCurrent.texture->Upload(normCurrTemp, GL_RGBA, GL_FLOAT);

//=====================================instanceFusion=====================================

		//>ori code
		//vertFern.texture->Upload(frames.at(minId)->initVerts, GL_RGBA, GL_FLOAT);
		//vertCurrent.texture->Upload(vertSmall.data, GL_RGBA, GL_FLOAT);
		//normFern.texture->Upload(frames.at(minId)->initNorms, GL_RGBA, GL_FLOAT);
		//normCurrent.texture->Upload(normSmall.data, GL_RGBA, GL_FLOAT);

//	colorFern.texture->Upload(frames.at(minId)->initRgb, GL_RGB, GL_UNSIGNED_BYTE);
//	colorCurrent.texture->Upload(imgSmall.data, GL_RGB, GL_UNSIGNED_BYTE);

		//WARNING initICP* must be called before initRGB*
		rgbd.initICPModel(&vertFern, &normFern, (float)maxDepth / 1000.0f, fernPose);
//	rgbd.initRGBModel(&colorFern);

		rgbd.initICP(&vertCurrent, &normCurrent, (float)maxDepth / 1000.0f);
//	rgbd.initRGB(&colorCurrent);

		Eigen::Vector3f trans = fernPose.topRightCorner(3, 1);
		Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = fernPose.topLeftCorner(3, 3);

		TICK("fernOdom");
		rgbd.getIncrementalTransformation(trans,
										  rot,
										  false,
										  100,
										  false,
										  false,
										  false);
		TOCK("fernOdom");

		estPose.topRightCorner(3, 1) = trans;
		estPose.topLeftCorner(3, 3) = rot;

		float photoError = photometricCheck(vertSmall, imgSmall, estPose, fernPose, frames.at(minId)->initRgb);		//photoThresh=115

		int icpCountThresh = lost ? 1400 : 2400;

//	std::cout << rgbd.lastICPError << ", " << rgbd.lastICPCount << ", " << photoError << std::endl;

		//std::cout<<"do fernOdom"<<std::endl;

		//globalDeformation  instICP style(instancefusion)
		//std::cout<<"rgbd.lastICPError:"<<rgbd.lastICPError<<" rgbd.lastICPCount:"<<rgbd.lastICPCount<<"photoError"<<photoError<<std::endl;
																									
		if(instICP)
		{
			
			//std::cout<<"		Ferns constraints(instancefusion)"<<std::endl;

			lastClosest = minId;

			int icr = afterNumC/50>0?afterNumC/50:1;
			for(int i = 0; i < afterNumC; i += icr )
			{
				if(vertCurrTemp2[i](2) > 0 && int(vertCurrTemp2[i](2) * 1000.0f) < maxDepth)
				{
					Eigen::Vector4f worldRawPoint = currPose * Eigen::Vector4f(vertCurrTemp2[i](0),
																			   vertCurrTemp2[i](1),
																			   vertCurrTemp2[i](2),
																			   1.0f);

					Eigen::Vector4f worldModelPoint = estPose * Eigen::Vector4f(vertCurrTemp2[i](0),
																				vertCurrTemp2[i](1),
																				vertCurrTemp2[i](2),
																				1.0f);

					constraints.push_back(SurfaceConstraint(worldRawPoint, worldModelPoint));
				}
			}
			
			//record to instanceFusion
			for(int i=0;i<matchNum;i++)
			{
				if(matchPair[i*3+2]>0.80)
				{
					int instIDFern = instListF[(int)matchPair[i*3+0]];
					int instIDCurr = instListC[(int)matchPair[i*3+1]]; 
					
					int l = std::min(instIDFern,instIDCurr);
					int h = std::max(instIDFern,instIDCurr);

					smallInstanceTable[5*h+4] = l;
				}
			}
		}
		else if(rgbd.lastICPError < 0.0003 && rgbd.lastICPCount > icpCountThresh && photoError < photoThresh)
		{
			//std::cout<<"Ferns constraints(elasticfusion)"<<std::endl;
			
			lastClosest = minId;

			for(int i = 0; i < num; i += num / 50)
			{
				if(vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(2) > 0 &&
				   int(vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(2) * 1000.0f) < maxDepth)
				{
					Eigen::Vector4f worldRawPoint = currPose * Eigen::Vector4f(vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(0),
																			   vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(1),
																			   vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(2),
																			   1.0f);

					Eigen::Vector4f worldModelPoint = estPose * Eigen::Vector4f(vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(0),
																				vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(1),
																				vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(2),
																				1.0f);

					constraints.push_back(SurfaceConstraint(worldRawPoint, worldModelPoint));
				}
			}
		}
		delete[] vertFernTemp;
		delete[] vertCurrTemp;
		delete[] normFernTemp;
		delete[] normCurrTemp;
		if(vertFernTemp2) delete[] vertFernTemp2;
		if(vertCurrTemp2) delete[] vertCurrTemp2;
		if(normFernTemp2) delete[] normFernTemp2;
		if(normCurrTemp2) delete[] normCurrTemp2;
	}

	delete frame;

	return estPose;
}

float Ferns::photometricCheck(const Img<Eigen::Vector4f> & vertSmall,
							  const Img<Eigen::Matrix<unsigned char, 3, 1>> & imgSmall,
							  const Eigen::Matrix4f & estPose,
							  const Eigen::Matrix4f & fernPose,
							  const unsigned char * fernRgb)
{
	float cx = Intrinsics::getInstance().cx() / factor;
	float cy = Intrinsics::getInstance().cy() / factor;
	float invfx = 1.0f / float(Intrinsics::getInstance().fx() / factor);
	float invfy = 1.0f / float(Intrinsics::getInstance().fy() / factor);

	Img<Eigen::Matrix<unsigned char, 3, 1>> imgFern(height, width, (Eigen::Matrix<unsigned char, 3, 1> *)fernRgb);

	float photoSum = 0;
	int photoCount = 0;

	for(int i = 0; i < num; i++)
	{
		if(vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(2) > 0 &&
		   int(vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(2) * 1000.0f) < maxDepth)
		{
			Eigen::Vector4f vertPoint = Eigen::Vector4f(vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(0),
														vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(1),
														vertSmall.at<Eigen::Vector4f>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(2),
														1.0f);

			Eigen::Matrix4f diff = fernPose.inverse() * estPose;

			Eigen::Vector4f worldCorrPoint = diff * vertPoint;

			Eigen::Vector2i correspondence((worldCorrPoint(0) * (1/invfx) / worldCorrPoint(2) + cx), (worldCorrPoint(1) * (1/invfy) / worldCorrPoint(2) + cy));

			if(correspondence(0) >= 0 && correspondence(1) >= 0 && correspondence(0) < width && correspondence(1) < height &&
			   (imgFern.at<Eigen::Matrix<unsigned char, 3, 1>>(correspondence(1), correspondence(0))(0) > 0 ||
				imgFern.at<Eigen::Matrix<unsigned char, 3, 1>>(correspondence(1), correspondence(0))(1) > 0 ||
				imgFern.at<Eigen::Matrix<unsigned char, 3, 1>>(correspondence(1), correspondence(0))(2) > 0))
			{
				photoSum += abs((int)imgFern.at<Eigen::Matrix<unsigned char, 3, 1>>(correspondence(1), correspondence(0))(0) - (int)imgSmall.at<Eigen::Matrix<unsigned char, 3, 1>>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(0));
				photoSum += abs((int)imgFern.at<Eigen::Matrix<unsigned char, 3, 1>>(correspondence(1), correspondence(0))(1) - (int)imgSmall.at<Eigen::Matrix<unsigned char, 3, 1>>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(1));
				photoSum += abs((int)imgFern.at<Eigen::Matrix<unsigned char, 3, 1>>(correspondence(1), correspondence(0))(2) - (int)imgSmall.at<Eigen::Matrix<unsigned char, 3, 1>>(conservatory.at(i).pos(1), conservatory.at(i).pos(0))(2));
				photoCount++;
			}
		}
	}

	return photoSum / float(photoCount);
}

float Ferns::blockHD(const Frame * f1, const Frame * f2)
{
	float sum = 0.0f;

	for(int i = 0; i < num; i++)
	{
		sum += f1->codes[i] == f2->codes[i];
	}

	sum /= (float)num;

	return sum;
}

float Ferns::blockHDAware(const Frame * f1, const Frame * f2)
{
	int count = 0;
	float val = 0;

	for(int i = 0; i < num; i++)
	{
		if(f1->codes[i] != badCode && f2->codes[i] != badCode)
		{
			count++;

			if(f1->codes[i] == f2->codes[i])
			{
				val += 1.0f;
			}
		}
	}

	return val / (float)count;
}

void Ferns::downloadVertex(std::string tag,int num,Eigen::Vector4f* vertex)
{
	std::string filename = "GlobalDeformation_"+tag+".ply";
	
	//std::cout<<"write DeformationVertex as ply : "<<filename<<std::endl;
	
	// Open file
	std::ofstream fs;
	fs.open (filename.c_str ());

	// Write header
	fs << "ply";
	fs << "\nformat " << "binary_little_endian" << " 1.0";

	// Vertices
	fs << "\nelement vertex "<< num;
	fs << "\nproperty float x"
		  "\nproperty float y"
		  "\nproperty float z";

	fs << "\nend_header\n";

	// Close the file
	fs.close ();

	// Open file in binary appendable
	std::ofstream fpout (filename.c_str (), std::ios::app | std::ios::binary);

	for(unsigned int i = 0; i < num; i++)
	{
		Eigen::Vector4f pos = vertex[i];

		float value;
		memcpy (&value, &pos[0], sizeof (float));
		fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

		memcpy (&value, &pos[1], sizeof (float));
		fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

		memcpy (&value, &pos[2], sizeof (float));
		fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

	}

	// Close file
	fpout.close();



}
