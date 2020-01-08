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

#ifndef LOGREADER_H_
#define LOGREADER_H_

#include <string>
#include <zlib.h>
#include <poll.h>
#include <Utils/Img.h>
#include <Utils/Resolution.h>

#include "JPEGLoader.h"

class LogReader
{
    public:
        LogReader(std::string file, bool flipColors)
         : flipColors(flipColors),  // bool
           timestamp(0), // int64_t
           depth(0), // unsigned short *
           rgb(0),  // unsigned char *
           currentFrame(0), // int
           decompressionBufferDepth(0), // Bytef *
           decompressionBufferImage(0),// Bytef *
           file(file),  // const std::string
           width(Resolution::getInstance().width()),
           height(Resolution::getInstance().height()),
           numPixels(width * height)
        {}

        virtual ~LogReader()
        {}

        virtual void getNext() = 0;  // 最后面的“=0”并不表示函数返回值为0，它只起形式上的作用，告诉编译系统“这是纯虚函数”; 纯虚函数只有函数的名字而不具备函数的功能，不能被调用。
                                     // 它只是通知编译系统: “在这里声明一个虚函数，留待派生类中定义”。
        virtual int getNumFrames() = 0;

        virtual bool hasDepthFilled() { return false;}

        virtual bool hasMore() = 0;

        virtual bool rewound() = 0;

        virtual void getBack() = 0;

        virtual void fastForward(int frame) = 0;

        virtual const std::string getFile() = 0;

        virtual void setAuto(bool value) = 0;

        virtual bool isLabeledFrame() { return false; }
        virtual bool is_valid() { return false; }
        virtual std::string getLabelFrameId() { return ""; }

        bool flipColors;
        int64_t timestamp;

        unsigned short * depth;
        unsigned char * rgb;
        int currentFrame;

	//SemanticFusion
        unsigned short * depthfilled;
	
	//InstanceFusion
        unsigned char * instanceGroundTruth;

    protected:
        Bytef * decompressionBufferDepth;
        Bytef * decompressionBufferImage;
        unsigned char * depthReadBuffer;
        unsigned char * imageReadBuffer;
        int32_t depthSize;
        int32_t imageSize;

        const std::string file;
        FILE * fp;
        int32_t numFrames;
        int width;
        int height;
        int numPixels;

        JPEGLoader jpeg;
};

#endif /* LOGREADER_H_ */
