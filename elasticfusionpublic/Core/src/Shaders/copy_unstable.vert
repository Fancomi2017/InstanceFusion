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

#version 450 core

layout (location = 0) in vec4 vPos;
layout (location = 1) in vec4 vCol;
layout (location = 2) in vec4 vNormR;
layout (location = 3) in vec4 vImgC;

layout (location = 4) in vec4 vInstInfoA_;
layout (location = 5) in vec4 vInstInfoB_;
layout (location = 6) in vec4 vInstInfoC_;
layout (location = 7) in vec4 vInstInfoD_;
layout (location = 8) in vec4 vInstInfoE_;
layout (location = 9) in vec4 vInstInfoF_;
layout (location = 10) in vec4 vInstInfoG_;
layout (location = 11) in vec4 vInstInfoH_;
layout (location = 12) in vec4 vInstInfoI_;
layout (location = 13) in vec4 vInstInfoJ_;
layout (location = 14) in vec4 vInstInfoK_;
layout (location = 15) in vec4 vInstInfoL_;


out vec4 vPosition;
out vec4 vColor;
out vec4 vNormRad;
out vec4 vImgCorr;
flat out int test;
flat out int vertexId;
out vec4 vInstInfoA;
out vec4 vInstInfoB;
out vec4 vInstInfoC;
out vec4 vInstInfoD;
out vec4 vInstInfoE;
out vec4 vInstInfoF;
out vec4 vInstInfoG;
out vec4 vInstInfoH;
out vec4 vInstInfoI;
out vec4 vInstInfoJ;
out vec4 vInstInfoK;
out vec4 vInstInfoL;


uniform int time;
uniform float scale;
uniform mat4 t_inv;
uniform vec4 cam; //cx, cy, fx, fy
uniform float cols;
uniform float rows;
uniform float confThreshold;
uniform usampler2D indexSampler;
uniform sampler2D vertConfSampler;
uniform sampler2D colorTimeSampler;
uniform sampler2D normRadSampler;
uniform sampler2D nodeSampler;
uniform sampler2D depthSampler;
uniform float nodes;
uniform float nodeCols;
uniform float maxDepth;
uniform int timeDelta;
uniform int isFern;

void main()
{
    vPosition = vPos;
    vColor = vCol;
    vNormRad = vNormR;
    vImgCorr = vImgC;

    vInstInfoA = vInstInfoA_;
    vInstInfoB = vInstInfoB_;
    vInstInfoC = vInstInfoC_;
    vInstInfoD = vInstInfoD_;
    vInstInfoE = vInstInfoE_;
    vInstInfoF = vInstInfoF_;
    vInstInfoG = vInstInfoG_;
    vInstInfoH = vInstInfoH_;
    vInstInfoI = vInstInfoI_;
    vInstInfoJ = vInstInfoJ_;
    vInstInfoK = vInstInfoK_;
    vInstInfoL = vInstInfoL_;


    test = 1;
    vertexId = gl_VertexID;

    vec3 localPos = (t_inv * vec4(vPosition.xyz, 1.0f)).xyz;
    
    float x = ((cam.z * localPos.x) / localPos.z) + cam.x;
    float y = ((cam.w * localPos.y) / localPos.z) + cam.y;
    
    vec3 localNorm = normalize(mat3(t_inv) * vNormRad.xyz);

    float indexXStep = (1.0f / (cols * scale)) * 0.5f;
    float indexYStep = (1.0f / (rows * scale)) * 0.5f;

    float windowMultiplier = 2;

    int count = 0;
    int zCount = 0;

    if(//abs(localNorm.z) > 0.85f &&
       time - vColor.w < timeDelta && localPos.z > 0 && x > 0 && y > 0 && x < cols && y < rows)
    {
        for(float i = x / cols - (scale * indexXStep * windowMultiplier); i < x / cols + (scale * indexXStep * windowMultiplier); i += indexXStep)
        {
            for(float j = y / rows - (scale * indexYStep * windowMultiplier); j < y / rows + (scale * indexYStep * windowMultiplier); j += indexYStep)
            {
               uint current = uint(textureLod(indexSampler, vec2(i, j), 0));
               
               if(current > 0U)
               {
                   vec4 vertConf = textureLod(vertConfSampler, vec2(i, j), 0);
                   vec4 colorTime = textureLod(colorTimeSampler, vec2(i, j), 0);

                   if(colorTime.z < vColor.z && 
                      vertConf.w > confThreshold && 
                      vertConf.z > localPos.z && 
                      vertConf.z - localPos.z < 0.01 &&
                      sqrt(dot(vertConf.xy - localPos.xy, vertConf.xy - localPos.xy)) < vNormRad.w * 1.4)
                   {
                       count++;
                   }
                   
                   if(colorTime.w == time &&
                      vertConf.w > confThreshold && 
                      vertConf.z > localPos.z && 
                      vertConf.z - localPos.z > 0.01 &&
                      abs(localNorm.z) > 0.85f)
                   {
                       zCount++;
                   }
               }
            }
        }
    }
    
    if(count > 8 || zCount > 4)
    {
        test = 0;
    }
    
    //New unstable point
    if(vColor.w == -2)
    {
        vColor.w = time;
    }
    
    //Degenerate case or too unstable
    if((vColor.w == -1 || ((time - vColor.w) > 20 && vPosition.w < confThreshold)))
    {
        test = 0;
    }
    
    if(vColor.w > 0 && time - vColor.w > timeDelta)
    {
        test = 1;
    }
    
    //This is probably really slow
    //We don't deform vColor.z == time because they were fused with the updated pose already!
    if(test == 1 && nodes > 0 && vColor.z != time)
    {
        const int k = 4;
        const int lookBack = 20;
        int nearNodes[lookBack];
        float nearDists[lookBack];
        
        for(int i = 0; i < lookBack; i++)
        {
            nearNodes[i] = -1;
            nearDists[i] = 16777216.0f;
        }
        
        int poseTime = int(vColor.z);

        int foundIndex = 0;

        int imin = 0;
        int imax = int(nodes) - 1;
        int imid = (imin + imax) / 2;

	//二分搜索
        while(imax >= imin)
        {
            imid = (imin + imax) / 2;

            int nodeTime = int(textureLod(nodeSampler, vec2(((imid * 16 + 15) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x);

            if(nodeTime < poseTime)
            {
                imin = imid + 1;
            }
            else if(nodeTime > poseTime)
            {
                imax = imid - 1;
            }
            else
            {
                break;
            }
        }

        imin = min(imin, int(nodes) - 1);

        int nodeMin = int(textureLod(nodeSampler, vec2(((imin * 16 + 15) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x);
        int nodeMid = int(textureLod(nodeSampler, vec2(((imid * 16 + 15) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x);
        int nodeMax = int(textureLod(nodeSampler, vec2(((imax * 16 + 15) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x);

        if(abs(nodeMin - poseTime) <= abs(nodeMid - poseTime) &&
           abs(nodeMin - poseTime) <= abs(nodeMax - poseTime))
        {
            foundIndex = imin;
        }
        else if(abs(nodeMid - poseTime) <= abs(nodeMin - poseTime) &&
                abs(nodeMid - poseTime) <= abs(nodeMax - poseTime))
        {
            foundIndex = imid;
        }
        else
        {
            foundIndex = imax;
        }

        if(foundIndex == int(nodes))
        {
            foundIndex = int(nodes) - 1;
        }
	//二分搜索end

	//保存附近点
        int nearNodeIndex = 0;

        int distanceBack = 0;
        
        for(int j = foundIndex; j >= 0; j--)
        {
            vec3 position = vec3(textureLod(nodeSampler, vec2(((j * 16) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x, 
                                 textureLod(nodeSampler, vec2(((j * 16 + 1) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x, 
                                 textureLod(nodeSampler, vec2(((j * 16 + 2) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x);

            nearNodes[nearNodeIndex] = j;
            nearDists[nearNodeIndex] = sqrt(dot(vPosition.xyz - position, vPosition.xyz - position));
            nearNodeIndex++;
            
            if(++distanceBack == lookBack / 2)
            {
                break;
            }
        }

        for(int j = foundIndex + 1; j < int(nodes); j++)
        {
            vec3 position = vec3(textureLod(nodeSampler, vec2(((j * 16) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x, 
                                 textureLod(nodeSampler, vec2(((j * 16 + 1) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x, 
                                 textureLod(nodeSampler, vec2(((j * 16 + 2) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x);

            nearNodes[nearNodeIndex] = j;
            nearDists[nearNodeIndex] = sqrt(dot(vPosition.xyz - position, vPosition.xyz - position));
            nearNodeIndex++;

            if(++distanceBack == lookBack)
            {
                break;
            }
        }

	//按距离排序	
        for(int i = 0; i < lookBack - 1; ++i)
        {
            for(int j = i + 1; j < lookBack; ++j)
            {
                if(nearDists[j] < nearDists[i])
                {
                    float t = nearDists[i];
                    nearDists[i] = nearDists[j];
                    nearDists[j] = t;
                    
                    int t2 = nearNodes[i];
                    nearNodes[i] = nearNodes[j];
                    nearNodes[j] = t2;
                }
            }
        }
	//保存附近点end
        
	        
	float dMax = nearDists[k];
        float nodeWeights[k];
        float weightSum = 0;
	//计算前k近的权重
        for(int j = 0; j < k; j++)
        {
            vec3 position = vec3(textureLod(nodeSampler, vec2(((nearNodes[j] * 16) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x, 
                                 textureLod(nodeSampler, vec2(((nearNodes[j] * 16 + 1) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x, 
                                 textureLod(nodeSampler, vec2(((nearNodes[j] * 16 + 2) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x);
    
            nodeWeights[j] = pow(1.0f - (sqrt(dot(vPosition.xyz - position, vPosition.xyz - position)) / dMax), 2);
            weightSum += nodeWeights[j];
        }

        for(int j = 0; j < k; j++)
        {
            nodeWeights[j] /= weightSum;
        }
        
        vec3 newPos = vec3(0, 0, 0);
        vec3 newNorm = vec3(0, 0, 0);
        //取出前k近的信息
        for(int i = 0; i < k; i++)
        {
            vec3 position = vec3(textureLod(nodeSampler, vec2(((nearNodes[i] * 16) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x, 
                                 textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 1) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x, 
                                 textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 2) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x);
                                 
	    vec3 column0 = vec3(textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 3) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x, 
	                        textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 4) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x,
	                        textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 5) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x);
	                            
	    vec3 column1 = vec3(textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 6) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x, 
	                        textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 7) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x,
	                        textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 8) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x);
	                            
	    vec3 column2 = vec3(textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 9) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x, 
	                        textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 10) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x,
	                        textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 11) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x);
	
	        mat3 rotation = mat3(column0, column1, column2);
	                      
	        vec3 translation = vec3(textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 12) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x, 
	                                textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 13) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x,
	                                textureLod(nodeSampler, vec2(((nearNodes[i] * 16 + 14) / nodeCols) + (1.0 / (nodeCols * 2)), 0.5), 0).x);
	
	        newPos += nodeWeights[i] * (rotation * (vPosition.xyz - position) + position + translation);
	        newNorm += nodeWeights[i] * (transpose(inverse(rotation)) * vNormRad.xyz);
        }
        //根据近点更新现在点
        vPosition.xyz = newPos;
        vNormRad.xyz = normalize(newNorm);

        if(vPosition.w > confThreshold && isFern == 0)
        {
            localPos = (t_inv * vec4(vPosition.xyz, 1.0f)).xyz;
            
            x = ((cam.z * localPos.x) / localPos.z) + cam.x;
            y = ((cam.w * localPos.y) / localPos.z) + cam.y;
            
            if(localPos.z > 0 && localPos.z < maxDepth && x > 0 && y > 0 && x < cols && y < rows)
            {
                float currentDepth = float(textureLod(depthSampler, vec2(x / cols, y / rows), 0));
            
                if(currentDepth > 0.0f && localPos.z < currentDepth + 0.1f)
                {
	                vColor.w = time;
                }
            }
        }
    }
}

