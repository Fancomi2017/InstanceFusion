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
#version 330 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec4 normal;
//layout (location = 3) in vec4 imgCorr;

uniform mat4 t_inv;
uniform vec4 cam; //cx, cy, fx, fy
uniform float cols;
uniform float rows;
uniform float maxDepth;
uniform int time;
uniform int timeDelta;
uniform float conf;

out vec4 vColor;
out vec4 vPosition;
out vec4 vNormRad;
out mat4 vT_inv;
out vec4 vCam;
out float vCols;
out float vRows;
out float vMaxDepth;
flat out int vertexId;

void main() {
    if(position.w > conf)
    {
        vertexId = gl_VertexID;
        vColor = color;
	vPosition = position;
	vNormRad = normal;

	    vT_inv = t_inv;
	    vCam = cam;
	    vCols = cols;
	    vRows = rows;
	    vMaxDepth = maxDepth;
    }
    else
    {
        vertexId = -1;
    }
}
