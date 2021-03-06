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

layout(points) in;
layout(points, max_vertices = 1) out;

in vec4 vPosition[];
in vec4 vColor[];
in vec4 vNormRad[];
in vec4 vImgCorr[];
flat in int updateId[];
in vec4 vInstInfoA[];
in vec4 vInstInfoB[];
in vec4 vInstInfoC[];
in vec4 vInstInfoD[];
in vec4 vInstInfoE[];
in vec4 vInstInfoF[];
in vec4 vInstInfoG[];
in vec4 vInstInfoH[];
in vec4 vInstInfoI[];
in vec4 vInstInfoJ[];
in vec4 vInstInfoK[];
in vec4 vInstInfoL[];

out vec4 vPosition0;
out vec4 vColor0;
out vec4 vNormRad0;
out vec4 vImgCorr0;
flat out int updateId0;
out vec4 vInstInfoA0;
out vec4 vInstInfoB0;
out vec4 vInstInfoC0;
out vec4 vInstInfoD0;
out vec4 vInstInfoE0;
out vec4 vInstInfoF0;
out vec4 vInstInfoG0;
out vec4 vInstInfoH0;
out vec4 vInstInfoI0;
out vec4 vInstInfoJ0;
out vec4 vInstInfoK0;
out vec4 vInstInfoL0;

void main() 
{
    //Emit a vertex if either we have an update to store, or a new unstable vertex to store
    if(updateId[0] > 0)
    {
	    vPosition0 = vPosition[0];
	    vColor0 = vColor[0];
	    vNormRad0 = vNormRad[0];
	    vImgCorr0 = vImgCorr[0];
	    updateId0 = updateId[0];

	    vInstInfoA0 = vInstInfoA[0];
	    vInstInfoB0 = vInstInfoB[0];
	    vInstInfoC0 = vInstInfoC[0];
	    vInstInfoD0 = vInstInfoD[0];
	    vInstInfoE0 = vInstInfoE[0];
	    vInstInfoF0 = vInstInfoF[0];
	    vInstInfoG0 = vInstInfoG[0];
	    vInstInfoH0 = vInstInfoH[0];
	    vInstInfoI0 = vInstInfoI[0];
	    vInstInfoJ0 = vInstInfoJ[0];
	    vInstInfoK0 = vInstInfoK[0];
	    vInstInfoL0 = vInstInfoL[0];

	    
	    //This will be -10, -10 (offscreen) for new unstable vertices, so they don't show in the fragment shader
	    gl_Position = gl_in[0].gl_Position;

	    EmitVertex();
	    EndPrimitive(); 
    }
}
