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

layout (location = 0) in vec4 vPosition;
layout (location = 1) in vec4 vColor;
layout (location = 2) in vec4 vNormRad;
//layout (location = 3) in vec4 vImgCorr;

out vec4 vPosition0;
out vec4 vColor0;
out vec4 vNormRad0;
out vec4 vImgCorr0;

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
    vPosition0 = vPosition;
    vColor0 = vColor;
    vColor0.y = 0; //Unused
    vColor0.z = 1; //This sets the vertex's initialisation time
    vNormRad0 = vNormRad;

    vImgCorr0 = vec4(-1.0,-1.0,-1.0,-1.0);

    vInstInfoA0 = vec4(-1.0,-1.0,-1.0,-1.0);
    vInstInfoB0 = vec4(-1.0,-1.0,-1.0,-1.0);
    vInstInfoC0 = vec4(-1.0,-1.0,-1.0,-1.0);
    vInstInfoD0 = vec4(-1.0,-1.0,-1.0,-1.0);
    vInstInfoE0 = vec4(-1.0,-1.0,-1.0,-1.0);
    vInstInfoF0 = vec4(-1.0,-1.0,-1.0,-1.0);
    vInstInfoG0 = vec4(-1.0,-1.0,-1.0,-1.0);
    vInstInfoH0 = vec4(-1.0,-1.0,-1.0,-1.0);
    vInstInfoI0 = vec4(-1.0,-1.0,-1.0,-1.0);
    vInstInfoJ0 = vec4(-1.0,-1.0,-1.0,-1.0);
    vInstInfoK0 = vec4(-1.0,-1.0,-1.0,-1.0);
    vInstInfoL0 = vec4(-1.0,-1.0,-1.0,-1.0);

}
