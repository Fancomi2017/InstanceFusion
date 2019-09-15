/*
 * This file is part of InstanceFusion.
 *
 */
#version 330 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec4 normal;

layout (location = 3) in vec4 vImgCorr;
layout (location = 4) in vec4 vInstInfoA;
layout (location = 5) in vec4 vInstInfoB;
layout (location = 6) in vec4 vInstInfoC;
layout (location = 7) in vec4 vInstInfoD;
layout (location = 8) in vec4 vInstInfoE;
layout (location = 9) in vec4 vInstInfoF;
layout (location = 10) in vec4 vInstInfoG;
layout (location = 11) in vec4 vInstInfoH;
layout (location = 12) in vec4 vInstInfoI;
layout (location = 13) in vec4 vInstInfoJ;
layout (location = 14) in vec4 vInstInfoK;
layout (location = 15) in vec4 vInstInfoL;

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
    //empty
    if(vInstInfoB == vInstInfoA && vInstInfoC == vInstInfoA && vInstInfoD == vInstInfoA &&
       vInstInfoE == vInstInfoA && vInstInfoF == vInstInfoA && vInstInfoG == vInstInfoA && 
       vInstInfoH == vInstInfoA && vInstInfoI == vInstInfoA && vInstInfoJ == vInstInfoA && 
       vInstInfoK == vInstInfoA && vInstInfoL == vInstInfoA )
    {
	vertexId = -1;
    }
    else if(position.w <= conf)
    {
        vertexId = -1;
    }
    else
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
}
