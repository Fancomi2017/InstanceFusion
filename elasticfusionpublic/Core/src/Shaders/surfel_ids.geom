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
layout(triangle_strip, max_vertices = 4) out;

in vec4 vColor[];
in vec4 vPosition[];
in vec4 vNormRad[];
in mat4 vT_inv[];
in vec4 vCam[]; //cx, cy, fx, fy
in float vCols[];
in float vRows[];
in float vMaxDepth[];
flat in int vertexId[];

flat out int vertexId0;
out vec3 v;
out vec3 n;
out vec2 texcoord;

#include "color.glsl"

// 3d -> 2d + depth
vec4 project_point(vec4 pos) {
    vec4 vPosHome = vT_inv[0] * pos;
    float xloc = ((((vCam[0].z * vPosHome.x) / vPosHome.z) + vCam[0].x) - (vCols[0] * 0.5)) / (vCols[0] * 0.5);
    float yloc = ((((vCam[0].w * vPosHome.y) / vPosHome.z) + vCam[0].y) - (vRows[0] * 0.5)) / (vRows[0] * 0.5);
    return vec4(xloc,yloc,vPosHome.z / vMaxDepth[0], 1.0);
}

void main() {
    if (vertexId[0] > -1) {
        vec4 point = project_point(vec4(vPosition[0].xyz, 1.0));
        if (point.z > 0.01) {
            vertexId0 = vertexId[0];
            vec3 x = normalize(vec3((vNormRad[0].y - vNormRad[0].z), 
                                    -vNormRad[0].x, vNormRad[0].x)) * 
                    vNormRad[0].w * 1.41421356 * 1.0;
            vec3 y = cross(vNormRad[0].xyz, x) * 1.0;
            n = vNormRad[0].xyz;

            texcoord = vec2(-1.0, -1.0);
            gl_Position = project_point(vec4(vPosition[0].xyz + x, 1.0));
            v = vPosition[0].xyz + x;
            EmitVertex();

            texcoord = vec2(1.0, -1.0);
            gl_Position = project_point(vec4(vPosition[0].xyz + y, 1.0));
            v = vPosition[0].xyz + y;
            EmitVertex();

            texcoord = vec2(-1.0, 1.0);
            gl_Position = project_point(vec4(vPosition[0].xyz - y, 1.0));
            v = vPosition[0].xyz - y;
            EmitVertex();

            texcoord = vec2(1.0, 1.0);
            gl_Position = project_point(vec4(vPosition[0].xyz - x, 1.0));
            v = vPosition[0].xyz - x;
            EmitVertex();
            EndPrimitive();
        }
    }
}
