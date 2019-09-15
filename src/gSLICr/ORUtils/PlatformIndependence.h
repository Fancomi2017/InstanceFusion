// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM
#pragma once

#include <cstdio>
#include <stdexcept>

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _CPU_AND_GPU_CODE_ __device__	// for CUDA device code
#else
#define _CPU_AND_GPU_CODE_ 
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define _CPU_AND_GPU_CONSTANT_ __constant__	// for CUDA device code
#else
#define _CPU_AND_GPU_CONSTANT_
#endif

#if defined(__METALC__) // for METAL device code
#define THREADPTR(x) thread x
#define DEVICEPTR(x) device x
#define THREADGRPPTR(x) threadgroup x
#define CONSTPTR(x) constant x
#else
#define THREADPTR(x) x
#define DEVICEPTR(x) x
#define THREADGROUPPTR(x) x
#define CONSTPTR(x) x
#endif

#ifdef ANDROID
#define DIEWITHEXCEPTION(x) { fprintf(stderr, "%s\n", x); exit(-1); }
#else
#define DIEWITHEXCEPTION(x) throw std::runtime_error(x)
#endif
