#pragma once
#ifndef  MULLTILABELGRAPH_H
#define  MULLTILABELGRAPH_H

#include <iostream>
#include "highgui.h"
#include "cv.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "GCoptimization.h"
#include "GMMModel.h"

using namespace std;
using namespace cv;

static  double e_gamma;
static  double e_lambda;
static  double e_beta;
static  Mat e_Img;


struct ForDataFn{
	int numLab;
	int *data;
};


class mymultilabelgraphcut
{

public:
	mymultilabelgraphcut(void);
	~mymultilabelgraphcut(void);

 	void DoMultiLabelGraphCut(const Mat& img, const Mat& mask,GMM *GMMModels,
 		int n_models,int iterCnt, Mat& dstImage );
	void MultiLabelGraphCut( InputArray _img, InputOutputArray _mask, int n_models, int iterCount,string Imamgename);
};

#endif 