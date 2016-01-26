#pragma once
#ifndef ShadowVegetationDetection_H
#define ShadowVegetationDetection_H

#include "com.h"
#include<opencv2\legacy\legacy.hpp>
#include "seeds2.h"
#include "qx_png.h"
#include "qx_basic.h"

void DoSuperpixelbySeeds(Mat srcImage, Size seedsSize, int nr_levels, int* SuperpixelsLabel, int& n_labels, bool isDraw, string imageName);
void ComputeSuperpixelMeanHSV(Mat srcImage, int* SuperpixelsLabel, int n_labels, Mat& meanHSV);

void ComputeHSMAPExtremeValue(Mat hs_map, vector<Vec2b> extreme_value);
void visulizeHSMAP(Mat hs_map, int deltathd, string imageName);
void Compute2Dhistrogram(Mat meanHS, int Hbins, int Sbins, Mat& hs_map);

void ShadowVegetationDetection(Mat srcImg, Mat& shadowMask, int& n_shadow,Mat& vegetationMask, int& n_vegetation);
void ShadowVegetationDetection(Mat srcImg, Mat samples, int* superpixelLabel, Mat& shadowMask, Mat& vegetationMask);
void EMSegmentation( Mat image, Mat VegetationIndex, int no_of_clusters, Mat& ShadowMask, int& n_shadow, Mat& VegetationMask, int& n_vegetation);
void EMSegmentationForSuperpixel( Mat image, Mat samples, int* superpixelLabel, Mat VegetationIndex, int no_of_clusters, Mat& ShadowMask, Mat& VegetationMask);
Mat asSamplesVectors( Mat& img );
void VegetationDetection(Mat srcImg,Mat& vegetatioCandidate );
void VisualizationObject(Mat& srcImage, Mat Mask, Vec3b color, string ImageName);
void ComputeVegetationIndex(Mat srcImg, Mat& phi );

void DrawContoursAroundSegments(UINT* img, int* labels, const int& width, const int& height, const UINT& color, bool internal);

#endif