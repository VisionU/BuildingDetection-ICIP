#include <io.h> 
#include <iostream>
#include "highgui.h"
#include "cv.h"
#include<opencv2\legacy\legacy.hpp>
#include "seeds2.h"
#include "qx_png.h"
#include "qx_basic.h"
#include "helper.h"

using namespace std;
using namespace cv;
typedef  unsigned int UINT;

#define PI 3.1415926
const uchar colormap_jet[15][3]= 
{
	255,      0,    255,  // magenta
	165,     42,     42,  // brown
	255,      0,      0,  // red
	238,     99,     99,  // indian red
	255,    193,    193,  // rocybrown
	184,    134,     11,  // dark goldenrod
	255,    255,      0,  // yellow
	255,    215,      0,  // gold
	0,    255,      0,  // green
	144,    238,    144,  // palegreen
	25,     25,    112,  // midnightblue
	0,      0,    255,  // blue
	0,    255,    255,  // cyan
	100,    149,    237,  // cornflowerblue
	220,    220,    220   // gainsboro

};
float transferMatrix[3][3] = 
{
	0.3, 0.58, 0.11,
	0.25, 0.25 -0.05,
	0.5, -0.05, 0 
};

struct  GMMComponentParas 
{
	Mat u;
	Mat cov;
	float weight;
};

void getFiles( string path, vector<string>& files );
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
void VisualizationObject(Mat srcImage, Mat Mask, Vec3b color, string ImageName);
void ComputeVegetationIndex(Mat srcImg, Mat& phi );
void VisualizeLabel(Mat srcImage,int* superpixelLabel,Mat bestLabel, string imageName);
void SpectralClustering(Mat srcImage, int* superpixelLabel, int n_labels, int n_features, int max_k_eigenvalue, int n_cluster, Mat& bestLabel);

void MultilabelGraphCuts(Mat srcImage, int n_masks, Mat *ObjectMask, int* n_samples);


void main()
{
	vector<string> srcImagePaths; 
	getFiles("srcImage\\",srcImagePaths);

	for (int it = 4; it < srcImagePaths.size(); it++)
	{
		Mat srcImg = imread(srcImagePaths[it].c_str());
		string ImageName;
		ImageName.assign(srcImagePaths[it].begin() + 10, srcImagePaths[it].end() - 4);

		int* SuperpixelsLabel = new int[srcImg.rows * srcImg.cols];
		int Nlabels = 0;
		Mat MeanHSV;
// 
		double start = clock();
// 
// 		DoSuperpixelbySeeds(srcImg,Size(2,3),4,SuperpixelsLabel,Nlabels,true,ImageName);
//		ComputeSuperpixelMeanHSV (srcImg,SuperpixelsLabel,Nlabels,MeanHSV);	
// 
// 		Mat HSMAP;
// 		Compute2Dhistrogram(MeanHS,10,10,HSMAP);	
// 		visulizeHSMAP(HSMAP,4,ImageName);

//  	Mat bestLabel(Nlabels,1,CV_16SC1);
// 		SpectralClustering(srcImg,SuperpixelsLabel,Nlabels,60,Nlabels - 10,10,bestLabel);
// 		VisualizeLabel(srcImg,SuperpixelsLabel,bestLabel,ImageName);

		Mat ShadowMask;
		Mat VegetationMask;
		int N_Shadow = 0;
		int N_Vegetation = 0;
		ShadowVegetationDetection(srcImg,ShadowMask,N_Shadow,VegetationMask,N_Vegetation); //,MeanHSV,SuperpixelsLabel

		Mat *ObjectMask = new Mat[3];
		ObjectMask[0] = ShadowMask;
		ObjectMask[1] = VegetationMask;
		ObjectMask[2] = 255 - (ShadowMask + VegetationMask);

		int *n_samples = new int[3];
		n_samples[0] = N_Shadow;
		n_samples[1] = N_Vegetation;
		n_samples[2] = srcImg.rows * srcImg.cols - N_Shadow - N_Vegetation;

		MultilabelGraphCuts(srcImg, 3, ObjectMask, n_samples);

		string shadowImageName = "shadowImage\\shadowImage_" + ImageName + ".jpg";
		string vegetationImageName = "vegetationImage\\vegetationImage_" + ImageName + ".jpg";
		VisualizationObject(srcImg,ShadowMask,Vec3b(255,0,0),shadowImageName);
		VisualizationObject(srcImg,VegetationMask,Vec3b(0,255,0),vegetationImageName);

		double end = clock();
		cout << "Time consuming is "<< (end - start) / CLOCKS_PER_SEC << endl;

 		delete[] SuperpixelsLabel;
	}
}

void getFiles( string path, vector<string>& files )  
{  
	//文件句柄  
	long   hFile   =   0;  
	//文件信息  
	struct _finddata_t fileinfo;  
	string p;  
	if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)  
	{  
		do  
		{  
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if((fileinfo.attrib &  _A_SUBDIR))  
			{  
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
				{
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name));
					getFiles( p.assign(path).append("\\").append(fileinfo.name), files );  
				}
			}
			else  
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
			}  
		}while(_findnext(hFile, &fileinfo)  == 0);  
		_findclose(hFile);  
	}  
}  

void DoSuperpixelbySeeds(Mat srcImage, Size seedsSize, int nr_levels, int* SuperpixelsLabel, int& n_labels, bool isDraw, string imageName)
{
	int width(0), height(0);
	int channel(3);
	width = srcImage.cols;
	height = srcImage.rows;
	UINT sz = height * width;

	uchar*** image = qx_allocu_3(  height, width, 3 );
	for(int j=0;j<height;j++)
		for(int i=0;i<width;i++)
		{
			image[j][i][0] = srcImage.at<Vec3b>(j,i)[2];
			image[j][i][1] = srcImage.at<Vec3b>(j,i)[1];
			image[j][i][2] = srcImage.at<Vec3b>(j,i)[0];
		}	

		UINT* ubuff = new UINT[sz];
		UINT* ubuff2 = new UINT[sz];
		UINT* dbuff = new UINT[sz];
		UINT pValue;
		UINT pdValue;
		char c;
		UINT r,g,b,d,dx,dy;
		int idx = 0;
		for(int j=0;j<height;j++)
			for(int i=0;i<width;i++)
			{
				if(channel == 3)
				{  
					b = image[j][i][2];
					g = image[j][i][1];
					r = image[j][i][0];
					pValue = b | (g << 8) | (r << 16);
				}
				else if( channel == 1)
				{
					c = image[i][j][0];
					pValue = c | (c << 8) | (c << 16);
				}
				else
				{
					printf("Unknown number of channels %d\n", channel );
					return;
				}          
				ubuff[idx] = pValue;
				ubuff2[idx] = pValue;
				idx++;
			}

		// SEEDS INITIALIZE
		int NR_BINS = 10;     // Number of bins in each histogram channel
		SEEDS seeds(width, height, 3, NR_BINS);
		if (width < height)
			swap(seedsSize.width,seedsSize.height);

		int nr_superpixels = (height / (float)(seedsSize.height * pow(2.0,nr_levels-1))) * (width / (float)(seedsSize.width * pow(2.0,nr_levels-1)));
		
		seeds.initialize(seedsSize.width, seedsSize.height, nr_levels);

		seeds.update_image_ycbcr(ubuff);

		seeds.iterate();

		n_labels = seeds.count_superpixels();
		printf("SEEDS produced %d labels\n", seeds.count_superpixels());

		for (int i = 0; i < sz; i++)
			SuperpixelsLabel[i] = (int)seeds.get_labels()[i];

		delete[] ubuff2;
		delete[] dbuff;


		// DRAW SEEDS OUTPUT
		if (isDraw)
		{
			UINT* output_buff = new UINT[sz];
			for (int i = 0; i < sz; i++)
			{
				int row = i / width;
				int col = i % width;
				UINT B = srcImage.at<Vec3b>(row,col)[0];
				UINT G = srcImage.at<Vec3b>(row,col)[1];
				UINT R = srcImage.at<Vec3b>(row,col)[2];
				output_buff[i] = (R << 16) | (G << 8) | B;
			}

			DrawContoursAroundSegments(output_buff,SuperpixelsLabel, width, height, 0xff0000, false);//0xff0000 draws white contours

			for(int j=0;j<height;j++)
				for(int i=0;i<width;i++)
				{
					int index = j*width+i;
					image[j][i][0] = (output_buff[index] >> 16) & 0xff;
					image[j][i][1] = (output_buff[index] >> 8) & 0xff;
					image[j][i][2] = (output_buff[index]) & 0xff;
				}

			string out_filename = imageName + "_SEEDS.png";

			Mat SuperpixelImage( height, width, CV_8UC3 );
			for( int y=0; y<height; y++ )
				for( int x=0; x<width; x++ )
				{
					SuperpixelImage.at<Vec3b>(y,x).val[0] = image[y][x][2];
					SuperpixelImage.at<Vec3b>(y,x).val[1] = image[y][x][1];
					SuperpixelImage.at<Vec3b>(y,x).val[2] = image[y][x][0];
				}
			imwrite( out_filename, SuperpixelImage );
			delete[] output_buff;
		}

		delete[] ubuff;		
		qx_freeu_3(image);
}

void ComputeSuperpixelMeanHSV(Mat srcImage, int* SuperpixelsLabel, int n_labels, Mat& meanHSV)
{
	Mat HSVImage;
	cvtColor(srcImage,HSVImage,CV_BGR2Lab);

	meanHSV = Mat(n_labels,3,CV_32FC1,Scalar(0));

	int* cnt = new int[n_labels];
	memset(cnt,0,n_labels*sizeof(int));

	for(int i = 0; i < srcImage.rows;i++)
	{
		for (int j = 0; j < srcImage.cols;j++)
		{
			int label_idx = SuperpixelsLabel[srcImage.cols*i + j];
			meanHSV.at<float>(label_idx,0) +=  (int)HSVImage.at<Vec3b>(i,j)[0];
			meanHSV.at<float>(label_idx,1) +=  (int)HSVImage.at<Vec3b>(i,j)[1];
			meanHSV.at<float>(label_idx,2) +=  (int)HSVImage.at<Vec3b>(i,j)[2];
			cnt[label_idx]++;
		}
	}

	for (int i = 0; i < n_labels; i++)
	{
		meanHSV.at<float>(i,0) /= cnt[i];
		meanHSV.at<float>(i,1) /= cnt[i];
		meanHSV.at<float>(i,2) /= cnt[i];
	}
}

void Compute2Dhistrogram(Mat meanHS, int Hbins, int Sbins, Mat& hs_map)
{
	int deltaH = floor(255.0 / Hbins);
	int deltaS = floor(255.0 / Sbins);
	hs_map = Mat(Hbins,Sbins,CV_32FC1,Scalar(0));
	for (int i = 0 ; i < meanHS.rows; i++)
	{
		int Hth = meanHS.at<float>(i,0) / deltaH;
		int Sth = meanHS.at<float>(i,1) / deltaS;
		hs_map.at<float>(Hth,Sth) ++;
	}
}

void visulizeHSMAP(Mat hs_map, int deltathd, string imageName)
{
	Size size_hs_map = hs_map.size();
	Mat visualHSMAP(Size(size_hs_map.width * 20, size_hs_map.height * 20),CV_8UC3,Scalar(0,0,0));

	for (int i = 0; i < visualHSMAP.rows; i++)
	{
		for (int j = 0; j< visualHSMAP.cols;j++)
		{
			int color_idx = hs_map.at<float>(i/20,j/20) / deltathd;
			if (color_idx >= 10)
			{
				color_idx = 9;
			}
			visualHSMAP.at<Vec3b>(i,j)[0] = colormap_jet[color_idx][0];
			visualHSMAP.at<Vec3b>(i,j)[1] = colormap_jet[color_idx][1];
			visualHSMAP.at<Vec3b>(i,j)[2] = colormap_jet[color_idx][2];
		}
	}
	imwrite(imageName + "_hsmap.jpg", visualHSMAP);
}

void ComputeHSMAPExtremeValue(Mat hs_map, vector<Vec2b> extreme_value)
{
	for (int i = 0; i < hs_map.rows; i++)
	{
		for (int j = 0; j < hs_map.cols;j++)
		{
			if (hs_map.at<float>(i,j) > hs_map.at<float>(i,j-1)
				&& hs_map.at<float>(i,j) > hs_map.at<float>(i,j+1)
				&& hs_map.at<float>(i,j) > hs_map.at<float>(i-1,j)
				&& hs_map.at<float>(i,j) > hs_map.at<float>(i+1,j))
			{
				extreme_value.push_back(Vec2b(i,j));
			}
		}
	}
}

void ShadowVegetationDetection(Mat srcImg, Mat& shadowMask, int& n_shadow,Mat& vegetationMask, int& n_vegetation)
{
	Mat LABImage;
	cvtColor(srcImg,LABImage,CV_BGR2Lab);
	Mat VegetationIndex(srcImg.size(),CV_32FC1);
	ComputeVegetationIndex(srcImg,VegetationIndex);
	EMSegmentation(LABImage,VegetationIndex,6,shadowMask,n_shadow,vegetationMask,n_vegetation);
}

void ShadowVegetationDetection(Mat srcImg, Mat samples, int* superpixelLabel, Mat& shadowMask, Mat& vegetationMask)
{
	Mat HSVImage;
	cvtColor(srcImg,HSVImage,CV_BGR2Lab);
	Mat VegetationIndex(srcImg.size(),CV_32FC1);
	ComputeVegetationIndex(srcImg,VegetationIndex);
	EMSegmentationForSuperpixel(HSVImage, samples, superpixelLabel, VegetationIndex,17,shadowMask,vegetationMask);
}

/**
* Create a sample vector out of RGB image
*/
Mat asSamplesVectors( Mat& img ) {
	Mat float_img;
	img.convertTo( float_img, CV_32F );

	Mat samples( img.rows * img.cols, 3, CV_32FC1 );

	/* Flatten  */
	int index = 0;
	for( int y = 0; y < img.rows; y++ ) {
		Vec3f * row = float_img.ptr<Vec3f>(y);
		for( int x = 0; x < img.cols; x++ )
			samples.at<Vec3f>(index++, 0) = row[x];
	}
	return samples;
}

/**
Perform segmentation (clustering) using EM algorithm
**/
void EMSegmentation( Mat image, Mat VegetationIndex, int no_of_clusters, Mat& ShadowMask, int& n_shadow, Mat& VegetationMask, int& n_vegetation)
{
	Mat samples = asSamplesVectors( image );

	float* meanIntensity = new float[no_of_clusters];
	memset(meanIntensity,0,no_of_clusters*sizeof(float));

	float* meanColorIndex = new float[no_of_clusters];
	memset(meanColorIndex,0,no_of_clusters*sizeof(float));

	int* cnt = new int[no_of_clusters];
	memset(cnt,0,no_of_clusters*sizeof(int));

	cout << "Starting EM training" << endl;
	TermCriteria term_criteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 0.1);

	EM em( no_of_clusters,1,term_criteria);
	em.train( samples );
	cout << "Finished training EM" << endl;

	vector<Mat> segmented;
	segmented.resize(no_of_clusters);
	for( int i = 0; i < no_of_clusters; i++ )
		segmented[i] = Mat::zeros( image.rows, image.cols, CV_8UC1 );

	int index = 0;
	for( int y = 0; y < image.rows; y++ ) {
		for( int x = 0; x < image.cols; x++ ) {
			int result = em.predict( samples.row(index++) )[1];
			segmented[result].at<uchar>(y, x) = 255;
			meanIntensity[result] +=  image.at<Vec3b>(y, x)[0]; 
			meanColorIndex[result] += VegetationIndex.at<float>(y,x); 
			cnt[result]++;
		}
	}

	// compute min mean Intensity
	float minIntensity = meanIntensity[0] / (float)cnt[0];
	int label_shadow = 0;
	for (int i = 1; i < no_of_clusters;i++)
	{
		meanIntensity[i] /= (float)cnt[i];
		if (meanIntensity[i] < minIntensity)
		{
			minIntensity = meanIntensity[i];
			label_shadow = i;
		}
	}

	// compute min mean color index
	float minColorIndex = meanColorIndex[0] / (float)cnt[0];
	int label_vegetation = 0;
	for (int i = 1; i < no_of_clusters;i++)
	{
		meanColorIndex[i] /= (float)cnt[i];
		if (meanColorIndex[i] > minColorIndex)
		{
			minColorIndex = meanColorIndex[i];
			label_vegetation = i;
		}
	}

	cout << "mean Intensity minval " << minIntensity << " shadow label "<< label_shadow << endl;
	cout << "mean color index  minval " << minColorIndex << " vegetation label "<< label_vegetation << endl;
	if (label_shadow == label_vegetation)
	{
		cout << "Warning! GMM components is not enough" <<endl;
	}

	ShadowMask = segmented[label_shadow];
	VegetationMask = segmented[label_vegetation];
	n_shadow = cnt[label_shadow];
	n_vegetation = cnt[label_vegetation];

	delete[] meanIntensity;
	delete [] meanColorIndex;
	delete[] cnt;
}

void EMSegmentationForSuperpixel( Mat image, Mat samples, int* superpixelLabel, Mat VegetationIndex, int no_of_clusters, Mat& ShadowMask, Mat& VegetationMask)
{
	float* meanIntensity = new float[no_of_clusters];
	memset(meanIntensity,0,no_of_clusters*sizeof(float));

	float* meanColorIndex = new float[no_of_clusters];
	memset(meanColorIndex,0,no_of_clusters*sizeof(float));

	int* cnt = new int[no_of_clusters];
	memset(cnt,0,no_of_clusters*sizeof(int));

	cout << "Starting EM training" << endl;
	TermCriteria term_criteria(TermCriteria::COUNT+TermCriteria::EPS, 300, 0.1);

	EM em( no_of_clusters,1,term_criteria);
	em.train( samples );
	cout << "Finished training EM" << endl;

	vector<Mat> segmented;
	segmented.resize(no_of_clusters);
	for( int i = 0; i < no_of_clusters; i++ )
		segmented[i] = Mat::zeros( image.size(), CV_8UC1 );

	int index = 0;
	for( int y = 0; y < image.rows; y++ ) {
		for( int x = 0; x < image.cols; x++ ) {
			int slabel = superpixelLabel[y*image.cols + x];
			int result = em.predict( samples.row(slabel) )[1];
			segmented[result].at<uchar>(y, x) = 255;
			meanIntensity[result] +=  image.at<Vec3b>(y, x)[0]; 
			meanColorIndex[result] += VegetationIndex.at<float>(y,x); 
			cnt[result]++;
		}
	}
	cout << meanColorIndex[0] <<endl;


	// compute min mean Intensity
	float minIntensity = meanIntensity[0] / (float)cnt[0];
	if(!cnt[0]) minIntensity = 1000;

	int label_shadow = 0;
	for (int i = 1; i < no_of_clusters;i++)
	{
		meanIntensity[i] /= (float)cnt[i];
		if (meanIntensity[i] < minIntensity)
		{
			minIntensity = meanIntensity[i];
			label_shadow = i;
		}
	}

	// compute min mean color index
	float minColorIndex = meanColorIndex[0] / (float)cnt[0];
	if(!cnt[0]) minColorIndex = 1000;
	cout << "minColorIndex_" << minColorIndex << " cnt[0]_" << cnt[0] << "meanColorIndex[0]_"<< meanColorIndex[0]<<endl;
	int label_vegetation = 0;
	for (int i = 1; i < no_of_clusters;i++)
	{
		meanColorIndex[i] /= (float)cnt[i];
		if (meanColorIndex[i] > minColorIndex)
		{
			minColorIndex = meanColorIndex[i];
			cout << "minColorIndex" << minColorIndex <<endl;
			label_vegetation = i;
		}
	}

	cout << "mean Intensity minval " << minIntensity << " shadow label "<< label_shadow << endl;
	cout << "mean color index  minval" << minColorIndex << " vegetation label "<< label_vegetation << endl;
	ShadowMask = segmented[label_shadow];
	VegetationMask = segmented[label_vegetation];

	delete[] meanIntensity;
	delete [] meanColorIndex;
	delete[] cnt;
}

void ComputeVegetationIndex(Mat srcImg, Mat& phi )
{
	for (int i = 0; i<srcImg.rows; i++)
	{
		for (int j = 0; j<srcImg.cols; j++)
		{
			float a = srcImg.at<Vec3b>(i,j)[1] - srcImg.at<Vec3b>(i,j)[0];
			float b = srcImg.at<Vec3b>(i,j)[1] + srcImg.at<Vec3b>(i,j)[0];
			phi.at<float>(i,j) = 4/PI * atan(a / b);
		}
	}
}

void VisualizationObject(Mat srcImage, Mat Mask, Vec3b color, string ImageName)
{
	for (int i = 0; i < srcImage.rows; i++)
	{
		for (int j = 0; j < srcImage.cols; j++)
		{
			if (Mask.at<uchar>(i,j) == 255)
			{
				srcImage.at<Vec3b>(i,j) = color;
			}
		}
	}
	imwrite(ImageName,srcImage);
}

Mat GetSamples(Mat srcImage, Mat Mask, int n_samples)
{
	// store samples 
	Mat Samples(n_samples,3,CV_32FC1,Scalar(0));

	int t = 0;
	for (int i = 0; i < srcImage.rows; i++)
	{
		for (int j = 0; j < srcImage.cols; j++)
		{
			if (Mask.at<uchar>(i,j) == 255)
			{
				Samples.at<Vec3f>(t++,0) = (Vec3f)srcImage.at<Vec3b>(i,j);
			}
		}
	}
	return Samples;
}

/* cluster smaples using kmeans
   return class label and number of each class
*/

void GetSampleClassLabelAndNumber(Mat samples,int k, Mat& classLabel, int* n_class)
{
	TermCriteria criteria;
	criteria.epsilon = 0.1; 
	criteria.maxCount = 100;
	kmeans(samples,k,classLabel,criteria,5,KMEANS_RANDOM_CENTERS);
	for (int i = 0; i < samples.rows; i++)
	{
		int la = classLabel.at<int>(i,0);
		n_class[la]++;
	}
}

/* divide original samples into k subMatrix
   
*/
void SplitkClasses(Mat samples, Mat classLabel, int k, Mat* k_sub_samples, int* n_class)
{
	for (int i = 0; i < k; i++)
	{
		k_sub_samples[i] = Mat(n_class[i],3,CV_32FC1,Scalar(0));
	}

	int *idx = new int[k];
	memset(idx,0,k*sizeof(int));
	for (int i = 0; i < samples.rows; i++)
	{
		int la = classLabel.at<int>(i,0);
		k_sub_samples[la].row(idx[la]++) = samples.row(i);
	}
}
/* compute mean and cov for Mat
*/
void ComputeMeanCovMatrixWeight(int n_samples, Mat subsamples, Mat& meanMat, Mat& covMat, float& weight)
{
	for (int i = 0; i < subsamples.cols; i++)
	{
		meanMat.at<float>(i,0) = (float)sum(subsamples.col(i))[0] / subsamples.rows;
		subsamples.col(i) -= meanMat.at<float>(i,0);
	}
	covMat = 1/(subsamples.rows - 1) * subsamples.t() * subsamples;
	weight = subsamples.rows / n_samples;
}

void CalcInitialGMMParas(Mat srcImage, Mat Mask, int n_samples, int k, GMMComponentParas* Paras)
{
	Mat samples = GetSamples(srcImage,Mask,n_samples);
	
	Mat classLabel(n_samples,3,CV_16SC1,Scalar(0));
	int *n_class = new int[k];
	memset(n_class,0,k*sizeof(int));
	GetSampleClassLabelAndNumber(samples,k,classLabel,n_class);

	Mat *k_sub_samples = new Mat[k];
	SplitkClasses(samples, classLabel, k, k_sub_samples, n_class);

	for (int i = 0; i < k; i++)
	{
		ComputeMeanCovMatrixWeight(n_samples, k_sub_samples[i], Paras[i].u, Paras[i].cov, Paras[i].weight);
	}

	delete [] n_class;
	delete [] k_sub_samples;
}

void calcDataTerm(Mat srcImage, GMMComponentParas **NGMMParas, int n_masks, int gmm_k, float* dataTerm)
{
	for (int i = 0; i < n_masks; i++)
	{

	}
}

void calcSmoothTerm(Mat srcImage,  float* smoothTerm)
{
	for (int i = 0; i < n_masks; i++)
	{
		
	}
}

void MultilabelGraphCuts(Mat srcImage, int n_masks, Mat *ObjectMask, int* n_samples) 
{
	GMMComponentParas **NGMMParas = new GMMComponentParas *[n_masks];
	for (int i = 0; i < n_masks; i++)
	{
		NGMMParas[i] = new GMMComponentParas[5];
		for(int j = 0; j < 5; j++)
		{
			NGMMParas[i][j].u = Mat(3,1,CV_32FC1,Scalar(0));
			NGMMParas[i][j].cov = Mat(3,3,CV_32FC1,Scalar(0));
			NGMMParas[i][j].weight = 0;
		}
	}

	for (int i = 0; i < n_masks; i++)
	{
		CalcInitialGMMParas(srcImage,ObjectMask[i],n_samples[i],5,NGMMParas[i]);
	}
};


//void ComputeSuperpixelColorHistogram(Mat srcImage, int* superpixelLabel,  Mat& superpixelHistogram)
//{
//	Mat HSVImage;
//	cvtColor(srcImage,HSVImage,CV_BGR2HSV_FULL);
//
//	int n_labels = superpixelHistogram.rows;
//	int n_feature = superpixelHistogram.cols;
//	int n_ch_bins = n_feature / 3; 
//	int lbin = ceil(256.0 / n_ch_bins);
//	int *n_pixels = new int[n_labels];
//	memset(n_pixels,0,n_labels*sizeof(float));
//
//	for (int i = 0; i < HSVImage.rows; i++)
//	{
//		for (int j = 0; j < HSVImage.cols; j++)
//		{
//			int slabel = superpixelLabel[i * HSVImage.rows + j];
//			
//			int hth = HSVImage.at<Vec3b>(i,j)[0] / lbin;
//			superpixelHistogram.at<float>(slabel,hth)++;
//
//			int sth = HSVImage.at<Vec3b>(i,j)[1] / lbin;
//			superpixelHistogram.at<float>(slabel,sth + n_ch_bins)++;
//
//			int vth = HSVImage.at<Vec3b>(i,j)[2] / lbin;
//			superpixelHistogram.at<float>(slabel,vth + n_ch_bins*2)++;
//
//			n_pixels[slabel]++;
//		}
//	}
//	
//	for (int i = 0; i<n_labels;i++)
//	{
//		for (int j =0; j < n_feature; j++)
//		{
//			superpixelHistogram.at<float>(i,j) /= n_pixels[i];
//		}
//	}
//}
//
//void ComputeSimilarityMatrix(Mat superpixelHistogram, Mat& similarityMatrix)
//{
//	int n_samples = superpixelHistogram.rows;
//	for (int i = 0; i < n_samples; i++)
//	{
////   		cout << superpixelHistogram.row(i) <<endl;
//
//		for (int j = 0; j < n_samples; j++)
//		{
////   			cout <<superpixelHistogram.row(j) <<endl;
// 			float sim = compareHist(superpixelHistogram.row(i),superpixelHistogram.row(j),0);
//
////  			cout <<"i_"<<i << " j_"<< j << " sim_" <<sim<<endl;
//			similarityMatrix.at<float>(i,j) = sim;
//		}
//	}
//}
//
//void SpectralClustering(Mat srcImage, int* superpixelLabel, int n_labels, int n_features, int max_k_eigenvalue, int n_cluster, Mat& bestLabel)
//{
//	Mat SuperpixelHistogram(n_labels,n_features,CV_32FC1,Scalar(0));
//	Mat SimilarityMatrix(n_labels,n_labels,CV_32FC1,Scalar(0));
//	ComputeSuperpixelColorHistogram(srcImage,superpixelLabel,SuperpixelHistogram);
//	ComputeSimilarityMatrix(SuperpixelHistogram,SimilarityMatrix);
//	
//	Mat D(n_labels,n_labels,CV_32FC1,Scalar(0));
//	for(int i = 0; i< n_labels;i++)
//		D.at<float>(i,i) = sum(SimilarityMatrix.row(i))[0];
//
//	Mat L = D - SimilarityMatrix;
//	Mat eigenVector;
//	Mat eigenValue;
//	eigen(L,eigenValue,eigenVector);
//
//	Mat samples(max_k_eigenvalue,n_labels,CV_32FC1);
//	for (int i = n_labels - 1; i >= n_labels - max_k_eigenvalue; i-- )
//	{
//		eigenVector.row(i).copyTo(samples.row(n_labels - 1 - i));
//	}
//	
//	TermCriteria criteria;
//	criteria.epsilon = 0.1;
//	criteria.maxCount = 100;
//
//	kmeans(samples.t(),n_cluster,bestLabel,criteria,5,KMEANS_RANDOM_CENTERS );
//}
//
//void VisualizeLabel(Mat srcImage,int* superpixelLabel,Mat bestLabel, string imageName)
//{
//	for (int i = 0; i<srcImage.rows; i++)
//	{
//		for (int j = 0; j < srcImage.cols; j++)
//		{
//			int slabel = superpixelLabel[i*srcImage.cols + j];
//			int color_idx = bestLabel.at<int>(slabel);
//			srcImage.at<Vec3b>(i,j)[2] = colormap_jet[color_idx][0];
//			srcImage.at<Vec3b>(i,j)[1] = colormap_jet[color_idx][1];
//			srcImage.at<Vec3b>(i,j)[0] = colormap_jet[color_idx][2];
//		//	cout <<"slabel_"<<slabel<<" color_idx_"<<color_idx<<endl;
//		}
//	}
//	imwrite("SpectralClustering_" + imageName + ".jpg",srcImage);
//}