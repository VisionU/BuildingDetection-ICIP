#include "ShadowVegetationDetection.h"
#include "mymultilabelgraphcut.h"
#include <cmath>

float transferMatrix[3][3] = 
{
	0.3, 0.58, 0.11,
	0.25, 0.25 -0.5,
	0.5, -0.5, 0 
};

float A[3][3] =
{
	27.07439, -22.80783,-1.806681,
	-5.646736, -7.722125, 12.86503,
	-4.163133, -4.579428, -4.576049
};
float B[3][3] = 
{
	0.9465229, 0.2946927, -0.1313419,
	-1.179179e-1, 9.929960e-1, 7.371554e-3,
	9.230461e-2, -4.645794e-2, 9.946464e-1
};

void getFiles( string path, vector<string>& files );
void convertToGaussianSpace(Mat srcImage, Mat TransferMatrix, Mat& dstImage);
void CvtPerceptionBasedColorSpace(Mat srcImage, Mat TMA, Mat TMB, Mat& dstImage);

void main()
{
	vector<string> srcImagePaths; 
	getFiles("srcImage\\",srcImagePaths);

	for (int it = 0; it < srcImagePaths.size(); it++)
	{
		Mat srcImg = imread(srcImagePaths[it].c_str());
		string ImageName;
		ImageName.assign(srcImagePaths[it].begin() + 10, srcImagePaths[it].end() - 4);

		int* SuperpixelsLabel = new int[srcImg.rows * srcImg.cols];
		int Nlabels = 0;
		Mat MeanHSV;

		double start = clock();

// 		Mat TMA(3,3,CV_32FC1,A);
// 		Mat TMB(3,3,CV_32FC1,B);
// 		Mat PerceptionImage(srcImg.size(),CV_8UC3);
// 		CvtPerceptionBasedColorSpace(srcImg,TMA,TMB,PerceptionImage);
// 		imwrite("PerceptionBasedColorSpaceImage"+ ImageName + ".jpg",PerceptionImage);

// 		Mat TransferMatrix(3,3,CV_32FC1,transferMatrix);
// 		Mat GaussianSpaceImage(srcImg.size(),CV_8UC3,Scalar(0,0,0));
// 		convertToGaussianSpace(srcImg,TransferMatrix,GaussianSpaceImage);
//  		imwrite("GaussianSpaceImage"+ ImageName + ".jpg",GaussianSpaceImage);

// 		DoSuperpixelbySeeds(srcImg,Size(2,3),4,SuperpixelsLabel,Nlabels,true,ImageName);
// 		ComputeSuperpixelMeanHSV(srcImg,SuperpixelsLabel,Nlabels,MeanHSV);	
// 
// 		Mat HSMAP;
// 		Compute2Dhistrogram(MeanHS,10,10,HSMAP);	
// 		visulizeHSMAP(HSMAP,4,ImageName);

//  	Mat bestLabel(Nlabels,1,CV_16SC1);
// 		SpectralClustering(srcImg,SuperpixelsLabel,Nlabels,60,Nlabels - 10,10,bestLabel);
// 		VisualizeLabel(srcImg,SuperpixelsLabel,bestLabel,ImageName);

// 		Mat ShadowMask;
// 		Mat VegetationMask;
// 		int N_Shadow = 0;
// 		int N_Vegetation = 0;
// 		ShadowVegetationDetection(srcImg,MeanHSV,SuperpixelsLabel,ShadowMask,VegetationMask); //

// 		ShadowMask = imread("ShadowMask" + ImageName + ".bmp",0);
// 		VegetationMask = imread("VegetationMask" + ImageName +".bmp",0);	

// 		Mat MaskImage(ShadowMask.size(),CV_8UC1);
//  		for (int i = 0; i<srcImg.rows; i++)
//  		{
//  			for (int j = 0; j < srcImg.cols; j++)
//  			{
//  				if (ShadowMask.at<uchar>(i,j) == 255)
//  				{
//  					MaskImage.at<uchar>(i,j) = 0;
//  				}
//  				else if (VegetationMask.at<uchar>(i,j) == 255)
//  				{
//  					MaskImage.at<uchar>(i,j) = 1;
//  				}
//  				else
//  					MaskImage.at<uchar>(i,j) = 2;
//  			}
//  		}
// 
//  		mymultilabelgraphcut mygraphcut;
//  		mygraphcut.MultiLabelGraphCut(srcImg,MaskImage,3,5,ImageName);

//  		Mat *ObjectMask = new Mat[3];
//  		ObjectMask[0] = ShadowMask;
//  		ObjectMask[1] = VegetationMask;
//  		ObjectMask[2] = 255 - (ShadowMask + VegetationMask);
//  
//  		int *n_samples = new int[3];
//  		n_samples[0] = N_Shadow;
//  		n_samples[1] = N_Vegetation;
//  		n_samples[2] = srcImg.rows * srcImg.cols - N_Shadow - N_Vegetation;
// 
// 		string shadowImageName = "shadowImage\\shadowImage_" + ImageName + ".jpg";
// 		string vegetationImageName = "vegetationImage\\vegetationImage_" + ImageName + ".jpg";
// 		VisualizationObject(srcImg,ShadowMask,Vec3b(255,0,0),shadowImageName);
// 		VisualizationObject(srcImg,VegetationMask,Vec3b(0,255,0),vegetationImageName);

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

void convertToGaussianSpace(Mat srcImage, Mat TransferMatrix, Mat& dstImage)
{
	int ROWS = srcImage.rows;
	int COLS = srcImage.cols;
	Mat TempC1 (ROWS,COLS,CV_32FC1);
	Mat TempC2 (ROWS,COLS,CV_32FC1);

	vector<Mat> dstC;
	split(dstImage,dstC);

	float tempc1_max = 0;
	float tempc2_max = 0;

	for (int i = 0; i < ROWS; i++)
	{
		for (int j = 0; j < COLS; j++)
		{
			Mat tmp(3,1,CV_32FC1);
			tmp.at<float>(0,0) = (float)srcImage.at<Vec3b>(i,j)[2];
			tmp.at<float>(1,0) = (float)srcImage.at<Vec3b>(i,j)[1];
			tmp.at<float>(2,0) = (float)srcImage.at<Vec3b>(i,j)[0];

			tmp = TransferMatrix * tmp;

			if (tmp.at<float>(0,0) != 0)
			{
				TempC1.at<float>(i,j) = tmp.at<float>(1,0) / tmp.at<float>(0,0);
				TempC2.at<float>(i,j) = tmp.at<float>(2,0) / tmp.at<float>(0,0);
			}
			else
			{
				TempC1.at<float>(i,j) = 0;
				TempC2.at<float>(i,j) = 0;
			}
		}
	}

	normalize(TempC1,dstC[1],0,255,NORM_MINMAX,CV_8UC1);
	normalize(TempC2,dstC[2],0,255,NORM_MINMAX,CV_8UC1);
	merge(dstC, dstImage);
}

void CvtPerceptionBasedColorSpace(Mat srcImage, Mat TMA, Mat TMB, Mat& dstImage)
{
	int ROWS = srcImage.rows;
	int COLS = srcImage.cols;

	Mat TempMat(ROWS,COLS,CV_32FC3);

	for (int i = 0; i < ROWS; i++)
	{
		for (int j = 0; j < COLS; j++)
		{
			Mat tmp(3,1,CV_32FC1);
			tmp.at<float>(0,0) = (float)srcImage.at<Vec3b>(i,j)[2];
			tmp.at<float>(1,0) = (float)srcImage.at<Vec3b>(i,j)[1];
			tmp.at<float>(2,0) = (float)srcImage.at<Vec3b>(i,j)[0];
			
			Mat t = TMB * tmp;
			t.at<float>(0,0) = log(t.at<float>(0,0));
			t.at<float>(1,0) = log(t.at<float>(1,0));
			t.at<float>(2,0) = log(t.at<float>(2,0));

			TempMat.at<Vec3f>(i,j)[0] = t.at<float>(0,0);
			TempMat.at<Vec3f>(i,j)[1] = t.at<float>(1,0);
			TempMat.at<Vec3f>(i,j)[2] = t.at<float>(2,0);
		}
	}
	TempMat.copyTo(dstImage);
	normalize(TempMat,dstImage,0,255,NORM_MINMAX,CV_8UC3);
// 	imshow("dstImage",dstImage);
// 	waitKey(0);
}