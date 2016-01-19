#include "ShadowVegetationDetection.h"
#include "mymultilabelgraphcut.h"

float transferMatrix[3][3] = 
{
	0.3, 0.58, 0.11,
	0.25, 0.25 -0.05,
	0.5, -0.05, 0 
};

void getFiles( string path, vector<string>& files );

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
// 		ShadowVegetationDetection(srcImg,ShadowMask,N_Shadow,VegetationMask,N_Vegetation); //,MeanHSV,SuperpixelsLabel
		ShadowMask = imread("ShadowMask.jpg",0);
		VegetationMask = imread("VegetationMask.jpg",0);


		Mat MaskImage(ShadowMask.size(),CV_8UC1);
		for (int i = 0; i<srcImg.rows; i++)
		{
			for (int j = 0; j < srcImg.cols; j++)
			{
				if (ShadowMask.at<uchar>(i,j) == 255)
				{
					MaskImage.at<uchar>(i,j) = 0;
				}
				else if (VegetationMask.at<uchar>(i,j) == 255)
				{
					MaskImage.at<uchar>(i,j) = 1;
				}
				else
					MaskImage.at<uchar>(i,j) = 2;
			}
		}

		mymultilabelgraphcut mygraphcut;
		mygraphcut.MultiLabelGraphCut(srcImg,MaskImage,3,10);

// 		Mat *ObjectMask = new Mat[3];
// 		ObjectMask[0] = ShadowMask;
// 		ObjectMask[1] = VegetationMask;
// 		ObjectMask[2] = 255 - (ShadowMask + VegetationMask);
// 
// 		int *n_samples = new int[3];
// 		n_samples[0] = N_Shadow;
// 		n_samples[1] = N_Vegetation;
// 		n_samples[2] = srcImg.rows * srcImg.cols - N_Shadow - N_Vegetation;
// 
// 		MultilabelGraphCuts(srcImg, 3, ObjectMask, n_samples);
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


