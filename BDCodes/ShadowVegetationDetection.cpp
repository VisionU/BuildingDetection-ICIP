#include "ShadowVegetationDetection.h"

extern Vec3b colorMap[10] =
{
	Vec3b(255,0,0),
	Vec3b(0,255,0),
	Vec3b(0,255,255),
	Vec3b(255,255,0),
	Vec3b(0,0,255),
	Vec3b(255,0,255),
	Vec3b(0,0,0),
	Vec3b(255,255,255),
	Vec3b(255,255,155),
	Vec3b(255,155,255)
};  
// detecte shadaw and vegetation from each pixel

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

void ComputeVegetationIndex(Mat srcImg, Mat& phi )
{
	for (int i = 0; i<srcImg.rows; i++)
	{
		for (int j = 0; j<srcImg.cols; j++)
		{
			float a = srcImg.at<Vec3b>(i,j)[1] - srcImg.at<Vec3b>(i,j)[0];
			float b = srcImg.at<Vec3b>(i,j)[1] + srcImg.at<Vec3b>(i,j)[0];
			if (b != 0)
			{
				phi.at<float>(i,j) = 4/PI * atan(a / b);
			}
			else 
				phi.at<float>(i,j) = 0;
		}
	}
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
	TermCriteria term_criteria(TermCriteria::COUNT+TermCriteria::EPS, 300, 0.001);

	EM em( no_of_clusters,1,term_criteria);
	em.train( samples );
	cout << "Finished training EM" << endl;

	vector<Mat> segmented;
	segmented.resize(no_of_clusters);
	for( int i = 0; i < no_of_clusters; i++ )
		segmented[i] = Mat::zeros( image.rows, image.cols, CV_8UC1 );

	int index = 0;
	Mat MskImage(image.size(),CV_8UC3);
	for( int y = 0; y < image.rows; y++ ) {
		for( int x = 0; x < image.cols; x++ ) {
			int result = em.predict( samples.row(index++) )[1];
			segmented[result].at<uchar>(y, x) = 255;

			MskImage.at<Vec3b>(y,x)[0] = colormap_jet[result][2];
			MskImage.at<Vec3b>(y,x)[1] = colormap_jet[result][1];
			MskImage.at<Vec3b>(y,x)[2] = colormap_jet[result][0];

			meanIntensity[result] +=  image.at<Vec3b>(y, x)[0]; 
			meanColorIndex[result] += VegetationIndex.at<float>(y,x); 
			cnt[result]++;
		}
	}

	imwrite("mskImage.jpg",MskImage);

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
// 
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

void ShadowVegetationDetection(Mat srcImg, Mat& shadowMask, int& n_shadow,Mat& vegetationMask, int& n_vegetation)
{
	Mat LABImage;
	cvtColor(srcImg,LABImage,CV_BGR2Lab);
	Mat VegetationIndex(srcImg.size(),CV_32FC1);
	ComputeVegetationIndex(srcImg,VegetationIndex);
	EMSegmentation(LABImage,VegetationIndex,10,shadowMask,n_shadow,vegetationMask,n_vegetation);
}

void VisualizationObject(Mat& srcImage, Mat Mask, Vec3b color, string ImageName)
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


// Superpixel Related

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

void DrawContoursAroundSegments(UINT* img, int* labels, const int& width, const int& height, const UINT& color, bool internal)
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	int sz = width*height;

	vector<bool> istaken(sz, false);

	int mainindex(0);
	int cind(0);
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			int np(0);
			for( int i = 0; i < 8; i++ )
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if( (x >= 0 && x < width) && (y >= 0 && y < height) )
				{
					int index = y*width + x;

					if (internal)
					{
						{
							if( labels[mainindex] != labels[index] ) np++;
						}
					} else {
						if( false == istaken[index] )   //comment this to obtain internal contours
						{
							if( labels[mainindex] != labels[index] ) np++;
						}
					}
				}
			}
			if( np > 2 )
			{
				istaken[mainindex] = true;
				img[mainindex] = color;
				cind++;
			}
			mainindex++;
		}
	}
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

void EMSegmentationForSuperpixel( Mat image, Mat samples, int* superpixelLabel, Mat VegetationIndex, int no_of_clusters, Mat& ShadowMask, Mat& VegetationMask)
{
	float* meanIntensity = new float[no_of_clusters];
	memset(meanIntensity,0,no_of_clusters*sizeof(float));

	float* meanColorIndex = new float[no_of_clusters];
	memset(meanColorIndex,0,no_of_clusters*sizeof(float));

	int* cnt = new int[no_of_clusters];
	memset(cnt,0,no_of_clusters*sizeof(int));

	cout << "Starting EM training" << endl;
	TermCriteria term_criteria(TermCriteria::COUNT+TermCriteria::EPS, 500, 0.001);

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
	cout << "mean color index  minval" << minColorIndex << " vegetation label "<< label_vegetation << endl;
	ShadowMask = segmented[label_shadow];
	VegetationMask = segmented[label_vegetation];

	delete[] meanIntensity;
	delete [] meanColorIndex;
	delete[] cnt;
}

void ShadowVegetationDetection(Mat srcImg, Mat samples, int* superpixelLabel, Mat& shadowMask, Mat& vegetationMask)
{
	Mat HSVImage;
	cvtColor(srcImg,HSVImage,CV_BGR2Lab);
	Mat VegetationIndex(srcImg.size(),CV_32FC1);
	ComputeVegetationIndex(srcImg,VegetationIndex);
	EMSegmentationForSuperpixel(HSVImage, samples, superpixelLabel, VegetationIndex,10,shadowMask,vegetationMask);
}


// compute extreme for 2D histogram

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
