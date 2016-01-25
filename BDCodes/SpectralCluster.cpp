
//void SpectralClustering(Mat srcImage, int* superpixelLabel, int n_labels, int n_features, int max_k_eigenvalue, int n_cluster, Mat& bestLabel);

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