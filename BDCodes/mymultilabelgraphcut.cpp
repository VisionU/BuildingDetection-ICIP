#include "mymultilabelgraphcut.h"
#include "com.h"

// static int e_img_width;
// static int e_img_height;
// static double e_gamma;
// static double e_lambda;
// static double e_beta;
// static Mat e_Img;

static double calcBeta( const Mat& img );
static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, 
						   Mat& uprightW, double beta, double gamma );
static void initGMMs( const Mat& img, const Mat& mask, GMM* GMMmodels, int n_models);


mymultilabelgraphcut::mymultilabelgraphcut(void)
{
}

mymultilabelgraphcut::~mymultilabelgraphcut(void)
{
}

int smoothFn(int p1, int p2, int l1, int l2)
{
	int cost = 0;
	if (l1 == l2 ) return cost;

	int e_img_width = e_Img.cols;

	int r1 = p1 / e_img_width;
	int c1 = p1 % e_img_width;

	int r2 = p2 / e_img_width;
	int c2 = p2 % e_img_width;
	Vec3b diff = e_Img.at<Vec3b>(r1,c1) - e_Img.at<Vec3b>(r2,c2);

	cost = e_gamma * exp(-e_beta * diff.dot(diff));

	return cost;
}

int dataFn(int p, int l, void *data)
{
	ForDataFn *myData = (ForDataFn *) data;
	int numLab = myData->numLab;

	return( myData->data[p*numLab+l] );
}


void mymultilabelgraphcut::DoMultiLabelGraphCut(const Mat& img, const Mat& mask,GMM *GMMModels,
												int n_models,int iterCnt, Mat& dstImage )
{
	int num_pixels = img.rows * img.cols;
	int width = img.cols;
	int height = img.rows;

	int *data = new int[num_pixels*n_models];
	int *result = new int[num_pixels];   // stores result of optimization

	Point p;
	for( p.y = 0; p.y < img.rows; p.y++ )
	{
		for( p.x = 0; p.x < img.cols; p.x++)
		{
			Vec3b color = img.at<Vec3b>(p);
			int pixelIdx = p.y * img.cols + p.x;
			for (int l = 0; l < n_models; l++)
			{
			//	cout << "p_" << p << " l_" << l ;

			//	cout << GMMModels[l](color);

 				if (abs(GMMModels[l](color)) < 1e-5)
 				{
 					data[pixelIdx * n_models + l] = 100000;
 				//	cout << l <<"  "<< data[pixelIdx * n_models + l] << endl;
 				}
 				else 
					data[pixelIdx * n_models + l] = -log(GMMModels[l](color));			
			}
		}
	}	

	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,n_models);

		ForDataFn toFn;
		toFn.data = data;
		toFn.numLab = n_models;

		gc->setDataCost(&dataFn,&toFn);
 		gc->setSmoothCost(&smoothFn);
		printf("\nBefore optimization energy is %d\n",gc->compute_energy());
		gc->expansion(iterCnt);			// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	int t = 0;
	img.copyTo(dstImage);
	for( p.y = 0; p.y < img.rows; p.y++ )
	{
		for( p.x = 0; p.x < img.cols; p.x++)
		{
			int color_idx = result[t++];
			if (color_idx == 0)
			{
				dstImage.at<Vec3b>(p) = Vec3b(255,0,0);
			}
			else if (color_idx == 1)
			{
				dstImage.at<Vec3b>(p) = Vec3b(0,255,0);
			}
		}
	}
	delete [] result;
	delete [] data;
}


void mymultilabelgraphcut::MultiLabelGraphCut( InputArray _img, InputOutputArray _mask, int n_models, int iterCount,string Imamgename)
{
	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();

	if( img.empty() )
		CV_Error( CV_StsBadArg, "image is empty" );
	if( img.type() != CV_8UC3 )
		CV_Error( CV_StsBadArg, "image mush have CV_8UC3 type" );

	GMM *GMMModels = new GMM[n_models];
	for (int i = 0; i < n_models; i++)
	{
		GMMModels[i] = GMM(Mat());
	}

	initGMMs( img, mask, GMMModels, n_models );

	if( iterCount <= 0)
		return;

	e_gamma = 50;
	e_lambda = 9 * e_gamma;
	e_beta = calcBeta( img );
	e_Img = img;

	Mat SegmentedImage(img.size(),CV_8UC3);
	DoMultiLabelGraphCut(img,mask,GMMModels,n_models,iterCount,SegmentedImage);
	imwrite("SegmentedImage" + Imamgename + ".jpg",SegmentedImage);

	delete [] GMMModels;
 }


//计算beta，也就是Gibbs能量项中的第二项（平滑项）中的指数项的beta，用来调整
//高或者低对比度时，两个邻域像素的差别的影响的，例如在低对比度时，两个邻域
//像素的差别可能就会比较小，这时候需要乘以一个较大的beta来放大这个差别，
//在高对比度时，则需要缩小本身就比较大的差别。
//所以我们需要分析整幅图像的对比度来确定参数beta，具体的见论文公式（5）。
/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
static double calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
			//计算四个方向邻域两像素的差别，也就是欧式距离或者说二阶范数
			//（当所有像素都算完后，就相当于计算八邻域的像素差了）
            Vec3d color = img.at<Vec3b>(y,x);
            if( x>0 ) // left  >0的判断是为了避免在图像边界的时候还计算，导致越界
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);  //矩阵的点乘，也就是各个元素平方的和
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) ); //论文公式（5）

    return beta;
}

//计算图每个非端点顶点（也就是每个像素作为图的一个顶点，不包括源点s和汇点t）与邻域顶点
//的边的权值。由于是无向图，我们计算的是八邻域，那么对于一个顶点，我们计算四个方向就行，
//在其他的顶点计算的时候，会把剩余那四个方向的权值计算出来。这样整个图算完后，每个顶点
//与八邻域的顶点的边的权值就都计算出来了。
//这个相当于计算Gibbs能量的第二个能量项（平滑项），具体见论文中公式（4）
/*
  Calculate weights of noterminal vertices of graph.
  beta and gamma - parameters of GrabCut algorithm.
 */
static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, 
							Mat& uprightW, double beta, double gamma )
{
    //gammaDivSqrt2相当于公式（4）中的gamma * dis(i,j)^(-1)，那么可以知道，
	//当i和j是垂直或者水平关系时，dis(i,j)=1，当是对角关系时，dis(i,j)=sqrt(2.0f)。
	//具体计算时，看下面就明白了
	const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
	//每个方向的边的权值通过一个和图大小相等的Mat来保存
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x-1>=0 ) // left  //避免图的边界
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}


//通过k-means算法来初始化背景GMM和前景GMM模型
/*
  Initialize GMM background and foreground models using kmeans algorithm.
*/
static void initGMMs( const Mat& img, const Mat& mask, GMM* GMMmodels, int n_models)
{
    const int kMeansItCount = 10;  //迭代次数
    const int kMeansType = KMEANS_PP_CENTERS; //Use kmeans++ center initialization by Arthur and Vassilvitskii

    Mat *ModelLabels = new Mat[n_models];

    vector<vector<Vec3f>> Samples;
	Samples.resize(n_models);
   
	Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
			for (int l = 0; l < n_models; l++)
			{
				//mask中标记为li 的像素存入相应的samples
				if( mask.at<uchar>(p) == l)
					Samples[l].push_back( (Vec3f)img.at<Vec3b>(p) );
			}
		}
    }
	for (int l = 0; l < n_models; l++)
	{
		CV_Assert(!Samples[l].empty());
	}
	
	//kmeans中参数_Samples为：每行一个样本
	//kmeans的输出为Labels，里面保存的是输入样本集中每一个样本对应的类标签（样本聚为componentsCount类后）
	for (int l = 0; l < n_models; l++)
	{
		Mat _Samples( (int)Samples[l].size(), 3, CV_32FC1, &Samples[l][0][0] );
		kmeans( _Samples, GMM::componentsCount, ModelLabels[l],
			TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
	}

    //经过上面的步骤后，每个像素所属的高斯模型就确定的了，那么就可以估计GMM中每个高斯模型的参数了。
	for (int l = 0; l < n_models; l++)
	{
		GMMmodels[l].initLearning();
		for( int i = 0; i < (int)Samples[l].size(); i++ )
			GMMmodels[l].addSample( ModelLabels[l].at<int>(i,0), Samples[l][i] );
		GMMmodels[l].endLearning();
	}
}

//论文中：迭代最小化算法step 1：为每个像素分配GMM中所属的高斯模型，kn保存在Mat compIdxs中
/*
  Assign GMMs components for each pixel.
*/
static void assignGMMsComponents( const Mat& img, const Mat& mask, GMM* GMMmodels, int n_models, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec3d color = img.at<Vec3b>(p);
			int label = mask.at<uchar>(p);
			compIdxs.at<int>(p) = GMMmodels[label].whichComponent(color);
        }
    }
}

//论文中：迭代最小化算法step 2：从每个高斯模型的像素样本集中学习每个高斯模型的参数
/*
  Learn GMMs parameters.
*/
static void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM* GMMmodels, int n_models )
{
	for (int l = 0; l < n_models; l++)
	{
		GMMmodels[l].initLearning();
	}

    Point p;
    for( int ci = 0; ci < GMM::componentsCount; ci++ )
    {
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    int label = mask.at<uchar>(p);
                        GMMmodels[label].addSample( ci, img.at<Vec3b>(p) );
                }
            }
        }
    }

	for (int l = 0; l < n_models; l++)
	{
		GMMmodels[l].endLearning();
	}    
}

