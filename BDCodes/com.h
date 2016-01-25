#pragma  once
#include <io.h> 
#include <iostream>
#include "highgui.h"
#include "cv.h"
using namespace std;
using namespace cv;
typedef  unsigned int UINT;

#define PI 3.1415926

// const uchar colormap_jet[5][3]= 
// {
// 	255,      0,      0,  // red
// 	255,    255,      0,  // yellow
// 	0,    255,      0,  // green
// 	0,      0,    255,  // blue
// 	0,    255,    255,  // cyan
// };

const uchar colormap_jet[10][3]= 
{	0, 0,170,
	0, 0,255,
	0,85,255,
	0,170,255,
	0,255,255,
	85,255,170,
	170,255,85,
	255,255, 0,
	255,170, 0,
	255,85, 0
};

// const uchar colormap_jet[15][3]= 
// {
// 	255,      0,    255,  // magenta
// 	165,     42,     42,  // brown
// 	255,      0,      0,  // red
// 	238,     99,     99,  // indian red
// 	255,    193,    193,  // rocybrown
// 	184,    134,     11,  // dark goldenrod
// 	255,    255,      0,  // yellow
// 	255,    215,      0,  // gold
// 	0,    255,      0,  // green
// 	144,    238,    144,  // palegreen
// 	25,     25,    112,  // midnightblue
// 	0,      0,    255,  // blue
// 	0,    255,    255,  // cyan
// 	100,    149,    237,  // cornflowerblue
// 	220,    220,    220   // gainsboro
// };
