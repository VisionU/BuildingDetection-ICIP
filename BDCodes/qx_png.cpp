#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <gdiplus.h>
#include <iostream>
#include "atlbase.h"
#include "atlstr.h"
#include "comutil.h"
#include <comdef.h>
#include "qx_png.h"
using namespace Gdiplus;
#pragma comment(lib, "gdiplus.lib")

static bool gGDIplusInit = false;

static GdiplusStartupInput gdiplusStartupInput;
static ULONG_PTR gdiplusToken;

typedef struct {
    UINT Flags;
    UINT Count;
    ARGB Entries[256];
} GrayColorPalette;


GrayColorPalette GCP;

void startGDIplus()
{
	if(GdiplusStartup(&gdiplusToken,&gdiplusStartupInput,NULL)==Ok) 
	{
		gGDIplusInit = true;
	} 
	else
	{
		return;
	}
	GCP.Flags = PaletteFlagsGrayScale;// initialize the color palette for grayscale images;
	GCP.Count = 256;
	for(int i=0;i<256;i++) 
	{
		GCP.Entries[i]=Color::MakeARGB(255,i,i,i);
	}
}

void shutDownGDIplus() {
	if (gGDIplusInit)
		GdiplusShutdown(gdiplusToken);
}


int GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
{
   UINT  num = 0;          // number of image encoders
   UINT  size = 0;         // size of the image encoder array in bytes

   ImageCodecInfo* pImageCodecInfo = NULL;

   GetImageEncodersSize(&num, &size);
   if(size == 0)
      return -1;  // Failure

   pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
   if(pImageCodecInfo == NULL)
      return -1;  // Failure

   GetImageEncoders(num, size, pImageCodecInfo);

   for(UINT j = 0; j < num; ++j)
   {
      if( wcscmp(pImageCodecInfo[j].MimeType, format) == 0 )
      {
         *pClsid = pImageCodecInfo[j].Clsid;
         free(pImageCodecInfo);
         return j;  // Success
      }    
   }

   free(pImageCodecInfo);
   return -1;  // Failure
}

bool qx_image_size_png(const char*filename,int &h,int &w)
{
	size_t origsize=strlen(filename)+1;
	const size_t newsize=500;
	size_t convertedChars=0;
	wchar_t wcstring[newsize]; 
	_bstr_t bstrt(filename);
	wcscpy(wcstring,(wchar_t*)bstrt);
	if(!gGDIplusInit) startGDIplus();
	Bitmap*image=new Bitmap(wcstring);
	if(image->GetLastStatus()!= Ok)
	{
		std::cout << "cvLoadImage:: Unable to load" <<  wcstring << "\n";
		h=w=0;
		delete image;
		return(false);
	}
	w =image->GetWidth();
	h =image->GetHeight();
	delete image;
	return(true);
}
bool qx_loadimage_png(const char*filename,unsigned char*image,int h,int w,bool use_rgb)
{
	size_t origsize=strlen(filename)+1;
	const size_t newsize=500;
	size_t convertedChars=0;
	wchar_t wcstring[newsize]; 
	_bstr_t bstrt(filename);
	wcscpy(wcstring,(wchar_t*)bstrt);
	if(!gGDIplusInit) startGDIplus();
	Bitmap*image_read=new Bitmap(wcstring);
	if(image_read->GetLastStatus()!= Ok)
	{
		std::cout << "cvLoadImage:: Unable to load" <<  wcstring << "\n";
		h=w=0;
		delete image_read;
		return(false);
	}
	BitmapData* bitmapData = new BitmapData;
	Rect rect(0, 0, w, h);
	// Lock the entire bitmap for reading.
	Status rst;
	if((rst=image_read->LockBits(&rect,ImageLockModeRead,PixelFormat24bppRGB,bitmapData))!= Ok)
	{
		printf("error locking bits in %s, %d\n", __FILE__, __LINE__);
		delete bitmapData;
		delete image_read;
		return(false);
	}

   
	UINT* pixels = (UINT*)bitmapData->Scan0;
	byte*p=(byte*)pixels;
	int nOffset=bitmapData->Stride-w*3;
	if(use_rgb)
	{
	   for(int y=0;y<h;y++)
	   {
		   for(int x=0;x<w;x++)
		   {
			   unsigned char b=*p++;
			   unsigned char g=*p++;
			   unsigned char r=*p++;
			   *image++=r; *image++=g; *image++=b;
		   }
		   p+=nOffset;
	   }
	} 
	else
	{
	   for(int y=0;y<h;y++)
	   {
		   for(int x=0;x<w;x++)
		   {
			   unsigned char b=*p++;
			   unsigned char g=*p++;
			   unsigned char r=*p++;
			   *image++=unsigned char(0.299*r+0.587*g+0.114*b+0.5);
		   }
		   p+=nOffset;
	   }
	}
	image_read->UnlockBits(bitmapData);
	delete bitmapData;
	delete image_read;
	return(true);
}

bool qx_loadimage_png(const char*filename,unsigned short*image,int h,int w,bool use_rgb)
{
	size_t origsize=strlen(filename)+1;
	const size_t newsize=500;
	size_t convertedChars=0;
	wchar_t wcstring[newsize]; 
	_bstr_t bstrt(filename);
	wcscpy(wcstring,(wchar_t*)bstrt);
	if(!gGDIplusInit) startGDIplus();
	Bitmap*image_read=new Bitmap(wcstring);
	if(image_read->GetLastStatus()!= Ok)
	{
		std::cout << "cvLoadImage:: Unable to load" <<  wcstring << "\n";
		h=w=0;
		delete image_read;
		return(false);
	}
	BitmapData* bitmapData = new BitmapData;
	Rect rect(0, 0, w, h);
	// Lock the entire bitmap for reading.
	Status rst;
	if((rst=image_read->LockBits(&rect,ImageLockModeRead,PixelFormat48bppRGB,bitmapData))!= Ok)
	{
		printf("error locking bits in %s, %d\n", __FILE__, __LINE__);
		delete bitmapData;
		delete image_read;
		return(false);
	}

   
	UINT* pixels = (UINT*)bitmapData->Scan0;
	unsigned short*p=(unsigned short*)pixels;
	int nOffset=bitmapData->Stride-w*2*3;
	if(use_rgb)
	{
	   for(int y=0;y<h;y++)
	   {
		   for(int x=0;x<w;x++)
		   {
			   unsigned short b=*p++;
			   unsigned short g=*p++;
			   unsigned short r=*p++;
			   *image++=r; *image++=g; *image++=b;
		   }
		   p+=nOffset;
	   }
	} 
	else
	{
	   for(int y=0;y<h;y++)
	   {
		   //printf("[%d]\n",y);
		   for(int x=0;x<w;x++)
		   {
			   unsigned short b=*p++;
			   unsigned short g=*p++;
			   unsigned short r=*p++;
			   *image++=unsigned short(0.299*r+0.587*g+0.114*b+0.5);
		   }
		   p+=nOffset;
	   }
	}
	image_read->UnlockBits(bitmapData);
	delete bitmapData;
	delete image_read;
	return(true);
}


qx_png::qx_png(int h,int w,int nr_channel,int nr_byte)
{
	if(h>0&&w>0&&nr_channel>0&&nr_byte>0) init(h,w,nr_channel,nr_byte);
}
qx_png::~qx_png()
{
	clean();
}
void qx_png::clean()
{
	delete [] m_image; m_image=NULL;
}
int qx_png::init(int h,int w,int nr_channel,int nr_byte)
{
	m_h=h; m_w_in=w; m_w=(((m_w_in+3)>>2)<<2); m_nr_channel=nr_channel; m_nr_byte=nr_byte;
	m_image=new unsigned char[m_h*m_w*m_nr_channel*m_nr_byte];
	memset(m_image,0,sizeof(char)*m_h*m_w*m_nr_channel*m_nr_byte);
	return(0);
}
bool qx_png::write(const char*filename,unsigned char*image)
{
	if(!gGDIplusInit) startGDIplus();
	size_t origsize=strlen(filename) + 1;// Convert to a wchar_t*
	size_t convertedChars=0;
	const size_t newsize=1024;
	
	//mbstowcs(wcstring,filename,100);
	//printf("%s\n",filename);

	DWORD dwNum=MultiByteToWideChar(CP_ACP,0,filename,-1,NULL,0);
	wchar_t*wcstring=new wchar_t[dwNum];
	if(!wcstring) delete[]wcstring;
	MultiByteToWideChar (CP_ACP,0,filename,-1,wcstring,dwNum );


	//wchar_t wcstring[newsize];
	//MultiByteToWideChar(CP_ACP,0,filename,strlen(filename),wcstring,newsize);
	//printf("%s\n",wcstring);
	BitmapData bitmapData;
	bitmapData.Height=m_h; bitmapData.Width=m_w;
	bitmapData.Stride=m_nr_channel*m_w;
	int nOffset=bitmapData.Stride-m_w_in*m_nr_channel;
	unsigned char*pt,*p;
	pt=m_image; p=image;
	memset(pt,255,sizeof(char)*m_h*m_w*m_nr_channel);
	for(int y=0;y<m_h;y++) 
	{
		for(int x=0;x<m_w_in;x++) 
		{
			unsigned char r,g,b,a;
			switch(m_nr_channel)
			{
				case 1:
					*pt++=*p++;
					break;
				case 3:
					r=*p++; g=*p++; b=*p++;
					*pt++=b; *pt++=g; *pt++=r;
					break;
				case 4:
					r=*p++; g=*p++; b=*p++; a=*p++;
					*pt++=a; *pt++=b; *pt++=g; *pt++=r;
					break;
				default:
					return(false);
					break;
			}
		}
		pt+=nOffset;
	}
   switch(m_nr_channel)
   {
   case 1:
	   bitmapData.PixelFormat = PixelFormat8bppIndexed;
	   break;
   case 3:
	   bitmapData.PixelFormat = PixelFormat24bppRGB;
	   break;
   case 4:
	bitmapData.PixelFormat = PixelFormat32bppARGB; 
	   break;
   default:
	   printf("only 1, 3, or 4 channel images are supported\n");
	   return(false);
	   break;
   }
   bitmapData.Scan0=(VOID*)m_image;
   bitmapData.Reserved=NULL;
   Bitmap bitmap(bitmapData.Width,bitmapData.Height,bitmapData.Stride,bitmapData.PixelFormat,(BYTE*)bitmapData.Scan0);
   if(m_nr_channel==1) bitmap.SetPalette((ColorPalette*)&GCP);
   CLSID encoderClsid;
   if(strstr(filename,"jpg"))
   {
		//ImageCodecInfo ici=GetImageCodec( "image/jpg" );
		//EncoderParameters eps=new EncoderParameters(1);
		//eps[0]=new EncoderParameter(Encoder.Quality,100); // or whatever other quality value you want
		//bmp.Save( yourStream, ici, eps );
		GetEncoderClsid(L"image/jpeg",&encoderClsid);
   }
   else if(strstr(filename,"bmp"))
   {
	   GetEncoderClsid(L"image/bmp",&encoderClsid);
   }
   else
   {
	   GetEncoderClsid(L"image/png",&encoderClsid);
   }
   if(bitmap.Save(wcstring,&encoderClsid,NULL) == Ok) return(true);
   else 
   {
	   printf("fail in final step! [%d, %d][%d: %s]\n",m_h,m_w,strlen(filename),filename);
	   return(false);
   }
   if(!wcstring) delete[]wcstring;
}

bool qx_png::write(const char*filename,unsigned short*image)
{
	if(!gGDIplusInit) startGDIplus();
	size_t origsize=strlen(filename) + 1;// Convert to a wchar_t*
	size_t convertedChars=0;
	const size_t newsize=1024;
	
	//mbstowcs(wcstring,filename,100);
	//printf("%s\n",filename);

	DWORD dwNum=MultiByteToWideChar(CP_ACP,0,filename,-1,NULL,0);
	wchar_t*wcstring=new wchar_t[dwNum];
	if(!wcstring) delete[]wcstring;
	MultiByteToWideChar (CP_ACP,0,filename,-1,wcstring,dwNum );


	//wchar_t wcstring[newsize];
	//MultiByteToWideChar(CP_ACP,0,filename,strlen(filename),wcstring,newsize);
	//printf("%s\n",wcstring);
	BitmapData bitmapData;
	bitmapData.Height=m_h*m_nr_byte; bitmapData.Width=m_w;
	bitmapData.Stride=m_nr_channel*m_w;
	int nOffset=bitmapData.Stride-m_w_in*m_nr_channel;
	unsigned char*pt,*pt2;
	unsigned short*p;
	pt=m_image; pt2=&(m_image[m_h*m_w*m_nr_channel]); p=image; 
	memset(pt,255,sizeof(char)*m_h*m_w*m_nr_channel*m_nr_byte);
	for(int y=0;y<m_h;y++) 
	{
		for(int x=0;x<m_w_in;x++) 
		{
			unsigned short r,g,b,a;
			switch(m_nr_channel)
			{
				case 1:
					r=*p++;
					*pt++=r&0xff;
					*pt2++=(r>>8)&0xff;
					break;
				case 3:
					r=*p++; g=*p++; b=*p++;
					*pt++=b&0xff; *pt++=g&0xff; *pt++=r&0xff;
					*pt2++=(b>>8)&0xff; *pt2++=(g>>8)&0xff; *pt2++=(r>>8)&0xff;
					break;
				case 4:
					r=*p++; g=*p++; b=*p++; a=*p++;
					*pt++=a&0xff; *pt++=b&0xff; *pt++=g&0xff; *pt++=r&0xff;
					*pt2++=(a>>8)&0xff; *pt2++=(b>>8)&0xff; *pt2++=(g>>8)&0xff; *pt2++=(r>>8)&0xff;
					break;
				default:
					return(false);
					break;
			}
		}
		pt+=nOffset;
	}
   switch(m_nr_channel)
   {
   case 1:
	   bitmapData.PixelFormat = PixelFormat8bppIndexed;
	   break;
   case 3:
	   bitmapData.PixelFormat = PixelFormat24bppRGB;
	   break;
   case 4:
	bitmapData.PixelFormat = PixelFormat32bppARGB; 
	   break;
   default:
	   printf("only 1, 3, or 4 channel images are supported\n");
	   return(false);
	   break;
   }
   bitmapData.Scan0=(VOID*)m_image;
   bitmapData.Reserved=NULL;
   Bitmap bitmap(bitmapData.Width,bitmapData.Height,bitmapData.Stride,bitmapData.PixelFormat,(BYTE*)bitmapData.Scan0);
   if(m_nr_channel==1) bitmap.SetPalette((ColorPalette*)&GCP);
   CLSID encoderClsid;
   if(strstr(filename,"jpg"))
   {
		//ImageCodecInfo ici=GetImageCodec( "image/jpg" );
		//EncoderParameters eps=new EncoderParameters(1);
		//eps[0]=new EncoderParameter(Encoder.Quality,100); // or whatever other quality value you want
		//bmp.Save( yourStream, ici, eps );
		GetEncoderClsid(L"image/jpeg",&encoderClsid);
   }
   else if(strstr(filename,"bmp"))
   {
	   GetEncoderClsid(L"image/bmp",&encoderClsid);
   }
   else
   {
	   GetEncoderClsid(L"image/png",&encoderClsid);
   }
   if(bitmap.Save(wcstring,&encoderClsid,NULL) == Ok) return(true);
   else 
   {
	   printf("fail in final step! [%d, %d][%d: %s]\n",m_h,m_w,strlen(filename),filename);
	   return(false);
   }
   if(!wcstring) delete[]wcstring;
}
