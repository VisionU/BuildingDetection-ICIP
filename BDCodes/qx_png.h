#ifndef QX_PNG_H
#define QX_PNG_H
bool qx_image_size_png(const char*filename,int &h,int &w);
bool qx_loadimage_png(const char*filename,unsigned char*image,int h,int w,bool use_rgb);//h,w need to be known!!
bool qx_loadimage_png(const char*filename,unsigned short*image,int h,int w,bool use_rgb);
class qx_png
{
public:
	qx_png(int h=0,int w=0,int nr_channel=0,int nr_byte=1);//nr_channel=1,3,4; nr_byte=2 for unsigned short images
	~qx_png();
	void clean();
	int init(int h,int w,int nr_channel,int nr_byte);
	bool write(const char*filename,unsigned char*image);
	bool write(const char*filename,unsigned short*image);
private:
	int m_h,m_w,m_w_in,m_nr_channel,m_nr_byte; unsigned char*m_image;
};
#endif