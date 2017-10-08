#include <CImg.h>
#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <math.h>
#include <time.h>

using cimg_library::CImg;
using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

// Constants
const bool displayImages = false;
const bool saveAllImages = true;
const unsigned int HISTOGRAM_SIZE = 256;
const unsigned int BAR_WIDTH = 4;
const unsigned int CONTRAST_THRESHOLD = 80;
const float filter[] = {	1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 
						1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 
						1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

// Global variables for timers
NSTimer rgbCudaTime = NSTimer("rgbCudaTime", false, false), histCudaTime = NSTimer("histCudaTime", false, false), contCudaTime = NSTimer("contCudaTime", false, false), smootCudaTime = NSTimer("smootCudaTime", false, false);
NSTimer rgbSeqTime = NSTimer("rgbSeqTime", false, false), histSeqTime = NSTimer("histSeqTime", false, false), contSeqTime = NSTimer("contSeqTime", false, false), smootSeqTime = NSTimer("smootSeqTime", false, false);

// rgb2gray
extern void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height);
extern unsigned char* rgb2grayCuda(unsigned char *inputImage, const int width, const int height, unsigned char *outcudagrayImage);

// histogram1D
extern void histogram1D(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int BAR_WIDTH);
extern unsigned int* histogram1DCuda(unsigned char *cudagrayImage,const int width, const int height,  const unsigned int HISTOGRAM_SIZE, const unsigned int BAR_WIDTH, unsigned int *outcudahistogram);

// contrast1D
extern void contrast1D(unsigned char *grayImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int CONTRAST_THRESHOLD);
void contrast1DCuda(unsigned char *cudagrayImage, unsigned int *cudahistogram, const int width, const int height, const unsigned int HISTOGRAM_SIZE, const unsigned int CONTRAST_THRESHOLD, unsigned char *outcudagrayImage );

// triangularSmooth
extern void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height, const float *filter);
void triangularSmoothCuda( unsigned char* outcudasmoothImage, unsigned char *cudagrayImage, const float filter[], const int width, const int height );


int main(int argc, char *argv[]) 
{
	if ( argc != 2 ) {
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1; }
    
	// Load the input image
	CImg< unsigned char > inputImage = CImg< unsigned char >(argv[1]);
	if ( displayImages ) {
		inputImage.display("Input Image"); }
	if ( inputImage.spectrum() != 3 ) {
		cerr << "The input must be a color image." << endl;
		return 1; }
    cout << "Kernel Runtimes:" << endl;
    
    /*****************************************************
     Convert the input image to grayscale
     ****************************************************/
	CImg< unsigned char > grayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);
    // Structure for the gray image computed in the GPU and accessible in the main function. Will be used after for contrast enhanced gray image
    CImg< unsigned char > outcudagrayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);

	rgb2gray(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height() );
    
    // Returns pointer for place in global memory where the gray image was stored, to decrease overhead of host to device copying in between kernel calls
	unsigned char *cudagrayImage = rgb2grayCuda(inputImage.data(), inputImage.width(), inputImage.height(), outcudagrayImage.data() );
    
    if ( displayImages ){
        grayImage.display("Grayscale Image");
        //outcudagrayImage.display("Grayscale Image");
    }
    if ( saveAllImages ){
        grayImage.save("./grayscale.bmp");
        //outcudagrayImage.save("./cudagrayscale.bmp");
    }
	
    
    
    /*****************************************************
     Compute 1D histogram
     ****************************************************/
	CImg< unsigned char > histogramImage = CImg< unsigned char >(BAR_WIDTH * HISTOGRAM_SIZE, HISTOGRAM_SIZE, 1, 1);
    
	unsigned int *histogram = new unsigned int [HISTOGRAM_SIZE];
    // Structure for the histogram data computed in the GPU and accessible in the main function
    unsigned int *outcudahistogram = new unsigned int [HISTOGRAM_SIZE];
    
	histogram1D(grayImage.data(), histogramImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, BAR_WIDTH);
    
    // Returns pointer for place in global memory where the histogram data was stored, to decrease overhead of host to device copying in between kernel calls
    unsigned int *cudahistogram = histogram1DCuda( cudagrayImage, grayImage.width(), grayImage.height(), HISTOGRAM_SIZE, BAR_WIDTH , outcudahistogram);
    
	if ( displayImages ) histogramImage.display("Histogram");
	if ( saveAllImages ) histogramImage.save("./histogram.bmp");
    
    /*
     // Cycle used to verify the calculated histogram data
     int a=0, i;
     for( i=0; i< HISTOGRAM_SIZE; i++){
        if( outcudahistogram[i] != histogram[i] ){
            a++;
            printf("%d %d\n", outcudahistogram[i], histogram[i]);
        }
     }
     printf("\nNumber of mismatched bins on the calculated histogram: %d\n\n", a);
    */
    
    
    
    /*****************************************************
     Contrast enhancement
     ****************************************************/
	contrast1D( grayImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, CONTRAST_THRESHOLD );
    contrast1DCuda( cudagrayImage, outcudahistogram, grayImage.width(), grayImage.height(), HISTOGRAM_SIZE, CONTRAST_THRESHOLD, outcudagrayImage.data() );

	if ( displayImages ) {
		grayImage.display("Contrast Enhanced Image");
        //outcudagrayImage.display("Contrast Enhanced Cuda Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./contrast.bmp");
        //outcudagrayImage.save("./cudacontrast.bmp");
	}
    
    // Deleting auxiliary data structures
	delete [] histogram;
    delete [] outcudahistogram;

    
    
    /*****************************************************
     Triangular smooth (convolution)
     ****************************************************/
	CImg< unsigned char > smoothImage = CImg< unsigned char >(grayImage.width(), grayImage.height(), 1, 1);
    // Structure for the final smooth image computed in the GPU and accessible in the main function
    CImg< unsigned char > outcudasmoothImage = CImg< unsigned char >(grayImage.width(), grayImage.height(), 1, 1);

	triangularSmooth(grayImage.data(), smoothImage.data(), grayImage.width(), grayImage.height(), filter);
    triangularSmoothCuda( outcudasmoothImage.data(), cudagrayImage, filter, grayImage.width(), grayImage.height() );
    
	if ( displayImages ) {
		smoothImage.display("Smooth Image");
        outcudasmoothImage.display("Cuda Smooth Image");
	}
	
	if ( saveAllImages ) {
		smoothImage.save("./smooth.bmp");
        outcudasmoothImage.save("./cudasmooth.bmp");
	}
    
    cout << fixed << setprecision(6);
    cout << "Total CPU Execution Time:  " << rgbSeqTime.getElapsed() + histSeqTime.getElapsed() + contSeqTime.getElapsed() + smootSeqTime.getElapsed() << " seconds." << endl;
    cout << "Total CUDA Execution Time:  " << rgbCudaTime.getElapsed() + histCudaTime.getElapsed() + contCudaTime.getElapsed() + smootCudaTime.getElapsed() << " seconds." << endl;
    
	return 0;
}
