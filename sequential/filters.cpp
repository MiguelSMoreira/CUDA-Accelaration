#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;



/*****************************************************
 Converting color image into grayscale
 Converts a color image to grayscale, by weighing, for
 each pixel, the Red, Green and Blue components of each
 pixel.
 ****************************************************/
void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height,
			  NSTimer &timer) 
{
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	kernelTime.start();
    
    
    
	// Kernel
    // For each of the pixels in the input image data structure we perform
	for ( int y = 0; y < height; y++ ) 
	{
		for ( int x = 0; x < width; x++ ) 
		{
			float grayPix = 0.0f;
            // We get the red, blue and green value for the pixel (x,y) from the data structure
			float r = static_cast< float >(inputImage[(y * width) + x]);
			float g = static_cast< float >(inputImage[(width * height) + (y * width) + x]);
			float b = static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

            // We compute the value of the pixel in the grey scale and store it
			grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);
			grayImage[(y * width) + x] = static_cast< unsigned char >(grayPix);
		}
	}
	// /Kernel
    
    
    
	kernelTime.stop();
	cout << fixed << setprecision(6);
	cout << "rgb2gray (kernel): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}



/*****************************************************
 Histogram Computation
 Measures how often a value of gray is used in an image.
 The value of each pixel has to be counted and for each
corresponding gray scale value, its repective counter
is incremented in the "histogram" vector
 ****************************************************/
void histogram1D(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height, 
				unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int BAR_WIDTH, 
				NSTimer &timer) 
{
	unsigned int max = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	
	memset(reinterpret_cast< void * >(histogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));
	kernelTime.start();
    
    
    
	// Kernel
	for ( int y = 0; y < height; y++ ) {
		for ( int x = 0; x < width; x++ ) {
            // For each of the pixels in the gray image data structure, we increment the position of the histogram respective to its value by one
			histogram[static_cast< unsigned int >(grayImage[(y * width) + x])] += 1;
		}
	}
	// /Kernel
    
    
    
	kernelTime.stop();

    // Calculates the darkest value of grey used
	for ( unsigned int i = 0; i < HISTOGRAM_SIZE; i++ ) {
		if ( histogram[i] > max ) {
			max = histogram[i];
		}
	}
    
    // Computes the histogram of bar size "BAR_WIDTH" to a data struture named "histogramImage"
	for ( int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH ) {
		unsigned int value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

		for ( unsigned int y = 0; y < value; y++ ) {
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ ) {
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 0;
			}
		}
        
		for ( unsigned int y = value; y < HISTOGRAM_SIZE; y++ ) {
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ ) {
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 255;
			}
		}
	}
	
	cout << fixed << setprecision(6);
	cout << "histogram1D (kernel): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}



/*****************************************************
 Contrast enhancement
 Runs through entire grayImage structure and switches
all pixels whose gray value is lower than threshold "min"
to black (0) and above "max" to white (256).
Pixels inside threshold are scaled to the entire possible
values (between 0 and 255) using the formula
          (255.0f * (pixel - min) / diff)
 ****************************************************/
void contrast1D(unsigned char *grayImage, const int width, const int height, unsigned int *histogram, 
				const unsigned int HISTOGRAM_SIZE, const unsigned int CONTRAST_THRESHOLD, NSTimer &timer) 
{
	unsigned int i = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	while ( (i < HISTOGRAM_SIZE) && (histogram[i] < CONTRAST_THRESHOLD) ) {
		i++;
	}
	unsigned int min = i;

	i = HISTOGRAM_SIZE - 1;
	while ( (i > min) && (histogram[i] < CONTRAST_THRESHOLD) ) {
		i--;
	}
	unsigned int max = i;
	float diff = max - min;

	kernelTime.start();
    
    
    
	// Kernel
	for ( int y = 0; y < height; y++ ) {
		for (int x = 0; x < width; x++ ) {
			unsigned char pixel = grayImage[(y * width) + x];

            // If below the threshold, turn pixel to black
			if ( pixel < min ) {
				pixel = 0;
			}
            // If above the threshold, turn pixel to white
			else if ( pixel > max ) {
				pixel = 255;
			}
            // If inside the threshold, scale the pixel to the original allowed values for grayscale (between 0 and 255)
			else {
				pixel = static_cast< unsigned char >(255.0f * (pixel - min) / diff);
			}
			// Save calculated pixel to the gray scale image structure
			grayImage[(y * width) + x] = pixel;
		}
	}
	// /Kernel
    
    
    
	kernelTime.stop();
	
	cout << fixed << setprecision(6);
	cout << "contrast1D (kernel): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}



/*****************************************************
 Smoothing
 Removes noise from image. To do so, each point is replaced
by a weighter average of its neighbors, in order to remove
small scale structures from image.
 ****************************************************/
void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, 
					  const int width, const int height, const float *filter, NSTimer &timer) 
{
	NSTimer kernelTime = NSTimer("kernelTime", false, false);
	kernelTime.start();
    
    
    
	// Kernel
    // For each of the pixels in the grayscale image
	for ( int y = 0; y < height; y++ ) {
		for ( int x = 0; x < width; x++ ) {
            
			unsigned int filterItem = 0;
			float filterSum = 0.0f;
			float smoothPix = 0.0f;

            // For each pixel, the two pixels below, above, to the left and right of it are considered. In total 5*5 pixels (25) will be analized
			for ( int fy = y - 2; fy < y + 3; fy++ ) {
				for ( int fx = x - 2; fx < x + 3; fx++ ) {
                    
                    // If one of the pixels we expect to consider is outside the bounds of the image (doesn't exist) we continue to the next pixel to consider
					if ( ((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= width)) ) {
						filterItem++;
						continue;
					}

                    // The value of the smoothed pixel is incremented with the value of the analized pixel times the weighted value it will be given from the filter
					smoothPix += grayImage[(fy * width) + fx] * filter[filterItem];
                    // Calculated the total weight to do a weighted average
					filterSum += filter[filterItem];
                    // filterItem allows to search filter[] for the weight supposed to be given to the particular pixel which is being analized
					filterItem++;
				}
			}

            // Divide the calculated smooth pixel by the total weight that originated it, to get an average weighted value
			smoothPix /= filterSum;
            // The pixel is replaced by its smoothed version
			smoothImage[(y * width) + x] = static_cast< unsigned char >(smoothPix);
		}
	}
	// /Kernel
    
    
    
	kernelTime.stop();
	cout << fixed << setprecision(6);
	cout << "triangularSmooth (kernel): \t" << kernelTime.getElapsed() << " seconds." << endl;
}
