#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

// Global variables for timers
extern NSTimer rgbCudaTime, histCudaTime, contCudaTime, smootCudaTime;
extern NSTimer rgbSeqTime, histSeqTime, contSeqTime, smootSeqTime;

/* Utility function/macro, used to do error checking.
Use this function/macro like this:
checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
And to check the result of a kernel invocation:
checkCudaCall(cudaGetLastError());
*/

#define checkCudaCall(result) {                                     \
    if (result != cudaSuccess){                                     \
        cerr << "cuda error: " << cudaGetErrorString(result);       \
        cerr << " in " << __FILE__ << " at line "<< __LINE__<<endl; \
        exit(1);                                                    \
    }                                                               \
}


/*****************************************************
Kernel Function for rgb2gray
****************************************************/
__global__
void rgb2grayCudaKernel(unsigned char *cudainputImage, unsigned char *cudagrayImage, const int width, const int height){

    // Identifies ID of each thread. One thread is run for each pixel in the image, so the ID is used to determine which pixel the kernel should operate with
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Because of the way we determine the number of blocks and threads to be run (to optimize performance) and the fact the images dont have the same dimensions, means that for the last block, some threads might not do usefull work. The if condition garantees that image position's that don't exist aren't acessed by those extra threads
    if( i < width*height ){
        // Accessing red, green and blue values for the same pixel
        float r = static_cast< float >(cudainputImage[i] );
        float g = static_cast< float >(cudainputImage[(width * height) + i] );
        float b = static_cast< float >(cudainputImage[(2 * width * height) + i] );

        // Formula used to calculate gray value
        int grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

        // Saving result of calculation. Because of the choosen implementation, no race-conditions can be raised, so no synchronizing (or use of atomic operations) was required
        cudagrayImage[i] = static_cast< unsigned char >( grayPix );
    }
}

/*****************************************************
Cuda implementation of rgb2gray funtion
****************************************************/
unsigned char* rgb2grayCuda(unsigned char *inputImage, const int width, const int height, unsigned char *grayImage){

    //----------------------------------
    // STEP 1: Create host buffers
    //----------------------------------
    // No host buffers were required

    //----------------------------------
    // STEP 2: Create device side memories
    //----------------------------------
    unsigned char *cudainputImage, *cudagrayImage;
    
    // Allocating space in the glocal memory for the inputImage. This is the only device side data struture that will not be carried to subsequent kernel calls
    if( cudaMalloc( (void**)&cudainputImage, sizeof(unsigned char)*width*height*3 ) != cudaSuccess ){
        printf("Error creating device side input image.\n");
        exit(0);
    }
    
    // Allocating space in the glocal memory for the computed grayImage. The memory for this data structure will not be freed in the end for use in subsequent CUDA functions
    if( cudaMalloc( (void**)&cudagrayImage, sizeof(unsigned char)*width*height ) != cudaSuccess ){
        printf("Error creating device side gray image data structure.\n");
        exit(0);
    }

    rgbCudaTime.start();
    // Copying input Image from host to device
    if( cudaMemcpy( cudainputImage, inputImage, sizeof(unsigned char)*width*height*3, cudaMemcpyHostToDevice) != cudaSuccess ){
        printf("Error copying to device the input image.\n");
        exit(0);
    }

    //----------------------------------
    // STEP 3: Calling the kernel
    //----------------------------------
    dim3 Db, Dg;
    NSTimer kernelTime = NSTimer("kernelTime", false, false);

    // Db - dimension of block (Db.x * Db.y * Db.z equals the number of threads per block). Because used card has Compute Capability 3.5, we can only have 2048 resident threads per SM and 16 resident blocks per SM, so we used the minimum size of block (128 threads) that garanties maximum thread utilization per SM. This way an unexpected slowdown in the processing of a block would garanty the computing of the other blocks as fast as possible, without hurting performance in a more realistic scenario (as we put equal strain in all threads and so don't expect particular slowdowns from any of them).
    Db.x = 128;
    
    // Dg - dimension of grid (Dg.x * Dg.y * Dg.z equals the number of blocks being launched). We compute the number of blocks we will require to garanty that one thread per block will be available
    Dg.x = (width*height)/128;
    
    // Because image sizes aren't necessarily multiple of 128, our last block will be requested even if we dont have necessity for all 128 threads it contains. As the defined number of threads per block isn't very high, this will not negatively affect the performance of the program
    if ( (width*height)%128 != 0 ) Dg.x ++;

    kernelTime.start();
    rgb2grayCudaKernel<<< Dg , Db >>>(cudainputImage, cudagrayImage, width, height);

    // Makes sure all the kernels finished executing before stoping the timer. This way we get an accurate representation of the run time
    cudaDeviceSynchronize();

    kernelTime.stop();
    cout << fixed << setprecision(6);
    cout << "(GPU):  " << kernelTime.getElapsed() << " seconds." << endl;

    //----------------------------------
    // STEP 4: Copying result back to host
    //----------------------------------
    // Copying computed image back to host to be able to visualize it
    /*if( cudaMemcpy( grayImage, cudagrayImage, sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost) != cudaSuccess ){
        printf("Error copying to host the resulting gray scale image.\n");
        exit(0);
    }*/

    rgbCudaTime.stop();

    //----------------------------------
    // STEP 5: Free memory
    //----------------------------------
    // The data struture allocated in global memory for the input image will no longer be necessary for computation in the GPU, so it is freed
    cudaFree(cudainputImage);
    
    // Pointer for global memory position of grayImage is returned so subsequent kernel calls can use it, removing aditional overhead associated with copying it back from host to device
    return cudagrayImage;
}

/*****************************************************
Sequential implementation of rgb2gray funtion
****************************************************/
void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height){

    rgbSeqTime.start();
    // Kernel
    for ( int y = 0; y < height; y++ ){
        for ( int x = 0; x < width; x++ ){
            float grayPix = 0.0f;
            float r = static_cast< float >(inputImage[(y * width) + x]);
            float g = static_cast< float >(inputImage[(width * height) + (y * width) + x]);
            float b = static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

            grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

            grayImage[(y * width) + x] = static_cast< unsigned char >(grayPix);
        }
    }
    // /Kernel
    rgbSeqTime.stop();

    cout << fixed << setprecision(6);
    cout << "rgb2gray (cpu): \t" << rgbSeqTime.getElapsed() << " seconds   ";
}





/*****************************************************
Kernel Function for histogram1D
****************************************************/
__global__ void histogram1DCudaKernel( unsigned char *cudagrayImage, unsigned int *cudahistogram, const int width, const int height) {

    // Identifies ID of each thread. One thread is run for each pixel in the image, so the ID is used to determine which pixel the kernel should operate with
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Because of the way we determine the number of blocks and threads to be run some threads from the last block might not do usefull work. If condition prevents incorrect memory access from those threads
    if( i < width*height ){
        // From Compute Capability 2.0 devices on, atomic adds in global memory are extremely fast because they are serviced by the L2 cache, rather than the global memory directly. Whats more, if the locations that are addressed in parallel map to completely different atomic units (which we found from testing the performance that they do) they will be performed in parallel (as opposed to in sequence when mapped to the same atomic units).
        atomicAdd( &cudahistogram[static_cast< unsigned int >( cudagrayImage[i] )], 1);
        
        // Finally with Kepler atomic performance is high enough (10x faster for worst case and 2x faster for best case, compared to Fermi) that we decided to implement atomics rather than some kind of parallel redution code
    }

// Alternative way to using atomic operations for every incrementing of a value on the histogram. In this approach we use sub-histograms for every block of threads in shared memory. We then compute sub-histograms in each block with atomic operations to prevent race-conditions in between threads. In the end all sub-histograms get added together to form the final complete histogram. Because of improvements in compute capability that we explain above, this would not bring us any improvements (sometimes we measured penalties even), so we choose the first implementation
/*
    __shared__ unsigned int temp[256];
    temp[ threadIdx.x ] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while (i < height*width){
        atomicAdd(&cudahistogram[static_cast< unsigned int >( cudagrayImage[ i ])], 1);
        i += offset;
    }

    __syncthreads();
    atomicAdd( &(cudahistogram[ threadIdx.x ]), temp[ threadIdx.x ] );
*/
}

/*****************************************************
Cuda implementation of histogram funtion
****************************************************/
unsigned int* histogram1DCuda( unsigned char *cudagrayImage, const int width, const int height, const unsigned int HISTOGRAM_SIZE, const unsigned int BAR_WIDTH, unsigned int *outcudahistogram ) {

    //----------------------------------
    // STEP 1: Create host buffers
    //----------------------------------
    // No host buffers necessary

    //----------------------------------
    // STEP 2: Create device side memories
    //----------------------------------
    unsigned int *cudahistogram;

    // Allocates global memory for the histogram data structure
    if( cudaMalloc( (void**)&cudahistogram, sizeof(unsigned int)*HISTOGRAM_SIZE) != cudaSuccess ){
        printf("Error creating device side histogram data structure.\n");
        exit(0);
    }
    cudaMemset( cudahistogram , 0 , sizeof(unsigned int)*HISTOGRAM_SIZE );

    // Copying from host to device
    // No structures need to be copied as the histogram structure will be created in the kernel and the grayImage structure already is in the GPU's Global Memory

    //----------------------------------
    // STEP 3: Calling the kernel
    //----------------------------------
    histCudaTime.start();
    dim3 Db, Dg;
    NSTimer kernelTime = NSTimer("kernelTime", false, false);

    // Db - dimension of block. Because used card has Compute Capability 3.5, we can only have 2048 resident threads per SM and 16 resident blocks per SM, so we used the minimum size of block (128 threads) that garanties maximum thread utilization per SM. This way an unexpected slowdown in the processing of a block would garanty the computing of the other blocks as fast as possible, without hurting performance in a more realistic scenario (as we put equal strain in all threads and so don't expect particular slowdowns from any of them).
    Db.x = 128;
    
    // Dg - dimension of grid.  We compute the number of blocks we will require to garanty that one thread per block will be available
Dg.x = (width*height)/128;
    
    // Because image sizes aren't necessarily multiple of 128, our last block will be requested even if we dont have necessity for all 128 threads it contains. As the defined number of threads per block isn't very high, this will not negatively affect the performance of the program
    if ( (width*height)%128 != 0 ) Dg.x ++;

    kernelTime.start();
    histogram1DCudaKernel<<< Dg , Db >>>( cudagrayImage, cudahistogram, width, height );

    // Makes sure all the kernels finished executing before stoping the timer. This way we get an accurate representation of the run time
    cudaDeviceSynchronize();
    kernelTime.stop();



//  Kernel calls for alternative way to compute the histogram. In this approach we use sub-histograms for every block of threads in shared memory. A full description of this implementation can be found on the respective kernel above, were the reason for why we don't use it can be found.
/*
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0 );
    // As an histogram with 256 bins maps well into this thread per block value
    Db.x = 256;
    // If we are not concerned with precision, we can use the processor count to accelarate the execution
    Dg.x = prop.multiProcessorCount *2;
    kernelTime.start();
    histogram1DCudaKernel<<< Dg , Db , Db.x * sizeof(int) >>>( cudagrayImage, cudahistogram, width, height );
    cudaDeviceSynchronize();
    kernelTime.stop();
*/



    cout << fixed << setprecision(6);
    cout << "(GPU):  " << kernelTime.getElapsed() << " seconds." << endl;

    //----------------------------------
    // STEP 4: Copying result back to host
    //----------------------------------
    if( cudaMemcpy( outcudahistogram, cudahistogram, sizeof(unsigned int)*HISTOGRAM_SIZE, cudaMemcpyDeviceToHost) != cudaSuccess ){
        printf("Error copying to host the resulting gray scale image.\n");
        exit(0);
    }

    histCudaTime.stop();

    //----------------------------------
    // STEP 5: Free memory
    //----------------------------------
    // Pointer for global memory position of histogram is returned so subsequent kernel calls can use it, removing aditional overhead associated with copying it back from host to device
    return cudahistogram;
}


/*****************************************************
Sequential Function for histogram1D
****************************************************/
void histogram1D(unsigned char *grayImage, unsigned char *histogramImage,const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int BAR_WIDTH) {

    histSeqTime.start();
    unsigned int max = 0;
    NSTimer kernelTime = NSTimer("kernelTime", false, false);

    memset(reinterpret_cast< void * >(histogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));

    kernelTime.start();
    // Kernel
    for ( int y = 0; y < height; y++ ) {
        for ( int x = 0; x < width; x++ ) {
            histogram[static_cast< unsigned int >(grayImage[(y * width) + x])] += 1;}}
    // /Kernel
    kernelTime.stop();

    for ( unsigned int i = 0; i < HISTOGRAM_SIZE; i++ ) {
        if ( histogram[i] > max ) {
            max = histogram[i];}}

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
    histSeqTime.stop();

    cout << fixed << setprecision(6);
    cout << "histogram1D (cpu): \t" << kernelTime.getElapsed() << " seconds   ";
}










/*****************************************************
Kernel Function for contrast1D
****************************************************/
__global__ void contrast1DKernel( unsigned char *cudagrayImage, unsigned int max, unsigned int min, float diff, const int width, const int height){

    // Identifies ID of each thread. One thread is run for each pixel in the image, so the ID is used to determine which pixel the kernel should operate with
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Because of the way we determine the number of blocks and threads to be run some threads from the last block might not do usefull work. If condition prevents incorrect memory access from those threads
    if( i < width*height ){
        unsigned char pixel = cudagrayImage[ i ];
        
        // If the analized pixel scores lower than the lowest gray values in the histogram (within a certain threshold), it is set to black
        if ( pixel < min ){
            pixel = 0;}
        // If it scores higher than the highest values of gray, it is set to white
        else if ( pixel > max ){
            pixel = 255;}
        // Otherwise, the pixels are scalled, which can be done by:
        else{
            pixel = static_cast< unsigned char >(255.0f * (pixel - min) / diff);}

        cudagrayImage[ i ] = pixel;
    }
}

/*****************************************************
Cuda implementation of constrast1D function
****************************************************/
void contrast1DCuda(unsigned char *cudagrayImage, unsigned int *cudahistogram, const int width, const int height, const unsigned int HISTOGRAM_SIZE, const unsigned int CONTRAST_THRESHOLD, unsigned char *outcudagrayImage ){

    //----------------------------------
    // STEP 1: Create host buffers
    //----------------------------------

    //----------------------------------
    // STEP 2: Create device side memories
    //----------------------------------

    //----------------------------------
    // STEP 2.1: Determining min and max
    //----------------------------------
    contCudaTime.start();
    NSTimer kernelTime = NSTimer("kernelTime", false, false);

    // Computing the lowest gray value that has "scored" in the histogram above a certain threshold
    unsigned int i = 0;
    while ( (i < HISTOGRAM_SIZE) && ( cudahistogram[i] < CONTRAST_THRESHOLD) ) i++;
    unsigned int min = i;
    
    // Computing the highest gray value that has "scored" in the histogram above a certain threshold
    i = HISTOGRAM_SIZE - 1;
    while ( (i > min) && ( cudahistogram[i] < CONTRAST_THRESHOLD) ) i--;
    unsigned int max = i;
    float diff = max - min;

    
    //----------------------------------
    // STEP 3: Calling the kernel
    //----------------------------------
    dim3 Db, Dg;
    // Db - dimension of block. Because used card has Compute Capability 3.5, we can only have 2048 resident threads per SM and 16 resident blocks per SM, so we used the minimum size of block (128 threads) that garanties maximum thread utilization per SM. This way an unexpected slowdown in the processing of a block would garanty the computing of the other blocks as fast as possible, without hurting performance in a more realistic scenario (as we put equal strain in all threads and so don't expect particular slowdowns from any of them).
    Db.x = 128;
    
    // Dg - dimension of grid.  We compute the number of blocks we will require to garanty that one thread per block will be available
    Dg.x = (width*height)/128 ;
    
    // Because image sizes aren't necessarily multiple of 128, our last block will be requested even if we dont have necessity for all 128 threads it contains. As the defined number of threads per block isn't very high, this will not negatively affect the performance of the program
    if ( (width*height)%128 != 0 ) Dg.x ++;

    kernelTime.start();
    contrast1DKernel<<< Dg , Db >>>( cudagrayImage, max, min, diff, width, height );

    // Makes sure all the kernels finished executing before stoping the timer. This way we get an accurate representation of the run time
    cudaDeviceSynchronize();
    kernelTime.stop();

    cout << fixed << setprecision(6);
    cout << "(GPU):  " << kernelTime.getElapsed() << " seconds." << endl;


    //----------------------------------
    // STEP 4: Copying result back to host
    //----------------------------------
    /*if( cudaMemcpy( outcudagrayImage, cudagrayImage, sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost) != cudaSuccess ){
        printf("Error copying to host the resulting gray scale image.\n");
        exit(0);
    }*/

    contCudaTime.stop();
    //----------------------------------
    // STEP 5: Free memory
    //----------------------------------
    // The struture allocated in global memory for the histogram data will no longer be necessary for computation in the GPU, so it is freed
    cudaFree(cudahistogram);
    return;
}


/*****************************************************
Sequential Function for constrast1D
****************************************************/
void contrast1D(unsigned char *grayImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int CONTRAST_THRESHOLD){

    contSeqTime.start();
    unsigned int i = 0;
    NSTimer kernelTime = NSTimer("kernelTime", false, false);

    while ( (i < HISTOGRAM_SIZE) && (histogram[i] < CONTRAST_THRESHOLD) ){
        i++;
    }
    unsigned int min = i;

    i = HISTOGRAM_SIZE - 1;
    while ( (i > min) && (histogram[i] < CONTRAST_THRESHOLD) ){
        i--;
    }
    unsigned int max = i;
    float diff = max - min;

    kernelTime.start();
    // Kernel
    for ( int y = 0; y < height; y++ ){
        for (int x = 0; x < width; x++ ){
            unsigned char pixel = grayImage[(y * width) + x];

            if ( pixel < min ){
                pixel = 0;
            }
            else if ( pixel > max ){
                pixel = 255;
            }
            else{
                pixel = static_cast< unsigned char >(255.0f * (pixel - min) / diff);
            }
            grayImage[(y * width) + x] = pixel;
        }
    }

    // /Kernel
    kernelTime.stop();
    contSeqTime.stop();

    cout << fixed << setprecision(6);
    cout << "contrast1D (cpu): \t" << kernelTime.getElapsed() << " seconds   ";
}











/*****************************************************
Kernel Function for triangularSmooth
****************************************************/
__global__ void triangularSmoothKernel( unsigned char *cudasmoothImage, const float filter[], unsigned char *cudagrayImage, const int width, const int height ) {

    // Identifies ID of each thread. One thread is run for each pixel in the image, so the ID is used to determine which pixel the kernel should operate with
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Because of the way we determine the number of blocks and threads to be run some threads from the last block might not do usefull work. If condition prevents incorrect memory access from those threads
    if( i < width*height ){

        unsigned int filterItem = 0; float filterSum = 0.0f; float smoothPix = 0.0f;

        // Converting ID of thread/pixel in the pixel's x and y coordinates, which make it easier to compute the following algorithm
        int y = i/width;
        int x = i%width;

        // The two for loops garanty that all elements inside a 5x5 matrix centered in the analyzed pixel are "covered" by the algorithm
        for ( int fy = y - 2; fy < y + 3; fy++ ){
            for ( int fx = x - 2; fx < x + 3; fx++ ){
                // If the pixel is close to one of the image boarders, a 5x5 matrix cannot be centered on it. In that case, the pixels that "fall outside" the boarders are ignored by the filterItem variable is incremented so next pixels are computed with the correct filter value
                if ( ((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= width)) ){
                    filterItem++;
                    continue;
                }
                // SmoothPix value is incremented with the weighted (acording to their filter value) values of the pixels inside the 5x5 matrix
                smoothPix += cudagrayImage[(fy * width) + fx] * filter[ filterItem ];
                // FilterSum keeps track of the total weight to be able to calculate a weighted average after (can be diferent in each case if the "if clause" above is used
                filterSum += filter[filterItem];
                // Filter value to be used in the next iteration of the for loops
                filterItem++;
            }
        }
        // Calculating final value of the smooth pixel by computing weighted average
        smoothPix /= filterSum;
        // Storing value
        cudasmoothImage[ i ] = static_cast< unsigned char >(smoothPix);

    }
}



/*****************************************************
Cuda implementation of triangularSmooth function
****************************************************/
void triangularSmoothCuda( unsigned char* outcudasmoothImage, unsigned char *cudagrayImage, const float filter[], const int width, const int height ) {

    //----------------------------------
    // STEP 1: Create host buffers
    //----------------------------------

    //----------------------------------
    // STEP 2: Create device side memories
    //----------------------------------
    float* cudafilter;
    unsigned char *cudasmoothImage;

    // Allocating, in global memory, space for the filter used, so its values can be accessed by the kernels
    if( cudaMalloc( (void**)&cudafilter, sizeof( float )*25) != cudaSuccess ){
        printf("Error creating device side histogram data structure.\n");
        exit(0);
    }

    // Allocating space in the glocal memory for the computed smoothImage
    if( cudaMalloc( (void**)&cudasmoothImage, sizeof(unsigned char)*width*height) != cudaSuccess ){
        printf("Error creating device side histogram data structure.\n");
        exit(0);
    }

    smootCudaTime.start();
    // Filling filter values
    if( cudaMemcpy( cudafilter, filter, sizeof( float )*25, cudaMemcpyHostToDevice) != cudaSuccess ){
        printf("Error copying to device the input image.\n");
        exit(0);
    }

    //----------------------------------
    // STEP 3: Calling the kernel
    //----------------------------------
    NSTimer kernelTime = NSTimer("kernelTime", false, false);

    dim3 Db, Dg;
    
    // Db - dimension of block. Because used card has Compute Capability 3.5, we can only have 2048 resident threads per SM and 16 resident blocks per SM, so we used the minimum size of block (128 threads) that garanties maximum thread utilization per SM. This way an unexpected slowdown in the processing of a block would garanty the computing of the other blocks as fast as possible, without hurting performance in a more realistic scenario (as we put equal strain in all threads and so don't expect particular slowdowns from any of them).
    Db.x = 128;
    
    // Dg - dimension of grid.  We compute the number of blocks we will require to garanty that one thread per block will be available
    Dg.x = (width*height)/128 ;
    
    // Because image sizes aren't necessarily multiple of 128, our last block will be requested even if we dont have necessity for all 128 threads it contains. As the defined number of threads per block isn't very high, this will not negatively affect the performance of the program
    if ( (width*height)%128 != 0 ) Dg.x ++;

    kernelTime.start();
    triangularSmoothKernel<<<Dg, Db >>>( cudasmoothImage, cudafilter, cudagrayImage, width, height );

    // Makes sure all the kernels finished executing before stoping the timer. This way we get an accurate representation of the run time
    cudaDeviceSynchronize();
    kernelTime.stop();

    cout << fixed << setprecision(6);
    cout << "(GPU):  " << kernelTime.getElapsed() << " seconds." << endl << endl;

    //----------------------------------
    // STEP 4: Copying result back to host
    //----------------------------------
    // Copying computed smooth image back to host
    if( cudaMemcpy( outcudasmoothImage, cudasmoothImage, sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost) != cudaSuccess ){
        printf("Error copying to host the resulting gray scale image.\n");
        exit(0);
    }

    smootCudaTime.stop();
    //----------------------------------
    // STEP 5: Free memory
    //----------------------------------
    // All data types allocated on global memory can now be freed as no further computations will be performed with them in the GPU
    cudaFree(cudafilter);
    cudaFree(cudasmoothImage);
    cudaFree(cudagrayImage);
    return;
}



/*****************************************************
Sequential Function for triangularSmooth
****************************************************/
void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height, const float *filter){

    smootSeqTime.start();
    // Kernel
    for ( int y = 0; y < height; y++ ){
        for ( int x = 0; x < width; x++ ){
            unsigned int filterItem = 0;
            float filterSum = 0.0f;
            float smoothPix = 0.0f;

            for ( int fy = y - 2; fy < y + 3; fy++ ){
                for ( int fx = x - 2; fx < x + 3; fx++ ){
                    if ( ((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= width)) ){
                        filterItem++;
                        continue;
                    }

                    smoothPix += grayImage[(fy * width) + fx] * filter[filterItem];
                    filterSum += filter[filterItem];
                    filterItem++;
                }
            }

            smoothPix /= filterSum;
            smoothImage[(y * width) + x] = static_cast< unsigned char >(smoothPix);
        }
    }
    // /Kernel
    smootSeqTime.stop();

    cout << fixed << setprecision(6);
    cout << "triangularSmooth (cpu): " << smootSeqTime.getElapsed() << " seconds   ";
}
