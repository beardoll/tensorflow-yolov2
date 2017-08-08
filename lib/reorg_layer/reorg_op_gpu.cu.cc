#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "reorg_op_gpu.h"

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
        i += blockDim.x * gridDim.x)

using namespace tensorflow;

__global__ void ReorgForward(const int nthreads, const int data_height, const int data_width, const int num_channels, 
        const int stride, const float *bottom_data, float *output) {
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        /* 
         * loop the entire input feature maps
         * the index is indicated by "index"
         */
        
        // compute the coords "b, h, w, c" under data format (b, h, w, c)
        int n = index;
        int c = n % num_channels;
        n /= num_channels;
        int w = n % data_width;
        n /= data_width;
        int h = n % data_height;
        n /= data_height;
        int b = n;

        // out_c is hard to explain ...
        int out_c = num_channels / (stride * stride);

        // the index of the output under data format (b, c, h, w)
        int in_index = w + data_width * (h + data_height * (c + b * num_channels));

        // convert in_index from format (b, c, h, w) to format (b, h, w, c)
        // the converted index is for output
        // first we should extract the coords "b, h, w, c" under data format (b, c, h, w)
        int real_out_c = num_channels * stride * stride;   // output
        int real_out_w = data_width / stride;              // output
        int real_out_h = data_height / stride;             // output
        int n1 = in_index;
        int w1 = n1 % real_out_w;
        n1 /= real_out_w;
        int h1 = n1 % real_out_h;
        n1 /= real_out_h;
        int c1 = n1 % real_out_c;
        n1 /= real_out_c;
        int b1 = n1;

        int real_out_index = c1 + real_out_c * (w1 + real_out_w * (h1 + real_out_h * b1));

        // compute the corresponding index in input
        int c2 = c % out_c;
        int offset = c / out_c;
        int w2 = w * stride + offset % stride;
        int h2 = h * stride + offset / stride;
        // the index for data format (b, c, h, w)
        int out_index = w2 + data_width * stride * (h2 + data_height * stride * (c2 + b*out_c));

        // get the coords "b, h, w, c" for out_index under format (b, c, h, w)
        int n3 = out_index;
        int w3 = n3 % data_width;
        n3 /= data_width;
        int h3 = n3 % data_height;
        n3 /= data_height;
        int c3 = n3 % num_channels;
        n3 /= num_channels;
        int b3 = n;

        // get the index under format (b, h, w, c)
        int real_in_index = c3 + num_channels * (w3 + data_width * (h3 + data_height * b3));

        output[real_out_index] = bottom_data[real_in_index];
    }
}

bool ReorgForwardLauncher(const int data_height, const int data_width, const int num_channels, const int batch_size, 
         const int stride, const float *bottom_data, float *output, const Eigen::GpuDevice &d)
{
    const int kThreadsPerBlock = 1024;   // threads per block
    cudaError_t err;
    int output_size = batch_size * data_height * data_width * num_channels; // the input and output has the same len

    ReorgForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>
        (output_size, data_height, data_width, num_channels, stride, bottom_data, output);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    return d.ok();
}

__global__ void ReorgBackward(const int nthreads, const int data_height, const int data_width, const int num_channels,
        const int stride, const float *grad_data, float *output)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        /*
         * index is for bottom data (input in the forward direction)
         * in the following, we name "input" and "output" in the aspect of forward
         */

        // the input map size if (_, th, tw, tc)
        int th = data_height * stride;
        int tw = data_width * stride;
        int tc = num_channels / (stride * stride);

        // get the coord "b, h, w, c" under data format (b, h, w, c)
        int n = index;
        int c = n % tc;
        n /= tc;
        int w = n % tw;
        n /= tw;
        int h = n % th;
        n /= th;
        int b = n;

        // the meaning of out_c is hard to explain
        int out_c = tc / (stride * stride);

        // the index of the grad under data format (b, c, h, w)
        // the index is for output (grad data)
        int out_index = w + tw * (h + th * (c + b * tc));

        // extract the coords "b, c, h, w"
        int n1 = out_index;
        int w1 = n1 % data_width;
        n1 /= data_width;
        int h1 = n1 % data_height;
        n1 /= data_height;
        int c1 = n1 % num_channels;
        n1 /= num_channels;
        int b1 = n1;

        // we can easily get the index w.r.t (b, h, w, c) format for output
        int real_out_index = c1 + num_channels * (w1 + data_width * (h1 + data_height * b1));

        // next we should get the index for input
        // first, get the index under data format (b, c, h, w)
        int c2 = c % out_c;
        int offset = c / out_c;
        int w2 = w * stride + offset % stride;
        int h2 = h * stride + offset / stride;
        int in_index = w2 + tw * stride * (h2 + th * stride * (c2 + out_c * b));

        // extract the coords "b, c, h, w" under data format (b, c, h, w)
        int n3 = in_index;
        int w3 = n3 % tw;
        n3 /= tw;
        int h3 = n3 % th;
        n3 /= th;
        int c3 = n3 % tc;
        n3 /= tc;
        int b3 = n3;

        // convert the coords into index under format (b, h, w, c), for input
        int real_in_index = c3 + tc * (w3 + tw * (h3 + th * b3));

        output[real_in_index] = grad_data[real_out_index];
    }
}

bool ReorgBackwardLauncher(const int data_height, const int data_width, const int num_channels, const int batch_size, 
        const int stride, const float *grad_data, float *output, const Eigen::GpuDevice& d)
{
    const int kThreadsPerBlock = 1024;  // threads per block
    const int output_size = batch_size * data_height * data_width * num_channels;
    cudaError_t err;

    ReorgBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>
        (output_size, data_height, data_width, num_channels, stride, grad_data, output);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failted: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return d.ok();
}


#endif
