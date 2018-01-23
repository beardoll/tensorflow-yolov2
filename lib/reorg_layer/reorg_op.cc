#include <stdio.h>
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "work_sharder.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Reorg")
    .Attr("T: {float, double}")
    .Attr("stride: int")
    .Input("bottom_data: T")
    .Output("output: T");

REGISTER_OP("ReorgGrad")
    .Attr("T: {float, double}")
    .Attr("stride: int")
    .Input("grad: T")
    .Output("output: T");

template<typename Device>
class ReorgOp: public OpKernel {
public:
    explicit ReorgOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
        OP_REQUIRES(context, stride_ >= 1, 
                errors::InvalidArgument("Need stride >= 1, got ", stride_));
    }

    void Compute(OpKernelContext* context) override
    {
        /***
         * The forward pass of reorg layer
        ***/

        const Tensor& bottom_data = context->input(0);
        auto bottom_data_flat = bottom_data.flat<float>();

        // data should have 4 dimensions
        OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

        // batch size    
        int batch = bottom_data.dim_size(0);
        // height
        int data_height = bottom_data.dim_size(1);
        // width
        int data_width = bottom_data.dim_size(2);
        // number of channels
        int num_channels = bottom_data.dim_size(3);

        // construct the output shape
        int dims[4];
        dims[0] = batch;
        dims[1] = data_height / stride_;
        dims[2] = data_width / stride_;
        dims[3] = num_channels * stride_ * stride_;

        TensorShape output_shape;
        TensorShapeUtils::MakeShape(dims, 4, &output_shape);

        // create output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        auto output = output_tensor->template flat<float>();

        int stride = stride_;
        auto shard = [data_height, data_width, num_channels, stride, &bottom_data_flat, &output]
            (int64 start, int64 limit) 
        {
            for (int64 g = start; g < limit; g++) 
            {
                // compute the coords "b, h, w, c" under data format (b, h, w, c)
                // g is for input data
                int n = g;
                int c = n % num_channels;
                n /= num_channels;
                int w = n % data_width;
                n /= data_width;
                int h = n % data_height;
                n /= data_height;
                int b = n;
                
                // out_c is hard to explain
                int out_c = num_channels / (stride * stride);

                // the index of the output, with format (b, c, h, w)
                // that means in_index is not real index for bottom_data_flat
                // in fact, we just project the index w.r.t (b, h, w, c) to (b, c, h, w)
                // for a given pixel, we believe that it's "b, h, w, c" stays unchanged
                int in_index = w + data_width * (h + data_height * (c + b * num_channels));
                
                // convert in_index from format (b, c, h, w) to format(b, h, w, c)
                // the output shape is (b, h/stride, w/stride, c*stride*stride)
                // first we should extract the coords "b, h, w, c" under data format (b, c, h, w)
                int real_out_c = num_channels * stride * stride;
                int real_out_w = data_width / stride;
                int real_out_h = data_height / stride;
                int n1 = in_index;
                int w1 = n1 % real_out_w;
                n1 /= real_out_w;
                int h1 = n1 % real_out_h;
                n1 /= real_out_h;
                int c1 = n1 % real_out_c;
                n1 /= real_out_c;
                int b1 = n1;
                // according to coords"b, h, w, c", we can get the index w.r.t to data format (b, h, w, c)
                int real_out_index = c1 + real_out_c * (w1 + real_out_w * (h1 + real_out_h * b1));

                // compute the corresponding index in input
                int c2 = c % out_c;
                int offset = c / out_c;
                int w2 = w * stride + offset % stride;
                int h2 = h * stride + offset / stride;
                // the index w.r.t data format (b, c, h, w)
                int out_index = w2 + data_width * stride * (h2 + data_height * stride * (c2 + b*out_c));

                // convert out_index to data format (b, h, w, c)
                // first extract coords "b, h, w, c" under data format (b, c, h, w)
                int n3 = out_index;
                int w3 = n3 % data_width;
                n3 /= data_width;
                int h3 = n3 % data_height;
                n3 /= data_height;
                int c3 = n3 % num_channels;
                n3 /= num_channels;
                int b3 = n;

                // the index w.r.t data format (b, h, w, c)
                int real_in_index = c3 + num_channels * (w3 + data_width * (h3 + data_height * b3));

                //printf("in_index: %d, out_index: %d\n", in_index, out_index);
                //printf("real_in_index: %d, real_out_index: %d\n", real_in_index, real_out_index);
                //printf("c2: %d, w2: %d, h2:%d\n", c2, w2, h2);
                //printf("\n");

                const float* bottom_data = bottom_data_flat.data();
                output(real_out_index) = bottom_data[real_in_index];
            }
        };
        const DeviceBase::CpuWorkerThreads& worker_threads = 
            *(context->device()->tensorflow_cpu_worker_threads());
        const int64 shard_cost = 1000;   // estimate
        Shard(worker_threads.num_threads, worker_threads.workers, output.size(), shard_cost, shard);
    }
private:
    int stride_;
};


bool ReorgForwardLauncher(const int data_height, const int data_width, const int num_channels, const int batch_size, 
        const int stride, const float *bottom_data, float *output, const Eigen::GpuDevice &d);

template< >
class ReorgOp<Eigen::GpuDevice>: public OpKernel {
public:
    explicit ReorgOp(OpKernelConstruction* context): OpKernel(context) {
        // get the stride
        OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));

        OP_REQUIRES(context, stride_ >= 1, 
                errors:: InvalidArgument("Need stride_ >= 1, get ", stride_));
    }

    void Compute(OpKernelContext* context) override
    {
        // get the input tensor
        const Tensor& bottom_data = context->input(0);
        
        // data should have 4 dimensions
        OP_REQUIRES(context, bottom_data.dims() == 4, 
                errors:: InvalidArgument("data must be 4-dimensional"));

        // batch size
        int batch_size = bottom_data.dim_size(0);
        // data height
        int data_height = bottom_data.dim_size(1);
        // data width
        int data_width = bottom_data.dim_size(2);
        // number of channels
        int num_channels = bottom_data.dim_size(3);

        // construct the output shape
        int dims[4];
        dims[0] = batch_size;
        dims[1] = data_height / stride_;
        dims[2] = data_width / stride_;
        dims[3] = num_channels * stride_ * stride_;

        TensorShape output_shape;
        TensorShapeUtils::MakeShape(dims, 4, &output_shape);

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        const Tensor* bottom_data_ptr = &bottom_data;

        if(!context->status().ok()) {
            return;
        }

        int stride = stride_;
        ReorgForwardLauncher(data_height, data_width, num_channels, batch_size, stride, 
                bottom_data_ptr->flat<float>().data(), output->flat<float>().data(), 
                context->eigen_device<Eigen::GpuDevice>());
    }
private:
    int stride_;
};


template <class Device>
class ReorgGradOp: public OpKernel {
public:
    explicit ReorgGradOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
        OP_REQUIRES(context, stride_ >= 1, 
                errors::InvalidArgument("Need stride >= 1, got ", stride_));

    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& grad_data = context->input(0);
        auto grad_data_flat = grad_data.flat<float>();

        // grad data should have 4 dimensions
        OP_REQUIRES(context, grad_data.dims() == 4,
                errors::InvalidArgument("grad must be 4-dimensional"))
        
        // batch size
        int batch = grad_data.dim_size(0);
        // grad data height
        int data_height = grad_data.dim_size(1);
        // grad data width
        int data_width = grad_data.dim_size(2);
        // number of channels
        int num_channels = grad_data.dim_size(3);
        
         // construct the output shape
        int dims[4];
        dims[0] = batch;
        dims[1] = data_height * stride_;
        dims[2] = data_width * stride_;
        dims[3] = num_channels / (stride_ * stride_);

        TensorShape output_shape;
        TensorShapeUtils::MakeShape(dims, 4, &output_shape);

        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        auto output = output_tensor->template flat<float>();

        int stride = stride_;
        auto shard = [data_height, data_width, num_channels, stride, &grad_data_flat, &output]
            (int64 start, int64 limit) 
        {
            /***
            here data_height, data_width and num_channels are for grad data
            grad is the output in the aspect of forward, and the corresponding output
            has shape of data_height * stride, data_width * stride, num_channels / (stride * stride)
            without implicity, in the following, we name 'input' and 'output' in the aspect of forward
            ***/

            for (int64 g = start; g < limit; g++) 
            {
                // g is for input maps
                // the input map size is (_, th, tw, tc)
                int th = data_height * stride;
                int tw = data_width * stride;
                int tc = num_channels / (stride * stride);

                // get the coord "b, h, w, c" w.r.t input data
                int n = g;
                int c = n % tc;
                n /= tc;
                int w = n % tw;
                n /= tw;
                int h = n % th;
                n /= th;
                int b = n;
                
                // the meaning is hard to explain ...
                int out_c = tc / (stride * stride);

                // the index of the grad under data format (b, c, h, w)
                int out_index = w + tw * (h + th * (c + b * tc));

                // get the coords "b, h, w, c" under format (b, c, h, w)
                // for gradients (output)
                int n1 = out_index;
                int w1 = n1 % data_width;
                n1 /= data_width;
                int h1 = n1 % data_height;
                n1 /= data_height;
                int c1 = n1 % num_channels;
                n1 /= num_channels;
                int b1 = n1;

                // index under format (b, h, w, c)
                int real_out_index = c1 + num_channels * (w1 + data_width * (h1 + data_height * b1));


                // we can easily get the index w.r.t (b, h, w, c) format
                // note that in_index and real_in_index are the same in coord "b, c, h, w"
                // the differences are only the order of coord
                // int real_in_index = c + tc * (w + tw * (h + th * b));

                // next we should get the index for output
                // first, get the out_index under data format (b, c, h, w)
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

                // convert the coords into index under data format (b, h, w, c)
                int real_in_index = c3 + tc * (w3 + tw * (h3 + th * b3));

                //printf("in_index: %d, out_index: %d\n", out_index, in_index);
                //printf("real_in_index: %d, real_out_index: %d\n", real_in_index, real_out_index);
                //printf("\n");

                const float* grad_data = grad_data_flat.data();
                output(real_in_index) = grad_data[real_out_index];
            }
        };
        const DeviceBase::CpuWorkerThreads& worker_threads = 
            *(context->device()->tensorflow_cpu_worker_threads());
        const int64 shard_cost = 1000;   // estimate
        Shard(worker_threads.num_threads, worker_threads.workers, output.size(), shard_cost, shard);
    }
private:
    int stride_;
};

bool ReorgBackwardLauncher(const int data_height, const int data_width, const int num_channels, const int batch_size, 
        const int stride, const float *grad_data, float *output, const Eigen::GpuDevice& d);

template<>
class ReorgGradOp<Eigen::GpuDevice> : public OpKernel {
public:
    explicit ReorgGradOp(OpKernelConstruction* context) : OpKernel(context) {
        // get the stride
        OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));

        OP_REQUIRES(context, stride_ >= 1,
                errors::InvalidArgument("Need stride_ >= 1 got ", stride_));
    }

    void Compute(OpKernelContext* context) override
    {
        // get the grad data
        const Tensor& grad_data = context->input(0);

        // grad_data should have 4 dimensions
        OP_REQUIRES(context, grad_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

        // batch_size
        int batch_size = grad_data.dim_size(0);
        int data_height = grad_data.dim_size(1);
        int data_width = grad_data.dim_size(2);
        int num_channels = grad_data.dim_size(3);

        // construct the output shape
        int dims[4];
        dims[0] = batch_size;
        dims[1] = data_height * stride_;
        dims[2] = data_width * stride_;
        dims[3] = num_channels / (stride_ * stride_);
        TensorShape output_shape;
        TensorShapeUtils::MakeShape(dims, 4, &output_shape);

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        if(!context->status().ok()) {
            return;
        }

        const Tensor *grad_data_ptr = &grad_data;

        int stride = stride_;
        ReorgBackwardLauncher(data_height, data_width, num_channels, batch_size, stride, 
                grad_data_ptr->flat<float>().data(), output->flat<float>().data(),
                context->eigen_device<Eigen::GpuDevice>());
    }
private:
    int stride_;

};

REGISTER_KERNEL_BUILDER(Name("Reorg").Device(DEVICE_CPU).TypeConstraint<float>("T"), ReorgOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("ReorgGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), ReorgGradOp<CPUDevice>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Reorg").Device(DEVICE_GPU).TypeConstraint<float>("T"), ReorgOp<Eigen::GpuDevice>);
REGISTER_KERNEL_BUILDER(Name("ReorgGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), ReorgGradOp<Eigen::GpuDevice>);
#endif

