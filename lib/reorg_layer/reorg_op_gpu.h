#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_REORG_OP_GPU_H_
#define TENSORFLOW_USER_OPS_REORG_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
bool ReorgForwardLauncher(
    const int data_height, const int data_width, const int num_channels, const int batch_size,
    const int stride, const float *bottom_data, float *output, const Eigen::GpuDevice &d);

bool ReorgBackwardLauncher(
    const int data_height, const int data_width, const int num_channels, const int batch_size, 
    const int stride, const float *grad_data, float *output, const Eigen::GpuDevice &d);

}

#endif
