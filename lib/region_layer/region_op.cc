#include <stdio.h>
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "work_sharder.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice

REGISTER_OP("Region")
    .Attr("T: {float, double})")
    .Attr("noobject_scale: int")
    .Attr("object_scale: int")
    .Attr("coord_scale: int")
    .Attr("class_scale: int")
    .Attr("class_num: int")
    .Attr("box_num: int")
    .Attr("seen: int")
    .Input("predicts: T")
    .Input("labels: T")
    .Input("prior_size: T")
    .Output("grad: T")

REGISTER_OP("RegionGrad")
    .Attr("T: {float, double}")
    .Attr("class_num: int")
    .Attr("box_num: int")
    .Input("grad: T")
    .Output("output: T")

typedef struct{
    float x;
    float y;
    float w;
    float h;
}box;

float overlap(float x1, float w1, float x2, float w2)
{
    // Compute overlap of one axis(Not only the horizontal axis)
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
}

float box_union(box a, box b) {
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float box_iou(box a, box b) {
    return box_intersection(a, b) / (box_union(a, b) + 0.00001);
}

box get_box(float *array, float prior_w, float prior_h, int i, int j, int w, int h) {
    // The first 4 elements in array corresponds to the coords
    // w, h: width and height of feature map
    // i, j: the left-top coord (from 0 - w/h)
    box b;
    b.x = (i + array[0]) / w;
    b.y = (j + array[1]) / h;
    b.w = exp(array[2]) * prior_w / w;
    b.h = exp(array[3]) * prior_h / h;
    return b;
}

void logistic_array(float *array, int len) {
    for(int i = 0; i < len; i++) {
        array[i] = 1. / (1. + exp(-array[i]));
    }
}


float delta_coord(box truth_box, float *predict_flat, float *output, int prior_w, int prior_h,
        int i, int j, int w, int h, scale){
    box pred_box = get_box(predict_flat, prior_w, prior_h, i, j, w, h);
    float iou = box_iou(pred_box, truth_box);

    float tx = (truth.x * w - i);
    float ty = (truth.y * h - j);
    float tw = log(truth.w*w / prior_w);
    float th = log(truth.h*h / prior_h);

    output[0] = scale * (tx - predict_flat[0]);
    output[1] = scale * (ty - predict_flat[1]);
    output[2] = scale * (tw - predict_flat[2]);
    output[3] = scale * (th - predict_flat[3]);

    return iou;

}

void delta_class(float *predict_ptr, float *output_ptr, int cls, int class_num, 
        int class_scale, float &avg_cat){

    for(int i = 0; i < class_num; i++) {
        output_dir[5 + i] = class_scale * (((i === cls?)1 : 0) - predict_ptr[5 + i]);
        if(i == cls) avg_cat += predict_ptr[5 + i];
    }
}

template<typename Device>
class RegionOp: public OpKernel{
public:
    explicit RegionOp(OpKernelConstruction *context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("noobject_scale", &noobject_scale_))
        OP_REQUIRES_OK(context, context->GetAttr("object_scale", &object_scale_))
        OP_REQUIRES_OK(context, context->GetAttr("coord_scale", &coord_scale_))
        OP_REQUIRES_OK(context, context->GetAttr("class_scale", &class_scale_))
        OP_REQUIRES_OK(context, context->GetAttr("seen", &seen_))
        OP_REQUIRES_OK(context, context->GetAttr("class_num", &class_num_))
        OP_REQUIRES_OK(context, context->GetAttr("box_num", &box_num_))
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor& predict = context->input(0);
        OP_REQUIRES(context, predict.dims() == 4,
                errors::InvalidArgument("Predictions must be 4-dimensional"));
        OP_REQUIRES(context, predict.dim_size(3) == box_num_ * (class_num_+ 5),
                errors::InvalidArgument("Inavalid channels number in prediction"));
        auto predict_flat = predict.flat<float>();

        const Tensor& label = context->input(1);
        OP_REQUIRES(context, label.dims() == 3,
                errors::InvalidArgument("Labels must be 3-dimensional"));
        OP_REQUIRES(context, label.dim_size(2) == 5,
                errors::InvalidArgument("Each label should have 5 elements"));
        auto label_flat = label.flat<float>();

        const Tensor& prior_size = context->input(2)
        OP_REQUIRES(context, prior_size.dims() == 1,
                errors::InvalidArgument("Prior size must be 1-dimensional"));
        OP_REQUIRES(context, prior_size.dim_size(0) == 2 * box_num_,
                errors::InvalidArgument("Prior size must be 2 times box_num"));
        auto prior_size_flat = prior_size.flat<float>();

        // batch size
        int batch = predicts.dim_size(0);
        // height of feature map
        int height = predicts.dim_size(1);
        // width of feature map
        int width = predicts.dim_size(2);
        // channels of feature map
        int channels = predicts.dim_size(3);

        // construct output shape
        int dims[4];
        dims[0] = batch;
        dims[1] = height;
        dims[2] = width;
        dims[3] = channels;

        TensorShape output_shape:
        TensorShapeUtils::MakeShape(dims, 4, &output_shape);

        // create output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        auto output = output_tensor->template flat<float>();


        float avg_iou = 0;
        float recall = 0;
        float avg_cat = 0;
        float avg_obj = 0;
        float avg_anyobj = 0;
        int count = 0;

        int box_num = box_num_;
        int box_info_len = box_num_ * (5 + class_num_);
        int noobject_scale = noobject_scale_;
        int object_scale = object_scale_;
        int coord_scale = coord_scale_;
        int class_scale = class_scale_;
        int seen = seen_;
        float *prior_size_ptr = prior_size_flat.data();

        for(int g = 0; g < batch; g++) {
            // For each batch
            int start_index_label = g * 5 * 30;
            for(int h = 0; h < height; h++) {
                for(int w = 0; w < width; w++) {
                    for(int k = 0; k < box_num; k++) {
                        float *predict_ptr = predict_flat.data();
                        float *label_ptr = label_flat.data();
                        float *output_ptr = output.data();

                        int start_index_pred = k*box_info_len + box_num * box_info_len * 
                            (w + width * (h + height * g));
                        
                        predict_ptr += start_index_pred;  // lock the current pos
                        output_ptr += start_index_pred;
                        label_ptr += start_index_label;

                        // The first two elements are tx, ty
                        logistic_array(predict_ptr, 2);
                        // The fifth element corresponds to obj
                        logistic_array(predict_ptr+4, 1);
                        
                        float prior_w = prior_size_ptr[2*k];
                        float prior_h = prior_size_ptr[2*k + 1];
                        
                        // (tx, ty, tw, th) -> (x, y, w, h)
                        box pred_box = get_box(predict_ptr, prior_w, prior_h, i, j,
                                width, height);
                        
                        float best_iou = 0;
                        for(int t = 0; t < 30; t++) {
                            box truth_box;
                            truth_box.x = label_ptr[t * 5 + 1];
                            truth_box.y = label_ptr[t * 5 + 2];
                            truth_box.w = label_ptr[t * 5 + 3];
                            truth_box.h = label_ptr[t * 5 + 4];
                            float iou = box_iou(pred_box, truth_box);
                            if(iou > best_iou) {
                                best_iou = iou;
                            }
                        }   
                        avg_anyobj += output[4];

                        // delta for obj
                        output[4] = noobject_scale * (0 - predict_ptr[4]);
                        if(best_iou > thresh) {
                            output_ptr[start_index_pred+4] = 0;
                        }

                        if(seen < 12800) {
                            box truth;
                            truth.x = (w + .5) / width;
                            truth.y = (h + .5) / height;
                            truth.w = prior_w;
                            truth.h = prior_h;
                            delta_coord(truth, predict_ptr, output_ptr, prior_w, prior_h, w, h, 
                                    width, height, 0.01);
                        }
                    }
                }
            }
            for(int t = 0; t < 30; t++) {
                // for each grounding truth bnox, find the corresponding
                // pred box
                box truth_box;
                float *label_ptr = label_flat.data();
                label_ptr += start_index_label + t * 5;
                
                truth_box.x = label[1];
                truth_box.y = label[2];
                truth_box.w = label[3];
                truth_box.h = label[4];
                if(truth_box.w == 0 || truth_box.h == 0) {
                    continue;
                } else {
                    float best_iou = 0;
                    int best_n;   // best pred box index
                    int w = truth_box.x * width;
                    int h = truth_box.y * height;
                    box truth_shift = truth_box;
                    truth_shift.x = 0;
                    truth_shift.y = 0;

                    float *predict_ptr = predict_flat.data();
                    predict_ptr += box_num * box_info_len * 
                        (w + width * (h + height * g));

                    for(int k = 0; k < box_num; k++) {
                        float prior_w = prior_size_ptr[2*k];
                        float prior_h = prior_size_ptr[2*k+1];
                        box pred_box = get_box(predict_ptr + k * box_info_len, prior_w, prior_h, 
                                w, h, width, height);
                        pred_box.x = 0;
                        pred_box.y = 0;
                        float iou = box_iou(pred_box, truth_shift);
                        if(iou > best_iou) {
                            best_iou = iou;
                            best_n = k;
                        }
                    }

                    predict_ptr += best_n * box_info_len;
                    float *output_ptr = output.data();

                    output_ptr += best_n * box_info_len + box_num * box_info_len * 
                        (w + width * (h + height * g));

                    float prior_w = prior_size_ptr[2 * best_n];
                    float prior_h = prior_size_ptr[2 * best_n + 1];

                    float iou = delta_coord(truth_box, predict_ptr, output_ptr, prior_w, prior_h, 
                            w, h, width, height, coord_scale * (2 - truth_box.w * truth_box.h));
    
                    avg_iou += iou;

                    if(iou > 0.5) recall += 1;
                    
                    avg_obj += output_ptr[4];

                    output_ptr[4] = object_scale * (iou - output_ptr[4]);

                    int cls = int(label[0]);

                    delta_class(predict_ptr, output_ptr, cls, class_num, class_scale, &avg_cat);
                
                    ++count;
                }
            }
        }
        printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f, count: %d\n", 
                avg_iou/count, avg_cat/count, avg_obj/count, avg_any_obj/(width*height*box_num*batch),
                recall/count, count);
    }
private:
    int noobject_scale_;
    int object_scale_;
    int coord_scale_;
    int class_scale_;
    int seen_;
    int class_num_;
    int box_num_;
}


template<class Device>
class RegionGradOp: public Opkernel {
public:
    explicit RegionGradOp(OpKernelConstruction *context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("box_num", &box_num_));
        OP_REQUIRES_OK(context, context->GetAttr("class_num", &class_num_));
    }

    void Compute(OpKernelContext* context) override
    {
        
    }

private:
    int box_num_;
    int class_num_;
}


