#include <stdio.h>
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("Region")
    .Attr("T: {float, double}")
    .Attr("noobject_scale: int")
    .Attr("object_scale: int")
    .Attr("coord_scale: int")
    .Attr("class_scale: int")
    .Attr("class_num: int")
    .Attr("box_num: int")
    .Attr("seen: int")
    .Attr("thresh: float")
    .Input("predicts: T")
    .Input("labels: T")
    .Input("prior_size: T")
    .Output("grad: T");

REGISTER_OP("RegionGrad")
    .Attr("T: {float, double}")
    .Attr("class_num: int")
    .Attr("box_num: int")
    .Input("bottom_data: T")
    .Input("grad: T")
    .Output("output: T");

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
    // add 1e-6 to avoid divided by zero
    return box_intersection(a, b) / (box_union(a, b) + 1e-6);
}

float logistic(float x) {
    return 1./(1. + exp(-x));
}

box get_box(const float *array, float prior_w, float prior_h, int i, int j, int w, int h) {
    // The first 4 elements in array corresponds to the coords
    // w, h: width and height of feature map
    // i, j: the left-top coord (from 0 -  w/h-1 )
    box b;
    float tx = array[0];
    float ty = array[1];


    b.x = (i + logistic(tx)) / w;
    b.y = (j + logistic(ty)) / h;
    b.w = exp(array[2]) * prior_w / w;
    b.h = exp(array[3]) * prior_h / h;
    
    //printf("w: %d, h: %d, xc: %f, yc: %f, wc: %f, hc: %f, prior_w: %f, prior_h: %f\n", 
    //        i, j, b.x, b.y, b.w, b.h, prior_w, prior_h);
    
    return b;
}


float delta_coord(box truth_box, const float *predict_ptr, float *output, float prior_w, float prior_h,
        int i, int j, int w, int h, float scale){
    
    // The function is for computing the coords loss
    
    box pred_box = get_box(predict_ptr, prior_w, prior_h, i, j, w, h);
    float iou = box_iou(pred_box, truth_box);

    // tx, ty: exactly it's the logistic(tx)
    float tx = (truth_box.x * w - i);
    float ty = (truth_box.y * h - j);
    float tw = log(truth_box.w*w / prior_w);
    float th = log(truth_box.h*h / prior_h);

    //printf("tx: %f, ty: %f, tw: %f, th: %f\n", tx, ty, tw, th);
    //printf("predx: %f, predy: %f, predw: %f, predh: %f\n", logistic(predict_ptr[0]), 
    //        logistic(predict_ptr[1]), predict_ptr[2], predict_ptr[3]);

    output[0] = -scale * (tx - logistic(predict_ptr[0]));
    output[1] = -scale * (ty - logistic(predict_ptr[1]));
    output[2] = -scale * (tw - predict_ptr[2]);
    output[3] = -scale * (th - predict_ptr[3]);

    return iou;

}

void delta_class(const float *predict_ptr, float *output_ptr, int cls, int class_num, 
        int class_scale, float &avg_cat){

    // The function is for computing the classifcation loss
    // avg_cat: the confidence in correct class

    for(int i = 0; i < class_num; i++) {
        output_ptr[5 + i] = -class_scale * (((i == cls)? 1 : 0) - predict_ptr[5 + i]);
        if(i == cls) avg_cat += predict_ptr[5 + i];
    }
}

template<typename Device>
class RegionOp: public OpKernel{
public:
    explicit RegionOp(OpKernelConstruction *context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("noobject_scale", &noobject_scale_));
        OP_REQUIRES_OK(context, context->GetAttr("object_scale", &object_scale_));
        OP_REQUIRES_OK(context, context->GetAttr("coord_scale", &coord_scale_));
        OP_REQUIRES_OK(context, context->GetAttr("class_scale", &class_scale_));
        OP_REQUIRES_OK(context, context->GetAttr("seen", &seen_));
        OP_REQUIRES_OK(context, context->GetAttr("class_num", &class_num_));
        OP_REQUIRES_OK(context, context->GetAttr("box_num", &box_num_));
        OP_REQUIRES_OK(context, context->GetAttr("thresh", &thresh_));
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

        const Tensor& prior_size = context->input(2);
        OP_REQUIRES(context, prior_size.dims() == 1,
                errors::InvalidArgument("Prior size must be 1-dimensional"));
        OP_REQUIRES(context, prior_size.dim_size(0) == 2 * box_num_,
                errors::InvalidArgument("Prior size must be 2 times box_num"));
        auto prior_size_flat = prior_size.flat<float>();

        // batch size
        int batch = predict.dim_size(0);
        // height of feature map
        int height = predict.dim_size(1);
        // width of feature map
        int width = predict.dim_size(2);
        // channels of feature map
        int channels = predict.dim_size(3);

        // construct output shape
        int dims[4];
        dims[0] = batch;
        dims[1] = height;
        dims[2] = width;
        dims[3] = channels;

        TensorShape output_shape;
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
        int class_num = class_num_;
        int box_info_len = 5 + class_num_;
        int noobject_scale = noobject_scale_;
        int object_scale = object_scale_;
        int coord_scale = coord_scale_;
        int class_scale = class_scale_;
        int seen = seen_;
        float thresh = thresh_;
        const float *prior_size_ptr = prior_size_flat.data();

        for(int g = 0; g < batch; g++) {
            // For each batch
            int start_index_label = g * 5 * 30;
            for(int h = 0; h < height; h++) {
                for(int w = 0; w < width; w++) {
                    for(int k = 0; k < box_num; k++) {
                        const float *predict_ptr = predict_flat.data();
                        const float *label_ptr = label_flat.data();
                        float *output_ptr = output.data();

                        // start index for current box info
                        int start_index_pred = k*box_info_len + box_num * box_info_len * 
                            (w + width * (h + height * g));
                   

                        // lock the current pos for predict, output and label
                        predict_ptr += start_index_pred;  
                        output_ptr += start_index_pred;
                        label_ptr += start_index_label;

                        float prior_w = prior_size_ptr[2*k];
                        float prior_h = prior_size_ptr[2*k + 1];
                        
                        // (tx, ty, tw, th) -> (x, y, w, h)
                        box pred_box = get_box(predict_ptr, prior_w, prior_h, w, h,
                                width, height);
                        
                        float best_iou = 0;
                        for(int t = 0; t < 30; t++) {
                            // find the best iou with labels
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

                        //printf("w: %d, h:%d, iou: %f\n", w, h, best_iou);

                        avg_anyobj += logistic(predict_ptr[4]);

                        output_ptr[4] = -noobject_scale * (0 - logistic(predict_ptr[4]));
                        if(best_iou > thresh) {
                            // If the iou for this box is over thresh, then the
                            // loss is 0
                            output_ptr[4] = 0;
                        }

                        if(seen < 12800) {
                            // In the early training, we correct the coords for
                            // each box using the man-made gt-box, the gt-box
                            // is exactly the grid of feature map
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
                const float *label_ptr = label_flat.data();
                
                label_ptr += start_index_label + t * 5;

                truth_box.x = label_ptr[1];
                truth_box.y = label_ptr[2];
                truth_box.w = label_ptr[3];
                truth_box.h = label_ptr[4];
                if(truth_box.w == 0 || truth_box.h == 0) {
                    // Eliminate the disabled label box
                    continue;
                } else {
                    int w = truth_box.x * width;
                    int h = truth_box.y * height;
                    
                    w = std::min(w, width-1);
                    h = std::min(h, height-1);
                    
                    box truth_shift;
                    truth_shift.x = 0;
                    truth_shift.y = 0;
                    truth_shift.w = truth_box.w;
                    truth_shift.h = truth_box.h;

                    // predict_ptr points to the beginning of selected boxes
                    // Note there are box_num boxes being selected
                    const float *predict_ptr = predict_flat.data();
                    predict_ptr += box_num * box_info_len * 
                        (w + width * (h + height * g));

                    float best_iou = 0;
                    int best_n = 0;   // best pred box index
                    for(int k = 0; k < box_num; k++) {
                        // check the best matching pred box
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

                    // point to the begining of the best matching pred box
                    predict_ptr += best_n * box_info_len;
                    
                    // output_ptr also point to the matching pred box
                    float *output_ptr = output.data();
                    output_ptr += best_n * box_info_len + box_num * box_info_len * 
                        (w + width * (h + height * g));

                    float prior_w = prior_size_ptr[2 * best_n];
                    float prior_h = prior_size_ptr[2 * best_n + 1];

                    //printf("best_n: %d, box_info_len: %d, w: %d, h: %d\n", best_n, box_info_len, w, h);

                    // re-calculate the iou
                    // and compute the coords loss
                    float iou = delta_coord(truth_box, predict_ptr, output_ptr, prior_w, prior_h, 
                            w, h, width, height, coord_scale * (2 - truth_box.w * truth_box.h));
    
                    //printf("w: %d, h: %d, best iou: %f\n", w, h, iou);
                    
                    avg_iou += iou;

                    if(iou > 0.5) recall += 1;
                    
                    avg_obj += logistic(predict_ptr[4]);

                    //printf("confidence: %f\n", logistic(predict_ptr[4]));

                    // obj loss
                    output_ptr[4] = -object_scale * (iou - logistic(predict_ptr[4]));

                    int cls = int(label_ptr[0]);

                    // class loss
                    delta_class(predict_ptr, output_ptr, cls, class_num, class_scale, avg_cat);
                
                    ++count;
                }
            }

            //const float *output_ptr = output.data();
            //for(int h = 0; h < height; h++){
            //    for(int w = 0; w < width; w++){
            //        for(int i = 0; i < 7; i++) {
            //            int index = i + 7*(w+width*h);
            //            printf("%d:%f ", index, output_ptr[index]);
            //        }
            //        printf("\n");
            //    }
            //}

        }
        printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f, count: %d\n", 
                avg_iou/count, avg_cat/count, avg_obj/count, avg_anyobj/(width*height*box_num*batch),
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
    float thresh_;
};


void logistic_gradient_array(const float *array, const float *grad, float *output, int len) {
    for(int i = 0; i < len; i++) {
        output[i] = ((1 - logistic(array[i])) * logistic(array[i])) * grad[i];
    }
}

template<class Device>
class RegionGradOp: public OpKernel {
public:
    explicit RegionGradOp(OpKernelConstruction *context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("box_num", &box_num_));
        OP_REQUIRES_OK(context, context->GetAttr("class_num", &class_num_));
    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& bottom_data = context->input(0);
        OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("Bottom data must be 4-dimensional"));
        OP_REQUIRES(context, bottom_data.dim_size(3) == box_num_ * (class_num_+ 5),
                errors::InvalidArgument("Inavalid channels number for bottom data"));
        auto bottom_data_flat = bottom_data.flat<float>();


        const Tensor& grad_data = context->input(1);
        OP_REQUIRES(context, grad_data.dims() == 4,
                errors::InvalidArgument("Gradients must be 4-dimensional"));
        OP_REQUIRES(context, grad_data.dim_size(3) == box_num_ * (class_num_+ 5),
                errors::InvalidArgument("Inavalid channels number in gradient data"));
        auto grad_data_flat = grad_data.flat<float>();
   
        
        // batch size
        int batch = bottom_data.dim_size(0);
        // height of feature map
        int height = bottom_data.dim_size(1);
        // width of feature map
        int width = bottom_data.dim_size(2);
        // channels of feature map
        int channels = bottom_data.dim_size(3);

        // construct output shape
        int dims[4];
        dims[0] = batch;
        dims[1] = height;
        dims[2] = width;
        dims[3] = channels;

        TensorShape output_shape;
        TensorShapeUtils::MakeShape(dims, 4, &output_shape);

        // create output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        auto output = output_tensor->template flat<float>();

        int box_num = box_num_;
        int class_num = class_num_;
        int box_info_len = 5 + class_num;

        for(int g = 0; g < batch; g++) {
            for(int h = 0; h < height; h++) {
                for(int w = 0; w < width; w++) {
                    for(int k = 0; k < box_num; k++) {
                        const float *bottom_data_ptr = bottom_data_flat.data();
                        const float *grad_data_ptr = grad_data_flat.data();
                        float *output_data_ptr = output.data();

                        int start_index_pred = k * box_info_len + box_num * box_info_len * (
                                w + width * (h + height * g));

                        bottom_data_ptr += start_index_pred;
                        grad_data_ptr += start_index_pred;
                        output_data_ptr += start_index_pred;
                        
                        // Apply logistic gradient for tx, ty
                        logistic_gradient_array(bottom_data_ptr, grad_data_ptr, output_data_ptr, 2);
                        
                        output_data_ptr[2] = grad_data_ptr[2];
                        output_data_ptr[3] = grad_data_ptr[3];

                        // Apply logistic gradient for obj
                        logistic_gradient_array(bottom_data_ptr+4, grad_data_ptr+4, output_data_ptr+4, 1);

                        //printf("gradients: %f\n", grad_data_ptr[4]);

                        for(int i = 0; i < class_num; i++) {
                            output_data_ptr[5 + i] = grad_data_ptr[5 + i];
                        }
                    }
                }
            }
        }
    }

private:
    int box_num_;
    int class_num_;
};

REGISTER_KERNEL_BUILDER(Name("Region").Device(DEVICE_CPU).TypeConstraint<float>("T"), RegionOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("RegionGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), RegionGradOp<CPUDevice>);
