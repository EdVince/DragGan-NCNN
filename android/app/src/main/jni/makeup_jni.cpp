// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <codecvt>

// ncnn & opencv
#include "net.h"
#include "benchmark.h"
#include "layer.h"
#include "layer_type.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cpu.h>

ncnn::Mat w_tmp;
ncnn::Mat w0_tmp;

ncnn::Net mapping;
ncnn::Net generator;

ncnn::Layer* gridsample;
ncnn::Option gridsample_opt;

ncnn::Layer* interp;
ncnn::Option interp_opt;

const int H = 512;
const int W = 512;

double X[H], Y[W];
double xx[H][W], yy[H][W];
double distance[H][W];

ncnn::Mat showwing;
double points[2] = { -1,-1 };
double targets[2] = { -1,-1 };

float lr = 0.1;
int r1 = 3, r2 = 12;
ncnn::Mat feat_refs;
ncnn::Mat w;
ncnn::Mat w0;

class Where : public ncnn::Layer
{
public:
    Where()
    {
        one_blob_only = true;
        support_packing = false;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        //printf("[Where] (%d,%d,%d)\n",channels,h,w);

        top_blob.create(w, h, channels, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const float* src = bottom_blob.channel(p);
            float* dst = top_blob.channel(p);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    if (src[0] >= 0)
                        dst[0] = 1.0f;
                    else
                        dst[0] = 0.2f;
                    src++;
                    dst++;
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Where)

class ConvTranspose2d1 : public ncnn::Layer
{
public:
    ConvTranspose2d1()
    {
        one_blob_only = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& _weight_data = bottom_blobs[1];
        ncnn::Mat& top_blob = top_blobs[0];

        //printf("[ConvTranspose2d1] (%d,%d,%d)*(%d,%d,%d,%d)\n", bottom_blob.c, bottom_blob.h, bottom_blob.w, _weight_data.c, _weight_data.d, _weight_data.h, _weight_data.w);

        const int _kernel_w = _weight_data.w;
        const int _kernel_h = _weight_data.h;
        const int _num_output = _weight_data.c * _weight_data.elempack;

        ncnn::Mat weight_data_flattened;
        ncnn::flatten(_weight_data, weight_data_flattened, opt);
        if (weight_data_flattened.empty())
            return -100;

        // weight_data_flattened as pack1
        weight_data_flattened.w *= weight_data_flattened.elempack;
        weight_data_flattened.elemsize /= weight_data_flattened.elempack;
        weight_data_flattened.elempack = 1;

        ncnn::Mat bias_data_flattened;

        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Deconvolution);

        const int dilation_w = 1;
        const int dilation_h = 1;
        const int stride_w = 2;
        const int stride_h = 2;
        const int pad_left = 0;
        const int pad_right = 0;
        const int pad_top = 0;
        const int pad_bottom = 0;
        const int bias_term = 0;
        const int output_pad_right = 0;
        const int output_pad_bottom = 0;

        ncnn::ParamDict pd;
        pd.set(0, _num_output);
        pd.set(1, _kernel_w);
        pd.set(11, _kernel_h);
        pd.set(2, dilation_w);
        pd.set(12, dilation_h);
        pd.set(3, stride_w);
        pd.set(13, stride_h);
        pd.set(4, pad_left);
        pd.set(15, pad_right);
        pd.set(14, pad_top);
        pd.set(16, pad_bottom);
        pd.set(18, output_pad_right);
        pd.set(19, output_pad_bottom);
        pd.set(5, bias_term);
        pd.set(6, weight_data_flattened.w);
        pd.set(9, 0);
        pd.set(10, ncnn::Mat());


        op->load_param(pd);

        ncnn::Mat weights[2];
        weights[0] = weight_data_flattened;
        weights[1] = bias_data_flattened;

        op->load_model(ncnn::ModelBinFromMatArray(weights));

        op->create_pipeline(opt);

        op->forward(bottom_blob, top_blob, opt);

        op->destroy_pipeline(opt);

        delete op;

        return 0;
    }
};

DEFINE_LAYER_CREATOR(ConvTranspose2d1)

class ConvTranspose2d2 : public ncnn::Layer
{
public:
    ConvTranspose2d2()
    {
        one_blob_only = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& _weight_data = bottom_blobs[1];
        ncnn::Mat& top_blob = top_blobs[0];

        // transpose wegiht from cdhw to dchw
        ncnn::Layer* transpose_op = ncnn::create_layer(ncnn::LayerType::Permute);
        ncnn::ParamDict transpose_pd;
        transpose_pd.set(0, 6); // WHDC->WHCD
        transpose_op->load_param(transpose_pd);
        transpose_op->create_pipeline(opt);
        ncnn::Mat _weight_data_T;
        transpose_op->forward(_weight_data, _weight_data_T, opt);

        //printf("[ConvTranspose2d2] (%d,%d,%d)*(%d,%d,%d,%d)\n", bottom_blob.c, bottom_blob.h, bottom_blob.w, _weight_data_T.c, _weight_data_T.d, _weight_data_T.h, _weight_data_T.w);

        const int _kernel_w = _weight_data_T.w;
        const int _kernel_h = _weight_data_T.h;
        const int _num_output = _weight_data_T.c * _weight_data_T.elempack;

        ncnn::Mat weight_data_flattened;
        ncnn::flatten(_weight_data_T, weight_data_flattened, opt);
        if (weight_data_flattened.empty())
            return -100;

        // weight_data_flattened as pack1
        weight_data_flattened.w *= weight_data_flattened.elempack;
        weight_data_flattened.elemsize /= weight_data_flattened.elempack;
        weight_data_flattened.elempack = 1;

        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Deconvolution);
        const int dilation_w = 1;
        const int dilation_h = 1;
        const int stride_w = 1;
        const int stride_h = 1;
        const int pad_left = 1;
        const int pad_right = 1;
        const int pad_top = 1;
        const int pad_bottom = 1;
        const int bias_term = 0;
        const int output_pad_right = 0;
        const int output_pad_bottom = 0;
        ncnn::ParamDict pd;
        pd.set(0, _num_output);
        pd.set(1, _kernel_w);
        pd.set(11, _kernel_h);
        pd.set(2, dilation_w);
        pd.set(12, dilation_h);
        pd.set(3, stride_w);
        pd.set(13, stride_h);
        pd.set(4, pad_left);
        pd.set(15, pad_right);
        pd.set(14, pad_top);
        pd.set(16, pad_bottom);
        pd.set(18, output_pad_right);
        pd.set(19, output_pad_bottom);
        pd.set(5, bias_term);
        pd.set(6, weight_data_flattened.w);
        pd.set(9, 0);
        pd.set(10, ncnn::Mat());
        op->load_param(pd);

        ncnn::Mat weights[2];
        ncnn::Mat bias_data_flattened;
        weights[0] = weight_data_flattened;
        weights[1] = bias_data_flattened;

        op->load_model(ncnn::ModelBinFromMatArray(weights));
        op->create_pipeline(opt);
        op->forward(bottom_blob, top_blob, opt);


        op->destroy_pipeline(opt);
        delete op;

        transpose_op->destroy_pipeline(opt);
        delete transpose_op;

        return 0;
    }
};

DEFINE_LAYER_CREATOR(ConvTranspose2d2)

class BConv2d1 : public ncnn::Layer
{
public:
    BConv2d1()
    {
        one_blob_only = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& _weight_data = bottom_blobs[1];

        //printf("[BConv2d1] (%d,%d,%d)%d*(%d,%d,%d,%d)%d\n", bottom_blob.c, bottom_blob.h, bottom_blob.w, bottom_blob.elempack, _weight_data.c, _weight_data.d, _weight_data.h, _weight_data.w, _weight_data.elempack);


        // 创建conv2d算子
        const int _kernel_w = _weight_data.w;
        const int _kernel_h = _weight_data.h;
        const int _num_output = _weight_data.c * _weight_data.elempack;
        const int _num_input = bottom_blob.c;

        ncnn::Mat weight_data_flattened;
        ncnn::flatten(_weight_data, weight_data_flattened, opt);
        if (weight_data_flattened.empty())
            return -100;

        weight_data_flattened.w *= weight_data_flattened.elempack;
        weight_data_flattened.elemsize /= weight_data_flattened.elempack;
        weight_data_flattened.elempack = 1;

        ncnn::Mat bias_data_flattened;
        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Convolution);
        ncnn::ParamDict pd;
        pd.set(0, _num_output);
        pd.set(1, _kernel_w);
        pd.set(11, _kernel_h);
        pd.set(2, 2);
        pd.set(21, 2);
        pd.set(3, 1);
        pd.set(31, 1);
        pd.set(4, 0);
        pd.set(15, 0);
        pd.set(14, 0);
        pd.set(16, 0);
        pd.set(18, 0);
        pd.set(5, 0);
        pd.set(6, weight_data_flattened.w);
        pd.set(8, 0);
        op->load_param(pd);

        ncnn::Mat weights[2];
        weights[0] = weight_data_flattened;
        weights[1] = bias_data_flattened;
        op->load_model(ncnn::ModelBinFromMatArray(weights));
        op->create_pipeline(opt);


        // 循环生成，注意，这里的bottom_blob和top_blob_set可能会有elempack的问题
        std::vector<ncnn::Mat> top_blob_set(_num_input);
        for (int i = 0; i < _num_input; i++) {
            op->forward(bottom_blob.channel(i), top_blob_set[i], opt);
        }

        // 拼接结果
        std::vector<ncnn::Mat> cat_out(1);
        ncnn::Layer* cat = ncnn::create_layer(ncnn::LayerType::Concat);
        ncnn::ParamDict cat_pd;
        cat_pd.set(0, 0);
        cat->load_param(cat_pd);
        cat->create_pipeline(opt);
        cat->forward(top_blob_set, cat_out, opt);

        // reshape
        ncnn::Layer* reshape = ncnn::create_layer(ncnn::LayerType::Reshape);
        ncnn::ParamDict reshape_pd;
        reshape_pd.set(0, cat_out[0].w);
        reshape_pd.set(1, cat_out[0].h);
        reshape_pd.set(11, _num_output);
        reshape_pd.set(2, _num_input);
        reshape->load_param(reshape_pd);
        reshape->create_pipeline(opt);
        reshape->forward(cat_out[0], top_blobs[0], opt);


        // 释放
        reshape->destroy_pipeline(opt);
        delete reshape;

        cat->destroy_pipeline(opt);
        delete cat;

        op->destroy_pipeline(opt);
        delete op;

        return 0;
    }
};

DEFINE_LAYER_CREATOR(BConv2d1)

class BConv2d2 : public ncnn::Layer
{
public:
    BConv2d2()
    {
        one_blob_only = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& _weight_data = bottom_blobs[1];
        ncnn::Mat& top_blob = top_blobs[0];

        //printf("[BConv2d2] (%d,%d,%d)%d*(%d,%d,%d,%d)%d\n", bottom_blob.c, bottom_blob.h, bottom_blob.w, bottom_blob.elempack, _weight_data.c, _weight_data.d, _weight_data.h, _weight_data.w, _weight_data.elempack);


        // 创建conv2d算子
        const int _kernel_w = _weight_data.w;
        const int _kernel_h = _weight_data.h;
        const int _num_output = _weight_data.c * _weight_data.elempack;
        const int _num_input = bottom_blob.c;

        ncnn::Mat weight_data_flattened;
        ncnn::flatten(_weight_data, weight_data_flattened, opt);
        if (weight_data_flattened.empty())
            return -100;

        weight_data_flattened.w *= weight_data_flattened.elempack;
        weight_data_flattened.elemsize /= weight_data_flattened.elempack;
        weight_data_flattened.elempack = 1;

        ncnn::Mat bias_data_flattened;
        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Convolution);
        ncnn::ParamDict pd;
        pd.set(0, _num_output);
        pd.set(1, _kernel_w);
        pd.set(11, _kernel_h);
        pd.set(2, 1);
        pd.set(21, 1);
        pd.set(3, 1);
        pd.set(31, 1);
        pd.set(4, 1);
        pd.set(15, 1);
        pd.set(14, 1);
        pd.set(16, 1);
        pd.set(18, 0);
        pd.set(5, 0);
        pd.set(6, weight_data_flattened.w);
        op->load_param(pd);

        ncnn::Mat weights[2];
        weights[0] = weight_data_flattened;
        weights[1] = bias_data_flattened;
        op->load_model(ncnn::ModelBinFromMatArray(weights));
        op->create_pipeline(opt);


        // 循环生成，注意，这里的bottom_blob和top_blob_set可能会有elempack的问题
        std::vector<ncnn::Mat> top_blob_set(_num_input);
        for (int i = 0; i < _num_input; i++) {
            op->forward(bottom_blob.channel(i), top_blob_set[i], opt);
        }

        // 拼接结果
        std::vector<ncnn::Mat> cat_out(1);
        ncnn::Layer* cat = ncnn::create_layer(ncnn::LayerType::Concat);
        ncnn::ParamDict cat_pd;
        cat_pd.set(0, 0);
        cat->load_param(cat_pd);
        cat->create_pipeline(opt);
        cat->forward(top_blob_set, cat_out, opt);

        // reshape
        ncnn::Mat reshape_out;
        ncnn::Layer* reshape = ncnn::create_layer(ncnn::LayerType::Reshape);
        ncnn::ParamDict reshape_pd;
        reshape_pd.set(0, cat_out[0].w);
        reshape_pd.set(1, cat_out[0].h);
        reshape_pd.set(11, _num_output);
        reshape_pd.set(2, _num_input);
        reshape->load_param(reshape_pd);
        reshape->create_pipeline(opt);
        reshape->forward(cat_out[0], top_blob, opt);

        // 释放
        reshape->destroy_pipeline(opt);
        delete reshape;

        cat->destroy_pipeline(opt);
        delete cat;

        op->destroy_pipeline(opt);
        delete op;

        return 0;
    }
};

DEFINE_LAYER_CREATOR(BConv2d2)

void linspace(double* arr, double start, double end, int size)
{
    double step = (end - start) / (size - 1);
    for (int i = 0; i < size; i++)
    {
        arr[i] = start + i * step;
    }
}

void meshgrid(double* X, double* Y, double(*xx)[W], double(*yy)[W])
{
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            xx[i][j] = X[i];
            yy[i][j] = Y[j];
        }
    }
}

ncnn::Mat rand(int seed)
{
    cv::Mat cv_x(cv::Size(512, 1), CV_32FC4);
    cv::RNG rng(seed);
    rng.fill(cv_x, cv::RNG::NORMAL, 0, 1);
    ncnn::Mat x_mat(512, 1, (void*)cv_x.data);
    return x_mat.clone();
}

template <typename T>
int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "makeup", "JNI_OnLoad");

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "makeup", "JNI_OnUnload");
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL
Java_com_tencent_makeup_StableDiffusion_Init(JNIEnv *env, jobject thiz, jobject assetManager) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    // 加载mapping
    mapping.opt.lightmode = true;
    mapping.opt.use_vulkan_compute = false;
    mapping.opt.use_winograd_convolution = false;
    mapping.opt.use_sgemm_convolution = false;
    mapping.opt.use_fp16_packed = false;
    mapping.opt.use_fp16_storage = false;
    mapping.opt.use_fp16_arithmetic = false;
    mapping.opt.use_packing_layout = true;
    mapping.load_param(mgr,"mapping.param");
    mapping.load_model(mgr,"mapping.bin");

    // 加载generator
    generator.opt.lightmode = true;
    generator.opt.use_vulkan_compute = false;
    generator.opt.use_winograd_convolution = true;
    generator.opt.use_sgemm_convolution = true;
    generator.opt.use_fp16_packed = false;
    generator.opt.use_fp16_storage = false;
    generator.opt.use_fp16_arithmetic = false;
    generator.opt.use_packing_layout = true;
    generator.register_custom_layer("Where", Where_layer_creator);
    generator.register_custom_layer("ConvTranspose2d1", ConvTranspose2d1_layer_creator);
    generator.register_custom_layer("ConvTranspose2d2", ConvTranspose2d2_layer_creator);
    generator.register_custom_layer("BConv2d1", BConv2d1_layer_creator);
    generator.register_custom_layer("BConv2d2", BConv2d2_layer_creator);
    generator.load_param(mgr,"generator.param");
    generator.load_model(mgr,"generator.bin");

    // 创建gridsample算子
    gridsample_opt = generator.opt;
    gridsample = ncnn::create_layer(ncnn::LayerType::GridSample);
    ncnn::ParamDict gridsample_pd;
    gridsample->load_param(gridsample_pd);
    gridsample->create_pipeline(gridsample_opt);

    // 创建interp算子
    interp_opt = generator.opt;
    interp = ncnn::create_layer(ncnn::LayerType::Interp);
    ncnn::ParamDict interp_pd;
    interp_pd.set(0, 2);
    interp_pd.set(3, 128);
    interp_pd.set(4, 128);
    interp->load_param(interp_pd);
    interp->create_pipeline(interp_opt);

    // 预计算网格
    linspace(X, 0, H, H);
    linspace(Y, 0, W, W);
    meshgrid(X, Y, xx, yy);

    __android_log_print(ANDROID_LOG_ERROR, "DragGan", "Init");

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_makeup_StableDiffusion_gen(JNIEnv *env, jobject thiz, jobject show_bitmap,
                                            jint seed) {
    // TODO: implement gen()
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, show_bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return JNI_FALSE;

    // 随机数mapping得到初始w
    ncnn::Mat z = rand(int(seed));
    ncnn::Mat w0(512, 16);
    w0.fill<float>(1.0f);
    {
        ncnn::Mat output;
        ncnn::Extractor ex = mapping.create_extractor();
        ex.input("/mapping/Cast_output_0", z);
        ex.extract("/mapping/Sub_2_output_0", output);
        float* src = output.row(0);
        for (int i = 0; i < 16; i++) {
            float* dst = w0.row(i);
            for (int j = 0; j < 512; j++) {
                dst[j] = src[j];
            }
        }
    }
    ncnn::Mat w = w0.row_range(0, 6).clone();

    // 保存一下w
    w_tmp = w.clone();
    w0_tmp = w0.clone();

    // 生成初始图像
    {
        ncnn::Mat ws(512, 16);
        {
            for (int i = 0; i < 6; i++) {
                float* src = w.row(i);
                float* dst = ws.row(i);
                for (int j = 0; j < 512; j++) {
                    dst[j] = src[j];
                }
            }
            for (int i = 6; i < 16; i++) {
                float* src = w0.row(i);
                float* dst = ws.row(i);
                for (int j = 0; j < 512; j++) {
                    dst[j] = src[j];
                }
            }
        }

        ncnn::Mat img;
        {
            ncnn::Extractor ex = generator.create_extractor();
//            ex.set_light_mode(true);
            ex.input("in0", ws);
            ex.extract("out0", img);
        }

        const float _mean_[3] = { -128.0f / 127.5f, -128.0f / 127.5f, -128.0f / 127.5f };
        const float _norm_[3] = { 127.5f, 127.5f, 127.5f };
        img.substract_mean_normalize(_mean_, _norm_);

        showwing = img.clone();
        img.to_android_bitmap(env,show_bitmap,ncnn::Mat::PIXEL_RGB);
    }

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_tencent_makeup_StableDiffusion_setPoint(JNIEnv *env, jobject thiz, jobject show_bitmap,
                                                 jint x, jint y) {
    // TODO: implement setPoint()
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, show_bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return JNI_FALSE;

    if (points[0] == -1) {
        cv::Mat show_point(512, 512, CV_8UC3);
        showwing.to_pixels(show_point.data, ncnn::Mat::PIXEL_RGB);
        cv::circle(show_point, cv::Point(x, y), 5, cv::Scalar(255, 0, 0), -1);

        points[0] = y;
        points[1] = x;

        ncnn::Mat show = ncnn::Mat::from_pixels(show_point.data,ncnn::Mat::PIXEL_RGB,512,512);
        show.to_android_bitmap(env,show_bitmap,ncnn::Mat::PIXEL_RGB);

        return JNI_TRUE;
    }


    cv::Mat show_point(512, 512, CV_8UC3);
    showwing.to_pixels(show_point.data, ncnn::Mat::PIXEL_RGB);
    cv::circle(show_point, cv::Point(int(points[1]), int(points[0])), 5, cv::Scalar(255, 0, 0), -1);
    cv::circle(show_point, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);

    targets[0] = y;
    targets[1] = x;

    ncnn::Mat show = ncnn::Mat::from_pixels(show_point.data,ncnn::Mat::PIXEL_RGB,512,512);
    show.to_android_bitmap(env,show_bitmap,ncnn::Mat::PIXEL_RGB);

    return JNI_TRUE;

}

JNIEXPORT jboolean JNICALL
Java_com_tencent_makeup_StableDiffusion_clean(JNIEnv *env, jobject thiz, jobject show_bitmap) {
    // TODO: implement clean()
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, show_bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return JNI_FALSE;

    points[0] = -1;
    points[1] = -1;
    targets[0] = -1;
    targets[1] = -1;

    showwing.to_android_bitmap(env,show_bitmap,ncnn::Mat::PIXEL_RGB);

    return JNI_TRUE;

}

}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_tencent_makeup_StableDiffusion_drag(JNIEnv *env, jobject thiz, jobject show_bitmap,
                                             jint step) {
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, show_bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return JNI_FALSE;

    // TODO: implement drag()
    if(step==0){
        feat_refs.release();
        w = w_tmp.clone();
        w0 = w0_tmp.clone();
    }

    ncnn::Mat ws(512, 16);
    {
        for (int i = 0; i < 6; i++) {
            float* src = w.row(i);
            float* dst = ws.row(i);
            for (int j = 0; j < 512; j++) {
                dst[j] = src[j];
            }
        }
        for (int i = 6; i < 16; i++) {
            float* src = w0.row(i);
            float* dst = ws.row(i);
            for (int j = 0; j < 512; j++) {
                dst[j] = src[j];
            }
        }
    }

    {
        ncnn::Mat feat5;
        ncnn::Extractor ex = generator.create_extractor();
//        ex.set_light_mode(false);
        ex.input("in0", ws);
        ex.extract("out1", feat5);

        ncnn::Mat feat_resize;
        ncnn::resize_bilinear(feat5, feat_resize, W, H);

        // 第一次的话要记录feature
        if (feat_refs.empty()) {
            feat_refs.create(256);
            for (int i = 0; i < 256; i++) {
                feat_refs[i] = feat_resize.channel(i).row(int(std::round(points[0])))[int(std::round(points[1]))];
            }
        }

        // Point tracking with feature matching
        int r = std::round(r2 / 512.0 * H);
        int up = std::max(points[0] - r, 0.0);
        int down = std::min(points[0] + r + 1, double(H));
        int left = std::max(points[1] - r, 0.0);
        int right = std::min(points[1] + r + 1, double(W));
        int height_patch = down - up;
        int width_patch = right - left;
        float min_value = 1e8;
        int min_y = -1, min_x = -1;
        for (int h = 0; h < height_patch; h++) {
            for (int w = 0; w < width_patch; w++) {
                float tmp = 0.0f;
                for (int c = 0; c < 256; c++) {
                    tmp += std::pow(feat_resize.channel(c).row(up + h)[left + w] - feat_refs[c], 2);
                }
                tmp = std::sqrt(tmp);
                if ((min_y == -1 && min_x == -1) || tmp < min_value) {
                    min_value = tmp;
                    min_y = up + h;
                    min_x = left + w;
                }
            }
        }
        points[0] = min_y;
        points[1] = min_x;

        __android_log_print(ANDROID_LOG_ERROR, "DragGan", "[Drag %d]current:(%d,%d), target:(%d,%d)",step,int(points[0]),int(points[1]),int(targets[0]),int(targets[1]));
        // show intermediate
        {
        	ncnn::Mat img;
        	ex.extract("out0", img);
        	const float _mean_[3] = { -128.0f / 127.5f, -128.0f / 127.5f, -128.0f / 127.5f };
        	const float _norm_[3] = { 127.5f, 127.5f, 127.5f };
        	img.substract_mean_normalize(_mean_, _norm_);
        	cv::Mat image(512, 512, CV_8UC3);
        	img.to_pixels(image.data, ncnn::Mat::PIXEL_RGB);
        	cv::circle(image, cv::Point(int(points[1]), int(points[0])), 5, cv::Scalar(255, 0, 0), -1);
        	cv::circle(image, cv::Point(int(targets[1]), int(targets[0])), 5, cv::Scalar(0, 255, 0), -1);
            ncnn::Mat show = ncnn::Mat::from_pixels(image.data,ncnn::Mat::PIXEL_RGB,512,512);
            show.to_android_bitmap(env,show_bitmap,ncnn::Mat::PIXEL_RGB);
        }

        // Motion supervision
        double direction[2] = { targets[1] - points[1],targets[0] - points[0] };
        if (std::sqrt(std::pow(direction[0], 2) + std::pow(direction[1], 2)) > 1) {

            std::vector<int> relis, reljs;
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    if (std::sqrt(std::pow(xx[h][w] - points[0], 2) + std::pow(yy[h][w] - points[1], 2)) < std::round(r1 / 512.0 * H)) {
                        relis.push_back(h);
                        reljs.push_back(w);
                    }
                }
            }

            double direction_norm = std::sqrt(std::pow(direction[0], 2) + std::pow(direction[1], 2));
            direction[0] /= direction_norm;
            direction[1] /= direction_norm;

            ncnn::Mat grid;
            grid.create(2, int(relis.size()), 1, (size_t)4u);
            for (int w = 0; w < relis.size(); w++) {
                grid.channel(0).row(w)[0] = (reljs[w] - direction[0]) / (W - 1) * 2 - 1;
                grid.channel(0).row(w)[1] = (relis[w] - direction[1]) / (H - 1) * 2 - 1;
            }

            std::vector<ncnn::Mat> inputs(2);
            inputs[0] = feat_resize;
            inputs[1] = grid;
            std::vector<ncnn::Mat> outputs(1);
            gridsample->forward(inputs, outputs, gridsample_opt);
            ncnn::Mat& target = outputs[0];

            ncnn::Mat feat5_grad(512, 512, 256, (size_t)4u);
            for (int i = 0; i < relis.size(); i++) {
                for (int c = 0; c < 256; c++) {
                    feat5_grad.channel(c).row(relis[i])[reljs[i]] = sign(feat_resize.channel(c).row(relis[i])[reljs[i]] - target.channel(c).row(0)[i]) / 256.0;
                }
            }
            ncnn::Mat feat5_grad_fit;
            interp->forward(feat5_grad, feat5_grad_fit, interp_opt);

            ex.input("in1", feat5_grad_fit);
            std::vector<ncnn::Mat> vg(6);
            ex.extract("out2", vg[0]);
            ex.extract("out3", vg[1]);
            ex.extract("out4", vg[2]);
            ex.extract("out5", vg[3]);
            ex.extract("out6", vg[4]);
            ex.extract("out7", vg[5]);

            // update w
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 512; j++) {
                    w.row(i)[j] = w.row(i)[j] - lr * vg[i].row(0)[j] * w.row(i)[j];
                }
            }

            return JNI_FALSE;
        }
        else {
            return JNI_TRUE;
        }
    }

}