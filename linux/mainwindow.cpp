#include "mainwindow.h"

const int H = 512;
const int W = 512;

double X[H], Y[W];
double xx[H][W], yy[H][W];
double distance[H][W];

class Where : public ncnn::Layer
{
public:
	Where()
	{
		one_blob_only = true;
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


		// ����conv2d����
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


		// ѭ�����ɣ�ע�⣬�����bottom_blob��top_blob_set���ܻ���elempack������
		std::vector<ncnn::Mat> top_blob_set(_num_input);
		for (int i = 0; i < _num_input; i++) {
			op->forward(bottom_blob.channel(i), top_blob_set[i], opt);
		}

		// ƴ�ӽ��
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


		// �ͷ�
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


		// ����conv2d����
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


		// ѭ�����ɣ�ע�⣬�����bottom_blob��top_blob_set���ܻ���elempack������
		std::vector<ncnn::Mat> top_blob_set(_num_input);
		for (int i = 0; i < _num_input; i++) {
			op->forward(bottom_blob.channel(i), top_blob_set[i], opt);
		}

		// ƴ�ӽ��
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

		// �ͷ�
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

ncnn::Mat rand(int seed)
{
	cv::Mat cv_x(cv::Size(512, 1), CV_32FC4);
	cv::RNG rng(seed);
	rng.fill(cv_x, cv::RNG::NORMAL, 0, 1);
	ncnn::Mat x_mat(512, 1, (void*)cv_x.data);
	return x_mat.clone();
}

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

template <typename T>
int sign(T val) {
	return (T(0) < val) - (val < T(0));
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

	// ����mapping
	mapping.load_param("assets/mapping.param");
	mapping.load_model("assets/mapping.bin");

	// ����generator
	generator.register_custom_layer("Where", Where_layer_creator);
	generator.register_custom_layer("ConvTranspose2d1", ConvTranspose2d1_layer_creator);
	generator.register_custom_layer("ConvTranspose2d2", ConvTranspose2d2_layer_creator);
	generator.register_custom_layer("BConv2d1", BConv2d1_layer_creator);
	generator.register_custom_layer("BConv2d2", BConv2d2_layer_creator);
	generator.load_param("assets/generator.param");
	generator.load_model("assets/generator.bin");

	// ����gridsample����
	gridsample_opt = generator.opt;
	gridsample = ncnn::create_layer(ncnn::LayerType::GridSample);
	ncnn::ParamDict gridsample_pd;
	gridsample->load_param(gridsample_pd);
	gridsample->create_pipeline(gridsample_opt);

	// ����interp����
	interp_opt = generator.opt;
	interp = ncnn::create_layer(ncnn::LayerType::Interp);
	ncnn::ParamDict interp_pd;
	interp_pd.set(0, 2);
	interp_pd.set(3, 128);
	interp_pd.set(4, 128);
	interp->load_param(interp_pd);
	interp->create_pipeline(interp_opt);

	// Ԥ��������
	linspace(X, 0, H, H);
	linspace(Y, 0, W, W);
	meshgrid(X, Y, xx, yy);
}

MainWindow::~MainWindow()
{
	gridsample->destroy_pipeline(gridsample_opt);
	delete gridsample;

	interp->destroy_pipeline(interp_opt);
	delete interp;
}

void MainWindow::showImage(cv::Mat in)
{
	cv::Mat show;
	cv::resize(in, show, cv::Size(1024, 1024));
	QImage qImage(show.data, show.cols, show.rows, static_cast<int>(show.step), QImage::Format_RGB888);
	ui.show->setPixmap(QPixmap::fromImage(qImage));
	ui.show->show();
}

void MainWindow::on_getBtn_clicked()
{
	int seed = QVariant(ui.seedEdit->text()).toInt();
	
	// �����mapping�õ���ʼw
	ncnn::Mat z = rand(seed);
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

	// ����һ��w
	w_tmp = w.clone();
	w0_tmp = w0.clone();

	// ���ɳ�ʼͼ��
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
			ex.set_light_mode(true);
			ex.input("in0", ws);
			ex.extract("out0", img);
		}
		const float _mean_[3] = { -128.0f / 127.5f, -128.0f / 127.5f, -128.0f / 127.5f };
		const float _norm_[3] = { 127.5f, 127.5f, 127.5f };
		img.substract_mean_normalize(_mean_, _norm_);
		cv::Mat image(512, 512, CV_8UC3);
		img.to_pixels(image.data, ncnn::Mat::PIXEL_RGB);
		showwing = image.clone();

		showImage(showwing);

		qDebug() << "[Init] seed:" << seed;
	}
}

void MainWindow::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {

		QPoint globalPos = event->globalPos();

		QPoint labelPos = ui.show->mapToGlobal(QPoint(0, 0));
		int labelWidth = ui.show->width();
		int labelHeight = ui.show->height();

		int relativeX = globalPos.x() - labelPos.x();
		int relativeY = globalPos.y() - labelPos.y();

		relativeX = relativeX / 2;
		relativeY = relativeY / 2;

		if (points[0] == -1) {
			cv::Mat show_point = showwing.clone();
			cv::circle(show_point, cv::Point(relativeX, relativeY), 3, cv::Scalar(255, 0, 0), -1);
			showImage(show_point);

			points[0] = relativeY;
			points[1] = relativeX;

			qDebug() << "[Choose] start point: (" << points[0] << "," << points[1] << ")";
		}
		else if (targets[0] == -1) {
			cv::Mat show_point = showwing.clone();
			cv::circle(show_point, cv::Point(int(points[1]), int(points[0])), 3, cv::Scalar(255, 0, 0), -1);
			cv::circle(show_point, cv::Point(relativeX, relativeY), 3, cv::Scalar(0, 255, 0), -1);
			showImage(show_point);

			targets[0] = relativeY;
			targets[1] = relativeX;

			qDebug() << "[Choose] target point: (" << targets[0] << "," << targets[1] << ")";
		}

	}
}

void MainWindow::on_cleanBtn_clicked()
{
	points[0] = -1;
	points[1] = -1;
	targets[0] = -1;
	targets[1] = -1;
	showImage(showwing);
	qDebug() << "[Clean] point";
}

void MainWindow::on_dragBtn_clicked()
{
	float lr = 0.1;
	ncnn::Mat feat_refs;
	int r1 = 3, r2 = 12;

	ncnn::Mat w = w_tmp.clone();
	ncnn::Mat w0 = w0_tmp.clone();

	for (int it = 0; it < 200; it++)
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

		{
			ncnn::Mat feat5;
			ncnn::Extractor ex = generator.create_extractor();
			ex.set_light_mode(false);
			ex.input("in0", ws);
			ex.extract("out1", feat5);

			ncnn::Mat feat_resize;
			ncnn::resize_bilinear(feat5, feat_resize, W, H);

			// ��һ�εĻ�Ҫ��¼feature
			if (feat_refs.empty()) {
				feat_refs.create(256);
				for (int i = 0; i < 256; i++) {
					feat_refs[i] = feat_resize.channel(i).row(int(std::round(points[0])))[int(std::round(points[1]))];
				}
			}

			// Point tracking with feature matching
			int r = std::round(r2 / 512.0 * H);
			int up = std::max(points[0] - r, (double)0.0);
			int down = std::min(points[0] + r + 1, (double)H);
			int left = std::max(points[1] - r, (double)0.0);
			int right = std::min(points[1] + r + 1, (double)W);
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

			qDebug() << "[Drag " << it << "]current:(" << int(points[0]) << "," << int(points[1]) << "), target:(" << targets[0] << "," << targets[1] << ")";

			//// save intermediate
			//{
			//	ncnn::Mat img;
			//	ex.extract("out0", img);
			//	const float _mean_[3] = { -128.0f / 127.5f, -128.0f / 127.5f, -128.0f / 127.5f };
			//	const float _norm_[3] = { 127.5f, 127.5f, 127.5f };
			//	img.substract_mean_normalize(_mean_, _norm_);
			//	cv::Mat image(512, 512, CV_8UC3);
			//	img.to_pixels(image.data, ncnn::Mat::PIXEL_RGB2BGR);
			//	cv::circle(image, cv::Point(int(points[1]), int(points[0])), 3, cv::Scalar(0, 0, 255), -1);
			//	cv::circle(image, cv::Point(int(targets[1]), int(targets[0])), 3, cv::Scalar(0, 255, 0), -1);
			//	cv::imwrite("images/" + std::to_string(it) + ".png", image);
			//}

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
			}
			else {
				qDebug() << "[Finish]";

				ncnn::Mat img;
				ex.extract("out0", img);
				const float _mean_[3] = { -128.0f / 127.5f, -128.0f / 127.5f, -128.0f / 127.5f };
				const float _norm_[3] = { 127.5f, 127.5f, 127.5f };
				img.substract_mean_normalize(_mean_, _norm_);
				cv::Mat image(512, 512, CV_8UC3);
				img.to_pixels(image.data, ncnn::Mat::PIXEL_RGB);
				showwing = image.clone();

				showImage(showwing);

				break;
			}

			if (it == 200 - 1) {
				ncnn::Mat img;
				ex.extract("out0", img);
				const float _mean_[3] = { -128.0f / 127.5f, -128.0f / 127.5f, -128.0f / 127.5f };
				const float _norm_[3] = { 127.5f, 127.5f, 127.5f };
				img.substract_mean_normalize(_mean_, _norm_);
				cv::Mat image(512, 512, CV_8UC3);
				img.to_pixels(image.data, ncnn::Mat::PIXEL_RGB);
				showwing = image.clone();

				showImage(showwing);
			}
		}
	}



}