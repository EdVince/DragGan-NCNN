#pragma once
#include <QtWidgets/QMainWindow>
#include "ui_mainwindow.h"
#include <QDebug>
#include <QVariant>
#include <QMouseEvent>

// AVOID moc parse error
#ifndef Q_MOC_RUN 
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <stack>
#include <fstream>
#include <map>
#include <math.h>
#include <algorithm>
#include <time.h>

#include <opencv2/opencv.hpp>

#include "ncnn/net.h"
#include "ncnn/layer.h"
#include "ncnn/layer_type.h"
#include "ncnn/benchmark.h"
#endif

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void showImage(cv::Mat in);

private slots:
    void on_getBtn_clicked();
    void on_cleanBtn_clicked();
    void on_dragBtn_clicked();

protected:
    void mousePressEvent(QMouseEvent* event);

private:
    Ui::MainWindowClass ui;

    cv::Mat showwing;
    double points[2] = { -1,-1 };
    double targets[2] = { -1,-1 };

    ncnn::Mat w_tmp;
    ncnn::Mat w0_tmp;

    ncnn::Net mapping;
    ncnn::Net generator;

    ncnn::Layer* gridsample;
    ncnn::Option gridsample_opt;

    ncnn::Layer* interp;
    ncnn::Option interp_opt;
};
