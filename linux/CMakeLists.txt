cmake_minimum_required(VERSION 3.10) 

set(ProjectName "DragGan-NCNN")
project(${ProjectName}) 
message("ProjectName: ${ProjectName}")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(ncnn_DIR ${PROJECT_SOURCE_DIR}/lib/cmake/ncnn)  
# add your own path to qt5 
set(CMAKE_PREFIX_PATH "[PATH TO]/qt5/5.14.2/gcc_64/lib/cmake/")

add_definitions("-fPIC")


find_package(ncnn REQUIRED)
find_package(Qt5 COMPONENTS
        Core
        Gui
        Widgets
        REQUIRED)

find_package(OpenCV REQUIRED)


add_executable(${ProjectName} 
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    mainwindow.qrc
) 

target_include_directories(${ProjectName} PUBLIC 
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${ProjectName} 
    ${OpenCV_LIBS}
    ncnn
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
) 

