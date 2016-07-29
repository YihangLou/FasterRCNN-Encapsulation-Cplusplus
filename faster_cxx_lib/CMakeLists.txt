
#This part is used for compile faster_rcnn_demo.cpp
cmake_minimum_required (VERSION 2.8)

project (main_demo)

add_executable(main main.cpp)

include_directories ( "${PROJECT_SOURCE_DIR}/../caffe-fast-rcnn/include"
    "${PROJECT_SOURCE_DIR}/../lib/nms" 
    "${PROJECT_SOURCE_DIR}/lib" 
    /share/apps/local/include
    /usr/local/include 
    /opt/python/include/python2.7
    /share/apps/opt/intel/mkl/include 
    /usr/local/cuda/include )

target_link_libraries(main /home/lyh1/workspace/py-faster-rcnn/faster_cxx_lib/lib/libfaster_rcnn.so
    /home/lyh1/workspace/py-faster-rcnn/caffe-fast-rcnn/build/lib/libcaffe.so
    /home/lyh1/workspace/py-faster-rcnn/lib/nms/gpu_nms.so 
    /share/apps/local/lib/libopencv_highgui.so 
    /share/apps/local/lib/libopencv_core.so 
    /share/apps/local/lib/libopencv_imgproc.so 
    /share/apps/local/lib/libopencv_imgcodecs.so
    /share/apps/local/lib/libglog.so
    /share/apps/local/lib/libboost_system.so
    /share/apps/local/lib/libboost_python.so
    /share/apps/local/lib/libglog.so
    /opt/rh/python27/root/usr/lib64/libpython2.7.so
    )


