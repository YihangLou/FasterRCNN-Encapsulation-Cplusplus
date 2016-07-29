#include "faster_rcnn.hpp"
int main()
{
	string model_file = "/home/lyh1/workspace/py-faster-rcnn/models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_alt_opt/faster_rcnn_test.pt";
	string weights_file = "/home/lyh1/workspace/py-faster-rcnn/output/default/yuanzhang_car/vgg_cnn_m_1024_fast_rcnn_stage2_iter_40000.caffemodel";
    int GPUID=0;
    int max_ret_num=30;

	Detector * handle = NULL;
    EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(), GPUID , handle);
    vector<cv::Rect> detection_result;
    cv::Mat inputimage = cv::imread("/home/lyh1/workspace/py-faster-rcnn/data/demo/car.jpg");
    handle->Detect(inputimage, detection_result);
    for(int i=0;i < detection_result.size(); i++){
        cv::rectangle(inputimage,cv::Point(detection_result[i].x,detection_result[i].y),
                                 cv::Point(detection_result[i].x + detection_result[i].width,detection_result[i].y + detection_result[i].height),
                                 cv::Scalar(0,255,0));

    }
    cv::imwrite("test.jpg",inputimage);
    return 0;
}
