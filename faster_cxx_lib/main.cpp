#include "faster_rcnn.hpp"
int main()
{
	string model_file = "/home/lyh1/workspace/py-faster-rcnn/models/pascal_voc/VGG_CNN_M_1024/faster_rcnn_alt_opt/faster_rcnn_test.pt";
	string weights_file = "/home/lyh1/workspace/py-faster-rcnn/output/default/yuanzhang_car/vgg_cnn_m_1024_fast_rcnn_stage2_iter_40000.caffemodel";
    int GPUID=0;
	Caffe::SetDevice(GPUID);
	Caffe::set_mode(Caffe::GPU);
	Detector det = Detector(model_file, weights_file);
	det.Detect("/home/lyh1/workspace/py-faster-rcnn/data/demo/car.jpg");
    return 0;
}
