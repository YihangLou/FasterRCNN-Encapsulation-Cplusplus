#ifndef FASTER_RCNN_HPP
#define FASTER_RCNN_HPP
#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <boost/python.hpp>
#include "caffe/caffe.hpp"
#include "gpu_nms.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace caffe;
using namespace std;

#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))
#define     EV2641_ERR_SUCCESS                          0
#define     EV2641_ERR_IMAGE                            -13
#define     EV2641_ERR_MODEL                            -14

int EV2641_InitCarDetector(char * cardetect_model, int ObjSize ,hcd_t* phcd);
int EV2641_ReleaseCarDetector(hcd_t hcd);
int EV2641_A_GetCarRect(hcd_t hcd, const EV2641Image * image, int &max_ret_num, EV2641Rect * rect);










//background and car
const int class_num=2;

/*
 * ===  Class  ======================================================================
 *         Name:  Detector
 *  Description:  FasterRCNN CXX Detector
 * =====================================================================================
 */
class Detector {
public:
	Detector(const string& model_file, const string& weights_file);
	void Detect(const string& im_name);
	void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width);
	void vis_detections(cv::Mat image, int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH);
	void boxes_sort(int num, const float* pred, float* sorted_pred);

private:
	shared_ptr<Net<float> > net_;
	Detector(){}
};

//Using for box sort
struct Info
{
	float score;
	const float* head;
};
bool compare(const Info& Info1, const Info& Info2)
{
	return Info1.score > Info2.score;
}

#endif
