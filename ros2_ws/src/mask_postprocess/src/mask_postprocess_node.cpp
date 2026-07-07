// Cleans up the raw predicted mask before it's trusted for anything
// downstream: a per-pixel classifier will always produce some salt-and-pepper
// misclassified pixels, especially at class boundaries. Keeping only the
// largest connected defect component removes that noise without needing any
// extra model complexity or postprocessing hyperparameters beyond "how many
// components to keep."
//
// Subscribes:  /segmentation/mask            (sensor_msgs/Image, mono8, class indices 0/1/2)
// Publishes:   /segmentation/mask_filtered   (sensor_msgs/Image, mono8, noise-filtered)
//              /segmentation/defect_contour_count (std_msgs/Int32)

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/int32.hpp>

namespace {
constexpr int kDefectClass = 2;
constexpr int kPanelClass = 1;
}  // namespace

class MaskPostprocessNode : public rclcpp::Node {
 public:
  MaskPostprocessNode() : Node("mask_postprocess_node") {
    declare_parameter<int>("min_component_area", 20);
    min_component_area_ = get_parameter("min_component_area").as_int();

    mask_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/segmentation/mask", 10,
        std::bind(&MaskPostprocessNode::onMask, this, std::placeholders::_1));
    filtered_pub_ = create_publisher<sensor_msgs::msg::Image>("/segmentation/mask_filtered", 10);
    contour_count_pub_ = create_publisher<std_msgs::msg::Int32>("/segmentation/defect_contour_count", 10);

    RCLCPP_INFO(get_logger(), "mask_postprocess_node ready (min_component_area=%d)", min_component_area_);
  }

 private:
  void onMask(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    const cv::Mat& raw_mask = cv_ptr->image;

    // Isolate the defect class as a binary mask for connected-component analysis.
    cv::Mat defect_binary = (raw_mask == kDefectClass);

    cv::Mat labels, stats, centroids;
    int num_components = cv::connectedComponentsWithStats(defect_binary, labels, stats, centroids, 8, CV_32S);

    // Component 0 is always background in OpenCV's convention — keep only
    // real components (labels 1..num_components-1) at or above the area
    // threshold, which drops single-pixel misclassification noise.
    cv::Mat filtered_defect_binary = cv::Mat::zeros(defect_binary.size(), CV_8U);
    int kept_components = 0;
    for (int label = 1; label < num_components; ++label) {
      int area = stats.at<int>(label, cv::CC_STAT_AREA);
      if (area >= min_component_area_) {
        filtered_defect_binary.setTo(255, labels == label);
        ++kept_components;
      }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(filtered_defect_binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Rebuild the class-indexed mask: start from the original (for the
    // panel/background classes, which this node doesn't touch), then
    // overwrite the defect class with only the filtered, noise-free regions.
    cv::Mat cleaned_mask = raw_mask.clone();
    cleaned_mask.setTo(kPanelClass, raw_mask == kDefectClass);  // demote all original defect pixels...
    cleaned_mask.setTo(kDefectClass, filtered_defect_binary == 255);  // ...then restore only the kept ones

    auto out_msg = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::MONO8, cleaned_mask).toImageMsg();
    filtered_pub_->publish(*out_msg);

    std_msgs::msg::Int32 count_msg;
    count_msg.data = static_cast<int32_t>(contours.size());
    contour_count_pub_->publish(count_msg);
  }

  int min_component_area_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr mask_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr filtered_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr contour_count_pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MaskPostprocessNode>());
  rclcpp::shutdown();
  return 0;
}
