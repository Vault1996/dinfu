#include <utility>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
#include "glog/logging.h"

using namespace kfusion;

struct DynamicFusionApp {

    static void KeyboardCallback(const cv::viz::KeyboardEvent &event, void *pthis) {
        DynamicFusionApp &dynamicFusionApp = *static_cast<DynamicFusionApp *>(pthis);

        if (event.action != cv::viz::KeyboardEvent::KEY_DOWN) {
            return;
        }

        if (event.code == 't' || event.code == 'T') {
            dynamicFusionApp.show_warp(*dynamicFusionApp.dynfu_);
        }

        if (event.code == 'i' || event.code == 'I') {
            dynamicFusionApp.interactive_mode_ = !dynamicFusionApp.interactive_mode_;
        }
    }

    explicit DynamicFusionApp(std::string dir) : exit_(false), interactive_mode_(false), pause_(false), directory(true),
                                                 dir_name(std::move(dir)) {
        KinFuParams params = KinFuParams::default_params_dynamicfusion();
        dynfu_ = KinFu::Ptr(new KinFu(params));

        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        viz.showWidget("cube", cube, params.volume_pose);
        viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
        viz.registerKeyboardCallback(KeyboardCallback, this);

    }

    static void show_depth(const cv::Mat &depth) {
        cv::Mat display;
        depth.convertTo(display, CV_8U, 255.0 / 4000);
        cv::imshow("Depth", display);
        cvWaitKey(10);
    }

    void show_raycasted(KinFu &kinfu, int i) {

        const int mode = 3;
        if (interactive_mode_)
            kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
        else
            kinfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);

#ifdef OUTPUT_PATH
        std::string path = TOSTRING(OUTPUT_PATH) + std::to_string(i) + ".jpg";
        // cv::viz::writeCloud(path, dynamic_fusion.getWarp().getNodesAsMat());
        cv::imwrite(path, view_host_);
#endif

        cv::imshow("Scene", view_host_);
        cvWaitKey(100);

    }

    void show_warp(KinFu &kinfu) {
        cv::Mat warp_host = kinfu.getWarp().getNodesAsMat();
        viz.showWidget("warp_field", cv::viz::WCloud(warp_host));
    }

    bool execute() {

        KinFu &dynamic_fusion = *dynfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image;
        std::vector<cv::String> depths;
        std::vector<cv::String> images;

        cv::glob(dir_name + "/depth", depths);
        cv::glob(dir_name + "/color", images);

        std::sort(depths.begin(), depths.end());
        std::sort(images.begin(), images.end());

        for (int i = 0; i < depths.size() && !exit_ && !viz.wasStopped(); i++) {
            image = cv::imread(images[i], CV_LOAD_IMAGE_COLOR);
            depth = cv::imread(depths[i], CV_LOAD_IMAGE_ANYDEPTH);
            // saves input data in cuda memory
            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

            has_image = dynamic_fusion(depth_device_);

            if (has_image) {
                show_raycasted(dynamic_fusion, i);
            }

            show_depth(depth);
            cv::imshow("Image", image);

            if (!interactive_mode_) {
                viz.setViewerPose(dynamic_fusion.getCameraPose());
            }

            int key = cv::waitKey(pause_ ? 0 : 3);
            show_warp(dynamic_fusion);
            switch (key) {
                case 'i':
                case 'I' :
                    interactive_mode_ = !interactive_mode_;
                    break;
                case 27:
                    exit_ = true;
                    break;
                case 32:
                    pause_ = !pause_;
                    break;
                default:
                    break;
            }
            viz.spinOnce(3, true);
            // exit_ = exit_ || i > 40;
        }

        return true;
    }

    bool pause_;
    bool exit_, interactive_mode_, directory;
    std::string dir_name;
    KinFu::Ptr dynfu_;
    cv::viz::Viz3d viz;

    cv::Mat view_host_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;


};

int main(int argc, char *argv[]) {

    int device = 0;
    cudaSafeCall(cudaSetDevice(0));
    kf::cuda::setDevice(device);
    kf::cuda::printShortCudaDeviceInfo(device);

    if (kf::cuda::checkIfPreFermiGPU(device)) {
        return std::cout << std::endl
                         << "DynamicFusion is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..."
                         << std::endl, 1;
    }

    assert(argc == 2 && "Usage: ./dynamicfusion <data-directory>");

    DynamicFusionApp app(argv[1]);

    try {
        app.execute();
    } catch (const std::exception &e) {
        DLOG(FATAL) << std::string("Exception") + e.what() + "\n";
        std::cerr << "Something went wrong.\n";
    }

    return 0;
}
