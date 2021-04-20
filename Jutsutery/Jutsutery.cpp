#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, const char** argv)
{
    Mat image;
    
    VideoCapture cap(1);
    if (!cap.isOpened()) {
        cerr << "No camera detected on this system" << endl;
        return -1;
    }

    while (true) {
        cap >> image;
        if (image.empty()) {
            cerr << "Frame invalid and skipped!" << endl;
            continue;
        }
        imshow("test", image);
        waitKey(5);
    }

    return 0;
}