#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

const int POSE_PAIRS[20][2] =
{
    {0,1}, {1,2}, {2,3}, {3,4},         // thumb
    {0,5}, {5,6}, {6,7}, {7,8},         // index
    {0,9}, {9,10}, {10,11}, {11,12},    // middle
    {0,13}, {13,14}, {14,15}, {15,16},  // ring
    {0,17}, {17,18}, {18,19}, {19,20}   // small
};

string protoFile = "hand/pose_deploy.prototxt";
string weightsFile = "hand/pose_iter_102000.caffemodel";

int nPoints = 22;

vector<Point2f> prevPoints;
vector<Point2f> nextPoints;
Mat prevInput;
Mat nextInput, frameCopy;
vector<uchar> status;
vector<float> err;



void detectPoints(Mat& img)
{
    cv::goodFeaturesToTrack(img, prevPoints, 500, 0.01, 1);
}

std::vector<cv::Point2f> purgePoints(std::vector<cv::Point2f>& points, std::vector<uchar>& status) {
    std::vector<cv::Point2f> result;
    for (int i = 0; i < points.size(); ++i) {
        if (status[i] > 0)result.push_back(points[i]);
    } return result;
}

void trackPoints()
{
    if (!prevInput.empty())
    {
        prevPoints = nextPoints;
        if (!(prevPoints.size() > 0))
        {
            detectPoints(prevInput);
            cout << nextPoints.size();
            if (!(prevPoints.size() > 0))
                return;
        }
        cv::calcOpticalFlowPyrLK(prevInput, nextInput, prevPoints, nextPoints, status, err);

        purgePoints(nextPoints, status);
        purgePoints(prevPoints, status);
    }
    
}

int main(int argc, char** argv)
{
    float thresh = 0.01;

    cv::VideoCapture cap(1);

    if (!cap.isOpened())
    {
        cerr << "Unable to connect to camera" << endl;
        return 1;
    }

    
    int frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    float aspect_ratio = frameWidth / (float)frameHeight;
    int inHeight = 64;
    int inWidth = (int(aspect_ratio * inHeight) * 8) / 8;

    cout << "inWidth = " << inWidth << " ; inHeight = " << inHeight << endl;

    VideoWriter video("Output-Skeleton.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frameWidth, frameHeight));

    Net net = readNetFromCaffe(protoFile, weightsFile);

    double t = 0;
    while (1)
    {
        double t = (double)cv::getTickCount();

        cap >> nextInput;
        frameCopy = nextInput.clone();
        Mat inpBlob = blobFromImage(nextInput, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);

        /*net.setInput(inpBlob);

        Mat output = net.forward();

        int H = output.size[2];
        int W = output.size[3];

        // find the position of the body parts
        vector<Point> points(nPoints);
        for (int n = 0; n < nPoints; n++)
        {
            // Probability map of corresponding body's part.
            Mat probMap(H, W, CV_32F, output.ptr(0, n));
            resize(probMap, probMap, Size(frameWidth, frameHeight));

            Point maxLoc;
            double prob;
            minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
            if (prob > thresh)
            {
                circle(frameCopy, cv::Point((int)maxLoc.x, (int)maxLoc.y), 8, Scalar(0, 255, 255), -1);
                cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)maxLoc.x, (int)maxLoc.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            }
            points[n] = maxLoc;
        }

        int nPairs = sizeof(POSE_PAIRS) / sizeof(POSE_PAIRS[0]);

        for (int n = 0; n < nPairs; n++)
        {
            // lookup 2 connected body/hand parts
            Point2f partA = points[POSE_PAIRS[n][0]];
            Point2f partB = points[POSE_PAIRS[n][1]];

            if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
                continue;

            line(frame, partA, partB, Scalar(0, 255, 255), 8);
            circle(frame, partA, 8, Scalar(0, 0, 255), -1);
            circle(frame, partB, 8, Scalar(0, 0, 255), -1);
        }*/

        cvtColor(nextInput, nextInput, COLOR_BGR2GRAY);
        trackPoints();
        prevInput = nextInput.clone();
        cvtColor(nextInput, nextInput, COLOR_GRAY2BGR);
        cout << nextPoints.size();
        for (size_t i = 0; i < prevPoints.size(); i++)
        {
            circle(nextInput, prevPoints[i], 10, Scalar(0, 0, 255));
            line(nextInput, prevPoints[i], nextPoints[i], Scalar(0, 255, 0));
            circle(nextInput, nextPoints[i], 10, Scalar(0, 0, 255));
        }

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "Time Taken for frame = " << t << endl;
        cv::putText(nextInput, cv::format("time taken = %.2f sec", t), cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(255, 50, 0), 2);
        // imshow("Output-Keypoints", frameCopy);
        imshow("Output-Skeleton", nextInput);
        video.write(nextInput);
        char key = waitKey(1);
        if (key == 27)
            break;
    }
    // When everything done, release the video capture and write object
    cap.release();
    video.release();

    return 0;
}
