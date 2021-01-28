#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/video/tracking.hpp>
#include <windows.h>

using namespace std;
using namespace cv;

void show(Mat img);
void show2(Mat img1, Mat img2);
Mat fft(Mat img);
bool cropGoodImagePartUsingKeyPoint(Mat img, Mat& result, int width, int height);

Mat poc(Mat ref_img, Mat tgt_img);
Mat ripoc(Mat ref_img, Mat tgt_img);
Mat ecc(Mat ref_img, Mat tgt_img, int number_of_iterations);

void test(Mat* ref_img, Mat* tgt_img, Mat* Dst);

Point2f getSubPixelUsingParabolaFitting(Mat match_result, Point2f &sub_pixel);
float getEvaluationValueOfAlignmentUsingSubPixel(Mat ref_img, Mat tgt_img, Mat match_img);

//cv::mat.type()を渡すと型を教えてくれる便利関数
string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

int main()
{
 //   //グレスケで読み込む
    Mat ref_img = imread(R"(C:\Users\Adacotech\Documents\GitHub\Python-Scripts\script\Landolt.png)", IMREAD_GRAYSCALE);
    Mat tgt_img = imread(R"(C:\Users\Adacotech\Documents\GitHub\Python-Scripts\script\ShiftLandolt.png)", IMREAD_GRAYSCALE);
    //Mat ref_img = imread(R"(C:\Users\Adacotech\Desktop\cpp_source\ref.png)", IMREAD_GRAYSCALE);
    //Mat tgt_img = imread(R"(C:\Users\Adacotech\Desktop\cpp_source\tgt.png)", IMREAD_GRAYSCALE);
    Mat aligned_img;

    //resize(ref_img, ref_img, cv::Size(), 10, 10);
    //resize(tgt_img, tgt_img, cv::Size(), 10, 10);

    show2(ref_img, tgt_img);
    
    //Mat output_ref, output_tgt;
    //Point pos_ref, pos_tgt; //探索結果の位置
    //Point2f sub_pixel_pos_ref, sub_pixel_pos_tgt; //サブピクセル

    ////テンプレートマッチング
    //matchTemplate(ref_img, tgt_img, output_ref, CV_TM_CCOEFF_NORMED);

    ////最大マッチ座標を取得
    //minMaxLoc(output_ref, NULL, NULL, NULL, &pos_ref);

    //rectangle(ref_img, Rect(pos_ref.x, pos_ref.y, tgt_img.cols, tgt_img.rows), Scalar(0, 0, 255), 3);

    //Mat c_ref = ref_img.clone();
    //Mat c_tgt = tgt_img.clone();
    //Mat aligned_img;

    //double response;
    ////8UC1 -> 64FC1に変換するときには255で割らないとサチる。以下参考url
    ////https://stackoverflow.com/questions/33299374/opencv-convert-cv-8u-to-cv-64f
    ////→Fはfloatの意味でmatの範囲は[0:1]となるため255で割る必要がある。
    //ref_img.convertTo(ref_img, CV_64FC1, 1.0 / 255.0);
    //tgt_img.convertTo(tgt_img, CV_64FC1, 1.0 / 255.0);

    //Point2d shift = phaseCorrelate(ref_img, tgt_img);
    //printf("Shift X: %g\nShift Y: %g\n", shift.x, shift.y);

    ////affine変換行列(2*3)、横にa,縦にb動かしたい時は[ 1, 0, a],[ 0, 1, b]とする
    //Mat	M = (Mat_<double>(2, 3) << 1.0, 0.0, -shift.x, 0.0, 1.0, -shift.y);

    //warpAffine(tgt_img, aligned_img, M, aligned_img.size());

    // QueryPerformanceCounter関数の1秒当たりのカウント数を取得する
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

    for (int i = 0; i < 1 ; i++) {
        //aligned_img = poc(ref_img, tgt_img);
        //aligned_img = ripoc(ref_img, tgt_img);
        aligned_img = ecc(ref_img, tgt_img, 20);
    }
    
    //cropGoodImagePartUsingKeyPoint(ref_img, aligned_img, 100, 100);

    QueryPerformanceCounter(&end);
    double time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    printf("time %lf[ms]\n", time);

    show(aligned_img);

   /* Mat a;
    cropGoodImagePartUsingKeyPoint(ref_img, a, 100, 100);
    imwrite("C:\\Users\\Adacotech\\Desktop\\cpp_source\\tes.png", a);

    show(a);*/


    //Mat aligned_img = poc(ref_img, tgt_img);
 //   //Mat fft_im = fft(ref_img);
 //   //show(fft_im);

 //   if (ref_img.empty() || tgt_img.empty()) {
 //       cout << "画像が読み込めていません。";
 //       return -1;
 //   }
    //Mat* refMat = new Mat();
    //Mat* tgtMat = new Mat();
    //Mat* Dst = new Mat();

    //*refMat = imread("C:/cpp_source/ref.png", IMREAD_GRAYSCALE);
    //*tgtMat = imread("C:/cpp_source/shift.png", IMREAD_GRAYSCALE);



    ////show(*refMat);

    //test(refMat, tgtMat, Dst);

    //show(*Dst);

    ////処理時間計測（http://www.sanko-shoko.net/note.php?id=rnfd）
    //// QueryPerformanceCounter関数の1秒当たりのカウント数を取得する
    //LARGE_INTEGER freq, start, end;
    //QueryPerformanceFrequency(&freq);
    //QueryPerformanceCounter(&start);

    ////Mat aligned_img = poc(ref_img, tgt_img);
    //Mat aligned_img = ripoc(ref_img, tgt_img);
    ////Mat aligned_img = ecc(ref_img, tgt_img, 100);

    //QueryPerformanceCounter(&end);
    //double time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    //printf("time %lf[ms]\n", time);

    ////imwrite("C:/cpp_source/ecc.png", aligned_img);

    ////show(aligned_img);

    ////評価に使う画像を適当に切り取り
    //Mat cropped = Mat(ref_img, Rect(220, 250, 80, 80));

    //float distance = getEvaluationValueOfAlignmentUsingSubPixel(ref_img, aligned_img, cropped);
    //printf("distance : %lf\n", distance);
}

//任意の幅高さの画像サイズで特徴点の多い箇所を抽出する関数
bool cropGoodImagePartUsingKeyPoint(Mat img, Mat& cropped, int width, int height) {
    int x = 0, y = 0;
    int good_X = 0, good_y = 0;
    int max_kp_num = 0;

    cv::Ptr<cv::AKAZE> algorithm = AKAZE::create();
    std::vector<cv::KeyPoint> keypoint;

    if (height > img.rows || width > img.cols) {
        return false;
    }
    while (y + height < img.rows) {
        while (x + width < img.cols) {
            //cropped = Mat(img, Rect(x, y, width, height));
            
            algorithm->detect(Mat(img, Rect(x, y, width, height)), keypoint);
            cout << keypoint.size() << endl;
            if (keypoint.size() > max_kp_num) {
                max_kp_num = keypoint.size();
                good_X = x;
                good_y = y;
            }
            x += width;
        }
        y += height;
        x = 0;
    }

    cropped = Mat(img, Rect(good_X, good_y, width, height));

    cout << good_X << good_y <<endl;

    return true;
}


//位相限定相関法
Mat poc(Mat ref_img, Mat tgt_img) {

    // 出力先変数定義
    Mat aligned_img;
    double response;

    // Float型に変換したため、matの範囲は[0:1]となり、255で割る必要がある。
    ref_img.convertTo(ref_img, CV_64FC1, 1.0 / 255.0);
    tgt_img.convertTo(tgt_img, CV_64FC1, 1.0 / 255.0);

    // 位相限定相関法
    Point2d shift = phaseCorrelate(ref_img, tgt_img);

    // affine変換による平行移動
    Mat M = (Mat_<double>(2, 3) << 1.0, 0.0, -shift.x, 0.0, 1.0, -shift.y);
    warpAffine(tgt_img, aligned_img, M, aligned_img.size());

    return aligned_img;
}

void test(Mat* ref_img, Mat* tgt_img, Mat* Dst) {
    Mat c_ref(*ref_img);
    Mat c_tgt(*tgt_img);

    c_ref.convertTo(c_ref, CV_64FC1, 1.0 / 255.0);
    c_tgt.convertTo(c_tgt, CV_64FC1, 1.0 / 255.0);

    cv::Point shift = phaseCorrelate(c_ref, c_tgt);
    //printf("Shift X: %g\nShift Y: %g\n", shift.x, shift.y);

    //affine変換行列(2*3)、横にa,縦にb動かしたい時は[ 1, 0, a],[ 0, 1, b]とする
    Mat	M = (Mat_<double>(2, 3) << 1.0, 0.0, -shift.x, 0.0, 1.0, -shift.y);

    warpAffine(*tgt_img, *Dst, M, c_ref.size());

    //return aligned_img;
    return ;

}

//回転不変位相限定相関法
//参考URL：https://python5.com/q/toahzppc
//https://www.soaristo.org/blog/archives/2015/02/150222.php
Mat ripoc(Mat ref_img, Mat tgt_img) {
    // 変数定義
    const int height = ref_img.rows;
    const int width = ref_img.cols;
    Point center = Point(width / 2, height / 2);
    const Size window_size(height, height);
    Mat aligned_img = Mat::zeros(tgt_img.size(), CV_64FC1);

    // グレースケールに変換＋Folat型に変換
    ref_img.convertTo(ref_img, CV_64FC1, 1.0 / 255.0);
    tgt_img.convertTo(tgt_img, CV_64FC1, 1.0 / 255.0);

    // フーリエ変換
    Mat ref_img_fft = fft(ref_img);
    Mat tgt_img_fft = fft(tgt_img);

    // 対数極座標変換された画像の入れ子
    Mat bg_ref = Mat::zeros(ref_img.size(), CV_64FC1);
    Mat bg_tgt = Mat::zeros(tgt_img.size(), CV_64FC1);

    // 窓関数定義
    Mat matHann;
    cv::createHanningWindow(matHann, window_size, CV_64FC1);

    // 対数極座標変換 (lanczos法補間)
    double l = sqrt(height * height + width * width);
    double m = l / log(l);
    int flags = INTER_LANCZOS4 + WARP_POLAR_LOG;
    warpPolar(ref_img_fft, bg_ref, window_size, center, m, flags);
    warpPolar(tgt_img_fft, bg_tgt, window_size, center, m, flags); 

    // 位相限定相関法
    Point2d pt = phaseCorrelate(bg_ref, bg_tgt);

    // 角度と大きさを導出
    float rotate = pt.y * 360 / width;
    float scale = exp(pt.x / m);

    // アフィン変換係数行列取得
    Mat M = getRotationMatrix2D(center, rotate, scale);
    Mat rotated_img = Mat::zeros(tgt_img.size(), CV_64FC1);

    // 角度と大きさを補正
    warpAffine(tgt_img, rotated_img, M, tgt_img.size());

    rotated_img.convertTo(rotated_img, CV_64FC1, 1.0 / 255.0);

    // 位相限定相関法
    pt = phaseCorrelate(ref_img, rotated_img);
    
    // affine変換行列を調整してaffine変換
    M.at<double>(0, 2) -= pt.x;
    M.at<double>(1, 2) -= pt.y;
    warpAffine(tgt_img, aligned_img, M, tgt_img.size());

    return aligned_img;
}

//ホモぐらら
//ver1. https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
Mat ecc(Mat ref_img, Mat tgt_img, int number_of_iterations) {
    Mat copy_ref = ref_img.clone();
    Mat copy_tgt = tgt_img.clone();
    Mat aligned_img = Mat(ref_img.rows, ref_img.cols, CV_32FC1);

    //cvtColor(copy_ref, copy_ref, CV_BGR2GRAY);
    //cvtColor(copy_tgt, copy_tgt, CV_BGR2GRAY);

    copy_ref.convertTo(copy_ref, CV_32FC1, 1.0 / 255.0);
    copy_tgt.convertTo(copy_tgt, CV_32FC1, 1.0 / 255.0);

    double termination_eps = 1e-10;
    int motionType = MOTION_HOMOGRAPHY;

    Mat warp_matrix = Mat::eye(3, 3, CV_32FC1);

    // criteriaを設定
    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations, termination_eps);
    
    // ECCアルゴリズムによる変換行列の作成       
    findTransformECC(
        copy_ref,
        copy_tgt,
        warp_matrix,
        motionType,
        criteria
    );

    warpPerspective(copy_ref, aligned_img, warp_matrix, tgt_img.size(), INTER_LINEAR + WARP_INVERSE_MAP);

    return aligned_img;
}

//テンプレートマッチングによる評価を行う関数
//テンプレートマッチング参考url
//http://imgprolab.sys.fit.ac.jp/~yama/imgproc/proc/Document_OpenCVforC_8_2017.pdf
float getEvaluationValueOfAlignmentUsingSubPixel(Mat ref_img, Mat tgt_img, Mat match_img) {
    Mat output_ref, output_tgt;
    Point pos_ref, pos_tgt; //探索結果の位置
    Point2f sub_pixel_pos_ref, sub_pixel_pos_tgt; //サブピクセル

    //テンプレートマッチング
    matchTemplate(ref_img, match_img, output_ref, CV_TM_CCOEFF_NORMED);
    matchTemplate(tgt_img, match_img, output_tgt, CV_TM_CCOEFF_NORMED);

    //最大マッチ座標を取得
    //minMaxLoc(output_ref, NULL, NULL, NULL, &pos_ref);
    //minMaxLoc(output_tgt, NULL, NULL, NULL, &pos_tgt);

    //パラボラフィッティングによるサブピクセル推定
    getSubPixelUsingParabolaFitting(output_ref, sub_pixel_pos_ref);
    getSubPixelUsingParabolaFitting(output_tgt, sub_pixel_pos_tgt);

    //入力画像に探索結果の位置に四角を描画
    //rectangle(tgt_img, Rect(sub_pixel_pos_tgt.x, sub_pixel_pos_tgt.y, match_img.cols, match_img.rows), Scalar(255, 255, 255), 2);
    //imwrite("C:/cpp_source/match_ripoc.png", tgt_img);

    //cout << sub_pixel_pos_tgt << endl;

    float distance = sqrt(pow((sub_pixel_pos_ref.x - sub_pixel_pos_tgt.x), 2) + pow((sub_pixel_pos_ref.y - sub_pixel_pos_tgt.y), 2));

    return distance;
}

//x座標の前後の点を使ったパラボラフィッティングによるサブピクセル推定
Point2f getSubPixelUsingParabolaFitting(Mat match_result, Point2f &sub_pixel) {
    Point pos;
    minMaxLoc(match_result, NULL, NULL, NULL, &pos);
    sub_pixel.y = pos.y;

    //cout << type2str(match_result.type()) << endl;

    float back_x, center_x, front_x;
    back_x   = match_result.at<float>(pos.y, pos.x - 1);
    center_x = match_result.at<float>(pos.y, pos.x);
    front_x  = match_result.at<float>(pos.y, pos.x + 1);

    sub_pixel.x = pos.x + ((back_x - front_x) / (2 * back_x - 4 * center_x + 2 * front_x));

    return sub_pixel;
}

void show(Mat img) {
    imshow("image", img);
    waitKey(0);
}
void show2(Mat img1, Mat img2) {
    imshow("image1", img1);
    imshow("image2", img2);
    waitKey(0);
}


//フーリエ変換する関数
//http://imgprolab.sys.fit.ac.jp/~yama/imgproc/proc/Document_OpenCVforC_7_2019.pdf
//https://qiita.com/exp/items/5c4bcd650e0e2ae26cc4
Mat fft(Mat src) {
    Mat padded;
    int m = getOptimalDFTSize(src.rows);
    int n = getOptimalDFTSize(src.cols);
    // 入力画像を中央に置き、周囲は0で埋める
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
    //dtfの結果は複素数（１要素につき２つ値がある）であり、また値の表現範囲が広い。
    //そのためfloatと複素数を保持するもので2枚作る
    Mat planes[] = { Mat_<double>(padded), Mat::zeros(padded.size(), CV_64F) };
    Mat complexI;
    // 2枚の画像を、2チャネルを持った1枚にする
    merge(planes, 2, complexI);
    // complexIにdftを適用し、complexIに結果を戻す
    dft(complexI, complexI);

    // 絶対値を計算し、logスケールにする
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
    //magI += Scalar::all(1);
    // 結果の値は大きすぎるものと小さすぎるものが混じっているので、logを適用して抑制する                  
    log(magI, magI);
    // 行・列が奇数の場合用。クロップする
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // 画像の中央に原点が来るように、象限を入れ替える(元のままだと画像の中心部分が４つ角の方向を向いているらしい？)
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    Mat q0(magI, Rect(0, 0, cx, cy));   // 左上（第二象限）
    Mat q1(magI, Rect(cx, 0, cx, cy));  // 右上（第一象限）
    Mat q2(magI, Rect(0, cy, cx, cy));  // 左下（第三象限）
    Mat q3(magI, Rect(cx, cy, cx, cy)); // 右下（第四象限）
    Mat tmp;
    // 入れ替え(左上と右下)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    // 右上と左下
    q2.copyTo(q1);
    tmp.copyTo(q2);

    //Mat mask(magI.size(), CV_64FC1, Scalar(1.0, 1.0));
    //circle(mask, Point(n / 2, m / 2), 16, Scalar(0.0), -1); //低周波成分を0に
    
    //imshow("Input Image", magI);
    //magI = magI.mul(mask);

    // 見ることができる値(float[0,1])に変換
    normalize(magI, magI, 0, 1, NORM_MINMAX, -1);
    //magI.convertTo(magI, CV_64FC1, 1.0 / 255.0);

    //表示     
    /*imshow("Input Image", src);
    imshow("spectrum magnitude", magI);
    waitKey();*/

    return magI;
}

