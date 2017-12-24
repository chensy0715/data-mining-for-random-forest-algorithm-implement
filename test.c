

#include <cv.h>       // opencv general include file
#include <ml.h>          // opencv machine learning include file
#include <stdio.h>

using namespace cv; // OpenCV API is in the C++ "cv" namespace

/******************************************************************************/
// global definitions (for speed and ease of use)
//手写体数字识别

#define NUMBER_OF_TRAINING_SAMPLES 60000
#define ATTRIBUTES_PER_SAMPLE 784
#define NUMBER_OF_TESTING_SAMPLES 10000

#define NUMBER_OF_CLASSES 10

// N.B. classes are integer handwritten digits in range 0-9

/******************************************************************************/

// loads the sample database from file (which is a CSV text file)

inline void revertInt(int&x)
{
    x=((x&0x000000ff)<<24)|((x&0x0000ff00)<<8)|((x&0x00ff0000)>>8)|((x&0xff000000)>>24);
};

int read_data_from_csv(const char* samplePath,const char* labelPath, Mat data, Mat classes,
                       int n_samples )
{
    FILE* sampleFile=fopen(samplePath,"rb");
    FILE* labelFile=fopen(labelPath,"rb");
    int mbs=0,number=0,col=0,row=0;
    fread(&mbs,4,1,sampleFile);
    fread(&number,4,1,sampleFile);
    fread(&row,4,1,sampleFile);
    fread(&col,4,1,sampleFile);
    revertInt(mbs);
    revertInt(number);
    revertInt(row);
    revertInt(col);
    fread(&mbs,4,1,labelFile);
    fread(&number,4,1,labelFile);
    revertInt(mbs);
    revertInt(number);
    unsigned char temp;
    for(int line = 0; line < n_samples; line++)
    {
        // for each attribute on the line in the file
        for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); attribute++)
        {
            if (attribute < ATTRIBUTES_PER_SAMPLE)
            {
                // first 64 elements (0-63) in each line are the attributes
                fread(&temp,1,1,sampleFile);
                //fscanf(f, "%f,", &tmp);
                data.at<float>(line, attribute) = static_cast<float>(temp);
                // printf("%f,", data.at<float>(line, attribute));
            }
            else if (attribute == ATTRIBUTES_PER_SAMPLE)
            {
                // attribute 65 is the class label {0 ... 9}
                fread(&temp,1,1,labelFile);
                //fscanf(f, "%f,", &tmp);
                classes.at<float>(line, 0) = static_cast<float>(temp);
                // printf("%f\n", classes.at<float>(line, 0));
            }
        }
    }
    fclose(sampleFile);
    fclose(labelFile);
    return 1; // all OK
}

/******************************************************************************/

int main( int argc, char** argv )
{
    
    for (int i=0; i< argc; i++)
        std::cout<<argv[i]<<std::endl;
    
    // lets just check the version first
    printf ("OpenCV version %s (%d.%d.%d)\n",
            CV_VERSION,
            CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
    
    //定义训练数据与标签矩阵
    Mat training_data = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

    //定义测试数据矩阵与标签
    Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);

    // define all the attributes as numerical
    // alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
    // that can be assigned on a per attribute basis

    Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
    var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical

    // this is a classification problem (i.e. predict a discrete number of class
    // outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL

    var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;

    double result; // value returned from a prediction

    //加载训练数据集和测试数据集
    if (read_data_from_csv(argv[1],argv[2], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
            read_data_from_csv(argv[3],argv[4], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
    {
      /********************************步骤1：定义初始化Random Trees的参数******************************/
        float priors[] = {1,1,1,1,1,1,1,1,1,1};  // weights of each classification for classes
        CvRTParams params = CvRTParams(20, // max depth
                                       50, // min sample count
                                       0, // regression accuracy: N/A here
                                       false, // compute surrogate split, no missing data
                                       15, // max number of categories (use sub-optimal algorithm for larger numbers)
                                       priors, // the array of priors
                                       false,  // calculate variable importance
                                       50,       // number of variables randomly selected at node and used to find the best split(s).
                                       100,     // max number of trees in the forest
                                       0.01f,                // forest accuracy
                                       CV_TERMCRIT_ITER |    CV_TERMCRIT_EPS // termination cirteria
                                      );

        /****************************步骤2：训练 Random Decision Forest(RDF)分类器*********************/
        printf( "\nUsing training database: %s\n\n", argv[1]);
        CvRTrees* rtree = new CvRTrees;
        bool train_result=rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,
                     Mat(), Mat(), var_type, Mat(), params);
//        float train_error=rtree->get_train_error();
//        printf("train error:%f\n",train_error);
        // perform classifier testing and report results
        Mat test_sample;
        int correct_class = 0;
        int wrong_class = 0;
        int false_positives [NUMBER_OF_CLASSES] = {0,0,0,0,0,0,0,0,0,0};

        printf( "\nUsing testing database: %s\n\n", argv[2]);

        for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
        {

            // extract a row from the testing matrix
            test_sample = testing_data.row(tsample);
        /********************************步骤3：预测*********************************************/
            result = rtree->predict(test_sample, Mat());

            printf("Testing Sample %i -> class result (digit %d)\n", tsample, (int) result);

            // if the prediction and the (true) testing classification are the same
            // (N.B. openCV uses a floating point decision tree implementation!)
            if (fabs(result - testing_classifications.at<float>(tsample, 0))
                    >= FLT_EPSILON)
            {
                // if they differ more than floating point error => wrong class
                wrong_class++;
                false_positives[(int) result]++;
            }
            else
            {
                // otherwise correct
                correct_class++;
            }
        }

        printf( "\nResults on the testing database: %s\n"
                "\tCorrect classification: %d (%g%%)\n"
                "\tWrong classifications: %d (%g%%)\n",
                argv[2],
                correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
                wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES);

        for (int i = 0; i < NUMBER_OF_CLASSES; i++)
        {
            printf( "\tClass (digit %d) false postives     %d (%g%%)\n", i,
                    false_positives[i],
                    (double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
        }

        // all matrix memory free by destructors

        // all OK : main returns 0
        return 0;
    }

    // not OK : main returns -1
    return -1;
}