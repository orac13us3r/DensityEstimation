#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <matio.h>
#include <boost/math/distributions.hpp>

using namespace std;
using namespace Eigen;

MatrixXd train_data, test_data;
VectorXi train_label, test_label;

void load_trainData(){
	mat_t *matfp = Mat_Open("train_data.mat", MAT_ACC_RDONLY);
	
	
	matvar_t *data_var = Mat_VarRead(matfp, "data");
	if (data_var -> data_type == MAT_T_DOUBLE){
		size_t row = data_var -> dims[0];
		size_t column = data_var -> dims[1];
		double *mat_data = static_cast<double *>(data_var -> data);
		
		trainData = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(mat_data, row, column);
		Mat_VarFree(data_var);
	}
	else if (data_var == nullptr){
		cerr << "Data not available" << endl;
	}
	else {
		cerr << "Error Reading Data" << endl;
	}
	
	matvar_t *label_var = Mat_VarRead(matfp, "label");
	if (label_var -> data_type == MAT_T_DOUBLE){
		size_t row = label_var -> dims[0];
		double *mat_label= static_cast<double *>(label_var -> data);
		
		trainData = Map<Matrix<int, Dynamic, 1>>(reinterpret_cast<int *>(mat_label), row);
		Mat_VarFree(label_var);
	}
	else if (data_var == nullptr){
		cerr << "Data not available" << endl;
	}
	else {
		cerr << "Error Reading Data" << endl;
	}
	Mat_Close(matfp);
}

void load_testData(){
	mat_t *matfp = Mat_Open("test_data.mat", MAT_ACC_RDONLY);
	
	
	matvar_t *data_var = Mat_VarRead(matfp, "data");
	if (data_var -> data_type == MAT_T_DOUBLE){
		size_t row = data_var -> dims[0];
		size_t column = data_var -> dims[1];
		double *mat_data = static_cast<double *>(data_var -> data);
		
		testData = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(mat_data, row, column);
		Mat_VarFree(data_var);
	}
	else if (data_var == nullptr){
		cerr << "Data not available" << endl;
	}
	else {
		cerr << "Error Reading Data" << endl;
	}
	
	matvar_t *label_var = Mat_VarRead(matfp, "label");
	if (label_var -> data_type == MAT_T_DOUBLE){
		size_t row = label_var -> dims[0];
		double *mat_label= static_cast<double *>(label_var -> data);
		
		testData = Map<Matrix<int, Dynamic, 1>>(reinterpret_cast<int *>(mat_label), row);
		Mat_VarFree(label_var);
	}
	else if (data_var == nullptr){
		cerr << "Data not available" << endl;
	}
	else {
		cerr << "Error Reading Data" << endl;
	}
	Mat_Close(matfp);
}

void standard_data(vector<vector<double>>& data){
	int samples = data.size();
	int features = data[0].size();
	
	// Calculate mean and standard deviation for each feature
    vector<double> mean(features, 0.0);
    vector<double> deviation(features, 0.0);

    // Calculate means
    for (int j = 0; j < features; j++) {
        for (int i = 0; i < samples; i++) {
            mean[j] = mean[j] + data[i][j];
        }
        mean[j] = mean[j] / samples;
    }

    // Calculate standard deviations
    for (int j = 0; j < features; j++) {
        for (int i = 0; i < samples; i++) {
            deviation[j] = deviation[j] + pow(data[i][j] - mean[j], 2);
        }
        deviation[j] = sqrt(deviation[j] / samples);
    }

    // Standardize each feature
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < features; j++) {
            data[i][j] = (data[i][j] - mean[j]) / deviation[j];
        }
    }
}

double accuracy_score(const vector<int>& y_true, const vector<int>& y_pred) {
    int correct = 0;
    int n = y_true.size();
    
    for (int i = 0; i < n; i++) {
        if (y_true[i] == y_pred[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / n;
}

double skew(const vector<double>& data) {
    int n = data.size();
    double mean = accumulate(data.begin(), data.end(), 0.0) / n;

    double m3 = 0.0, m2 = 0.0;
    for (const auto& x : data) {
        double diff = x - mean;
        m3 += diff * diff * diff;
        m2 += diff * diff;
    }
    m3 /= n;
    m2 = pow(m2 / n, 1.5);
    
    return m3 / m2;
}

vector<VectorXd> multivariate_normal(int n, const VectorXd& mean, const MatrixXd& cov) {
    int dim = mean.size();
    MatrixXd L = cov.llt().matrixL(); // Cholesky decomposition

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0, 1);

    vector<VectorXd> samples;
    for (int i = 0; i < n; i++) {
        VectorXd z(dim);
        for (int j = 0; j < dim; j++) {
            z[j] = d(gen);
        }
        samples.push_back(mean + L * z); // Transform standard normal to multivariate normal
    }
    return samples;
}

// Feature extraction function
vector<vector<double>> extract_features(const vector<vector<vector<double>>>& data, double threshold) {
    vector<vector<double>> features;

    for (const auto& sample : data) {
        // Flatten the 2D sample to a 1D vector
        vector<double> flat_data;
        for (const auto& row : sample) {
            flat_data.insert(flat_data.end(), row.begin(), row.end());
        }

        // Calculate skewness using your pre-defined function
        double skew_value = skew(flat_data);

        // Calculate ratio of bright to dark pixels based on the threshold
        int bright_count = 0;
        for (const auto& pixel : flat_data) {
            if (pixel > threshold) {
                bright_count++;
            }
        }
        double bright_to_dark_ratio = static_cast<double>(bright_count) / (flat_data.size() - bright_count);

        // Store the extracted features (skewness and ratio) for this sample
        features.push_back({skew_value, bright_to_dark_ratio});
    }

    return features;
}


void normalize_data(vector<vector<double>>& data) {
    int samples = data.size();
    int features = data[0].size();
    
    // Find the min and max values across all standardized data for each feature
    vector<double> min_val(features, numeric_limits<double>::max());
    vector<double> max_val(features, numeric_limits<double>::lowest());

    for (int j = 0; j < features; j++) {
        for (int i = 0; i < samples; i++) {
            min_val[j] = min(min_val[j], data[i][j]);
            max_val[j] = max(max_val[j], data[i][j]);
        }
    }

    // Normalize each feature using the calculated min and max values
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < features; j++) {
            if (max_val[j] != min_val[j]) { // To avoid division by zero
                data[i][j] = (data[i][j] - min_val[j]) / (max_val[j] - min_val[j]);
            } else {
                data[i][j] = 0.0; // Set to 0 if all values are the same
            }
        }
    }
}

//MLE Function
pair<double, double> maximum_likelihood_estimation(const vector<vector<vector<double>>>& train_data, const vector<int>& train_labels, double threshold) {
    vector<vector<double>> train_data_feature = extract_features(train_data, threshold);
    normalize_extracted_features(train_data_feature);

    // Step 2: Separate data by label (0 and 1)
    vector<VectorXd> zero, one;
    for (size_t i = 0; i < train_labels.size(); i++) {
        VectorXd feature_vector = Map<VectorXd>(train_data_feature[i].data(), train_data_feature[i].size());
        if (train_labels[i] == 0) {
            three.push_back(feature_vector);
        } else if (train_labels[i] == 1) {
            seven.push_back(feature_vector);
        }
    }

    // Step 3: Calculate mean vectors
    VectorXd one_mean = VectorXd::Zero(zero[0].size());
    VectorXd seven_mean = VectorXd::Zero(one[0].size());

    for (const auto& vec : zero) {
        zero_mean += vec;
    }
    zero_mean /= zero.size();

    for (const auto& vec : one) {
        one_mean += vec;
    }
    one_mean /= one.size();

    // Step 4: Calculate covariance matrices
    MatrixXd one_cov = MatrixXd::Zero(one[0].size(), zero[0].size());
    MatrixXd seven_cov = MatrixXd::Zero(seven[0].size(), one[0].size());

    for (const auto& vec : zero) {
        VectorXd diff = vec - zero_mean;
        zero_cov += diff * diff.transpose();
    }
    zero_cov /= (zero.size() - 1);

    for (const auto& vec : one) {
        VectorXd diff = vec - one_mean;
        one_cov += diff * diff.transpose();
    }
    one_cov /= (one.size() - 1);

    // Step 5: Calculate normal distributions
    // Using multivariate normal density function (assumes `multivariate_normal` function is defined)
    double three_distribution = multivariate_normal(zero_mean, zero_cov, zero_mean);
    double seven_distribution = multivariate_normal(one_mean, one_cov, one_mean);

    // Output the calculated MLE
    cout << "Mean of 0: " << zero_mean.transpose() << ", Mean of 1: " << one_mean.transpose() << endl;
    cout << "Covariance of 0:\n" << zero_cov << "\nCovariance of 1:\n" << one_cov << endl;

    return {zero_distribution, one_distribution};
}

pair<double, double> classify(
    const vector<vector<vector<double>>>& data_feature, 
    const vector<int>& label_feature,
    double threshold, 
    double zero_probability, 
    double one_probability,
    const VectorXd& zero_mean, 
    const MatrixXd& zero_cov,
    const VectorXd& one_mean, 
    const MatrixXd& one_cov
) {
    // Step 1: Extract features and normalize
    vector<vector<double>> extracted_data = extract_features(data_feature, threshold);
    normalize_extracted_features(extracted_data);

    // Display the top 5 normalized features
    cout << "Normalized features: ";
    for (size_t i = 0; i < min(5, (int)extracted_data.size()); i++) {
        cout << "[" << extracted_data[i][0] << ", " << extracted_data[i][1] << "] ";
    }
    cout << endl;

    // Step 2: Predict labels using Bayesian Classification
    vector<int> predicted_labels;
    for (const auto& features : extracted_data) {
        VectorXd feature_vector = Map<const VectorXd>(features.data(), features.size());

        // Calculate the posterior probability for each class
        double P0 = zero_probability * multivariate_normal(zero_mean, zero_cov, feature_vector);
        double P1 = one_probability * multivariate_normal(one_mean, one_cov, feature_vector);

        // Classify based on the higher posterior probability
        if (P0 > P1) {
            predicted_labels.push_back(0);
        } else {
            predicted_labels.push_back(1);
        }
    }

    // Step 3: Calculate accuracy and error rate
    double accuracy = accuracy_score(label_feature, predicted_labels);
    double error_rate = 1.0 - accuracy;

    // Output the accuracy and error rate
    cout << "Accuracy: " << accuracy << ", Error Rate: " << error_rate << endl;

    return {accuracy, error_rate};
}

int main() {
    // Placeholder: Load or define your train and test data here
    vector<vector<vector<double>>> train_data;  // 3D vector for train data
    vector<int> train_labels;                   // Vector for train labels
    vector<vector<vector<double>>> test_data;   // 3D vector for test data
    vector<int> test_labels;                    // Vector for test labels

    double threshold = 200.0;
    double one_probability = 0.5;
    double seven_probability = 0.5;

    // Step 1: Extract features and normalize for the training data
    vector<vector<double>> train_features = extract_features(train_data, threshold);
    normalize_extracted_features(train_features);

    // Step 2: Separate features based on labels (0 and 1) and calculate mean and covariance
    vector<VectorXd> three, seven;
    for (size_t i = 0; i < train_labels.size(); ++i) {
        VectorXd feature_vector = Map<VectorXd>(train_features[i].data(), train_features[i].size());
        if (train_labels[i] == 0) {
            three.push_back(feature_vector);
        } else if (train_labels[i] == 1) {
            seven.push_back(feature_vector);
        }
    }

    // Calculate mean for class 0 and 1
    VectorXd one_mean = VectorXd::Zero(zero[0].size());
    VectorXd seven_mean = VectorXd::Zero(one[0].size());
    for (const auto& vec : zero) zero_mean += vec;
    zero_mean /= zero.size();
    for (const auto& vec : one) one_mean += vec;
    one_mean /= one.size();

    // Calculate covariance for class 0 and 1
    MatrixXd zero_cov = MatrixXd::Zero(zero[0].size(), zero[0].size());
    MatrixXd one_cov = MatrixXd::Zero(one[0].size(), one[0].size());
    for (const auto& vec : zero) {
        VectorXd diff = vec - zero_mean;
        zero_cov += diff * diff.transpose();
    }
    zero_cov /= (zero.size() - 1);
    for (const auto& vec : one) {
        VectorXd diff = vec - one_mean;
        one_cov += diff * diff.transpose();
    }
    one_cov /= (one.size() - 1);

    // Print normalized features (first 5 for verification)
    cout << "Normalized features: " << endl;
    for (size_t i = 0; i < min(5, (int)train_features.size()); i++) {
        cout << "[" << train_features[i][0] << ", " << train_features[i][1] << "]" << endl;
    }

    // Print the mean and covariance for class 0
    cout << "\nMean of class 0: " << zero_mean.transpose() << endl;
    cout << "Covariance of class 0:\n" << zero_cov << endl;

    // Print the mean and covariance for class 1
    cout << "\nMean of class 1: " << one_mean.transpose() << endl;
    cout << "Covariance of class 1:\n" << one_cov << endl;

    // Step 3: Classify test data and print accuracy and error rate
    pair<double, double> result = classify(
        test_data, test_labels, threshold, zero_probability, zero_probability, zero_mean, one_cov, one_mean, one_cov
    );

    cout << "\nAccuracy: " << result.first * 100 << "%" << endl;
    cout << "Error Rate: " << result.second * 100 << "%" << endl;

    return 0;
}
