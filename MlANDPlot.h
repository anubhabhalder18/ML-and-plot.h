// RegressionAndPlot.h

#ifndef REGRESSIONANDPLOT_H
#define REGRESSIONANDPLOT_H

#include <vector>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

class RegressionAndPlot {
public:
    // Plotting functions
    void plot(const std::vector<double>& x, const std::vector<double>& y) {
        plt::scatter(x, y);
        plt::show();
    }

    // Linear Regression
    Eigen::VectorXd linearRegression(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        Eigen::MatrixXd X_b(X.rows(), X.cols() + 1);
        X_b << Eigen::MatrixXd::Ones(X.rows(), 1), X;
        return (X_b.transpose() * X_b).inverse() * X_b.transpose() * y;
    }

    // Polynomial Regression
    Eigen::VectorXd polynomialRegression(const Eigen::VectorXd& x, const Eigen::VectorXd& y, int degree) {
        Eigen::MatrixXd X(x.size(), degree + 1);
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                X(i, j) = std::pow(x(i), j);
            }
        }
        return (X.transpose() * X).inverse() * X.transpose() * y;
    }

    // Logistic Regression
    Eigen::VectorXd logisticRegression(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int iterations = 1000, double alpha = 0.01) {
        Eigen::MatrixXd X_b(X.rows(), X.cols() + 1);
        X_b << Eigen::MatrixXd::Ones(X.rows(), 1), X;
        Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_b.cols());
        
        for (int i = 0; i < iterations; ++i) {
            Eigen::VectorXd predictions = 1 / (1 + (-X_b * theta).array().exp());
            Eigen::VectorXd errors = y - predictions;
            theta += alpha * X_b.transpose() * errors;
        }

        return theta;
    }
};

#endif // REGRESSIONANDPLOT_H
