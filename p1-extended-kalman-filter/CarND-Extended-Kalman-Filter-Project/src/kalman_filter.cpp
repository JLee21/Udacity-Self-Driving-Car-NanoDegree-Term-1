#include "kalman_filter.h"
//#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

#define PI 3.14159265

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in; // includes noise from measurment
  Q_ = Q_in; // includes noise from acceleration
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
//  std::cout << "\nPrediction\n" << x_ << std::endl;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  std::cout << "\nInside Laser Update()";
//  std::cout << "Raw measurments = \n" << z;

  //update calculations
  VectorXd z_pred = H_ * x_;
//  std::cout << "\nz_pred = \n" << z_pred << std::endl;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
//  std::cout << "\nHt\n" << Ht;
//  std::cout << "\nR_\n" << R_;
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

//  std::cout << "y = " << y << std::endl;
//  std::cout << "updated x_ = \n" << x_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
//  std::cout << "\nInside Radar Update\n";
//  std::cout << "Raw measurments = \n" << z;

  //update calculations
  //recover state parameters
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  //check division by zero
  if(fabs(px) < 0.0001){
    px = 0.0001;
  }
  //map the predicted state to polar coord.
  // NOTE!
  // You'll need to make sure to normalize phi in the y vector
  // so that its angle is between -pi and pi;
  // in other words, add or subtract 2pi from phi until it is between -pi and pi.
  float phi = atan2(py, px);
  std::cout << "\nx   \t" << px << std::endl;
  std::cout << "y   \t" << py << std::endl;
  std::cout << "phi\t" << phi << std::endl;
//  if (px < 0 && py < 0){
//    std::cout << "\nPhi is greater than PI\n";
//    phi -= 6.28;
//  }
//  if (phi < -3.14){
//    std::cout << "\nGreater than PI\n";
//    phi += 2*3.14;
//  }
  std::cout << "\nchange2\n";

  // add polar coords.
  Eigen::VectorXd z_pred = VectorXd(3);
  z_pred << sqrt(pow(px,2)+pow(py,2)),
            phi,
            (px*vx + py*vy)/sqrt(pow(px,2)+pow(py,2));

//  VectorXd z_pred =  H_*x_;
//  std::cout << "\nz_pred = \n" << z_pred << std::endl;
  VectorXd y = z - z_pred;
  if (y(1) > 3.14){
    std::cout << "\nPhi is greater than PI\n";
    y(1) = y(1) - 6.28;
  }
  if (y(1) < -3.14){
    std::cout << "\nGreater than PI\n";
    y(1) = y(1) + 2*3.14;
  }
  std::cout << "y_phi\t" << y(1);
  MatrixXd Ht = H_.transpose();
//  std::cout << "\nHt\n" << Ht;
//  std::cout << "\nR_\n" << R_;
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

//  std::cout << "y = " << y << std::endl;
//  std::cout << "updated x_ = \n" << x_;

}
