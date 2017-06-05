#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
    * Similar to Lesson 5: Part 13 in 'Tracking.cpp'
    * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
  */
  noise_ax = 9;
  noise_ay = 9;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  I N I T I A L I Z A T I O N
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    //state covariance matrix P
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1000, 0,
               0, 0, 0, 1000;
    //measurement matrix
    H_laser_ = MatrixXd(2, 4);
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      std::cout << "Inside RADAR Init" << endl;
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float p_x = rho*cos(phi);
      float p_y = rho*sin(phi);
      ekf_.x_ << p_x, p_y, 0, 0;
      cout << "ekf_.x_ = \n" << ekf_.x_ << endl;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      std::cout << "Inside LASER Init" << endl;
      //set the state with the initial location and zero velocity (we can't measure velocity)
      ekf_.x_ << measurement_pack.raw_measurements_[0],
                 measurement_pack.raw_measurements_[1],
                 0, 0;
      // the initial Laser's F transition matrix
      ekf_.F_ = MatrixXd(4, 4);
      ekf_.F_ << 1, 0, 1, 0,
                 0, 1, 0, 1,
                 0, 0, 1, 0,
                 0, 0, 0, 1;
    }
    //save the first time entry
    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  P R E D I C T I O N
   ****************************************************************************/
  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  //compute the time elapsed between the current and previous measurements
  //dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //helper variables for Q matrix
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  //set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
              0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
              dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
              0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  //do I need to call KalmanFilter::Init() here?
  ekf_.Init(ekf_.x_,
            ekf_.P_,
            ekf_.F_,
            H_laser_,
            R_laser_,
            ekf_.Q_);

  // Predict() will return a predicted State Vector and the corrsp. Covariance Matrix
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Calculate Jacobian Matrix
    // Hj_ is a matrix of coeffiencints. These coeffs represent the
    // derivative coeffs need for the First Order Taylor Series Approx.
    Tools jacob;
    Hj_ = jacob.CalculateJacobian(ekf_.x_);

    //do I need to call KalmanFilter::Init() here?
    ekf_.Init(ekf_.x_,
              ekf_.P_,
              ekf_.F_,
              Hj_,
              R_radar_,
              ekf_.Q_);

    // make measurement update
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {

    //do I need to call KalmanFilter::Init() here?
    ekf_.Init(ekf_.x_,
              ekf_.P_,
              ekf_.F_,
              H_laser_,
              R_laser_,
              ekf_.Q_);

    // Laser updates
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
//  cout << "\nx_ = \n" << ekf_.x_ << endl;
//  cout << "P_ = \n" << ekf_.P_ << endl;
}
