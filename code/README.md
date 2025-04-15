# Visual-Inertial SLAM (ECE276A Project 3)

This project implements a Visual-Inertial SLAM system using an Extended Kalman Filter (EKF) to perform IMU localization and landmark mapping. The system fuses inertial measurements with stereo camera observations to estimate the IMU trajectory and map environmental landmarks.

## File Structure

- **pr3_utils.py**  
  Contains utility functions for:
  - Loading data (IMU measurements, stereo feature observations, camera calibration, and extrinsic parameters).
  - Performing SE(3) operations and transformations.
  - Visualizing trajectories in 2D.

- **ekf_predict.py**  
  Implements the EKF prediction step:
  - Uses IMU measurements (linear and angular velocities) along with SE(3) kinematics to propagate the pose.
  - Computes the predicted IMU trajectory and the associated state covariance.

- **ekf_update.py**  
  Implements the EKF update for landmark mapping:
  - Triangulates landmark positions from stereo camera measurements.
  - Updates landmark states and covariances using visual observations.
  - Filters out outlier landmarks based on a distance threshold.

- **main.py**  
  The main entry point for the visual-inertial SLAM pipeline:
  - Loads datasets (dataset00, dataset01, and dataset02).
  - Performs EKF-based IMU pose prediction and subsequently corrects the pose using visual landmark observations.
  - Updates landmark positions and visualizes both the IMU trajectory and the landmark map.
  
## How to Run

### 1. Dependencies

Make sure you have Python 3 installed along with the following packages:
- NumPy
- Matplotlib
- transforms3d

You can install the required packages using pip:

```bash
pip install numpy matplotlib transforms3d
