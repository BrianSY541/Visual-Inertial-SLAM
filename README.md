# Visual-Inertial SLAM for Mobile Robot Navigation

This repository implements a robust Visual-Inertial Simultaneous Localization and Mapping (SLAM) system combining inertial measurement unit (IMU) data and stereo camera observations using an Extended Kalman Filter (EKF). The system is designed for accurate navigation and consistent landmark mapping in complex, GPS-denied environments.

## 🌟 Key Features
- **SE(3) Kinematics EKF Prediction:** High-frequency IMU data integration to predict robot poses.
- **Stereo Vision EKF Update:** Accurate landmark mapping using stereo camera measurements.
- **Adaptive Noise Modeling:** Custom noise parameters to handle sensor uncertainties.
- **Threshold-based Outlier Rejection:** Ensures robust landmark tracking and map consistency.

## 🔧 Technical Details

### IMU Localization via EKF Prediction
- **State Representation:** SE(3) kinematics are used to represent the robot's pose, incorporating both position and orientation.
- **Process Noise Modeling:** Custom noise covariance matrices accurately reflect sensor-specific uncertainties.
- **Incremental Pose Update:** Pose predictions utilize exponential mapping from twist vectors derived from IMU measurements.
- **Covariance Propagation:** Efficient covariance propagation via first-order Taylor expansion for state transition.

### Landmark Mapping via EKF Update
- **Stereo Triangulation:** Initial landmark positions obtained via stereo triangulation from left-right camera image pairs.
- **Measurement Prediction:** Predicts stereo pixel coordinates from estimated 3D landmark positions.
- **Jacobian Computation:** Jacobian matrices calculated analytically to facilitate EKF updates.
- **Landmark Refinement:** Updates landmark positions iteratively using EKF and visual observations.
- **Outlier Rejection:** Implements threshold-based strategy to discard inconsistent landmarks and improve robustness.

### Visual-Inertial Fusion
- **Pose Correction:** Combines visual landmark updates with IMU predictions to mitigate drift.
- **Numerical Jacobian Estimation:** Performs numerical differentiation for accurate Jacobian estimation in pose correction.
- **Iterative Refinement:** Continuously refines both robot pose and landmark estimates for enhanced accuracy.

## 🗂️ Repository Structure

```
Visual-Inertial-SLAM/
├── code/
│   ├── ekf_prediction.py
│   ├── ekf_update.py
│   ├── main.py
│   ├── pr3_utils.py
│   └── README.md
├── plot/
│   └── [Generated trajectory plots and landmark maps]
├── report/
│   └── ECE276A_Project3_Report.pdf
└── README.md
```

### Main Scripts
- `ekf_prediction.py`: EKF prediction using IMU data.
- `ekf_update.py`: EKF landmark update with stereo camera measurements.
- `main.py`: Integrates prediction and update steps for full SLAM.

## 📊 Results
Generated results, including trajectory plots and landmark maps, are stored in the `plot` folder.

## 📝 Report
Detailed project description, methodology, and results can be found in [Project Report](ECE276A_Project3_Report.pdf).

---

## 📧 Contact
- **Brian (Shou-Yu) Wang**  
  - Email: briansywang541@gmail.com  
  - LinkedIn: [linkedin.com/in/sywang541](https://linkedin.com/in/sywang541)
  - GitHub: [BrianSY541](https://github.com/BrianSY541)

---

**Project developed as part of ECE 276A: Sensing & Estimation in Robotics at UC San Diego.**

