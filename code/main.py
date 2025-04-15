import numpy as np
import matplotlib.pyplot as plt
from pr3_utils import *
from ekf_predict import predict_trajectory
from ekf_update import ekf_update_landmarks, predict_measurement

def numerical_pose_jacobian(T, m, K_l, K_r, extL_T_imu, extR_T_imu, delta=1e-5):
    """
    Estimate the numerical Jacobian of the observation function with respect to the IMU pose:
      J = ∂(predict_measurement(m, T, ...))/∂(T) (expressed in minimal 6D form)
    Output: A 4x6 Jacobian matrix.
    """
    z0 = predict_measurement(m, T, K_l, K_r, extL_T_imu, extR_T_imu)
    J = np.zeros((4, 6))
    for i in range(6):
        dx = np.zeros(6)
        dx[i] = delta
        # Convert dx into a 4x4 perturbation matrix via the exponential map.
        delta_T = axangle2pose(dx[np.newaxis, :])[0]
        T_perturbed = T @ delta_T
        z_perturbed = predict_measurement(m, T_perturbed, K_l, K_r, extL_T_imu, extR_T_imu)
        J[:, i] = (z_perturbed - z0) / delta
    return J

def update_imu_pose(T, P, features_t, landmarks, K_l, K_r, extL_T_imu, extR_T_imu, sigma_px):
    """
    Update the IMU pose using numerical Jacobian based on all valid visual observations at the current timestep.
    
    Inputs:
      T         : Predicted IMU pose (4x4 SE(3) matrix)
      P         : Predicted 6x6 error covariance for the IMU pose
      features_t: Stereo features for the current timestep (4 x M), with -1 indicating missing observations
      landmarks : Dictionary of updated landmarks (keys: landmark index, values: (m, P_m))
      Other parameters: Camera intrinsic, extrinsic calibration, and pixel standard deviation (sigma_px)
    
    Outputs:
      T_updated : Updated IMU pose
      P_updated : Updated 6x6 pose error covariance
    """
    H_total = []
    r_total = []
    M = features_t.shape[1]
    for i in range(M):
        z_obs = features_t[:, i]
        # Skip if the landmark is not observed or not initialized.
        if np.all(z_obs == -1) or i not in landmarks:
            continue
        m, _ = landmarks[i]
        z_pred = predict_measurement(m, T, K_l, K_r, extL_T_imu, extR_T_imu)
        r_i = z_obs - z_pred  # 4D residual
        # Estimate the numerical Jacobian (4x6) with respect to the IMU pose.
        J_i = numerical_pose_jacobian(T, m, K_l, K_r, extL_T_imu, extR_T_imu)
        H_total.append(J_i)
        r_total.append(r_i)
    # If there are no valid observations, do not update.
    if len(H_total) == 0:
        return T, P
    # Stack all the observations.
    H = np.vstack(H_total)           # Shape: (4*N, 6)
    r = np.hstack(r_total)           # Shape: (4*N, )
    # Assume each measurement has noise covariance sigma_px^2 (4x4 block), forming a big diagonal R.
    num_obs = len(r_total)
    R_meas = np.diag([sigma_px**2] * (4 * num_obs))
    # EKF update: K = P H^T (H P H^T + R)^(-1)
    S = H @ P @ H.T + R_meas
    try:
        invS = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return T, P
    K_gain = P @ H.T @ invS
    delta_x = K_gain @ r  # 6D increment
    # Update the pose: right-multiply by the exponential map perturbation (update on the Lie group)
    delta_T = axangle2pose(delta_x[np.newaxis, :])[0]
    T_updated = T @ delta_T
    # Update the error covariance.
    P_updated = (np.eye(6) - K_gain @ H) @ P
    return T_updated, P_updated

def main():
    datanum = ['00', '01', '02']
    for j in datanum:
        filename = f"../data/dataset{j}/dataset{j}.npy"
        v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(filename)

        # (a) IMU Localization via EKF Prediction
        sigma_v = 0.1       # Standard deviation for linear velocity (m/s)
        sigma_w = 0.01745   # Standard deviation for angular velocity (rad/s)
        Q_default = np.diag([sigma_v**2, sigma_v**2, sigma_v**2,
                             sigma_w**2, sigma_w**2, sigma_w**2])
        
        # Compute the IMU trajectory and the corresponding error covariances using EKF prediction.
        trajectory, covariances = predict_trajectory(v_t, w_t, timestamps, Q_default)
        trajectory_init = trajectory.copy()
        
        # (b) Landmark Mapping via EKF Update + IMU Update = (c) Visual-Inertial SLAM
        sigma_px = 4  # Pixel standard deviation
        R = np.diag([sigma_px**2, sigma_px**2, sigma_px**2, sigma_px**2])
        init_cov = np.eye(3) * 10000.0
        threshold = 20
        landmarks = {}
        
        T_total = features.shape[2]
        # For each timestep, we:
        # 1. Update landmarks based on the predicted T.
        # 2. Then, use visual observations to correct T (and its covariance).
        for t in range(T_total):
            # Extract the predicted pose and covariance for the current timestep.
            T_pred = trajectory[t]
            P_pred = covariances[t]
            features_t = features[:, :, t]  # Stereo features at the current timestep (4 x M)
            
            # Update landmarks using EKF (or initialize them) based on the current IMU pose and features.
            landmarks = ekf_update_landmarks(landmarks, features_t, T_pred,
                                             K_l, K_r, extL_T_imu, extR_T_imu,
                                             R, init_cov, threshold)
            # Update the IMU pose using visual observations.
            T_updated, P_updated = update_imu_pose(T_pred, P_pred, features_t,
                                                   landmarks, K_l, K_r, extL_T_imu, extR_T_imu,
                                                   sigma_px)
            # Update the trajectory and covariance.
            trajectory[t] = T_updated
            covariances[t] = P_updated

        # Plot the final landmarks (only the x-y plane).
        landmark_positions = np.array([landmarks[i][0] for i in sorted(landmarks.keys())])
        plt.figure(figsize=(6,6))
        plt.scatter(landmark_positions[:, 0], landmark_positions[:, 1],
                    marker='.', s=3, c='b', label='Landmarks')
        plt.plot(trajectory_init[:, 0, 3], trajectory_init[:, 1, 3], color='g', label='Predicted IMU Trajectory')
        plt.plot(trajectory[:, 0, 3], trajectory[:, 1, 3], color='r', label='Updated IMU Trajectory')
        plt.scatter(trajectory[0, 0, 3], trajectory[0, 1, 3], marker='s', c='r', label='Start')
        plt.scatter(trajectory[-1, 0, 3], trajectory[-1, 1, 3], marker='o', c='orange', label='End')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"Visual-Inertial SLAM for dataset{j}")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    main()
