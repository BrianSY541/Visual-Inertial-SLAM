import numpy as np
import matplotlib.pyplot as plt
from pr3_utils import *
from ekf_predict import predict_trajectory

def triangulate_landmark(z, T_imu, K_l, K_r, extL_T_imu, extR_T_imu):
    """
    Triangulate a landmark's 3D position using stereo measurements.
    
    Inputs:
        z: a 4-vector [u_l, v_l, u_r, v_r] representing pixel coordinates 
           from the left and right cameras respectively.
        T_imu: 4x4 current IMU pose (T_imu in the world frame) at the time of observation.
        K_l, K_r: 3x3 intrinsic matrices for the left and right cameras.
        extL_T_imu, extR_T_imu: 4x4 extrinsic calibration matrices, 
                                 representing the transformation from the IMU frame to the left/right camera frame.
    Output:
        m: the 3D landmark position in the world coordinate frame (a 3-vector).
    """
    # Compute the poses of the left and right cameras by combining the IMU pose with the extrinsic calibration.
    # Note: The multiplication order depends on the definition of the extrinsics.
    T_left = T_imu @ extL_T_imu
    T_right = T_imu @ extR_T_imu

    # Compute the inverse pose (from world frame to camera frame).
    T_left_inv = inversePose(T_left)
    T_right_inv = inversePose(T_right)

    # Construct the projection matrices P = K * [R|t] using the intrinsic matrices.
    P_left = K_l @ T_left_inv[:3, :]   # 3x4 projection matrix for the left camera
    P_right = K_r @ T_right_inv[:3, :]   # 3x4 projection matrix for the right camera

    u_l, v_l, u_r, v_r = z

    # Build the linear system A * m_h = 0, where m_h is the homogeneous landmark position.
    A = np.zeros((4, 4))
    A[0, :] = u_l * P_left[2, :] - P_left[0, :]
    A[1, :] = v_l * P_left[2, :] - P_left[1, :]
    A[2, :] = u_r * P_right[2, :] - P_right[0, :]
    A[3, :] = v_r * P_right[2, :] - P_right[1, :]

    # Solve the homogeneous system via SVD and dehomogenize the result.
    U, S, Vh = np.linalg.svd(A)
    m_h = Vh[-1, :]  # Use the solution corresponding to the smallest singular value.
    m = m_h[:3] / m_h[3]  # Convert from homogeneous to Cartesian coordinates.
    return m

def predict_measurement(m, T_imu, K_l, K_r, extL_T_imu, extR_T_imu):
    """
    Predict the stereo measurement for a landmark based on its 3D position and the current IMU pose.
    
    Inputs:
        m: (3,) landmark position in the world coordinate frame.
        T_imu: 4x4 current IMU pose.
        K_l, K_r: intrinsic matrices for the left and right cameras.
        extL_T_imu, extR_T_imu: extrinsic calibration matrices from the IMU frame to the left/right camera frame.
    Output:
        z_pred: the predicted measurement, a 4-vector [u_l, v_l, u_r, v_r].
    """
    # Compute the poses of the left and right cameras (same as in triangulate_landmark).
    T_left = T_imu @ extL_T_imu
    T_right = T_imu @ extR_T_imu
    T_left_inv = inversePose(T_left)
    T_right_inv = inversePose(T_right)
    P_left = K_l @ T_left_inv[:3, :]  # Left camera projection matrix.
    P_right = K_r @ T_right_inv[:3, :]  # Right camera projection matrix.

    m_h = np.hstack([m, 1])  # Convert to homogeneous coordinates.
    p_left = P_left @ m_h    # Project into the left camera (homogeneous 3-vector).
    p_right = P_right @ m_h  # Project into the right camera.

    # Convert to pixel coordinates by performing perspective division.
    z_left = p_left[:2] / p_left[2]
    z_right = p_right[:2] / p_right[2]
    return np.hstack([z_left, z_right])

def compute_jacobian(m, T_imu, K_l, K_r, extL_T_imu, extR_T_imu):
    """
    Compute the 4x3 Jacobian of the stereo measurement function with respect to the landmark position m.
    
    Inputs: Same as predict_measurement.
    Output:
        H: a 4x3 Jacobian matrix.
    """
    # Compute the projection matrices for both cameras.
    T_left = T_imu @ extL_T_imu
    T_right = T_imu @ extR_T_imu
    T_left_inv = inversePose(T_left)
    T_right_inv = inversePose(T_right)
    P_left = K_l @ T_left_inv[:3, :]   # Left camera projection matrix.
    P_right = K_r @ T_right_inv[:3, :]   # Right camera projection matrix.

    m_h = np.hstack([m, 1])
    p_left = P_left @ m_h   # Resulting homogeneous coordinates: [u_l, v_l, w_l]
    p_right = P_right @ m_h # Resulting homogeneous coordinates: [u_r, v_r, w_r]

    # For the left camera: compute the derivative of the projection (perspective division).
    u_l, v_l, w_l = p_left
    d_proj_left = np.array([[1/w_l, 0, -u_l/(w_l**2)],
                            [0, 1/w_l, -v_l/(w_l**2)]])
    # Only the first three columns of P_left affect m.
    J_left = d_proj_left @ P_left[:, :3]
    
    # For the right camera: similar computation.
    u_r, v_r, w_r = p_right
    d_proj_right = np.array([[1/w_r, 0, -u_r/(w_r**2)],
                             [0, 1/w_r, -v_r/(w_r**2)]])
    J_right = d_proj_right @ P_right[:, :3]
    
    # Stack the Jacobians vertically to form a 4x3 matrix.
    H = np.vstack([J_left, J_right])
    return H

def ekf_update_one_landmark(m, P, z, T_imu, K_l, K_r, extL_T_imu, extR_T_imu, R, dist_threshold):
    """
    Perform EKF update for a single landmark. If the updated landmark is too far from the IMU,
    it is considered an outlier.
    
    Inputs:
        m: (3,) current landmark state (position in the world frame).
        P: (3,3) current error covariance of the landmark.
        z: (4,) stereo measurement [u_l, v_l, u_r, v_r].
        T_imu: 4x4 current IMU pose.
        K_l, K_r: intrinsic matrices for the left and right cameras.
        extL_T_imu, extR_T_imu: extrinsic calibration matrices from the IMU frame to the left/right camera frame.
        R: (4,4) measurement noise covariance matrix.
        dist_threshold: if the distance between the landmark and the IMU exceeds this threshold, 
                        it is considered an outlier.
    Outputs:
        m_new: updated landmark state.
        P_new: updated error covariance.
    """
    # 1) Predict measurement and compute residual
    h = predict_measurement(m, T_imu, K_l, K_r, extL_T_imu, extR_T_imu)
    r = z - h

    # 2) Compute measurement Jacobian
    H = compute_jacobian(m, T_imu, K_l, K_r, extL_T_imu, extR_T_imu)
    S = H @ P @ H.T + R

    # 3) If S is not invertible, skip the update.
    try:
        invS = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return m, P

    # 4) Perform standard EKF update.
    K_gain = P @ H.T @ invS
    m_new = m + K_gain @ r
    P_new = (np.eye(3) - K_gain @ H) @ P

    # 5) Check the distance between the updated landmark and the IMU.
    #    The IMU position in the world frame is T_imu[:3, 3].
    imu_pos = T_imu[:3, 3]
    dist = np.linalg.norm(m_new - imu_pos)
    if dist > dist_threshold:
        # If too far, return the original m and P (or remove the landmark in the caller).
        return m, P

    # Return the updated landmark state and covariance.
    return m_new, P_new

def ekf_update_landmarks(landmarks, features, T_imu, K_l, K_r, extL_T_imu, extR_T_imu, R, init_cov, dist_threshold):
    """
    Perform EKF updates for all landmarks in a single timestep.
    
    Inputs:
        landmarks: dictionary with keys as landmark indices and values as (m, P) tuples.
        features: a (4, M) stereo measurement matrix where each column corresponds to a landmark.
                  For landmarks that are not observed, the column is [-1, -1, -1, -1].
        T_imu: 4x4 current IMU pose.
        K_l, K_r: intrinsic matrices for the left and right cameras.
        extL_T_imu, extR_T_imu: extrinsic calibration matrices from the IMU frame to the left/right camera frame.
        R: (4,4) measurement noise covariance matrix.
        init_cov: (3,3) initial error covariance for newly initialized landmarks.
        dist_threshold: distance threshold (in the world frame) to consider a landmark as an outlier.
    Output:
        Updated landmarks dictionary.
    """
    M = features.shape[1]
    for i in range(M):
        z = features[:, i]
        if np.all(z == -1):
            continue

        elif i not in landmarks:
            # First-time initialization via triangulation.
            m_init = triangulate_landmark(z, T_imu, K_l, K_r, extL_T_imu, extR_T_imu)
            # Check if the initial estimate is too far from the IMU; if so, skip initialization.
            imu_pos = T_imu[:3, 3]
            if np.linalg.norm(m_init - imu_pos) > dist_threshold:
                continue
            landmarks[i] = (m_init, init_cov.copy())
        else:
            m, P = landmarks[i]
            # Update using EKF.
            m_new, P_new = ekf_update_one_landmark(m, P, z, T_imu, K_l, K_r, extL_T_imu, extR_T_imu, R,
                                                     dist_threshold=dist_threshold)
            # If the update did not change the landmark (e.g., due to a large distance), remove it.
            if np.allclose(m_new, m, atol=1e-9):
                del landmarks[i]
            else:
                landmarks[i] = (m_new, P_new)
    return landmarks

def run_ekf_update():
    """
    Main function: Perform Landmark Mapping via EKF Update.
    
    Process:
    1. Load data, which includes IMU measurements, visual features, and calibration parameters.
    2. Compute the IMU trajectory using EKF Prediction (from ekf_predict.py); 
       here we assume the IMU estimated poses are correct for landmark updates.
    3. For each timestep:
         - Extract the current IMU pose and corresponding stereo feature measurements (shape: (4, M)).
         - Call ekf_update_landmarks() to update the states and covariances of all landmarks.
    4. After all updates, plot the landmarks (mainly the x-y coordinates) along with the IMU trajectory for visualization.
    """
    # Load data (iterate over 2 datasets, 00 & 01, because 02 doesn't provide features)
    datanum = ['00', '01']
    for j in datanum:
        filename = f"../data/dataset{j}/dataset{j}.npy"
        v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(filename)
        
        # Compute the IMU trajectory using EKF Prediction (assuming the trajectory is correct)
        # Set the process noise covariance Q; adjust parameters as needed
        sigma_v = 0.1       # Standard deviation for linear velocity (m/s)
        sigma_w = 0.01745   # Standard deviation for angular velocity (rad/s), approximately 1° in radians
        Q_default = np.diag([sigma_v**2, sigma_v**2, sigma_v**2,
                             sigma_w**2, sigma_w**2, sigma_w**2])
        trajectory, _ = predict_trajectory(v_t, w_t, timestamps, Q_default)
        
        # Set the measurement noise covariance (assuming a pixel standard deviation of 4)
        sigma_px = 4
        R = np.diag([sigma_px**2, sigma_px**2, sigma_px**2, sigma_px**2])
        
        # Set a large initial covariance for newly initialized landmarks
        init_cov = np.eye(3) * 10000.0
        
        # Set a threshold for outlier rejection
        threshold = 20
        
        # Initialize a dictionary to store landmarks (keys: landmark indices, values: (m, P) tuples)
        landmarks = {}
        
        # The shape of features is (4, M, T): 4 pixel values, M landmarks, T timesteps.
        T_total = features.shape[2]
        
        # Update landmarks for each timestep using the current IMU pose and stereo measurements.
        for t in range(T_total):
            T_imu = trajectory[t]
            features_t = features[:, :, t]  # Shape: (4, M)
            landmarks = ekf_update_landmarks(landmarks, features_t, T_imu,
                                             K_l, K_r, extL_T_imu, extR_T_imu, R, init_cov, threshold)
        
        # After updates, extract the x-y coordinates of all landmarks for plotting.
        landmark_positions = np.array([landmarks[i][0] for i in sorted(landmarks.keys())])

        plt.figure(figsize=(6,6))
        # Plot landmarks as blue dots.
        plt.scatter(landmark_positions[:, 0], landmark_positions[:, 1],
                    marker='.', s=3, c='b', label='landmarks')
        # Plot the IMU trajectory as a red line.
        x_imu = trajectory[:, 0, 3]
        y_imu = trajectory[:, 1, 3]
        plt.plot(x_imu, y_imu, 'r-', label='IMU Trajectory')
        # Mark the start and end points.
        plt.scatter(x_imu[0],  y_imu[0],  marker='s', c='r', label='start')
        plt.scatter(x_imu[-1], y_imu[-1], marker='o', c='orange', label='end')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Landmark mapping via EKF update for dataset{j}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    
    return landmarks

if __name__ == '__main__':
    run_ekf_update()
