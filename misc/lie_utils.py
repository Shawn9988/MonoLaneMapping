#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import numpy as np
from scipy.spatial.transform import Rotation as R

def skew_symmetric(v):
    """Convert a 3-vector to a skew-symmetric matrix such that 
        dot(skew_symmetric(v), w) = cross(v, w)"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def unskew_symmetric(mat):
    """Convert a skew-symmetric matrix to a 3-vector"""
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])

def se3_log(transform):
    """Compute the logarithm of an SE(3) transformation matrix."""
    R = transform[:3, :3]
    t = transform[:3, 3]
    
    # Get rotation angle (theta) from trace of R
    cos_theta = (np.trace(R) - 1) / 2
    # Clamp to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # Handle special cases
    if np.isclose(theta, 0):
        # Small angle approximation
        omega = np.zeros(3)
        v = t
    elif np.isclose(theta, np.pi, atol=1e-6):
        # Angle near pi
        # Find rotation axis
        omega_mat = np.log(R)  # This is actually skew symmetric
        omega = unskew_symmetric(omega_mat)
        # Compute v vector
        # In this case, we use the general formula
        W = np.eye(3) - theta / 2 * skew_symmetric(omega) + (1 - theta / np.tan(theta / 2)) / theta**2 * (skew_symmetric(omega) @ skew_symmetric(omega))
        v = np.linalg.solve(W, t)
    else:
        # General case
        omega_mat = theta / (2 * np.sin(theta)) * (R - R.T)
        omega = unskew_symmetric(omega_mat)
        
        # Compute W matrix for translation
        W = np.eye(3) - theta / 2 * skew_symmetric(omega) + (1 - theta / np.tan(theta / 2)) / theta**2 * (skew_symmetric(omega) @ skew_symmetric(omega))
        v = np.linalg.solve(W, t)
    
    return np.concatenate([omega, v])

def se3_exp(se3_vec):
    """Compute the exponential of an se(3) vector."""
    omega = se3_vec[:3]
    v = se3_vec[3:]
    
    theta = np.linalg.norm(omega)
    omega_mat = skew_symmetric(omega)
    
    if np.isclose(theta, 0):
        # Small angle approximation
        R = np.eye(3) + omega_mat
        t = v
    else:
        # General case
        omega_unit = omega / theta
        omega_unit_mat = skew_symmetric(omega_unit)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        R = np.eye(3) + sin_theta * omega_unit_mat + (1 - cos_theta) * (omega_unit_mat @ omega_unit_mat)
        
        # Compute translation
        W = np.eye(3) + ((1 - cos_theta) / theta) * omega_unit_mat + (theta - sin_theta) / (theta**2) * (omega_unit_mat @ omega_unit_mat)
        t = W @ v
    
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform

def make_noisy_pose2d(pose, yaw_std, trans_std):
    # pose: 4x4
    ypr = [np.random.normal(0, yaw_std), 0, 0]
    pose_noisy = np.eye(4)
    pose_noisy[:3, :3] = R.from_euler('zyx', ypr, degrees=True).as_matrix()
    pose_noisy[:2, 3] += np.random.normal(0, trans_std, 2)
    pose_add_noise = pose @ pose_noisy
    return pose_add_noise

def get_pose2d_noise(yaw_std, trans_std):
    # pose: 4x4
    ypr = [np.random.normal(0, yaw_std), 0, 0]
    pose_noisy = np.eye(4)
    pose_noisy[:3, :3] = R.from_euler('zyx', ypr, degrees=True).as_matrix()
    pose_noisy[:2, 3] += np.random.normal(0, trans_std, 2)
    return pose_noisy

def inv_se3(transform):
    return np.linalg.inv(transform)
    # transform: (4, 4)
    # points: (N, 3)
    inv_transform = np.eye(4)
    inv_transform[:3, :3] = transform[:3, :3].T
    inv_transform[:3, 3] = -np.dot(transform[:3, :3].T, transform[:3, 3])
    return inv_transform

def so3_to_quat(rot):
    r = R.from_matrix(rot)
    return r.as_quat()

def so3_to_rotvec(rot) -> np.ndarray:
    r = R.from_matrix(rot)
    rotvec = r.as_rotvec()
    return rotvec

def rad2deg(rad):
    return rad * 180.0 / np.pi

def se3_to_euler_xyz(transform):
    # transform: (4, 4)
    # points: (N, 3)
    r = R.from_matrix(transform[:3, :3])
    euler = r.as_euler('zyx', degrees=True)
    t = transform[:3, 3]
    return np.concatenate([euler, t]).tolist()

def compute_rpe(pose_est_i, pose_est_j, pose_gt_i, pose_gt_j):
    gt_ij = inv_se3(pose_gt_j) @ pose_gt_i
    est_ij = inv_se3(pose_est_j) @ pose_est_i
    delta_ij = inv_se3(est_ij) @ gt_ij
    delta_deg = rot_to_angle(delta_ij[:3, :3], deg=True)
    delta_xyz = np.linalg.norm(delta_ij[:3, 3])
    return delta_deg, delta_xyz

def rot_to_angle(rot, deg=True):
    rotvec = so3_to_rotvec(rot)
    angle = np.linalg.norm(rotvec)
    if deg:
        angle = rad2deg(angle)
    return angle