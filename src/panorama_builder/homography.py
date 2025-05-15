import cv2
import numpy as np


def compute_homography(kp_base, kp_target, matches, ransac_thresh = 5.0, ransac_max_iter = 500):
    """
    Compute the homography matrix using RANSAC.
    
    Args:
        kp_base (list): Keypoints from the base image.
        kp_target (list): Keypoints from the target image.
        matches (list): Matches between the two sets of keypoints.
        ransac_thresh (float): RANSAC reprojection threshold.
        ransac_max_iter (int): Maximum number of RANSAC iterations.

    Returns:
        tuple: A tuple containing the homography matrix and the mask of inliers.
    """
    # Check if the number of matches is sufficient
    if len(matches) < 4:
        print("ERROR | Not enough matches to compute homography.")
        return None, None
    
    # Extract the matched keypoints from both images
    base_matchpoints = np.float32([kp_base[match.queryIdx].pt for match in matches])
    target_matchpoints = np.float32([kp_target[match.trainIdx].pt for match in matches])
    
    # Compute the homography using RANSAC
    #   With the reprojection threshold and maximum iterations

    H, status = cv2.findHomography(
        base_matchpoints, 
        target_matchpoints, 
        cv2.RANSAC, 
        ransac_thresh, 
        maxIters = ransac_max_iter
    )

    # Return the homography matrix and the mask of inliers
    return H, status


def compute_chain_homographies(features, matches_all, base_idx):
    """
    Compute the homography matrices for a chain of images.
    
    Args:
        features (list): List of features for each image.
        matches (list): List of matches between images.
        base_idx (int): Index of the base image.

    Returns:
        list: List of homography matrices.
    """
    # Compute the homographies for a chain of images.
    # The base image is the one at index base_idx.
    n = len(features)
    homographies = [None] * n
    homographies[base_idx] = np.eye(3)

    # Left to right
    for i in range(base_idx - 1, -1, -1):
        kp1 = features[i]["keypoints"]
        kp2 = features[i + 1]["keypoints"]
        match_pair = matches_all[i]
        H, _ = compute_homography(kp1, kp2, match_pair)
        if H is None:
            print(f"WARNING | Homography failed for images {i+1} → {i}")
            continue
        homographies[i] = homographies[i + 1] @ H

    # Right to left
    for i in range(base_idx + 1, n):
        kp1 = features[i - 1]["keypoints"]
        kp2 = features[i]["keypoints"]
        match_pair = matches_all[i - 1]
        H, _ = compute_homography(kp1, kp2, match_pair)
        if H is None:
            print(f"WARNING | Homography failed for images {i-1} → {i}")
            continue
        homographies[i] = homographies[i - 1] @ np.linalg.inv(H)

    print("SUCCESS | Homographies computed for all images.")

    return homographies