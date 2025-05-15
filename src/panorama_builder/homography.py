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


def compute_all_homographies(features, base_idx, matches_with_base):
    
    # Initialize an empty list to store homographies
    homographies = [None] * len(features)
    # Get the base image features
    homographies[base_idx] = np.eye(3)  # Identity matrix for the base image

    # Get the keypoints from the base image
    kp_base = features[base_idx]["keypoints"]

    # Loop through each image and compute homography
    for i, matches in enumerate(matches_with_base):
        if i == base_idx:
            continue  # Skip the base image

        # Get the keypoints from the target image
        kp_target = features[i]["keypoints"]

        # Compute the homography
        H, status = compute_homography(kp_base, kp_target, matches)

        # Check if the homography is valid
        if H is not None:
            homographies[i] = H
            print(f"INFO | Homography computed for image {i} with base image {base_idx}.")
        else:
            print(f"WARNING | Homography computation failed for image {i} with base image {base_idx}.")

    return homographies