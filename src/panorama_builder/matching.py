import cv2
import numpy as np


def match_features(desc1, desc2, ratio=0.75):
    """
    Match features between two sets of descriptors using the brute force matcher
    and the Lowe's ratio test.

    Args:
        desc1 (numpy.ndarray): The first set of descriptors.
        desc2 (numpy.ndarray): The second set of descriptors.
        ratio (float): The ratio threshold for matching.

    Returns:
        list: A list of matched keypoints.
    """
    # Define the brute force matcher object
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Initialize the matches
    matches = []

    # Find the matches between the two feature descriptors
    raw_matches = bf_matcher.knnMatch(desc1, desc2, 2)

    # Filter the matches using the Lowes ratio test
    for m, n in raw_matches:
        # Check if the distance of the first match is less than 
        # the ratio of the second match
        if m.distance < ratio * n.distance:
            matches.append(m)

    # Sort the matches based on the distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Return the matches
    return matches


def match_to_base(features, base_index, ratio=0.75):
    """
    Match features from all images to a base image.
    
    Args:
        features (list): A list of dictionaries containing image features.
        base_index (int): The index of the base image in the features list.
        ratio (float): The ratio threshold for matching.

    Returns:
        list: A list of dictionaries containing the base image index, 
              current image index, and matched keypoints.
    """
    # Initialize an empty list to store the matches
    matches = []

    # Get the base image descriptors
    base_desc = features[base_index]["descriptors"]

    # Loop through each feature set and match to the base image
    for i, feature in enumerate(features):
        # Skip the base image itself
        if i == base_index:
            continue

        # Get the current image descriptors
        curr_desc = feature["descriptors"]

        # Match the features between the base and current image
        matched_keypoints = match_features(base_desc, curr_desc, ratio)

        # Append the matches to the list
        matches.append({
            "base_index": base_index,
            "curr_index": i,
            "matches": matched_keypoints
        })

    # Return the list of matches
    return matches
