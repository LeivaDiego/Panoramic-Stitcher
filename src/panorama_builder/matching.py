import cv2


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
