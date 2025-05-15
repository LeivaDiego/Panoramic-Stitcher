import cv2


def extract_sift_features(image):
    """
    Extract SIFT features from an image.
    
    Args:
        image (np.ndarray): The input image from which to extract features.

    Returns:
        tuple: A tuple containing the keypoints and descriptors.

    Raises:
        Exception: If the image is not valid or if feature extraction fails.
    """
    try:
        # Convert the image to grayscale for SIFT
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Initialize SIFT detector
        feature_detector = cv2.SIFT.create()
       
        # Detect keypoints and compute descriptors
        keypoints, descriptors = feature_detector.detectAndCompute(grayscale_image, None)
        return keypoints, descriptors
    
    except Exception as e:
        print(f"ERROR | An error occurred while extracting features: {e}")
        return None, None


def compute_all_features(images):
    """
    Compute features for all images in a collection.

    Args:
        images (list): A list of images to process.

    Returns:
        list: A list of dictionaries containing the image, keypoints, and descriptors.
    """
    print("INFO | Extracting features from images...")
    # Initialize an empty list to store features
    features = []

    # Loop through each image and extract features
    for image in images:
        # Extract SIFT or ORB features
        keypoints, descriptors = extract_sift_features(image)
        
        # Check if descriptors are None
        if descriptors is None:
            print("WARNING: No descriptors found for one of the images. Skipping this image.")
            continue

        # Append the image, keypoints, and descriptors to the features list
        features.append({
            "image": image,
            "keypoints": keypoints,
            "descriptors": descriptors
        })
        print(f"INFO | Extracted features from image with {len(keypoints)} keypoints.")
    
    print("SUCCESS | SIFT features extracted successfully.")
    
    # Return the list of features
    return features
