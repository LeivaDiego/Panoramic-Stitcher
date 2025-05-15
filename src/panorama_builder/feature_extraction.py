import cv2
import numpy as np

def extract_sift_features(image, descriptor_type):
    """
    Extract SIFT or ORB features from an image.
    
    Args:
        image (np.ndarray): The input image from which to extract features.
        descriptor_type (str): The type of descriptor to use ('SIFT' or 'ORB').

    Returns:
        tuple: A tuple containing the keypoints and descriptors.

    Raises:
        ValueError: If the descriptor type is not 'SIFT' or 'ORB'.
    """
    try:
        # Convert the image to grayscale for SIFT
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Initialize SIFT or ORB detector
        if descriptor_type == 'ORB':
            feature_detector = cv2.ORB.create()
        elif descriptor_type == 'SIFT':
            feature_detector = cv2.SIFT.create()
        else:
            raise ValueError("Invalid descriptor type. Use 'SIFT' or 'ORB'.")
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = feature_detector.detectAndCompute(grayscale_image, None)
        return keypoints, descriptors
    
    except Exception as e:
        print(f"ERROR | An error occurred while extracting features: {e}")
        return None, None


def compute_all_features(images, descriptor_type='SIFT'):
    """
    Compute features for all images in a collection.

    Args:
        images (list): A list of images to process.
        descriptor_type (str): The type of descriptor to use ('SIFT' or 'ORB').

    Returns:
        list: A list of dictionaries containing the image, keypoints, and descriptors.
    """
    print("INFO | Extracting features from images...")
    # Initialize an empty list to store features
    features = []

    # Loop through each image and extract features
    for image in images:
        # Extract SIFT or ORB features
        keypoints, descriptors = extract_sift_features(image, descriptor_type)
        
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
    
    print("SUCCESS | Features extracted successfully.")
    
    # Return the list of features
    return features
