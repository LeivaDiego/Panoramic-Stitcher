import numpy as np
import cv2

def get_transformed_corners(image, H):
    """
    Transforms the corners of an image using a homography matrix.
    
    Args:
        image (np.ndarray): The input image.
        H (np.ndarray): The homography matrix.
    
    Returns:
        np.ndarray: The transformed corners of the image.
    """
    # Get the dimensions of the image
    h, w = image.shape[:2]

    # Define the corners of the image
    #   and reshape them to be in the correct format
    corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1, 1, 2)
    
    # Transform the corners using the homography matrix
    transformed = cv2.perspectiveTransform(corners, H)

    # Return the transformed corners
    return transformed


def compute_offset_and_canvas_size(all_corners):
    """
    Computes the offset matrix and canvas size based on the transformed corners of all images.
    
    Args:
        all_corners (list[np.ndarray]): List of transformed corners from all images.

    Returns:
        tuple: A tuple containing the offset matrix (3x3) and the canvas size (width, height).
    """
    # Get the minimum and maximum x and y coordinates from all transformed corners
    #   and convert them to integers
    all_pts = np.vstack(all_corners)
    min_x = np.floor(np.min(all_pts[:, 0, 0])).astype(int)
    min_y = np.floor(np.min(all_pts[:, 0, 1])).astype(int)
    max_x = np.ceil(np.max(all_pts[:, 0, 0])).astype(int)
    max_y = np.ceil(np.max(all_pts[:, 0, 1])).astype(int)

    # Calculate the offset for x and y coordinates
    #   to ensure all transformed corners are within the canvas
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0
    
    # Create the offset matrix to shift the image
    #   to the positive quadrant of the canvas
    offset_matrix = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # Calculate the canvas size
    #   by adding the offset to the maximum coordinates
    canvas_width = max_x + offset_x
    canvas_height = max_y + offset_y

    # Return the offset matrix and the canvas size
    #   as a tuple of (width, height)
    return offset_matrix, (canvas_width, canvas_height)
