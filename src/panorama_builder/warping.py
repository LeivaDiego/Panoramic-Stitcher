import cv2
from panorama_builder.utils import get_transformed_corners, compute_offset_and_canvas_size

def warp_images(images, homographies):
    """
    Warps a list of images using their respective homography matrices.

    Args:
        images (list[np.ndarray]): List of images to be warped.
        homographies (list[np.ndarray]): List of homography matrices corresponding to the images.

    Returns:
        tuple: A tuple containing:
            - warped_images (list[np.ndarray]): List of warped images.
            - offset_matrix (np.ndarray): The offset matrix used for warping.
            - canvas_size (tuple): The size of the canvas (width, height).
    """
    # Get the transformed corners of all images
    #   using the homography matrices
    all_corners = []
    for img, H in zip(images, homographies):
        corners = get_transformed_corners(img, H)
        all_corners.append(corners)

    # Compute the offset matrix and canvas size
    offset_matrix, canvas_size = compute_offset_and_canvas_size(all_corners)

    # Initialize a list to store the warped images
    #   and apply the offset matrix to each homography
    warped_images = []
    for img, H in zip(images, homographies):
        H_total = offset_matrix @ H # Combine the offset matrix with the homography
        # Apply the homography to warp the image
        #   and store the warped image in the list
        warped = cv2.warpPerspective(img, H_total, canvas_size)
        warped_images.append(warped)

    # Return the list of warped images, offset matrix, and canvas size
    return warped_images, offset_matrix, canvas_size
