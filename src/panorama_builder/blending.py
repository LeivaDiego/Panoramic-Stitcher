import numpy as np
import cv2

def estimate_blur_size(image, smoothing_percent=0.10, min_size=51, max_size=301):
    """
    Estimate the Gaussian blur kernel size based on a percentage of the image width.

    Args:
        image: The input image (already warped to canvas).
        smoothing_percent: Fraction of the image width to determine blur size.
        min_size: Minimum kernel size allowed.
        max_size: Maximum kernel size allowed.

    Returns:
        An odd integer for the blur kernel size.
    """
    # Get the width of the image
    width = image.shape[1]
    # Calculate the estimated kernel size based on the width
    #   and the smoothing percentage
    estimated = int(smoothing_percent * width)
    estimated = max(min_size, min(estimated, max_size))
    # Ensure the kernel size is odd
    estimated = estimated + 1 if estimated % 2 == 0 else estimated
    # Return the estimated kernel size
    return estimated


def generate_gaussian_mask(image, smoothing_percent=0.10):
    """
    Generate a smooth Gaussian blending mask for a warped image.

    Args:
        image: Warped RGB image.
        smoothing_percent: Determines blur strength based on image width.

    Returns:
        A 2D mask with values in [0, 1], same width and height as the image.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Generate a binary mask where pixels are greater than 0
    mask = (gray > 0).astype(np.float32)

    # Compute the Gaussian blur size based on the image width
    #   and the smoothing percentage
    blur_size = estimate_blur_size(image, smoothing_percent)
    blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    # Normalize the blurred mask to the range [0, 1]
    normalized = blurred / (blurred.max() + 1e-8)
    
    # Return the normalized mask
    return normalized


def blend_images(warped_images, masks):
    """
    Blend a list of warped images using Gaussian masks.

    Args:
        warped_images: List of images aligned to the same canvas.
        masks: List of normalized masks corresponding to each image.

    Returns:
        The final blended panorama image.
    """
    # Initialize the panorama image and total weights
    height, width = warped_images[0].shape[:2]
    panorama = np.zeros((height, width, 3), dtype=np.float32)
    total_weights = np.zeros((height, width), dtype=np.float32)

    # Loop through each warped image and its corresponding mask
    #   to blend them into the panorama
    for img, mask in zip(warped_images, masks):
        for c in range(3):  # For each color channel
            panorama[:, :, c] += img[:, :, c] * mask
        total_weights += mask

    # Normalize the panorama by the total weights to avoid artifacts
    total_weights = np.clip(total_weights, 1e-8, None)
    panorama /= total_weights[:, :, np.newaxis]

    # Return the blended panorama image
    #   and convert it to uint8 format
    return panorama.astype(np.uint8)
