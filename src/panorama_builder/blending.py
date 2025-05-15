import numpy as np
import cv2

def single_weights_matrix(shape):
    """
    Create a weight mask with higher weights in the center and lower on edges.
    The mask is a 2D tent function that decreases linearly from the center to the edges.

    Args:
        shape: Shape of the image (height, width, channels).

    Returns:
        A 2D weight mask with values in [0, 1].
    """
    # Create a 2D tent function
    h, w = shape[:2]
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    # Create a meshgrid for 2D tent function
    xv, yv = np.meshgrid(x, y)
    # Calculate the weight based on the distance from the center
    weight = (1 - np.abs(xv)) * (1 - np.abs(yv))  # 2D tent function
    # Return the weight mask
    return weight.astype(np.float32)

def blend_images(warped_images):
    """
    Blend warped images using center-weighted interpolation inspired by multi-band blending.

    Args:
        warped_images: List of warped images (aligned to same canvas).

    Returns:
        Blended panorama image.
    """
    print("INFO | Blending images...")
    # Get the dimensions of the first warped image
    height, width = warped_images[0].shape[:2]
    # Initialize panorama and weights
    panorama = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    # Loop through each warped image
    for img in warped_images:
        # Convert to float32 and normalize
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = (gray > 0).astype(np.float32) # Create a mask for valid pixels

        # Generate base weights for the current image
        base_weights = single_weights_matrix(img.shape)
        base_weights *= mask

        # Normalize the base weights and generate 3-channel weights
        base_weights_3c = np.repeat(base_weights[:, :, np.newaxis], 3, axis=2)
        mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Update the weights and panorama using the mask
        weight_sum = weights + base_weights[:, :, np.newaxis]
        normalized_weights = np.divide(weights, weight_sum, where=weight_sum != 0)

        # Update the panorama using the normalized weights
        panorama = np.where(
            mask_3c == 0,
            panorama,
            img * (1 - normalized_weights) + panorama * normalized_weights
        )

        # Update the weights
        weights += base_weights[:, :, np.newaxis]

    # Normalize the panorama to [0, 255]
    panorama = np.clip(panorama, 0, 255)

    print("SUCCESS | Blending completed.")
    # Convert to uint8
    return panorama.astype(np.uint8)
