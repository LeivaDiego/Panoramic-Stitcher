import numpy as np
import cv2

def apply_color_correction(valid_images, base_idx):
    """
    Apply color gain compensation to a list of images.
    This function adjusts the color of each image to match the average color 
    of a reference image (the base image).

    Args:
        valid_images (list): List of images to be adjusted.
        base_idx (int): Index of the reference image in the list.

    Returns:
        list: List of color-adjusted images.
    """
    print("INFO | Applying color gain compensation...")
    reference_image = valid_images[base_idx]  # Middle image as reference
    reference_mean = compute_average_color(reference_image)
    print(f"INFO | Reference image mean color: {reference_mean}")

    # Loop through all images and apply color gain
    for i in range(len(valid_images)):
        # Get the image and compute its mean color
        img = valid_images[i]
        # Skip the reference image
        if i == base_idx:
            continue
        # Compute the mean color of the image
        # and apply color gain compensation
        image_mean = compute_average_color(img)
        adjusted_img = apply_color_gain(img, reference_mean, image_mean)
        valid_images[i] = adjusted_img
    
    print("SUCCESS | Color gain adjustment completed.")
    # Return the adjusted images
    return valid_images

    

def compute_average_color(image, mask=None):
    """
    Compute the average color (R, G, B) of an image within a mask.
    If no mask is provided, the average color of the entire image is computed.
    
    Args:
        image (np.ndarray): Input image.
        mask (np.ndarray, optional): Binary mask to specify the region of interest.

    Returns:
        np.ndarray: Average color (R, G, B) of the image.
    """
    # Check if mask is provided
    if mask is None:
        # Create a mask where all pixels are considered
        mask = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) > 0).astype(np.uint8)
    
    # Initialize mean color array
    mean_colors = []

    # Loop through each channel (R, G, B)
    # and compute the mean color using the mask
    for c in range(3):  # R, G, B
        channel = image[:, :, c]
        mean = np.sum(channel * mask) / np.sum(mask)
        # Append the mean color to the list
        mean_colors.append(mean)
    
    # Convert the list to a numpy array and return it
    return np.array(mean_colors)


def apply_color_gain(image, reference_mean, image_mean):
    """
    Apply per-channel gain to match the reference mean color.

    Args:
        image (np.ndarray): Input image.
        reference_mean (np.ndarray): Target average color.
        image_mean (np.ndarray): Current average color.

    Returns:
        np.ndarray: Color-adjusted image.
    """
    # Compute the gain for each channel
    # to match the reference mean color
    gain = reference_mean / (image_mean + 1e-5)
    # Ensure gain is a 1D array with 3 elements
    adjusted = image.astype(np.float32)
    # Apply the gain to each channel
    for c in range(3):
        adjusted[:, :, c] *= gain[c]

    # Clip the values to be in the range [0, 255]
    # and convert back to uint8 format
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def create_feather_mask(image, direction="left"):
    """
    Create a feathering mask for blending images.
    The mask is a gradient that fades from 1 to 0 in the specified direction.

    Args:
        image (np.ndarray): Input image.
        direction (str): Direction of feathering ("left", "right", "center").

    Returns:
        np.ndarray: Feathering mask.
    """
    
    # Get the height and width of the image
    h, w = image.shape[:2]
    # Initialize a mask with zeros
    mask = np.zeros((h, w), dtype=np.float32)

    # Create a feathering mask based on the specified direction
    if direction == "left":
        mask = np.tile(np.linspace(1, 0, w), (h, 1))
    elif direction == "right":
        mask = np.tile(np.linspace(0, 1, w), (h, 1))
    elif direction == "center":
        left = np.linspace(0, 1, w // 2)
        right = np.linspace(1, 0, w - w // 2)
        mask = np.tile(np.concatenate([left, right]), (h, 1))

    # Return the mask
    return mask


def blend_images(warped_images):
    """
    Blend images using feathering masks.
    This function takes a list of warped images and blends them together
    using feathering masks to create a seamless panorama.
    The blending is done by applying a feathering mask to each image
    and accumulating the results.

    Args:
        warped_images (list): List of warped images to be blended.

    Returns:
        np.ndarray: Blended panorama image.
    """
    print("INFO | Blending with feathering masks...")
    # Get the height and width of the first warped image
    height, width = warped_images[0].shape[:2]
    # Initialize an accumulator for the blended image
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width, 1), dtype=np.float32)

    # Get the number of warped images
    n = len(warped_images)

    # Loop through each warped image
    for i, img in enumerate(warped_images):
        # Skip completely black images
        if np.sum(img) == 0:
            continue

        # Convert the image to grayscale and create a binary mask
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask_bin = (gray > 0).astype(np.float32)

        # Decide direction of feathering
        if i == 0:
            mask = create_feather_mask(img, direction="left")
        elif i == n - 1:
            mask = create_feather_mask(img, direction="right")
        else:
            mask = create_feather_mask(img, direction="center")

        # Apply the binary mask to the feathering mask
        mask *= mask_bin  # Mask out non-overlapping pixels
        mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Accumulate the weighted image and the weight sum
        accumulator += img.astype(np.float32) * mask_3c
        weight_sum += mask[:, :, np.newaxis]

    # Normalize the accumulated image by the weight sum
    # Avoid division by zero by setting zero weights to 1
    weight_sum[weight_sum == 0] = 1
    result = accumulator / weight_sum

    # Clip the values to be in the range [0, 255]
    # and convert back to uint8 format
    return np.clip(result, 0, 255).astype(np.uint8)