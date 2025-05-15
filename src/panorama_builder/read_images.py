import cv2
import os


def load_images_from_directory(collection_path):
    """
    Load images from a specified directory into a list.
    
    Args:
        collection_path (str): The path to the directory containing the images.

    Returns:
        list: A list of images read from the directory.

    Raises:
        ValueError: If the directory does not exist, is empty, or contains no valid images.
    """
    print("INFO | Loading images from directory...")
    # Define the valid image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Check if the directory exists
    if not os.path.exists(collection_path):
        raise ValueError(f"The directory {collection_path} does not exist. Please check the path and try again.")
    
    # Check if the directory is empty
    if len(os.listdir(collection_path)) == 0:
        raise ValueError(f"The directory {collection_path} is empty. Please add images and try again.")
    
    # Get a sorted list of all image files in the directory
    # and filter out any non-image files
    image_files = sorted(
        [i for i in os.listdir(collection_path) if os.path.splitext(i)[1].lower() in valid_extensions]
        )

    # Check if there are any valid image files in the directory
    if len(image_files) == 0:
        raise ValueError(f"No valid images found in {collection_path}. Please check the directory and try again.")

    # Initialize an empty list to store the images
    images = []

    # Loop through the image files and read them 
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(collection_path, image_file)
        # Read the image using OpenCV
        image = cv2.imread(image_path)
         # Check if the image was read successfully
        if image is  None:
            print(f"WARNING: {image_file} could not be read and will be skipped.")
            continue
        
         # Convert the image from BGR to RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Append the image to the list
        images.append(rgb_image)
        print(f"INFO | Loaded image: {image_file}")

    print("SUCCESS | Images loaded successfully.")

    # Return the list of images
    return images
    