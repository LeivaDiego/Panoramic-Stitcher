import os
import argparse
import cv2
from panorama_builder.read_images import load_images_from_directory
from panorama_builder.feature_extraction import compute_all_features
from panorama_builder.matching import match_features
from panorama_builder.homography import compute_chain_homographies
from panorama_builder.warping import warp_images
from panorama_builder.blending import blend_images, apply_color_correction

def parse_arguments():
    """
    Parse command-line arguments using argparse.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Panorama builder: stitch a set of overlapping images into a single panoramic image."
    )

    parser.add_argument(
        "--input",
        default="data/input/",
        type=str,
        help="Path to the folder containing input images."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/output/panorama.jpg",
        help="Path to save the output panorama image (default: 'data/output/panorama.jpg')."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    folder_path = args.input

    if not os.path.exists(folder_path):
        raise(f"ERROR | The folder '{folder_path}' does not exist.")

    if not os.path.isdir(folder_path):
        raise(f"ERROR | '{folder_path}' is not a directory.")

    print(f"- Image folder found: {folder_path}")
    print("--- Starting panorama building ---")
    
    # Step 1: Load images
    images = load_images_from_directory(folder_path)

    # Step 2: Extract features
    features = compute_all_features(images)

    # Step 3: Select base image (middle one)
    base_idx = len(features) // 2
    print(f"Using image {base_idx+1} as base image.")

    # Step 4: Match features between images
    matches_all = []
    for i in range(len(features)-1):
        desc1 = features[i]["descriptors"]
        desc2 = features[i+1]["descriptors"]
        matches = match_features(desc1, desc2)
        print(f"INFO | Matches between image {i} and {i+1}: {len(matches)}")
        matches_all.append(matches)

    # Step 5: Compute homographies in a chain
    homographies = compute_chain_homographies(features, matches_all, base_idx)
    # Filter out None homographies
    #   and keep only valid images
    valid_images = []
    valid_homographies = []
    for feat, H in zip(features, homographies):
        if H is not None:
            valid_images.append(feat["image"])
            valid_homographies.append(H)
        else:
            print("WARNING | Skipping image due to missing homography.")

    # Step 6: Apply color gain compensation
    apply_color_correction(valid_images, base_idx)

    # Step 7: Warp all images to base view
    warped_images, _, _ = warp_images(valid_images, valid_homographies)

    # Step 8: Blend images
    panorama = blend_images(warped_images)

    # Step 9: Save output
    cv2.imwrite(args.output, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
    
    print(f"SUCCESS | Panorama saved as '{args.output}'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{e}")
        exit(1)
    finally:
        print("Panorama building completed.")