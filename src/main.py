import os
import argparse
import cv2
from panorama_builder.read_images import load_images_from_directory
from panorama_builder.feature_extraction import compute_all_features
from panorama_builder.matching import match_to_base
from panorama_builder.homography import compute_all_homographies
from panorama_builder.warping import warp_images
from panorama_builder.blending import generate_gaussian_mask, blend_images

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

    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.10,
        help="Smoothing percent for blending masks (default: 0.10)."
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
    print(f"- Blending smoothing: {args.smoothing * 100:.1f}%")
    print("--- Starting panorama building ---")
    
    # Step 1: Load images
    images = load_images_from_directory(folder_path)

    # Step 2: Extract features
    features = compute_all_features(images)

    # Step 3: Select base image (middle one)
    base_idx = len(features) // 2
    print(f"Using image {base_idx+1} as base image.")

    # Step 4: Match features to base
    matches_with_base = match_to_base(features, base_idx)

    # Step 5: Compute homographies
    homographies = compute_all_homographies(features, base_idx, [m["matches"] for m in matches_with_base])

    # Step 6: Warp all images to base view
    warped_images, _, _ = warp_images([f["image"] for f in features], homographies)

    # Step 7: Generate blending masks
    masks = [generate_gaussian_mask(warped, smoothing_percent=args.smoothing) for warped in warped_images]

    # Step 8: Blend images
    panorama = blend_images(warped_images, masks)

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