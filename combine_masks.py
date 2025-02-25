import cv2
import numpy as np
import os
import glob
from collections import defaultdict


def combine_masks(base_mask_path, disease_mask_path, output_path):
    """
    Combine base and disease masks with the following mapping:
    - background in base mask (0) => black (0,0,0)
    - white in base mask (255) => blue (255,0,0)
    - white in disease mask (255) that overlaps with white in base mask => green (0,255,0)
    """
    base_mask = cv2.imread(base_mask_path, cv2.IMREAD_GRAYSCALE)
    disease_mask = cv2.imread(disease_mask_path, cv2.IMREAD_GRAYSCALE)

    if base_mask.shape != disease_mask.shape:
        raise ValueError(f"Mask dimensions do not match for {base_mask_path} and {disease_mask_path}")

    # Create a 3-channel mask
    combined_mask = np.zeros((base_mask.shape[0], base_mask.shape[1], 3), dtype=np.uint8)

    # Convert binary masks
    base_mask_binary = (base_mask == 255).astype(np.uint8)
    disease_mask_binary = (disease_mask == 255).astype(np.uint8)

    # Set blue regions (category 1)
    combined_mask[base_mask_binary == 1] = [255, 0, 0]  # BGR format

    # Set green regions (category 2)
    combined_mask[(base_mask_binary == 1) & (disease_mask_binary == 1)] = [0, 255, 0]  # BGR format

    cv2.imwrite(output_path, combined_mask)
    return combined_mask


def get_base_index_from_disease_mask(filename):
    """Extract the base mask index from a disease mask filename"""
    return int(filename.split('_')[0])


def process_mask_directories(base_mask_dir, disease_mask_dir, output_dir):
    """
    Process all matching masks in the base and disease directories
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all base masks
    base_masks = glob.glob(os.path.join(base_mask_dir, "*.png"))
    base_masks.sort()

    # Get all disease masks
    disease_masks = glob.glob(os.path.join(disease_mask_dir, "*.png"))
    disease_masks.sort()

    # Group disease masks by their base mask index
    disease_masks_by_base = defaultdict(list)
    for disease_mask in disease_masks:
        base_idx = get_base_index_from_disease_mask(os.path.basename(disease_mask))
        disease_masks_by_base[base_idx].append(disease_mask)

    # Process each base mask
    for base_mask_path in base_masks:
        base_idx = int(os.path.splitext(os.path.basename(base_mask_path))[0])

        # Get all corresponding disease masks
        corresponding_disease_masks = disease_masks_by_base.get(base_idx, [])

        if not corresponding_disease_masks:
            print(f"No disease masks found for base mask {base_idx}")
            continue

        # Process each disease mask for this base mask
        for disease_mask_path in corresponding_disease_masks:
            disease_mask_name = os.path.basename(disease_mask_path)
            output_name = f"{disease_mask_name}"
            output_path = os.path.join(output_dir, output_name)

            try:
                combine_masks(base_mask_path, disease_mask_path, output_path)
                print(f"Processed {disease_mask_name} with base mask {base_idx}")
            except Exception as e:
                print(f"Error processing {disease_mask_name}: {str(e)}")


if __name__ == "__main__":
    base_mask_dir = 'data/masks'
    disease_mask_dir = 'output/vitiligo_sim/masks'
    output_dir = 'output/combined_masks'

    process_mask_directories(base_mask_dir, disease_mask_dir, output_dir)
