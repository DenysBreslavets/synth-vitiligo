import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm
from patch_process import extract_patches
from multiprocessing import Pool


def create_directory_structure(base_dir):
    """Create the required directory structure."""
    directories = [
        'train/images/original',
        'train/masks/original',
        'train/images/patches',
        'train/masks/patches',
        'val/images/original',
        'val/masks/original',
        'val/images/patches',
        'val/masks/patches'
    ]

    try:
        # Clear existing directory if it exists
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

        # Create new directories
        for dir_path in directories:
            os.makedirs(os.path.join(base_dir, dir_path), exist_ok=True)

    except PermissionError:
        print(f"Error: Permission denied when accessing {base_dir}")
        raise


def get_dataset_files(images_dir, masks_dir):
    """Get sorted lists of image and mask files."""
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    return image_files, mask_files


def split_dataset(files, train_split):
    """Split files into train and validation sets."""
    num_files = len(files)
    num_train = int(num_files * train_split)

    indices = np.arange(num_files)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    return train_indices, val_indices


def process_single_image(args):
    """Process a single image and its mask."""
    idx, image_file, mask_file, input_images_dir, input_masks_dir, output_base_dir, \
        split_name, patch_size, overlap_percent, scale_factor = args

    # Process image
    image = cv2.imread(os.path.join(input_images_dir, image_file))
    image_patches, shape, effective_scale = extract_patches(
        image=image,
        patch_size=patch_size,
        overlap_percent=overlap_percent,
        scale_factor=scale_factor
    )

    # Process mask
    mask = cv2.imread(os.path.join(input_masks_dir, mask_file))
    mask_patches, _, _ = extract_patches(
        image=mask,
        patch_size=patch_size,
        overlap_percent=overlap_percent,
        scale_factor=scale_factor
    )

    scale_dir = f"{scale_factor:.2f}"
    base_name = os.path.splitext(image_file)[0]

    # Save patches
    for patch_idx, (img_patch, mask_patch) in enumerate(zip(image_patches, mask_patches)):
        # Save image patch
        img_patch_path = os.path.join(
            output_base_dir, split_name, 'images/patches', scale_dir,
            f'{base_name}_{scale_factor:.2f}_{patch_idx}.jpg'
        )
        cv2.imwrite(img_patch_path, img_patch)

        # Save mask patch
        mask_patch_path = os.path.join(
            output_base_dir, split_name, 'masks/patches', scale_dir,
            f'{base_name}_{scale_factor:.2f}_{patch_idx}.png'
        )
        cv2.imwrite(mask_patch_path, mask_patch)


def copy_original_files(args):
    """Copy original image and mask files."""
    idx, image_file, mask_file, input_images_dir, input_masks_dir, output_base_dir, split_name = args

    # Copy image
    src_img = os.path.join(input_images_dir, image_file)
    dst_img = os.path.join(output_base_dir, split_name, 'images/original', image_file)
    shutil.copy2(src_img, dst_img)

    # Copy mask
    src_mask = os.path.join(input_masks_dir, mask_file)
    dst_mask = os.path.join(output_base_dir, split_name, 'masks/original', mask_file)
    shutil.copy2(src_mask, dst_mask)


def process_dataset(input_images_dir, input_masks_dir, output_base_dir, train_split,
                    patch_size, overlap_percent, scale_factors):
    """
    Process a dataset of images and masks by splitting into train/validation sets and generating scaled patches.

    The function:
    1. Reads JPG images and PNG masks from separate input directories
    2. Splits data into train/validation sets while keeping image-mask pairs together
    3. Creates patches at multiple scales with specified overlap
    4. Saves both original files and generated patches in organized directory structure

    Directory Structure Created:
    processed_dataset/
    ├── train/
    │   ├── images/
    │   │   ├── original/      # Original training images
    │   │   └── patches/
    │   │       └── {scale}/   # Patches at each scale factor
    │   └── masks/
    │       ├── original/      # Original training masks
    │       └── patches/
    │           └── {scale}/   # Patches at each scale factor
    └── val/                   # Same structure as train/

    File Naming:
    - Original files: {number}.jpg (images) and {number}.png (masks)
    - Patches: {original_number}_{scale}_{patch_number}.jpg/png
    Example: 1_0.01_0.jpg for first patch of image 1 at 0.01 scale

    Args:
      input_images_dir (str): Path to directory containing input images (*.jpg)
      input_masks_dir (str): Path to directory containing input masks (*.png)
      output_base_dir (str): Path where processed dataset will be created
      train_split (float): Fraction of data to use for training (0.0-1.0)
      patch_size (int): Size of patches to generate (width=height)
      overlap_percent (float): Overlap between patches (0.0-1.0)
      scale_factors (tuple): Scale factors for generating patches (e.g., (0.01, 0.2, 0.4))

    Note:
      - Existing content in output_base_dir will be deleted
      - Progress bars show processing status for each operation
      - Images and masks must have matching numeric prefixes (e.g., 1.jpg and 1.png)
    """

    # Create directory structure
    create_directory_structure(output_base_dir)

    # Get sorted file lists
    image_files, mask_files = get_dataset_files(input_images_dir, input_masks_dir)

    # Split dataset
    train_indices, val_indices = split_dataset(image_files, train_split)

    # Process training and validation sets
    for split_name, indices in [('train', train_indices), ('val', val_indices)]:
        print(f"\nProcessing {split_name} set...")

        # Prepare arguments for copying original files
        copy_args = [
            (idx, image_files[idx], mask_files[idx], input_images_dir, input_masks_dir,
             output_base_dir, split_name)
            for idx in indices
        ]

        # Copy original files in parallel
        with Pool() as pool:
            list(tqdm(
                pool.imap(copy_original_files, copy_args),
                total=len(indices),
                desc=f"Copying original {split_name} files"
            ))

        # Generate and save patches for each scale factor
        for scale_factor in scale_factors:
            scale_dir = f"{scale_factor:.2f}"

            # Create scale-specific directories
            os.makedirs(os.path.join(output_base_dir, split_name, 'images/patches', scale_dir), exist_ok=True)
            os.makedirs(os.path.join(output_base_dir, split_name, 'masks/patches', scale_dir), exist_ok=True)

            print(f"\nGenerating patches for scale factor {scale_factor}")

            # Prepare arguments for parallel processing
            process_args = [
                (idx, image_files[idx], mask_files[idx], input_images_dir, input_masks_dir,
                 output_base_dir, split_name, patch_size, overlap_percent, scale_factor)
                for idx in indices
            ]

            # Process images in parallel
            with Pool() as pool:
                list(tqdm(
                    pool.imap(process_single_image, process_args),
                    total=len(indices),
                    desc=f"Processing {split_name} patches"
                ))


if __name__ == "__main__":
    # Configuration
    train_split = 0.8
    input_images_dir = "output/vitiligo_sim/images"
    input_masks_dir = "output/combined_masks"
    output_base_dir = "output/processed_dataset"
    patch_size = 512
    overlap_percent = 0.2
    scale_factors = (0.01, 0.2, 0.4)

    # Process the dataset
    process_dataset(
        input_images_dir=input_images_dir,
        input_masks_dir=input_masks_dir,
        output_base_dir=output_base_dir,
        train_split=train_split,
        patch_size=patch_size,
        overlap_percent=overlap_percent,
        scale_factors=scale_factors
    )
