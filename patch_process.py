import os
from math import ceil
import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2

"""
Patch Extraction and Reconstruction
==============================================

This provides efficient tools for extracting overlapping patches from images
and reconstructing images from these patches.

Key Features
-----------
- Memory-efficient patch extraction using numpy stride tricks
- Automatic image scaling with aspect ratio preservation
- Configurable patch overlap
- Special handling for mask images
- Proper reconstruction with overlap averaging
- Support for both grayscale and multi-channel images

Basic Usage
----------
# Read an image
image = cv2.imread('input.png')

# Extract patches with 20% overlap, scaled to 50% size
patches, original_shape, effective_scale = extract_patches(
    image=image,
    patch_size=256,        # Size of each square patch
    overlap_percent=0.2,   # 20% overlap between patches
    scale_factor=0.5,      # Scale image to 50% before extraction
    is_mask=False          # Set to True for segmentation masks
)

# Process patches independently...

# Reconstruct the image from processed patches
reconstructed = reconstruct_image(
    patches=patches,
    original_shape=original_shape,
    patch_size=256,
    overlap_percent=0.2
)

# Save individual patches if needed
for i, patch in enumerate(patches):
    cv2.imwrite(f'patch_{i}.png', patch)

Detailed Behavior
---------------
1. Scaling Behavior:
   - Maintains aspect ratio during scaling
   - Automatically adjusts scale if resulting image would be too small
   - Uses INTER_AREA interpolation for regular images
   - Uses INTER_NEAREST for mask images to preserve discrete values

2. Patch Extraction:
   - Uses numpy stride tricks for memory efficiency
   - Automatically pads image if needed to fit complete patches
   - Centers padding to maintain spatial relationship
   - Handles both 2D (grayscale) and 3D (multi-channel) images

3. Edge Cases:
   - Small Images: Automatically scales up if smaller than patch_size
   - Large Images: Warns if creating >1000 patches
   - Invalid Parameters: Raises ValueError with descriptive messages
   - Zero-size Images: Raises ValueError
   - Invalid Overlap: Prevents negative stride values

4. Reconstruction:
   - Averages overlapping regions for smooth blending
   - Preserves original image dimensions
   - Handles multi-channel images correctly
   - Uses efficient numpy operations for speed

Performance Considerations
------------------------
1. Memory Efficiency:
   - Uses views instead of copies where possible
   - Efficient stride-based patch extraction
   - Memory usage depends on:
     * Original image size
     * Patch size
     * Overlap percentage
     * Scale factor

2. Speed Optimization:
   - Vectorized operations for patch placement
   - Efficient numpy-based reconstruction
   - Minimal memory copying
   - Uses stride tricks for zero-copy views

3. Scaling Considerations:
   - Lower scale_factor reduces memory usage
   - Higher overlap increases computation time
   - Large patch_size reduces total patch count

Parameter Guidelines
------------------
patch_size:
    - Power of 2 recommended (64, 128, 256, etc.)
    - Consider GPU memory limitations
    - Typical range: 128-512 pixels

overlap_percent:
    - Higher values (0.2-0.5) for smoother reconstruction
    - Lower values (0.1-0.2) for efficiency
    - Must be < 1.0 to prevent zero stride

scale_factor:
    - Use 1.0 for no scaling
    - 0.5 for 50% size reduction
    - Minimum 0.01 (1% of original size)

Error Handling
------------
The implementation includes comprehensive error checking:
- Input validation for all parameters
- Descriptive error messages
- Warnings for potentially problematic configurations
- Graceful handling of edge cases

Future Considerations
-------------------
Potential improvements could include:
- Multi-threading support for large images
"""


def extract_patches_strided(image: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    """
    Extract patches from image using numpy stride tricks for memory-efficient patch extraction.
    This method avoids copying data by creating a view into the original array.

    :param image: Input image array of shape (H, W) or (H, W, C)
    :type image: np.ndarray
    :param patch_size: Size of square patches
    :type patch_size: int
    :param stride: Stride between patches
    :type stride: int
    :returns: Array of patches with shape (N, patch_size, patch_size[, C])
    :rtype: np.ndarray
    :raises ValueError: If input parameters are invalid
    """
    # Validate input types and dimensions
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")

    image_dims = len(image.shape)
    if image_dims not in [2, 3]:
        raise ValueError("Image must be 2D or 3D array")

    if patch_size <= 0 or stride <= 0:
        raise ValueError("Patch size and stride must be positive")

    min_image_dim = min(image.shape[:2])
    if patch_size > min_image_dim:
        raise ValueError("Patch size cannot be larger than image dimensions")

    # Calculate number of complete patches that fit in each dimension
    patches_height = (image.shape[0] - patch_size) // stride + 1
    patches_width = (image.shape[1] - patch_size) // stride + 1

    if patches_height <= 0 or patches_width <= 0:
        raise ValueError("Invalid patch_size/stride combination for image size")

    # Configure shape and strides for efficient patch extraction
    if image_dims == 3:
        # For RGB/multi-channel images
        output_shape = (patches_height, patches_width, patch_size, patch_size, image.shape[2])
        stride_config = (
            stride * image.strides[0],  # Vertical stride between patches
            stride * image.strides[1],  # Horizontal stride between patches
            *image.strides  # Original strides for within-patch access
        )
    else:
        # For grayscale/single-channel images
        output_shape = (patches_height, patches_width, patch_size, patch_size)
        stride_config = (
            stride * image.strides[0],  # Vertical stride between patches
            stride * image.strides[1],  # Horizontal stride between patches
            *image.strides  # Original strides for within-patch access
        )

    # Create view into the array using stride tricks
    patches = as_strided(image, shape=output_shape, strides=stride_config, writeable=False)

    # Reshape to consolidate all patches into a single dimension
    final_shape = (-1, patch_size, patch_size, *((image.shape[2],) if image_dims == 3 else ()))
    return patches.reshape(final_shape)


def extract_patches(image, patch_size, overlap_percent, scale_factor, is_mask=False):
    """
    Extract overlapping patches from an image with automatic scaling and padding.
    Maintains aspect ratio while ensuring patches are of the specified size.

    :param image: Input image array
    :type image: np.ndarray
    :param patch_size: Size of square patches
    :type patch_size: int
    :param overlap_percent: Overlap between patches (0 to 1)
    :type overlap_percent: float
    :param scale_factor: Desired scale factor (0.01 to 1)
    :type scale_factor: float
    :param is_mask: Whether the input is a mask image
    :type is_mask: bool
    :returns: Tuple of (patches, padded_shape, effective_scale)
    :rtype: tuple
    :raises ValueError: If input parameters are invalid
    """
    # Input validation
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty")

    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("Patch size must be a positive integer")

    if not 0 <= overlap_percent < 1:
        raise ValueError("Overlap must be between 0 and 1")

    if not 0.01 <= scale_factor <= 1:
        raise ValueError("Scale factor must be between 0.01 and 1")

    # Calculate stride based on overlap
    image_height, image_width = image.shape[:2]
    patch_stride = int(patch_size * (1 - overlap_percent))
    if patch_stride <= 0:
        raise ValueError("Overlap too large, resulting in zero or negative stride")

    # Determine appropriate scaling factor
    smallest_dimension = min(image_height, image_width)
    largest_dimension = max(image_height, image_width)
    scaled_min_dim = smallest_dimension * scale_factor

    # Adjust scaling to ensure patches are valid
    if largest_dimension < patch_size:
        # Scale up if image is smaller than patch size
        effective_scale = patch_size / largest_dimension
    elif scaled_min_dim < patch_size:
        # Ensure scaling doesn't make image too small
        effective_scale = patch_size / largest_dimension
        print(f"Warning: Using minimum safe scale {effective_scale:.3f} instead of requested {scale_factor}")
    else:
        effective_scale = scale_factor

    # Apply scaling if necessary
    if effective_scale != 1.0:
        new_width = int(image_width * effective_scale)
        new_height = int(image_height * effective_scale)
        interpolation_method = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        try:
            scaled_image = cv2.resize(image, (new_width, new_height),
                                      interpolation=interpolation_method)
        except Exception as e:
            raise ValueError(f"Scaling failed: {str(e)}")
    else:
        scaled_image = image

    # Calculate required padding dimensions
    scaled_height, scaled_width = scaled_image.shape[:2]
    patches_across = max(1, ceil((scaled_width - patch_size) / patch_stride) + 1)
    patches_down = max(1, ceil((scaled_height - patch_size) / patch_stride) + 1)

    total_patches = patches_across * patches_down
    if total_patches > 1000:
        print(f"Warning: Creating {total_patches} patches")

    # Calculate required dimensions for complete patch coverage
    required_height = (patches_down - 1) * patch_stride + patch_size
    required_width = (patches_across - 1) * patch_stride + patch_size

    # Calculate padding amounts for each dimension
    padding_amounts = [
        (max(0, req - curr) // 2, max(0, req - curr) - max(0, req - curr) // 2)
        for req, curr in zip((required_height, required_width),
                             (scaled_height, scaled_width))
    ]

    # Apply padding to accommodate all patches
    try:
        padding_config = [*padding_amounts, *[(0, 0)] * (len(image.shape) - 2)]
        padded_image = np.pad(scaled_image, padding_config, mode='constant')
    except Exception as e:
        raise ValueError(f"Padding failed: {str(e)}")

    # Extract patches using strided implementation
    patches = extract_patches_strided(padded_image, patch_size, patch_stride)

    if not patches.any():
        raise ValueError("No patches were created")

    return patches, padded_image.shape[:2], effective_scale


def reconstruct_image(patches, original_shape, patch_size, overlap_percent):
    """
    Reconstruct the original image from overlapping patches by averaging overlapped regions.

    :param patches: Array of image patches
    :type patches: np.ndarray
    :param original_shape: Shape of padded image (h, w)
    :type original_shape: tuple
    :param patch_size: Size of square patches
    :type patch_size: int
    :param overlap_percent: Overlap between patches (0 to 1)
    :type overlap_percent: float
    :returns: Reconstructed image
    :rtype: np.ndarray
    """
    # Calculate stride between patches
    patch_stride = int(patch_size * (1 - overlap_percent))
    padded_height, padded_width = original_shape

    # Determine output shape based on whether image is multi-channel
    has_channels = len(patches.shape) == 4
    num_channels = patches.shape[-1] if has_channels else None
    output_shape = (*original_shape, num_channels) if has_channels else original_shape

    # Initialize accumulation arrays
    reconstructed = np.zeros(output_shape)  # Accumulates patch values
    overlap_counter = np.zeros_like(reconstructed)  # Tracks overlap count

    # Generate grid of patch positions
    position_grid = np.mgrid[
                    0:padded_height - patch_size + 1:patch_stride,
                    0:padded_width - patch_size + 1:patch_stride
                    ].reshape(2, -1).T

    # Place patches back into image
    for patch_idx, (y_pos, x_pos) in enumerate(position_grid):
        if patch_idx >= len(patches):
            break
        patch_region = slice(y_pos, y_pos + patch_size), slice(x_pos, x_pos + patch_size)
        reconstructed[patch_region] += patches[patch_idx]
        overlap_counter[patch_region] += 1

    # Average overlapping regions
    return np.divide(reconstructed, overlap_counter, where=overlap_counter != 0)


if __name__ == '__main__':
    patch_size = 512
    overlap_percent = 0.2
    scale_factor = 0.5

    output_dir = 'data/patches'
    image_idx = 0
    image = cv2.imread(f'data/masks/{image_idx}.png')

    patches, shape, effective_scale = extract_patches(
        image=image,
        patch_size=patch_size,
        overlap_percent=overlap_percent,
        scale_factor=scale_factor,
    )
    print(f"Effective scale used: {effective_scale}")
    print(f"Number of patches generated: {len(patches)}")
    print('Padded shape:', shape)

    os.makedirs(output_dir, exist_ok=True)
    # Save patches
    for patch_idx, patch in enumerate(patches):
        cv2.imwrite(f'{output_dir}/0_{scale_factor}_{patch_idx}.png', patch)

    reconstructed = reconstruct_image(
        patches=patches,
        original_shape=shape,
        patch_size=patch_size,
        overlap_percent=overlap_percent
    )
    cv2.imwrite(f'{output_dir}/{scale_factor}_{image_idx}_reconstructed.png', reconstructed)
