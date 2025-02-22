import os
import numpy as np
import cv2

from vitiligo_progression_simulator import VitiligoProgressionSimulator
from vitiligo_skin_blend import VitiligoBlendSimulator


def center_crop(mask, target_width, target_height):
    """
    Crop the mask from the center while maintaining aspect ratio.
    Returns the largest possible portion that fits the target dimensions.
    """
    # Calculate aspect ratios
    mask_aspect = mask.shape[1] / mask.shape[0]
    target_aspect = target_width / target_height

    if mask_aspect > target_aspect:
        # Mask is wider - crop width
        new_width = int(mask.shape[0] * target_aspect)
        start_x = (mask.shape[1] - new_width) // 2
        cropped = mask[:, start_x:start_x + new_width]
    else:
        # Mask is taller - crop height
        new_height = int(mask.shape[1] / target_aspect)
        start_y = (mask.shape[0] - new_height) // 2
        cropped = mask[start_y:start_y + new_height, :]

    # Resize to target dimensions
    return cv2.resize(
        cropped,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST
    )


def process_masks(
        follicle_mask_path,
        dimensions=(1024, 1024),
        depigment_steps=50,
        depigment_severity=10,
        repigment_steps=50,
        repigment_severity=5,
        progression_seed=42,
        depigmentation_params=None,
        repigmentation_params=None,
        confetti_params=None,
        koebner_params=None,
        hypochromic_params=None,
        anisotropy_params=None,
        edge_params=None):
    """Process only the masks without blending"""
    # Load follicle mask
    follicle_mask = cv2.imread(follicle_mask_path, cv2.IMREAD_GRAYSCALE)
    if follicle_mask is None:
        raise FileNotFoundError(f"Follicle mask image not found at path: {follicle_mask_path}")

    follicle_mask = cv2.resize(follicle_mask, dimensions, interpolation=cv2.INTER_NEAREST)

    # Create progression simulator instance
    progression_simulator = VitiligoProgressionSimulator(
        follicle_mask=follicle_mask,
        dimensions=dimensions,
        depigment_steps=depigment_steps,
        depigment_severity=depigment_severity,
        repigment_steps=repigment_steps,
        repigment_severity=repigment_severity,
        seed=progression_seed,
        depigmentation_params=depigmentation_params,
        repigmentation_params=repigmentation_params,
        confetti_params=confetti_params,
        koebner_params=koebner_params,
        hypochromic_params=hypochromic_params,
        anisotropy_params=anisotropy_params,
        edge_params=edge_params
    )

    progression_simulator.simulate_depigmentation()
    progression_simulator.simulate_repigmentation()

    return progression_simulator.depigmentation_masks, progression_simulator.repigmentation_masks


def blend_only(masks_dict, **blend_params):
    """Run only the blending on existing masks"""
    blend_simulator = VitiligoBlendSimulator(
        depigmentation_masks=masks_dict['depigmentation_masks'],
        repigmentation_masks=masks_dict['repigmentation_masks'],
        **blend_params
    )

    blend_simulator.depigmentation()
    blend_simulator.repigmentation()

    return blend_simulator.depigmentation_images, blend_simulator.repigmentation_images


def simulate_vitiligo(
        # Input image parameters
        input_image_path,
        skin_mask_path,
        follicle_mask_path,
        dimensions=(1024, 1024),
        original_dimensions=(1024, 1024),

        # Progression simulation parameters
        depigment_steps=50,
        depigment_severity=10,
        repigment_steps=50,
        repigment_severity=5,
        progression_seed=42,

        # Output directories
        output_dir_base=None,

        # Simulation parameters
        depigmentation_params=None,
        repigmentation_params=None,
        confetti_params=None,
        koebner_params=None,
        hypochromic_params=None,
        anisotropy_params=None,
        edge_params=None,

        # Blend parameters
        blur_size_vitiligo=(9, 9),
        s_reduction_min=0.2,
        s_reduction_max=0.35,
        v_increase_min=0.5,
        v_increase_max=0.85,
        h_shift_min=0,
        h_shift_max=-1.5,
        pink_alpha_ranges=[0.015, 0.04],
        white_alpha_ranges=[0.01, 0.05],
        pink_color=np.array([255, 192, 203], dtype=np.uint8),
        white_color=np.array([255, 255, 255], dtype=np.uint8),
        decay_factor=0.95,
        probability_threshold=0.6,
        blur_size_darkening=(5, 5),
        dark_factor=0.35,
        save_outputs=False,
):
    """
    Unified function to run both vitiligo progression and blend simulation.
    """
    # Create output directories
    if output_dir_base:
        depigmentation_output_dir = os.path.join(output_dir_base, 'depigmentation')
        repigmentation_output_dir = os.path.join(output_dir_base, 'repigmentation')
        os.makedirs(depigmentation_output_dir, exist_ok=True)
        os.makedirs(repigmentation_output_dir, exist_ok=True)
    else:
        if save_outputs:
            raise ValueError("Output directory must be provided to save outputs.")

        depigmentation_output_dir = None
        repigmentation_output_dir = None

    # Load follicle mask
    follicle_mask = cv2.imread(follicle_mask_path, cv2.IMREAD_GRAYSCALE)
    if follicle_mask is None:
        raise FileNotFoundError(f"Follicle mask image not found at path: {follicle_mask_path}")

    follicle_mask = cv2.resize(follicle_mask, dimensions, interpolation=cv2.INTER_NEAREST)

    # Create progression simulator instance
    progression_simulator = VitiligoProgressionSimulator(
        follicle_mask=follicle_mask,
        dimensions=dimensions,
        depigment_steps=depigment_steps,
        depigment_severity=depigment_severity,
        repigment_steps=repigment_steps,
        repigment_severity=repigment_severity,
        seed=progression_seed,
        depigmentation_params=depigmentation_params,
        repigmentation_params=repigmentation_params,
        confetti_params=confetti_params,
        koebner_params=koebner_params,
        hypochromic_params=hypochromic_params,
        anisotropy_params=anisotropy_params,
        edge_params=edge_params
    )

    # Run progression simulation to generate masks
    progression_simulator.simulate_depigmentation(save_masks=save_outputs)
    progression_simulator.simulate_repigmentation(save_masks=save_outputs)

    # Crop depigmentation masks
    cropped_depigmentation_masks = [
        center_crop(mask, original_dimensions[1], original_dimensions[0])
        for mask in progression_simulator.depigmentation_masks
    ]

    # Crop repigmentation masks
    cropped_repigmentation_masks = [
        center_crop(mask, original_dimensions[1], original_dimensions[0])
        for mask in progression_simulator.repigmentation_masks
    ]

    # Create blend simulator instance
    blend_simulator = VitiligoBlendSimulator(
        depigmentation_masks=cropped_depigmentation_masks,
        repigmentation_masks=cropped_repigmentation_masks,
        depigmentation_output_dir=depigmentation_output_dir,
        repigmentation_output_dir=repigmentation_output_dir,
        dimensions=dimensions,
        input_image_path=input_image_path,
        skin_mask_path=skin_mask_path,
        blur_size_vitiligo=blur_size_vitiligo,
        s_reduction_min=s_reduction_min,
        s_reduction_max=s_reduction_max,
        v_increase_min=v_increase_min,
        v_increase_max=v_increase_max,
        h_shift_min=h_shift_min,
        h_shift_max=h_shift_max,
        pink_alpha_ranges=pink_alpha_ranges,
        white_alpha_ranges=white_alpha_ranges,
        pink_color=pink_color,
        white_color=white_color,
        decay_factor=decay_factor,
        probability_threshold=probability_threshold,
        blur_size_darkening=blur_size_darkening,
        dark_factor=dark_factor
    )

    # Run blend simulation
    blend_simulator.depigmentation(save_images=save_outputs)
    blend_simulator.repigmentation(save_images=save_outputs)

    # Outputs
    return {
        'depigmentation_masks': progression_simulator.depigmentation_masks,
        'repigmentation_masks': progression_simulator.repigmentation_masks,
        'depigmentation_images': blend_simulator.depigmentation_images,
        'repigmentation_images': blend_simulator.repigmentation_images
    }


# Example usage:
if __name__ == "__main__":
    depigmentation_params = {
        'max_seeds': 50,  # Maximum number of seed points
        'min_seeds': 20,  # Minimum number of seed points
        'seed_radius': 1,  # Radius of initial seed points
        'growth_factor': 1.0,  # Growth rate multiplier
    }
    repigmentation_params = {
        'distance_decay': 2.5,  # Controls how quickly probability decreases with distance to hair follicles
        'repigment_rate': 2,  # Base rate of repigmentation
    }

    confetti_params = {
        'probability': 0.5,  # Probability of confetti depigmentation occurring each step
        'num_range': (5, 30),  # Range of number of confetti spots
        'max_distance': 50,  # Maximum distance from depigmented areas to place confetti spots
        'radius_range': (1, 3),  # Range of confetti spot sizes
    }
    koebner_params = {
        'probability': 0,  # Probability of Koebner phenomenon occurring each step
        'num_streaks_range': (1, 3),  # Range of number of streaks
        'thickness_range': (1, 3),  # Range of streak thicknesses
        'length_range': (20, 100),  # Range of streak lengths
        'num_points_range': (5, 15),  # Range of points to define the curved path
    }
    hypochromic_params = {
        'probability': 0.3,  # Probability of hypochromic areas occurring each step
        'num_areas_range': (1, 5),  # Number of hypochromic areas to add
        'size_range': (10, 50),  # Size range for hypochromic areas
        'scale': 100.0,
        'octaves': 3,
        'persistence': 0.5,
        'lacunarity': 2.0,
        'threshold': 0.5,
        'salt_pepper_prob': 0.75,
    }

    anisotropy_params = {
        'connection_distance': 200,  # Distance within which blobs tend to grow towards each other
        'connection_multiplier': 1,  # Multiplier to increase growth probability towards other blobs
        'connection_seed_radius': 1,  # Radius of new seed points between connected blobs
        'min_patch_size': 100,  # Minimum number of pixels for a patch
        'x_range_shift': (-5, 5),  # Maximum shift perpendicular to the vector
        'y_range_shift': (-5, 5),  # Maximum shift in the direction of the vector
        'seed_proximity': 0.05,  # Fraction of the distance to place seeds near the boundary of patch
        'num_seeds_range': (0, 5),  # Range of number of seeds to place between connected patches
    }

    edge_params = {
        'edge_noise': 0.5,  # Amount of edge noise (0.0 for no noise)
    }

    simulate_vitiligo(
        # Input image parameters
        input_image_path='data/images/input_1.jpg',
        skin_mask_path='data/images/skin_mask_1.png',
        follicle_mask_path='data/follicle_masks/forearm_mask.png',
        dimensions=(1024, 1024),

        # Progression simulation parameters
        depigment_steps=50,
        depigment_severity=10,
        repigment_steps=50,
        repigment_severity=5,
        progression_seed=42,

        # Output directories
        output_dir_base=None,

        # Simulation parameters
        depigmentation_params=depigmentation_params,
        repigmentation_params=repigmentation_params,
        confetti_params=confetti_params,
        koebner_params=koebner_params,
        hypochromic_params=hypochromic_params,
        anisotropy_params=anisotropy_params,
        edge_params=edge_params,

        # Blend parameters
        blur_size_vitiligo=(9, 9),
        s_reduction_min=0.2,
        s_reduction_max=0.35,
        v_increase_min=0.5,
        v_increase_max=0.85,
        h_shift_min=0,
        h_shift_max=-1.5,
        pink_alpha_ranges=[0.015, 0.04],
        white_alpha_ranges=[0.01, 0.05],
        pink_color=np.array([255, 192, 203], dtype=np.uint8),
        white_color=np.array([255, 255, 255], dtype=np.uint8),
        decay_factor=0.95,
        probability_threshold=0.6,
        blur_size_darkening=(5, 5),
        dark_factor=0.35
    )
