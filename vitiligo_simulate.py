import os
import cv2
import numpy as np

from vitiligo_sim_main import simulate_vitiligo

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

# Read images and masks
input_images_dir = 'data/images'
input_masks_dir = 'data/masks'


def read_and_sort_images_masks(images_dir, masks_dir):
    """Read and sort all images and masks by their index."""
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')],
                         key=lambda x: int(x.split('.')[0]))
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')],
                        key=lambda x: int(x.split('.')[0]))

    image_paths = [os.path.join(images_dir, f) for f in image_files]
    mask_paths = [os.path.join(masks_dir, f) for f in mask_files]

    return list(zip(image_paths, mask_paths))


def save_progression_images(results, output_base_dir, image_idx, sim_idx, original_dims):
    """Save the extracted progression images at specified indices with original dimensions."""
    depigmentation_indices = [19, 39, 59, 79, 99]
    # depigmentation_indices = [25, 50, 75]
    # repigmentation_indices = [19, 39, 59, 79, 99]
    repigmentation_indices = [20, 30]

    # TODO: Full results object:
    #  {
    #    'depigmentation_masks': progression_simulator.depigmentation_masks,
    #    'repigmentation_masks': progression_simulator.repigmentation_masks,
    #    'depigmentation_images': blend_simulator.depigmentation_images,
    #    'repigmentation_images': blend_simulator.repigmentation_images
    #   }

    # TODO: Save masks as well. Consider saving masks in a separate dir, like /masks

    # Save depigmentation images
    for idx in depigmentation_indices:
        # Resize to original dimensions
        resized_image = cv2.resize(results['depigmentation_images'][idx],
                                   (original_dims[1], original_dims[0]),
                                   interpolation=cv2.INTER_LINEAR)
        # output_path = os.path.join(depig_dir, f'sim_{sim_idx}_idx_{idx}.jpg')
        output_path = os.path.join(output_base_dir, f'{image_idx}_depig_{sim_idx}_idx_{idx}.jpg')
        cv2.imwrite(output_path, resized_image)

    # Save repigmentation images
    for idx in repigmentation_indices:
        # Resize to original dimensions
        resized_image = cv2.resize(results['repigmentation_images'][idx],
                                   (original_dims[1], original_dims[0]),
                                   interpolation=cv2.INTER_LINEAR)
        # output_path = os.path.join(repig_dir, f'sim_{sim_idx}_idx_{idx}.jpg')
        output_path = os.path.join(output_base_dir, f'{image_idx}_repig_{sim_idx}_idx_{idx}.jpg')
        cv2.imwrite(output_path, resized_image)


if __name__ == "__main__":
    # Read all available images and masks
    image_mask_pairs = read_and_sort_images_masks(input_images_dir, input_masks_dir)

    print('Number of image/mask pairs:', len(image_mask_pairs))

    # Start index
    start_idx = 0

    # Output base directory
    output_base_dir = 'output/vitiligo_sim'
    os.makedirs(output_base_dir, exist_ok=True)

    for idx, (image_path, mask_path) in enumerate(image_mask_pairs[start_idx:], start=start_idx):
        print(f'Processing image {idx + 1}/{len(image_mask_pairs)}')
        # Read original image to get dimensions
        original_image = cv2.imread(image_path)
        original_dims = original_image.shape[:2]  # (height, width)

        # Perform 5 simulations for each pair
        for sim_idx in range(3):
            print(f'Processing simulation {sim_idx + 1}/3')
            # Use a different, but reproducible seed for each pair and simulation instance
            progression_seed = 10 + idx + sim_idx

            results = simulate_vitiligo(
                # Input image parameters
                input_image_path=image_path,
                skin_mask_path=mask_path,
                follicle_mask_path='data/follicle_masks/forearm_mask.png',
                dimensions=(1000, 1000),  # Keep square processing dimension
                # dimensions=(int(original_dims[1] * blend_scale_factor), int(original_dims[0] * blend_scale_factor)),
                original_dimensions=original_dims,

                # Progression simulation parameters
                depigment_steps=100,
                depigment_severity=10,
                repigment_steps=50,
                repigment_severity=5,

                progression_seed=progression_seed,

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
                # s_reduction_max=0.35,
                s_reduction_max=0.4,
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

            # Save the progression images with original dimensions
            save_progression_images(results, output_base_dir, idx, sim_idx, original_dims)
