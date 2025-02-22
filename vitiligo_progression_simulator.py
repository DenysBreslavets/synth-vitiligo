import numpy as np
import cv2
import random
import os
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from noise import pnoise2

OUTPUT_DIR_DEPIG = 'data/output_masks/vitiligo_sim/depigmentation'
OUTPUT_DIR_REPIG = 'data/output_masks/vitiligo_sim/repigmentation'


class VitiligoProgressionSimulator:
    """
    Simulates the depigmentation and repigmentation processes of vitiligo, incorporating hair follicle distribution
    and specific patterns observed in the disease progression.

    The simulator uses a follicle mask to influence the probability of depigmentation and repigmentation in different
    regions, mimicking the slower depigmentation and faster repigmentation around hair follicles due to higher
    concentrations of melanocyte stem cells.

    :param follicle_mask: Grayscale image where hair follicles are white (255), and inter-hair regions are black (0).
    :type follicle_mask: numpy.ndarray
    :param dimensions: Dimensions of the simulation area (height, width).
    :type dimensions: tuple
    :param depigment_steps: Number of steps in the depigmentation simulation.
    :type depigment_steps: int
    :param depigment_severity: Severity level of depigmentation (0-10).
    :type depigment_severity: float
    :param repigment_steps: Number of steps in the repigmentation simulation.
    :type repigment_steps: int
    :param repigment_severity: Severity level of repigmentation (0-10).
    :type repigment_severity: float
    :param seed: Seed for random number generators to ensure reproducibility.
    :type seed: int
    :param depigmentation_params: Dictionary of depigmentation parameters.
    :type depigmentation_params: dict
    :param repigmentation_params: Dictionary of repigmentation parameters.
    :type repigmentation_params: dict
    :param confetti_params: Dictionary of confetti depigmentation parameters.
    :type confetti_params: dict
    :param koebner_params: Dictionary of Koebner phenomenon parameters.
    :type koebner_params: dict
    :param hypochromic_params: Dictionary of hypochromic area parameters.
    :type hypochromic_params: dict
    :param edge_params: Dictionary of edge processing parameters.
    :type edge_params: dict
    """

    def __init__(
            self,
            follicle_mask,
            dimensions=(1024, 1024),
            depigment_steps=10,
            depigment_severity=5,
            repigment_steps=10,
            repigment_severity=1,
            seed: int = 42,
            depigmentation_params: dict = None,
            repigmentation_params: dict = None,
            confetti_params: dict = None,
            koebner_params: dict = None,
            hypochromic_params: dict = None,
            anisotropy_params: dict = None,
            edge_params: dict = None,
    ):
        self.follicle_mask = follicle_mask
        self.dimensions = dimensions
        self.depigment_steps = depigment_steps
        self.depigment_severity: float = depigment_severity
        self.repigment_steps = repigment_steps
        self.repigment_severity: float = repigment_severity
        self.seed = seed

        self.depigmentation_masks = []
        self.repigmentation_masks = []

        # Initialize random seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Process the follicle mask to create a probability mask
        # Inter-hair regions have higher probability (1), hair follicles lower (0)
        # Normalize the follicle mask to [0, 1]
        self.follicle_mask = cv2.normalize(self.follicle_mask, None, 0, 1, cv2.NORM_MINMAX)
        # Invert the follicle mask to get probability mask
        inverted_follicle_mask = (1 - self.follicle_mask).astype(np.uint8)
        # Apply distance transform to binary inverted_follicle_mask mask
        probability_mask_dist = cv2.distanceTransform(inverted_follicle_mask, cv2.DIST_L2, 5)
        # Normalize the distance transformed mask to [0, 1] range
        # Start at 0.05, to make it possible for hair follicles to depigment
        self.probability_mask = cv2.normalize(probability_mask_dist, None, 0.05, 1, cv2.NORM_MINMAX)

        # Generate a per-pixel noise map for organic growth
        x = np.linspace(0, 5, self.dimensions[1])
        y = np.linspace(0, 5, self.dimensions[0])
        x_grid, y_grid = np.meshgrid(x, y)
        vectorized_pnoise2 = np.vectorize(
            lambda x, y: pnoise2(x, y, octaves=6, repeatx=1024, repeaty=1024, base=0)
        )
        self.noise_map = vectorized_pnoise2(x_grid, y_grid)
        # Normalize the noise map to [0,1]
        self.noise_map = (self.noise_map - self.noise_map.min()) / (self.noise_map.max() - self.noise_map.min())

        # Define default depigmentation parameters
        default_depigmentation_params = {
            'max_seeds': 100,  # Maximum number of seed points
            'min_seeds': 20,  # Minimum number of seed points
            'seed_radius': 3,  # Radius of initial seed points
            'growth_factor': 1.0,  # Growth rate multiplier
        }

        # Define default repigmentation parameters
        default_repigmentation_params = {
            'distance_decay': 5.0,  # Controls how quickly probability decreases with distance to hair follicles
            'repigment_rate': 1.0,  # Base rate of repigmentation
        }

        # Define default confetti parameters
        default_confetti_params = {
            'probability': 0.3,  # Probability of confetti depigmentation occurring each step
            'num_range': (1, 5),  # Range of number of confetti spots
            'max_distance': 50,  # Maximum distance from depigmented areas to place confetti spots
            'radius_range': (1, 3),  # Range of confetti spot sizes
        }

        # Define default Koebner parameters
        default_koebner_params = {
            'probability': 0.2,  # Probability of Koebner phenomenon occurring each step
            'num_streaks_range': (1, 3),  # Range of number of streaks
            'thickness_range': (1, 3),  # Range of streak thicknesses
            'length_range': (20, 100),  # Range of streak lengths
            'num_points_range': (5, 15),  # Range of points to define the curved path
        }

        # Define default hypochromic parameters
        default_hypochromic_params = {
            'probability': 0.3,  # Probability of hypochromic areas occurring each step
            'num_areas_range': (1, 5),  # Number of hypochromic areas to add
            'size_range': (10, 50),  # Size range for hypochromic areas
            'scale': 50.0,  # Scale for noise frequency
            'octaves': 3,  # Octaves for Perlin noise
            'persistence': 0.5,
            'lacunarity': 2.0,
            'threshold': 0.5,  # Threshold for blob density
            'salt_pepper_prob': 0.75,  # Probability of a pixel being removed
        }

        # Define default anisotropy parameters
        default_anisotropy_params = {
            'connection_distance': 15,  # Distance within which blobs tend to grow towards each other
            'connection_multiplier': 1,  # Multiplier to increase growth probability towards other blobs
            'connection_seed_radius': 1,  # Radius of new seed points between connected blobs
            'min_patch_size': 500,  # Minimum number of pixels for a patch
            'x_range_shift': (-50, 50),  # Maximum shift perpendicular to the vector
            'y_range_shift': (-50, 50),  # Maximum shift in the direction of the vector
            'seed_proximity': 0.05,  # Fraction of the distance to place seeds near the boundary of patch
            'num_seeds_range': (5, 15),  # Range of number of seeds to place between connected patches
        }

        # Define default edge parameters
        default_edge_params = {
            'edge_noise': 0.0,  # Amount of edge noise (0.0 for no noise)
        }

        # Update parameters with any provided ones
        self.depigmentation_params = default_depigmentation_params
        if depigmentation_params is not None:
            self.depigmentation_params.update(depigmentation_params)
        self.repigmentation_params = default_repigmentation_params
        if repigmentation_params is not None:
            self.repigmentation_params.update(repigmentation_params)
        self.confetti_params = default_confetti_params
        if confetti_params is not None:
            self.confetti_params.update(confetti_params)
        self.koebner_params = default_koebner_params
        if koebner_params is not None:
            self.koebner_params.update(koebner_params)
        self.hypochromic_params = default_hypochromic_params
        if hypochromic_params is not None:
            self.hypochromic_params.update(hypochromic_params)
        self.anisotropy_params = default_anisotropy_params
        if anisotropy_params is not None:
            self.anisotropy_params.update(anisotropy_params)
        self.edge_params = default_edge_params
        if edge_params is not None:
            self.edge_params.update(edge_params)

    def simulate_depigmentation(self, return_masks=False, save_masks=False):
        """Simulates depigmentation over depigment_steps, incorporating various vitiligo patterns."""
        # Step 1: Find inter-hair regions (probability_mask >= 0.9)
        inter_hair_coords = np.column_stack(np.where(self.probability_mask >= 0.9))

        # Determine number of seed points based on depigment_severity
        max_seeds = self.depigmentation_params['max_seeds']
        min_seeds = self.depigmentation_params['min_seeds']
        N_seeds = int(self.depigment_severity / 10 * (max_seeds - min_seeds) + min_seeds)
        N_seeds = max(min_seeds, min(N_seeds, max_seeds))

        # Adjust N_seeds if available inter-hair regions are fewer
        available_seeds = len(inter_hair_coords)
        if available_seeds < N_seeds:
            print(
                f"Adjusting number of depigmentation seeds from {N_seeds} to {available_seeds} due to limited inter-hair regions.")
            N_seeds = available_seeds

        if N_seeds == 0:
            print("No available inter-hair regions to place depigmentation seeds.")
            return

        # Randomly select N_seeds from inter_hair_coords
        seed_indices = np.random.choice(available_seeds, N_seeds, replace=False)
        seed_points = inter_hair_coords[seed_indices]

        # Initialize the depigmentation mask
        depigment_mask = np.zeros(self.dimensions, dtype=np.uint8)

        # Set seed points in depigment_mask
        seed_radius = self.depigmentation_params['seed_radius']
        for point in seed_points:
            cv2.circle(depigment_mask, (point[1], point[0]), seed_radius, 1, -1)

        total_area = self.dimensions[0] * self.dimensions[1]

        # Simulate growth over depigment_steps
        for step in range(self.depigment_steps):
            time_factor = step / self.depigment_steps

            # Dilate the depigment_mask to get potential growth areas
            dilated_mask = cv2.dilate(depigment_mask, None, iterations=1)
            potential_growth = dilated_mask - depigment_mask

            # Get indices of potential growth pixels
            potential_coords = np.column_stack(np.where(potential_growth == 1))

            if len(potential_coords) == 0:
                print(f"No new depigmented pixels added at step {step + 1}.")
                break

            # Compute probabilities for potential growth pixels
            follicle_probs = self.probability_mask[potential_coords[:, 0], potential_coords[:, 1]]
            noise_probs = self.noise_map[potential_coords[:, 0], potential_coords[:, 1]]
            adjusted_probs = (follicle_probs * (1 - noise_probs) * time_factor *
                              self.depigmentation_params['growth_factor'])

            # ----- Anisotropic Extension (Morphological Bringing) Start -----
            # Identify blobs and compute their contours
            contours, _ = cv2.findContours(depigment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in contours if
                                 cv2.contourArea(cnt) >= self.anisotropy_params['min_patch_size']]
            # print(f"Number of significant blobs: {len(filtered_contours)}")
            # Extract all boundary points from the filtered contours
            # boundary_points = [cnt[:, 0, :] for cnt in filtered_contours]

            # Compute centroids for informational purposes (optional)
            centroids = []
            for cnt in filtered_contours:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    centroids.append((cX, cY))
            centroids = np.array(centroids)
            # print(f"Blob centroids: {centroids}")

            # If multiple significant blobs exist, compute connections
            if len(filtered_contours) >= 2:
                # Create a list of all centroids for KD-tree
                centroids_tree = cKDTree(centroids)

                # Find all unique pairs within connection_distance
                connection_distance = self.anisotropy_params['connection_distance']
                pairs = centroids_tree.query_pairs(r=connection_distance)
                connected_pairs = list(pairs)

                # print('Connected Pairs:', len(connected_pairs))

                for (i, j) in connected_pairs:
                    contour_i = filtered_contours[i]
                    contour_j = filtered_contours[j]

                    # Extract perimeter points
                    points_i = contour_i[:, 0, :]  # Shape (N_i, 2)
                    points_j = contour_j[:, 0, :]  # Shape (N_j, 2)

                    # Build KDTree for contour_j points to find the closest point to each point in contour_i
                    tree_j = cKDTree(points_j)
                    distances, indices = tree_j.query(points_i, k=1)
                    min_idx = np.argmin(distances)
                    closest_point_i = tuple(points_i[min_idx])
                    closest_point_j = tuple(points_j[indices[min_idx]])

                    # print(f"Blob {i} closest to Blob {j}: {closest_point_i} -> {closest_point_j}")

                    # Compute direction vector from closest_point_i to closest_point_j
                    direction_vector = np.array(closest_point_j) - np.array(closest_point_i)
                    distance_between_points = np.linalg.norm(direction_vector)

                    if distance_between_points == 0:
                        # print(f"Closest points for blobs {i} and {j} are identical. Skipping seed placement.")
                        continue  # Avoid division by zero

                    direction_unit_vector = direction_vector / distance_between_points  # Normalize

                    # Define the number of new seeds and the range for placement
                    num_new_seeds = int(random.randint(self.anisotropy_params['num_seeds_range'][0],
                                                       self.anisotropy_params['num_seeds_range'][1]) *
                                        self.anisotropy_params['connection_multiplier'])
                    connection_seed_radius = self.anisotropy_params['connection_seed_radius']
                    for _ in range(num_new_seeds):
                        # Place new seed at a short distance from closest_point_i or closest_point_j with deviation
                        # Define a small fraction of the distance to place seeds near the boundary
                        if random.choice([0, 1]) == 0:
                            fraction_distance = random.uniform(0, self.anisotropy_params['seed_proximity'])
                        else:
                            fraction_distance = random.uniform(1 - self.anisotropy_params['seed_proximity'], 1)

                        seed_distance = fraction_distance * distance_between_points
                        base_seed = np.array(closest_point_i) + direction_unit_vector * seed_distance

                        # Compute a perpendicular unit vector
                        perpendicular_unit_vector = np.array([-direction_unit_vector[1], direction_unit_vector[0]])

                        # Generate random shifts within the specified ranges
                        shift_along = random.uniform(self.anisotropy_params['y_range_shift'][0],
                                                     self.anisotropy_params['y_range_shift'][1])
                        shift_perp = random.uniform(self.anisotropy_params['x_range_shift'][0],
                                                    self.anisotropy_params['x_range_shift'][1])

                        # Calculate the deviation relative to the vector's coordinate system
                        deviation = (direction_unit_vector * shift_along) + (perpendicular_unit_vector * shift_perp)
                        # Apply the deviation to the base seed
                        new_seed = base_seed + deviation
                        new_seed = new_seed.astype(int)

                        # Ensure new seed is within image bounds
                        new_seed[0] = np.clip(new_seed[0], 0, self.dimensions[0] - 1)
                        new_seed[1] = np.clip(new_seed[1], 0, self.dimensions[1] - 1)

                        # Add new seed point to the mask
                        cv2.circle(depigment_mask, (new_seed[0], new_seed[1]),
                                   connection_seed_radius, 1, -1)
            # ----- Anisotropic Extension (Morphological Bringing) End -----

            # Ensure probabilities are between 0 and 1
            adjusted_probs = np.clip(adjusted_probs, 0, 1)

            # Generate random numbers to decide which pixels to depigment
            rand_vals = np.random.rand(len(adjusted_probs))
            pixels_to_add = rand_vals < adjusted_probs

            # Update depigment_mask
            depigment_mask[potential_coords[pixels_to_add, 0], potential_coords[pixels_to_add, 1]] = 1

            # ----- Hypochromic Areas Start -----
            # Implement hypochromic areas as irregular random blobs using noise
            if random.uniform(0, 1) < self.hypochromic_params['probability']:
                num_areas = random.randint(*self.hypochromic_params['num_areas_range'])
                for _ in range(num_areas):
                    center_x = random.randint(0, self.dimensions[1] - 1)
                    center_y = random.randint(0, self.dimensions[0] - 1)
                    area_size = random.randint(*self.hypochromic_params['size_range'])

                    # Define the bounding box for the hypochromic area
                    x_min = max(center_x - area_size, 0)
                    x_max = min(center_x + area_size + 1, self.dimensions[1])
                    y_min = max(center_y - area_size, 0)
                    y_max = min(center_y + area_size + 1, self.dimensions[0])

                    # Create coordinate grid within the bounding box
                    x_range = np.arange(x_min, x_max)
                    y_range = np.arange(y_min, y_max)
                    x_grid, y_grid = np.meshgrid(x_range, y_range)

                    # Calculate distance from the center to create a circular mask
                    distance = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
                    circular_mask = distance <= area_size

                    # Generate Perlin noise within the circular region
                    scale = self.hypochromic_params['scale']
                    octaves = self.hypochromic_params['octaves']
                    persistence = self.hypochromic_params['persistence']
                    lacunarity = self.hypochromic_params['lacunarity']

                    # Normalize coordinates for noise generation
                    x_norm = x_grid / scale
                    y_norm = y_grid / scale

                    # Vectorized Perlin noise generation
                    noise = np.zeros_like(x_grid, dtype=np.float32)
                    for i in range(x_grid.shape[0]):
                        for j in range(x_grid.shape[1]):
                            noise[i, j] = pnoise2(x_norm[i, j], y_norm[i, j],
                                                  octaves=octaves,
                                                  persistence=persistence,
                                                  lacunarity=lacunarity,
                                                  repeatx=1024,
                                                  repeaty=1024,
                                                  base=0)

                    # Normalize the noise to [0, 1]
                    noise_normalized = (noise - noise.min()) / (noise.max() - noise.min())

                    # Threshold the noise to create an irregular mask
                    threshold = self.hypochromic_params['threshold']
                    irregular_mask = (noise_normalized > threshold) & circular_mask

                    # Introduce salt-and-pepper noise (randomly remove some pixels)
                    salt_pepper_prob = self.hypochromic_params['salt_pepper_prob']
                    salt_pepper_mask = np.random.rand(*irregular_mask.shape) > salt_pepper_prob
                    irregular_mask = irregular_mask & salt_pepper_mask

                    # Convert to uint8
                    irregular_mask = irregular_mask.astype(np.uint8)

                    # Apply morphological operations to enhance blob shape
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    irregular_mask = cv2.morphologyEx(irregular_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                    irregular_mask = cv2.morphologyEx(irregular_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

                    # Add the irregular hypochromic blob to the depigment_mask
                    depigment_mask[y_min:y_max, x_min:x_max] = np.maximum(depigment_mask[y_min:y_max, x_min:x_max],
                                                                          irregular_mask)

            # ----- Hypochromic Areas End -----

            # ----- Confetti-Like Depigmentation Start -----
            # Implement confetti-like depigmentation
            if random.uniform(0, 1) < self.confetti_params['probability']:
                num_confetti = random.randint(*self.confetti_params['num_range'])
                for _ in range(num_confetti):
                    depigmented_coords = np.column_stack(np.where(depigment_mask == 1))
                    if len(depigmented_coords) == 0:
                        continue
                    idx = np.random.choice(len(depigmented_coords))
                    base_point = depigmented_coords[idx]
                    max_distance = self.confetti_params['max_distance']
                    angle = np.random.uniform(0, 2 * np.pi)
                    distance = np.random.uniform(10, max_distance)
                    dx = int(distance * np.cos(angle))
                    dy = int(distance * np.sin(angle))
                    confetti_x = base_point[0] + dy
                    confetti_y = base_point[1] + dx
                    # Ensure confetti points are within image boundaries
                    confetti_x = np.clip(confetti_x, 0, self.dimensions[0] - 1)
                    confetti_y = np.clip(confetti_y, 0, self.dimensions[1] - 1)
                    radius = random.randint(*self.confetti_params['radius_range'])
                    cv2.circle(depigment_mask, (confetti_y, confetti_x), radius, 1, -1)
            # ----- Confetti-Like Depigmentation End -----

            # ----- Koebner Phenomenon Start -----
            # TODO: Koebner phenomenon should be limited to a max value. As they are the regions of previously damaged
            #  skin, like a scratch or a scar. A counter can be used to limit how many times it can occur in total.
            #  - Shape needs to be more straight, and not curled, as it is a scratch. Make sure points are in a
            #    relatively straight line.
            # Implement Koebner phenomenon with random curves
            if random.uniform(0, 1) < self.koebner_params['probability']:
                num_streaks = random.randint(*self.koebner_params['num_streaks_range'])
                for _ in range(num_streaks):
                    hair_coords = np.column_stack(np.where(self.probability_mask < 0.1))  # Hair regions
                    available_hair = len(hair_coords)
                    if available_hair == 0:
                        continue

                    streak_seed_idx = np.random.choice(available_hair, 1, replace=False)[0]
                    start_point = hair_coords[streak_seed_idx]

                    # Generate a random curved path
                    num_points = random.randint(*self.koebner_params['num_points_range'])
                    angles = np.cumsum(np.random.uniform(-np.pi / 4, np.pi / 4, num_points))
                    lengths = np.random.uniform(5, 5, num_points)
                    dxs = (lengths * np.cos(angles)).astype(int)
                    dys = (lengths * np.sin(angles)).astype(int)

                    x_points = np.clip(np.cumsum(dxs) + start_point[1], 0, self.dimensions[1] - 1)
                    y_points = np.clip(np.cumsum(dys) + start_point[0], 0, self.dimensions[0] - 1)

                    streak_points = np.column_stack((x_points, y_points)).astype(np.int32)

                    # Draw the streak
                    thickness = random.randint(*self.koebner_params['thickness_range'])
                    for i in range(len(streak_points) - 1):
                        cv2.line(depigment_mask,
                                 (streak_points[i][0], streak_points[i][1]),
                                 (streak_points[i + 1][0], streak_points[i + 1][1]),
                                 1, thickness=thickness)
            # ----- Koebner Phenomenon End -----

            # -----  Edge Processing Start -----
            # Edge Processing (Optional)
            if self.edge_params['edge_noise'] > 0.0:
                depigment_mask_float = depigment_mask.astype(np.float32)
                depigment_mask_blurred = gaussian_filter(depigment_mask_float, sigma=self.edge_params['edge_noise'])
                depigment_mask = (depigment_mask_blurred > 0.5).astype(np.uint8)
            # -----  Edge Processing End -----

            # Calculate and print the current depigmented area
            current_area = np.sum(depigment_mask)
            area_percentage = (current_area / total_area) * 100
            print(f"Depigmentation Step {step + 1}: {area_percentage:.2f}% depigmented.")

            # Save the depigment_mask at this step
            self.depigmentation_masks.append(depigment_mask.copy())
            if save_masks:
                cv2.imwrite(f'{OUTPUT_DIR_DEPIG}/depigmentation_step_{step + 1}.png', depigment_mask * 255)

            # Early termination if depigmentation reaches high coverage
            if area_percentage > (self.depigment_severity / 10) * 100:
                print(f"Desired depigmentation severity achieved at step {step + 1}.")
                break

        if return_masks:
            return self.depigmentation_masks

    def simulate_repigmentation(self, return_masks=False, save_masks=False):
        """Simulates repigmentation over repigment_steps, starting from the final depigmented mask."""
        if len(self.depigmentation_masks) == 0:
            print("Depigmentation must be simulated before repigmentation.")
            return

        depigment_mask = self.depigmentation_masks[-1].copy()
        total_area = self.dimensions[0] * self.dimensions[1]

        # Identify hair follicle regions within depigmented areas
        hair_follicle_mask = (self.probability_mask < 0.1).astype(np.uint8)
        print(hair_follicle_mask.shape, depigment_mask.shape)
        depigmented_hair_follicles = cv2.bitwise_and(hair_follicle_mask, depigment_mask)

        # If no depigmented hair follicle regions are found, consider nearby areas
        if np.sum(depigmented_hair_follicles) == 0:
            print("No depigmented hair follicle regions to start repigmentation.")
            return

        # Initialize the repigment_mask
        repigment_mask = np.zeros(self.dimensions, dtype=np.uint8)

        # Set initial repigmentation seeds at depigmented hair follicles
        repigment_mask[depigmented_hair_follicles == 1] = 1
        depigment_mask[depigmented_hair_follicles == 1] = 0

        # Compute distance from every pixel to the nearest hair follicle
        # Invert hair_follicle_mask to get distance to hair follicles
        inverted_hair_follicle_mask = 1 - hair_follicle_mask
        distance_to_hair = cv2.distanceTransform(inverted_hair_follicle_mask, cv2.DIST_L2, 3)

        # Normalize distance_to_hair to [0, 1]
        max_distance = np.max(distance_to_hair)
        if max_distance == 0:
            distance_to_hair_normalized = np.zeros_like(distance_to_hair)
        else:
            distance_to_hair_normalized = distance_to_hair / max_distance

        # Compute per-pixel repigmentation probability
        # Closer to hair follicles have higher probability
        alpha = self.repigmentation_params.get('distance_decay', 5.0)
        prob_map = np.exp(-alpha * distance_to_hair_normalized)
        # Modulate with noise map
        prob_map = prob_map * self.noise_map

        # Mask to only consider depigmented areas
        prob_map = prob_map * depigment_mask

        # Simulate repigmentation over steps
        for step in range(self.repigment_steps):
            time_factor = (step + 1) / self.repigment_steps

            # Adjust the probability map for current time step
            current_prob_map = prob_map * time_factor * self.repigmentation_params.get('repigment_rate', 1.0)

            # Ensure probabilities are between 0 and 1
            current_prob_map = np.clip(current_prob_map, 0, 1)

            # Generate random numbers to decide which pixels to repigment
            random_vals = np.random.rand(*self.dimensions)
            new_repigmented = ((random_vals < current_prob_map) & (depigment_mask == 1)).astype(np.uint8)

            # Update depigment_mask and repigment_mask
            depigment_mask[new_repigmented == 1] = 0
            repigment_mask[new_repigmented == 1] = 1

            # Edge Processing (Optional)
            if self.edge_params['edge_noise'] > 0.0:
                repigment_mask_float = repigment_mask.astype(np.float32)
                repigment_mask_blurred = gaussian_filter(repigment_mask_float, sigma=self.edge_params['edge_noise'])
                repigment_mask = (repigment_mask_blurred > 0.5).astype(np.uint8)

            # Save the current depigment_mask (remaining depigmented areas)
            self.repigmentation_masks.append(depigment_mask.copy())

            # Calculate and print the current depigmented area
            depigment_area = np.sum(depigment_mask)
            depigment_percentage = (depigment_area / total_area) * 100
            print(f"Repigmentation Step {step + 1}: {depigment_percentage:.2f}% remaining depigmented.")

            # Save the repigmentation mask at this step
            if save_masks:
                cv2.imwrite(f'{OUTPUT_DIR_REPIG}/repigmentation_step_{step + 1}.png', depigment_mask * 255)

            if depigment_area == 0:
                print(f"Full repigmentation achieved at step {step + 1}.")
                break

        if return_masks:
            return self.repigmentation_masks

    # Add any additional methods if needed


# Example Usage
if __name__ == "__main__":
    # Load the hair follicle mask
    follicle_mask_path = 'data/follicle_masks/forearm_mask.png'
    follicle_mask = cv2.imread(follicle_mask_path, cv2.IMREAD_GRAYSCALE)

    # Validate follicle_mask
    if follicle_mask is None:
        raise FileNotFoundError(f"Follicle mask image not found at path: {follicle_mask_path}")

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

    # Create output directories
    os.makedirs(OUTPUT_DIR_DEPIG, exist_ok=True)
    os.makedirs(OUTPUT_DIR_REPIG, exist_ok=True)

    # Create an instance of the simulator
    simulator = VitiligoProgressionSimulator(
        follicle_mask=follicle_mask,
        dimensions=follicle_mask.shape,
        depigment_steps=50,
        depigment_severity=10,
        repigment_steps=50,
        repigment_severity=5,
        seed=42,
        depigmentation_params=depigmentation_params,
        repigmentation_params=repigmentation_params,
        confetti_params=confetti_params,
        koebner_params=koebner_params,
        hypochromic_params=hypochromic_params,
        anisotropy_params=anisotropy_params,
        edge_params=edge_params
    )

    # Run depigmentation simulation
    simulator.simulate_depigmentation(save_masks=True)

    # Run repigmentation simulation
    simulator.simulate_repigmentation(save_masks=True)