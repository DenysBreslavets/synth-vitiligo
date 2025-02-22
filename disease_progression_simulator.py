import os
import cv2
import random
import numpy as np

from noise import pnoise2
from scipy.ndimage import gaussian_filter


class DiseaseProgressionSimulator:
    """
    Simulates the progression of a skin disease over a specified number of iterations with proportional erosion.

    This class generates a series of binary masks representing disease progression based on provided parameters.
    It utilizes Perlin noise for initial mask generation and applies morphological operations to simulate disease
    erosion or minor expansions. Various 'holes' are introduced for more realistic, patchy patterns.

    :param width: Width of the mask.
    :type width: int
    :param height: Height of the mask.
    :type height: int
    :param start_params: Starting parameters for mask generation.
    :type start_params: dict
    :param end_severity_level: Final severity level (0 to 1).
    :type end_severity_level: float
    :param iterations: Number of simulation steps.
    :type iterations: int
    :param deviation: Controls randomness in healing or expansion (0 means consistent healing).
    :type deviation: float
    :param elongated: If True, generates elongated disease patterns. Defaults to False.
    :type elongated: bool, optional
    :param edge_noise: Amount of noise to add to edges (0.0 for no noise). Defaults to 0.0.
    :type edge_noise: float, optional
    :param seed: Seed for random number generators to ensure reproducibility. Defaults to 42.
    :type seed: int, optional
    """

    def __init__(
            self,
            width: int,
            height: int,
            start_params: dict,
            end_severity_level: float,
            iterations: int,
            deviation: float,
            elongated: bool = False,
            edge_noise: float = 0.0,
            seed: int = 42
    ):
        self.width = width
        self.height = height
        self.start_params = start_params
        self.end_severity_level = end_severity_level
        self.iterations = iterations
        self.deviation = deviation
        self.elongated = elongated
        self.edge_noise = edge_noise
        self.seed = seed

        # Initialize random seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Unpack starting parameters with default values
        self.scale = self.start_params.get('scale', 100)
        self.threshold = self.start_params.get('threshold', 0.5)
        self.blur_radius = self.start_params.get('blur_radius', 0)
        self.num_seeds = self.start_params.get('num_seeds', 20)
        self.max_distance = self.start_params.get('max_distance', 50)
        self.noise_scale = self.start_params.get('noise_scale', 10)

        # Pre-generate a Perlin noise function that we can reuse
        self.vectorized_pnoise2 = np.vectorize(
            lambda x, y: pnoise2(x, y, octaves=6, repeatx=1024, repeaty=1024, base=0)
        )

    def _generate_holes_mask(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        """
        Generate an organic 'holes' mask with rough, noisy edges to produce patchy disease areas.

        :param x_grid: Meshgrid of x-coordinates, shape=(height, width).
        :param y_grid: Meshgrid of y-coordinates, shape=(height, width).
        :return: A float mask with 'holes' where intensity < 1 in some areas.
        :rtype: np.ndarray
        """
        holes_mask = np.ones((self.height, self.width), dtype=np.float32)

        # ----- 1) Generate Two Noise Fields at Different Scales -----
        base_noise = self.vectorized_pnoise2(
            x_grid / random.uniform(30, 60),
            y_grid / random.uniform(30, 60)
        )
        base_noise = (base_noise - base_noise.min()) / (base_noise.max() - base_noise.min())

        edge_noise = self.vectorized_pnoise2(
            x_grid / random.uniform(5, 15),
            y_grid / random.uniform(5, 15)
        )
        edge_noise = (edge_noise - edge_noise.min()) / (edge_noise.max() - edge_noise.min())

        # ----- 2) Create a Few Large Organic Holes -----
        num_holes = random.randint(3, 5)
        for _ in range(num_holes):
            center_x = random.randint(0, self.width)
            center_y = random.randint(0, self.height)

            # Random scale factors that control the shape's squishiness
            scale_x = self.width * random.uniform(0.1, 0.3)
            scale_y = self.height * random.uniform(0.1, 0.3)

            dist_field = np.sqrt(
                ((x_grid - center_x + edge_noise * 20) / scale_x) ** 2 +
                ((y_grid - center_y + edge_noise * 20) / scale_y) ** 2
            )

            # Where dist_field is less than some threshold, we cut holes
            boundary_threshold = random.uniform(0.7, 1.3) + edge_noise * random.uniform(0.2, 0.4)

            # We combine both noise layers to decide hole shape
            hole_shape = (dist_field < boundary_threshold) & (
                    (base_noise + edge_noise * 0.5) < random.uniform(0.4, 0.6)
            )

            # Intensity defines how "deep" the hole is: <1 means it erodes the mask
            # We'll keep the random factor for the entire hole,
            # then multiply by (1 + local_edge_noise*0.3) for each pixel in that hole.
            intensity_val = random.uniform(0.3, 0.7)
            holes_mask[hole_shape] *= intensity_val * (1.0 + edge_noise[hole_shape] * 0.3)

        # ----- 3) Scatter Additional Small Holes -----
        scattered_threshold = random.uniform(0.2, 0.3) + edge_noise * 0.15
        small_holes = (base_noise < scattered_threshold) & (edge_noise < random.uniform(0.5, 0.7))
        holes_mask[small_holes] *= random.uniform(0.5, 0.8)

        # ----- 4) Slight "Lift" of the Mask at Some Edges -----
        # Making those regions less likely to be holes:
        edge_lift_threshold = random.uniform(0.6, 0.8)
        edge_mask = (edge_noise > edge_lift_threshold)
        holes_mask[edge_mask] = np.maximum(holes_mask[edge_mask], random.uniform(0.8, 0.95))

        # ----- 5) Random Morphological Operations for Extra Variation -----
        kernel_size = random.randint(2, 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

        if random.random() < 0.5:
            holes_mask = cv2.erode(holes_mask, kernel, iterations=1)
        else:
            holes_mask = cv2.dilate(holes_mask, kernel, iterations=1)

        # Smooth slightly
        holes_mask = gaussian_filter(holes_mask, sigma=0.5)

        return holes_mask

    def _generate_base_noise_mask(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        """
        Generates a base noise mask with optional elongation.

        :param x_grid: Meshgrid of x-coordinates, shape=(height, width).
        :param y_grid: Meshgrid of y-coordinates, shape=(height, width).
        :return: Base noise mask in float form (0..1).
        :rtype: np.ndarray
        """
        if self.elongated:
            scale_x, scale_y = self.scale * 0.5, self.scale * 2
        else:
            scale_x = scale_y = self.scale

        noise = self.vectorized_pnoise2(x_grid / scale_x, y_grid / scale_y)
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        return noise

    def _generate_distance_mask(self, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        """
        Generate a distance-based mask using random seed points.
        Pixels within a certain distance to any seed are set to 1, else 0.

        :param x_grid: Meshgrid of x-coordinates.
        :param y_grid: Meshgrid of y-coordinates.
        :return: A float distance mask.
        :rtype: np.ndarray
        """
        seed_points = np.random.rand(self.num_seeds, 2) * [self.width, self.height]
        distances = np.full((self.height, self.width), np.inf, dtype=np.float32)

        for point in seed_points:
            dist = np.sqrt((x_grid - point[0]) ** 2 + (y_grid - point[1]) ** 2)
            distances = np.minimum(distances, dist)

        # Add Perlin noise to distances so edges become more uneven
        noise_distance = self.vectorized_pnoise2(x_grid / self.noise_scale, y_grid / self.noise_scale)
        noise_distance = (noise_distance - noise_distance.min()) / (noise_distance.max() - noise_distance.min())

        # Weighted sum to push the distance cutoff
        distances_with_noise = distances + (noise_distance * self.max_distance)

        distance_mask = (distances_with_noise < self.max_distance).astype(np.float32)
        return distance_mask

    def _generate_initial_mask(self) -> np.ndarray:
        """
        Generates the initial binary mask with combined base noise,
        a distance-based patch mask, and internal holes.

        :return: A binary mask (0 or 1) as np.ndarray of shape (height, width).
        :rtype: np.ndarray
        """
        # Create a base coordinate mesh
        x = np.linspace(0, self.width, self.width)
        y = np.linspace(0, self.height, self.height)
        x_grid, y_grid = np.meshgrid(x, y)  # shape: (height, width)

        # 1) Base noise mask
        base_noise_mask = self._generate_base_noise_mask(x_grid, y_grid)
        base_mask = (base_noise_mask > self.threshold).astype(np.float32)

        # 2) Distance-based patch mask
        distance_mask = self._generate_distance_mask(x_grid, y_grid)

        # 3) Combine base noise + distance-based patches
        combined_mask = np.minimum(base_mask, distance_mask)

        # 4) Add holes
        holes_mask = self._generate_holes_mask(x_grid, y_grid)
        combined_mask *= holes_mask

        # 5) Threshold to get a binary mask
        #    Setting the threshold somewhat randomly between 0.4 and 0.6
        combined_mask = (combined_mask > random.uniform(0.4, 0.6)).astype(np.uint8)

        # 6) Optional final blur -> re-threshold for softer edges
        if self.blur_radius > 0:
            combined_mask = gaussian_filter(combined_mask.astype(float), sigma=self.blur_radius)
            combined_mask = (combined_mask > 0.5).astype(np.uint8)

        return combined_mask

    def simulate(self) -> list:
        """
        Runs the disease progression simulation, generating a list of binary masks
        representing the disease's appearance at each iteration.

        :return: List of binary masks (np.ndarrays) representing the simulated disease progression.
        :rtype: list
        """
        # Edge-case check
        if self.iterations <= 0:
            raise ValueError("Number of iterations must be positive.")

        # ---- (1) Generate the initial mask ----
        initial_mask = self._generate_initial_mask()

        # Calculate how many pixels we start with and how many we want at the end
        initial_area = np.sum(initial_mask)
        final_area = initial_area * self.end_severity_level

        # If the final area is effectively zero, handle that edge-case
        if final_area < 1:
            final_area = 1  # ensures at least some area can remain

        # ---- (2) Figure out how many total erosions needed to reach final area ----
        temp_mask = initial_mask.copy()
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        total_erosions_needed = 0
        current_area_temp = np.sum(temp_mask)
        while current_area_temp > final_area and np.any(temp_mask):
            temp_mask = cv2.erode(temp_mask, kernel_small, iterations=1)
            current_area_temp = np.sum(temp_mask)
            total_erosions_needed += 1

        # If no erosion is needed (e.g., end_severity_level = 1):
        if total_erosions_needed == 0:
            # We won't do any erosion at all, but we might still do expansions/deviations
            pass

        # ---- (3) Simulate each iteration ----
        current_mask = initial_mask.copy()
        results = []
        erosions_applied_so_far = 0

        for i in range(self.iterations):
            # The cumulative erosions needed by iteration i:
            # E.g. if total_erosions_needed=10, iterations=5, by iteration i we apply i*(10/5)= 2 erosions
            # Use round() so that across all iterations we sum to total_erosions_needed.
            cumulative_erosions_needed = int(round((i + 1) * total_erosions_needed / self.iterations))
            erosions_to_apply = cumulative_erosions_needed - erosions_applied_so_far

            # Decide whether to erode/dilate/do nothing, based on `deviation`
            if self.deviation > 0:
                rand_value = random.random()
                if rand_value < self.deviation:
                    # Randomly choose to dilate or do nothing
                    action = random.choice(['dilate', 'none'])
                else:
                    action = 'erode'
            else:
                action = 'erode'

            # ---- Erode / Dilate / None ----
            if action == 'erode' and erosions_to_apply > 0:
                current_mask = cv2.erode(current_mask, kernel_small, iterations=erosions_to_apply)
                erosions_applied_so_far += erosions_to_apply
            elif action == 'dilate':
                current_mask = cv2.dilate(current_mask, kernel_small, iterations=1)
                # We do not recalculate erosions_applied_so_far because we didn't erode.

            # ---- Edge Processing (Blur + Noise) ----
            if self.blur_radius > 0:
                mask_float = current_mask.astype(np.float32)
                mask_blurred = gaussian_filter(mask_float, sigma=self.blur_radius)
                current_mask = (mask_blurred > 0.5).astype(np.uint8)

            if self.edge_noise > 0.0:
                # 1) Extract edges
                edges = cv2.Canny(current_mask * 255, 100, 200) / 255.0

                # 2) Create random toggles for edge pixels
                noise = np.random.rand(self.height, self.width)
                # Where edge > 0, randomly flip bits if noise < self.edge_noise
                flip_mask = (edges > 0) & (noise < self.edge_noise)

                # 3) Flip those bits
                current_mask[flip_mask] = 1 - current_mask[flip_mask]

            # Optional morphological closing to unify small holes or cracks
            # (This often helps produce a more "organic" boundary.)
            if random.random() < 0.2:
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

            results.append(current_mask.copy())

            # Stop early if we've reached or gone below final area
            if np.sum(current_mask) <= final_area or not np.any(current_mask):
                print(f"Reached target severity at iteration {i + 1}.")
                break

        return results


def example_usage():
    """
    Example usage of the DiseaseProgressionSimulator class.
    Generates and saves a single mask (iterations=1) at full severity (end_severity_level=1).
    """
    import os

    # Define starting parameters
    start_params = {
        'scale': 200,  # Controls the size of noise features
        'threshold': 0.65,  # Controls the initial coverage area
        'blur_radius': 0,  # Controls edge softness
        'num_seeds': 100,  # Number of seeds in distance mask
        'max_distance': 50,  # Max distance for patches
        'noise_scale': 20  # Controls edge irregularity
    }

    # Simulation parameters
    width = 1024
    height = 1024
    end_severity_level = 1.0  # 0 => fully healed, 1 => no erosion
    iterations = 1  # Only one iteration => single mask
    deviation = 0.2  # Randomness in healing steps
    elongated = False  # Keep standard shapes
    edge_noise = 0.95  # High noise => more toggling at edges

    # Create simulator
    simulator = DiseaseProgressionSimulator(
        width=width,
        height=height,
        start_params=start_params,
        end_severity_level=end_severity_level,
        iterations=iterations,
        deviation=deviation,
        elongated=elongated,
        edge_noise=edge_noise
    )

    # Run the simulation
    results = simulator.simulate()

    # Save output
    output_dir = 'data/output_masks/disease_progression'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, mask in enumerate(results):
        cv2.imwrite(os.path.join(output_dir, f"iteration_{idx + 1}.png"), mask * 255)


if __name__ == "__main__":
    example_usage()
