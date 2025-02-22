import numpy as np
import cv2
import sys
from math import sqrt, ceil, pi, cos, sin
import random


def poisson_disk_sampling(width, height, radius, k=10):
    """
    Generates points in a 2D space using Poisson Disk Sampling (Bridson's algorithm).

    Parameters:
    - width (float): Width of the space in pixels.
    - height (float): Height of the space in pixels.
    - radius (float): Minimum distance between points in pixels.
    - k (int): Maximum number of attempts before rejection.

    Returns:
    - positions (list of tuples): Generated points as (x, y) coordinates.
    """
    cell_size = radius / sqrt(2)
    grid_width = int(ceil(width / cell_size))
    grid_height = int(ceil(height / cell_size))

    # Initialize the grid with None
    grid = np.full((grid_height, grid_width), None)

    positions = []
    spawn_points = []

    # Initial point
    x0 = random.uniform(0, width)
    y0 = random.uniform(0, height)
    positions.append((x0, y0))
    spawn_points.append((x0, y0))

    grid_x = int(x0 / cell_size)
    grid_y = int(y0 / cell_size)
    grid[grid_y, grid_x] = (x0, y0)

    while spawn_points:
        idx = random.randint(0, len(spawn_points) - 1)
        x_center, y_center = spawn_points[idx]
        accepted = False

        for _ in range(k):
            angle = random.uniform(0, 2 * pi)
            r = random.uniform(radius, 2 * radius)
            x_new = x_center + r * cos(angle)
            y_new = y_center + r * sin(angle)

            if x_new < 0 or x_new >= width or y_new < 0 or y_new >= height:
                continue

            grid_x = int(x_new / cell_size)
            grid_y = int(y_new / cell_size)

            # Define the search range
            x_min = max(grid_x - 2, 0)
            x_max = min(grid_x + 3, grid_width)
            y_min = max(grid_y - 2, 0)
            y_max = min(grid_y + 3, grid_height)

            ok = True
            for i in range(x_min, x_max):
                for j in range(y_min, y_max):
                    neighbor = grid[j, i]
                    if neighbor is not None:
                        dx = neighbor[0] - x_new
                        dy = neighbor[1] - y_new
                        if dx * dx + dy * dy < radius * radius:
                            ok = False
                            break
                if not ok:
                    break

            if ok:
                positions.append((x_new, y_new))
                spawn_points.append((x_new, y_new))
                grid[grid_y, grid_x] = (x_new, y_new)
                accepted = True
                break

        if not accepted:
            spawn_points.pop(idx)

    return positions


def generate_hair_follicle_mask(body_part, width_cm, height_cm, pixels_per_cm):
    """
    Generates a hair follicle mask for a specified body part using Poisson Disk Sampling.

    Parameters:
    - body_part (str): The body part to generate the mask for.
                        Options: 'forehead', 'back', 'thorax', 'upper_arm',
                                 'forearm', 'thigh', 'calf'
    - width_cm (float): Width of the image in centimeters.
    - height_cm (float): Height of the image in centimeters.
    - pixels_per_cm (float): Resolution in pixels per centimeter.

    Returns:
    - mask (numpy.ndarray): The generated hair follicle mask image.
    - actual_density (float): The actual density of follicles per cm².
    - diameter_pixels (int): Diameter of each follicle in pixels.
    - expected_density (float): Expected density of follicles per cm².
    """
    min_distance_factor = 1  # Lower increases density, higher decreases density

    # Define hair follicle data for each body part
    body_part_data = {
        'forehead': {'density': 292, 'diameter_um': 66},
        'back': {'density': 29, 'diameter_um': 70},  # Estimated diameter
        'thorax': {'density': 22, 'diameter_um': 70},  # Estimated diameter
        'upper_arm': {'density': 32, 'diameter_um': 70},  # Estimated diameter
        'forearm': {'density': 18, 'diameter_um': 78},
        'thigh': {'density': 17, 'diameter_um': 70},  # Estimated diameter
        'calf': {'density': 14, 'diameter_um': 90},
    }

    # Validate body part
    if body_part not in body_part_data:
        raise ValueError(f"Unsupported body part: {body_part}")

    # Retrieve density and diameter for the specified body part
    density = body_part_data[body_part]['density']  # follicles per cm²
    diameter_um = body_part_data[body_part]['diameter_um']  # in micrometers

    # Calculate image dimensions in pixels
    width_px = int(width_cm * pixels_per_cm)
    height_px = int(height_cm * pixels_per_cm)

    # Calculate image area in cm²
    image_area_cm2 = width_cm * height_cm

    # Calculate follicle diameter in cm
    diameter_cm = diameter_um / 10000  # Convert micrometers to centimeters

    # Calculate follicle diameter in pixels
    diameter_pixels = diameter_cm * pixels_per_cm
    diameter_pixels = max(int(round(diameter_pixels)), 1)  # Ensure minimum size of 1 pixel

    # Adjusted Minimum Distance Calculation
    # To account for Poisson Disk Sampling efficiency (~54.3%), adjust the formula
    # min_distance_cm = sqrt(2 / (pi * D))
    min_distance_cm = sqrt(2 / (pi * density)) * min_distance_factor
    min_distance_pixels = min_distance_cm * pixels_per_cm

    # Generate follicle positions using Poisson Disk Sampling
    positions = poisson_disk_sampling(width_px, height_px, min_distance_pixels, k=50)

    num_follicles = len(positions)

    # Create a black mask image
    mask = np.zeros((height_px, width_px), dtype=np.uint8)

    # Convert positions to integer pixel coordinates
    positions_int = np.array(positions).astype(int)

    # Draw follicles as white circles using OpenCV
    # Vectorization can be used with cv2.circle by iterating, but it's already efficient
    for (x, y) in positions_int:
        cv2.circle(mask, (x, y), diameter_pixels // 2, 255, -1)  # Filled circle

    # Calculate actual density
    actual_density = num_follicles / image_area_cm2

    return mask, actual_density, diameter_pixels, density


def calculate_expected_follicles(width_cm, height_cm, density):
    """
    Calculates the expected number of follicles based on image area and density.

    Parameters:
    - width_cm (float): Width of the image in centimeters.
    - height_cm (float): Height of the image in centimeters.
    - density (float): Density of follicles per cm².

    Returns:
    - expected_number (int): Expected number of follicles.
    """
    return int(width_cm * height_cm * density)


def main():
    """
    Main function to generate and save hair follicle masks with specified parameters.
    """

    side_cm = 30
    pixels_per_cm = 25

    # Define test cases with expected outputs
    test_cases = [
        {
            'body_part': 'forearm',
            'width_cm': side_cm,
            'height_cm': side_cm,
            'pixels_per_cm': pixels_per_cm,
            'output_filename': 'data/follicle_masks/forearm_mask.png'
        },
        # {
        #     'body_part': 'forehead',
        #     'width_cm': side_cm,
        #     'height_cm': side_cm,
        #     'pixels_per_cm': pixels_per_cm,
        #     'output_filename': 'data/follicle_masks/forehead_mask.png'
        # },
        # {
        #     'body_part': 'thigh',
        #     'width_cm': side_cm,
        #     'height_cm': side_cm,
        #     'pixels_per_cm': pixels_per_cm,
        #     'output_filename': 'data/follicle_masks/thigh_mask.png'
        # },
        # Add more test cases as needed
    ]

    for test in test_cases:
        body_part = test['body_part']
        width_cm = test['width_cm']
        height_cm = test['height_cm']
        pixels_per_cm = test['pixels_per_cm']
        output_filename = test['output_filename']

        try:
            # Generate the mask
            mask, actual_density, diameter_pixels, expected_density = generate_hair_follicle_mask(
                body_part=body_part,
                width_cm=width_cm,
                height_cm=height_cm,
                pixels_per_cm=pixels_per_cm
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            continue

        # Calculate expected number of follicles
        expected_number = calculate_expected_follicles(width_cm, height_cm, expected_density)

        # Calculate actual number of follicles
        actual_number = int(actual_density * width_cm * height_cm)

        # Print expected and actual densities and numbers
        print(f"Body Part: {body_part.capitalize()}")
        print(f"Expected density: {expected_density} follicles/cm²")
        print(f"Actual density: {actual_density:.2f} follicles/cm²")
        print(f"Expected number of follicles: {expected_number}")
        print(f"Actual number of follicles: {actual_number}")
        print(f"Follicle diameter in pixels: {diameter_pixels}")
        print(f"Hair follicle mask saved as {output_filename}\n")

        # Save the mask image
        success = cv2.imwrite(output_filename, mask)
        if success:
            print(f"Hair follicle mask saved as {output_filename}")
        else:
            print(f"Error: Could not save the image to {output_filename}", file=sys.stderr)


if __name__ == "__main__":
    main()
