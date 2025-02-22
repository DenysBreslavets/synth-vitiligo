import cv2
import numpy as np
import os


class VitiligoBlendSimulator:
    def __init__(self,
                 depigmentation_input_dir='data/output_masks/vitiligo_sim/depigmentation',
                 repigmentation_input_dir='data/output_masks/vitiligo_sim/repigmentation',
                 depigmentation_output_dir='data/output_images/depigmentation',
                 repigmentation_output_dir='data/output_images/repigmentation',
                 depigmentation_masks=None,
                 repigmentation_masks=None,
                 dimensions=(1024, 1024),
                 input_image_path='data/images/input_1.jpg',
                 skin_mask_path='data/images/skin_mask_1.png',
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
                 depigmentation_steps=50,
                 repigmentation_steps=50,
                 decay_factor=0.95,
                 probability_threshold=0.6,
                 blur_size_darkening=(5, 5),
                 dark_factor=0.35):
        # Set all parameters as instance variables
        self.depigmentation_input_dir = depigmentation_input_dir
        self.repigmentation_input_dir = repigmentation_input_dir
        self.depigmentation_output_dir = depigmentation_output_dir
        self.repigmentation_output_dir = repigmentation_output_dir
        self.depigmentation_masks = depigmentation_masks
        self.repigmentation_masks = repigmentation_masks
        self.repigmentation_images = []
        self.depigmentation_images = []
        self.dimensions = dimensions
        self.input_image_path = input_image_path
        self.skin_mask_path = skin_mask_path
        self.blur_size_vitiligo = blur_size_vitiligo
        self.s_reduction_min = s_reduction_min
        self.s_reduction_max = s_reduction_max
        self.v_increase_min = v_increase_min
        self.v_increase_max = v_increase_max
        self.h_shift_min = h_shift_min
        self.h_shift_max = h_shift_max
        self.pink_alpha_ranges = pink_alpha_ranges
        self.white_alpha_ranges = white_alpha_ranges
        self.pink_color = pink_color
        self.white_color = white_color
        self.depigmentation_steps = depigmentation_steps
        self.repigmentation_steps = repigmentation_steps
        self.decay_factor = decay_factor
        self.probability_threshold = probability_threshold
        self.blur_size_darkening = blur_size_darkening
        self.dark_factor = dark_factor

        # Load the input images
        self.load_images()

        # Process the masks
        self.process_masks()

    def process_masks(self):
        """
        Process masks either from filesystem or from passed arrays.
        Converts all masks to the correct format and stores them for later use.
        """

        # Process both depigmentation and repigmentation masks
        self.processed_depigmentation_masks = self._process_mask_set(
            self.depigmentation_masks,
            self.depigmentation_input_dir,
            self.depigmentation_steps,
            "depigmentation",
        )

        self.processed_repigmentation_masks = self._process_mask_set(
            self.repigmentation_masks,
            self.repigmentation_input_dir,
            self.repigmentation_steps,
            "repigmentation",
        )

    def _process_mask_set(self, masks, input_dir, steps, mask_type):
        """
        Helper function to process a set of masks either from passed arrays or filesystem.

        Parameters:
            masks: List of masks or None
            input_dir: Directory containing mask files
            steps: Number of steps
            mask_type: String identifier for mask type

        Returns:
            List of processed masks
        """
        processed_masks = []

        if masks is not None:
            # Process directly passed masks
            for mask in masks:
                processed_mask = self.process_single_mask(mask)
                if processed_mask is not None:
                    processed_masks.append(processed_mask)
        else:
            # Process masks from filesystem
            for idx in range(1, steps + 1):
                mask_path = f'{input_dir}/{mask_type}_step_{idx}.png'
                try:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    processed_mask = self.process_single_mask(mask)
                    if processed_mask is not None:
                        processed_masks.append(processed_mask)
                except Exception as e:
                    print(f"Warning: Could not process {mask_type} mask {idx}: {str(e)}")

        return processed_masks

    def process_single_mask(self, mask):
        """
        Process a single mask to the correct format.
        """
        if mask is None:
            return None

        # Create a copy to avoid modifying the original
        mask = mask.copy()

        # Handle uint8 masks with 0-1 values (from progression simulator)
        if mask.dtype == np.uint8 and mask.max() <= 1:
            mask = mask.astype(np.float32)
        # Handle uint8 masks with 0-255 values (from files)
        elif mask.dtype == np.uint8:
            mask = mask / 255.0
        # Handle float masks (already normalized)
        elif mask.dtype == np.float32 or mask.dtype == np.float64:
            pass

        # Check if mask dimensions match self.dimensions
        if mask.shape[:2] != self.dimensions:
            mask = cv2.resize(mask, self.dimensions, interpolation=cv2.INTER_NEAREST)

        # Maintain mask only on skin
        mask = mask * self.input_image_skin

        return mask

    def load_images(self):
        # Read input image and skin mask
        self.input_image = cv2.imread(self.input_image_path, cv2.IMREAD_COLOR)
        if self.input_image is None:
            raise FileNotFoundError(f"Input image not found at {self.input_image_path}")
        self.input_image_skin = cv2.imread(self.skin_mask_path, cv2.IMREAD_GRAYSCALE)
        if self.input_image_skin is None:
            raise FileNotFoundError(f"Skin mask image not found at {self.skin_mask_path}")

        # Resize images
        self.input_image = cv2.resize(self.input_image, self.dimensions)
        self.input_image_skin = cv2.resize(self.input_image_skin, self.dimensions, interpolation=cv2.INTER_NEAREST)

        # Normalize mask to 0-1 range
        self.input_image_skin = self.input_image_skin / 255.0

        # Ensure output directories exist
        if self.depigmentation_output_dir and self.repigmentation_output_dir:
            if not os.path.exists(self.depigmentation_output_dir):
                os.makedirs(self.depigmentation_output_dir)
            if not os.path.exists(self.repigmentation_output_dir):
                os.makedirs(self.repigmentation_output_dir)

    def apply_vitiligo_effect(self, input_image, mask):
        """
        Applies the vitiligo effect to the input image based on the provided mask.
        Preserves skin texture details using high-frequency detail pass.
        Parameters:
            input_image (np.ndarray): The original BGR image.
            mask (np.ndarray): A single-channel mask with values in [0, 1].
        Returns:
            np.ndarray: The image with the vitiligo effect applied.
        """
        # Feather the mask edges for a soft transition
        feathered_mask = cv2.GaussianBlur(mask, self.blur_size_vitiligo, 0)
        feathered_mask_3c = cv2.merge([feathered_mask] * 3)

        # Extract high-frequency details before color transformations
        input_image_f32 = input_image.astype(np.float32)
        low_freq = cv2.GaussianBlur(input_image_f32, (3, 3), 0)
        high_freq_details = input_image_f32 - low_freq

        # Convert input image to HSV color space
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)

        # Create a mask of pixels where the mask is applied
        mask_pixels = feathered_mask > 0

        # Compute mean V (brightness) and S (saturation) in the masked area
        if np.any(mask_pixels):
            mean_v = np.mean(v[mask_pixels])
            mean_s = np.mean(s[mask_pixels])
        else:
            mean_v = 255
            mean_s = 255

        # Compute the factors based on mean_v (brightness)
        s_reduction_factor = self.s_reduction_min + (self.s_reduction_max - self.s_reduction_min) * (
                255 - mean_v) / 255.0
        v_increase_factor = self.v_increase_min + (self.v_increase_max - self.v_increase_min) * (255 - mean_v) / 255.0
        h_shift = self.h_shift_min + (self.h_shift_max - self.h_shift_min) * (255 - mean_v) / 255.0

        # Apply the adjustments in the masked area
        s_vitiligo = s * (1 - s_reduction_factor * feathered_mask)
        v_vitiligo = v * (1 + v_increase_factor * feathered_mask)
        h_vitiligo = h + h_shift * feathered_mask

        # Ensure the values are within [0,255]
        s_vitiligo = np.clip(s_vitiligo, 0, 255)
        v_vitiligo = np.clip(v_vitiligo, 0, 255)
        h_vitiligo = np.mod(h_vitiligo, 180)

        # Merge the adjusted channels
        h_vitiligo = h_vitiligo.astype(np.uint8)
        s_vitiligo = s_vitiligo.astype(np.uint8)
        v_vitiligo = v_vitiligo.astype(np.uint8)
        hsv_vitiligo = cv2.merge([h_vitiligo, s_vitiligo, v_vitiligo])

        # Convert back to BGR color space
        vitiligo_image = cv2.cvtColor(hsv_vitiligo, cv2.COLOR_HSV2BGR)

        # Add Pink Tint Overlay
        pink_overlay = np.full_like(input_image, self.pink_color)

        # Calculate pink intensity factor based on mean brightness
        pink_intensity = self.pink_alpha_ranges[0] + (self.pink_alpha_ranges[1] - self.pink_alpha_ranges[0]) * (
                255 - mean_v) / 255.0

        # Add White Tint Overlay
        white_overlay = np.full_like(input_image, self.white_color)

        # Calculate white intensity factor based on mean brightness
        white_intensity = self.white_alpha_ranges[0] + (self.white_alpha_ranges[1] - self.white_alpha_ranges[0]) * (
                255 - mean_v) / 255.0

        # Blend the pink and white overlays with the vitiligo image using the feathered mask
        vitiligo_image = vitiligo_image.astype(np.float32)
        pink_overlay = pink_overlay.astype(np.float32)
        white_overlay = white_overlay.astype(np.float32)

        # Apply the pink overlay
        vitiligo_image = vitiligo_image * (1 - pink_intensity * feathered_mask_3c) + pink_overlay * (
                pink_intensity * feathered_mask_3c)

        # Apply the white overlay
        vitiligo_image = vitiligo_image * (1 - white_intensity * feathered_mask_3c) + white_overlay * (
                white_intensity * feathered_mask_3c)

        # Reintroduce high-frequency details in the masked area
        detail_factor = 0.95  # Control the strength of detail preservation
        masked_details = high_freq_details * feathered_mask_3c * detail_factor
        vitiligo_image = vitiligo_image + masked_details

        # Ensure the values are within the valid range
        vitiligo_image = np.clip(vitiligo_image, 0, 255)

        # Final Blend with Original Image
        input_image_f = input_image.astype(np.float32)
        vitiligo_image_final = vitiligo_image * feathered_mask_3c + input_image_f * (1 - feathered_mask_3c)
        vitiligo_image_final = np.clip(vitiligo_image_final, 0, 255).astype(np.uint8)

        return vitiligo_image_final

    def depigmentation(self, return_images=False, save_images=False):
        """
        Depigmentation Process
        """
        if not self.processed_depigmentation_masks:
            print("No depigmentation masks available to process.")
            return

        for idx, mask in enumerate(self.processed_depigmentation_masks, 1):
            # Apply the vitiligo effect
            vitiligo_image = self.apply_vitiligo_effect(self.input_image, mask)

            # Save the depigmented image
            if save_images:
                if not self.depigmentation_output_dir:
                    raise ValueError("Depigmentation output directory not set.")
                output_path = f'{self.depigmentation_output_dir}/step_{idx}.png'
                cv2.imwrite(output_path, vitiligo_image)
                print(f"Depigmentation step {idx} saved to {output_path}")

            self.depigmentation_images.append(vitiligo_image.copy())

        if return_images:
            return self.depigmentation_images

    def repigmentation(self, return_images=False, save_images=False):
        """
        Repigmentation Process
        """
        if not self.processed_repigmentation_masks:
            print("No repigmentation masks available to process.")
            return

        # Initialize cumulative mask and previous mask for repigmentation
        cumulative_mask = np.zeros(self.dimensions, dtype=np.float32)
        previous_mask = np.zeros(self.dimensions, dtype=np.float32)

        # Generate random noise mask with float values between 0-1
        noise_mask = np.random.rand(*self.dimensions).astype(np.float32)

        # Create a fixed binary mask where darkening should be applied based on noise
        darkening_probability_mask = (noise_mask < self.probability_threshold).astype(np.float32)

        for idx, mask in enumerate(self.processed_repigmentation_masks, 1):
            # Compute the areas that have been repigmented in this iteration
            mask_diff = mask - previous_mask
            mask_diff = np.maximum(mask_diff, 0)  # Ensure no negative values

            # Apply decay to cumulative mask to simulate healing
            cumulative_mask = cumulative_mask * self.decay_factor + mask_diff

            # Clip cumulative_mask to [0,1]
            cumulative_mask = np.clip(cumulative_mask, 0, 1)

            # Create the darkening mask based on fixed probability mask
            darkening_mask = cumulative_mask * darkening_probability_mask

            # Expand to 3 channels
            darkening_mask_3c = cv2.merge([darkening_mask] * 3)

            # Apply blur to the darkening mask
            darkening_mask_3c = cv2.GaussianBlur(darkening_mask_3c, self.blur_size_darkening, sigmaX=0, sigmaY=0)

            # Apply the vitiligo effect to the current mask
            vitiligo_image = self.apply_vitiligo_effect(self.input_image, mask)

            # Apply darkening to the vitiligo_image based on darkening_mask
            vitiligo_image = self.apply_darkening(vitiligo_image, darkening_mask_3c)

            # Update previous_mask for the next iteration
            previous_mask = mask.copy()

            # Save the repigmented image
            if save_images:
                if not self.repigmentation_output_dir:
                    raise ValueError("Repigmentation output directory not set.")
                output_path = f'{self.repigmentation_output_dir}/step_{idx}.png'
                cv2.imwrite(output_path, vitiligo_image)
                print(f"Repigmentation step {idx} saved to {output_path}")

            self.repigmentation_images.append(vitiligo_image.copy())

        if return_images:
            return self.repigmentation_images

    def apply_darkening(self, image, darkening_mask_3c):
        """
        Applies darkening to the image based on the darkening mask.
        """
        # Convert image to float for blending
        image = image.astype(np.float32)

        # Apply darkening where darkening_mask > 0
        image = image * (1 - darkening_mask_3c * (1 - self.dark_factor))

        # Ensure the values are within the valid range and convert back to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image


if __name__ == "__main__":
    simulator = VitiligoBlendSimulator()
    simulator.depigmentation(save_images=True)
    simulator.repigmentation(save_images=True)
