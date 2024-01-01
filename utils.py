import numpy as np
import torch
import os
import random


def remove_specular_component(image, K=0.4, epsilon=1e-8):
    """
    Remove the specular component from an image based on an equation,
    with additional safeguards for division by zero.
    """
    if image.dtype != np.uint8:
        raise ValueError("The image should be an 8-bit RGB image")

    image_float = image.astype(np.float32)

    # Calculate I_tilde and C_tilde for the entire image
    I_tilde = np.max(image_float, axis=-1, keepdims=True)
    sum_pixels = np.sum(image_float, axis=-1, keepdims=True) + epsilon
    C_tilde = I_tilde / sum_pixels

    # Compute the second term for the entire image
    second_term = (sum_pixels - ((I_tilde * (3 * C_tilde - 1)) / (C_tilde * (3 * K - 1) + epsilon))) / 3

    # Remove the specular component
    specular_free_image = image_float - second_term

    # Clip values to be in the range [0, 255] and convert back to uint8
    specular_free_image_clipped = np.clip(specular_free_image, 0, 255).astype(np.uint8)

    return specular_free_image_clipped


def apply_threshold_segmentation(image, lambda_factor=0.15):
    """
    Apply threshold segmentation based on horizontal projection while retaining color.
    """
    h = np.sum(image, axis=2)

    T = lambda_factor * np.max(h)

    mask = h >= T

    segmented_image = np.zeros_like(image)

    segmented_image[mask] = image[mask]

    binary_image = np.any(segmented_image, axis=-1).astype(np.uint8) * 255

    return binary_image


def find_most_informative_region(binary_image):
    """
    Identifies the most informative horizontal region in a binary image.
    This region is determined by finding the row with the highest sum of pixel values.
    """
    row_sums = np.sum(binary_image, axis=1)
    most_informative_row = np.argmax(row_sums)
    return most_informative_row


def extract_roi(image, most_informative_row, height=80):
    """
    Extracts a region of interest (ROI) from an image based on the most informative row.
    The ROI is a horizontal slice of the specified height centered around the most informative row.
    """
    start_row = max(most_informative_row - height // 2, 0)
    end_row = min(most_informative_row + height // 2, image.shape[0])
    roi = image[start_row:end_row, :]
    return roi


def get_image_paths(directory):
    """
    Collects and returns a list of paths to image files within the specified directory.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, file)) and file.lower().endswith(('.png', '.jpg', '.jpeg'))]


def extract_features(model, data_loader):
    """
    Extracts features from a batch of images using a pre-trained model.
    """
    mps_device = torch.device("mps")
    with torch.no_grad():
        features = []
        for inputs in data_loader:
            inputs = inputs.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            elif torch.backends.mps.is_available():
                inputs = inputs.to(mps_device)
            outputs = model(inputs)
            pooled_outputs = torch.mean(outputs, dim=[2, 3])
            features.append(pooled_outputs.cpu().numpy())
        features = np.concatenate(features, axis=0)
    return features


def get_random_image_paths(directory, num_images=8):
    """
    Retrieves a specified number of random image file paths from a given directory.
    This function is used to randomly sample a set of images from a larger collection.
    """
    all_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(all_files) >= num_images:
        return random.sample(all_files, num_images)
    else:
        return all_files
