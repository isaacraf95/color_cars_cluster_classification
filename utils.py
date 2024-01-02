import numpy as np
import torch
import os
import random
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


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


# Use ColorCarsDataset for preprocessing to the model
class ColorCarsDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Preprocess functions
        image_r = remove_specular_component(image)
        binary_image = apply_threshold_segmentation(image_r)
        most_informative_row = find_most_informative_region(binary_image)
        valid_image = cv2.bitwise_and(image, image, mask=binary_image)
        final_image = extract_roi(valid_image, most_informative_row, height=70)

        # Move to PIL data
        final_image_pil = Image.fromarray(final_image.astype('uint8'), 'RGB')

        # Apply transformations
        if self.transform:
            final_image_transformed = self.transform(final_image_pil)
        else:
            final_image_transformed = transforms.ToTensor()(final_image_pil)

        return final_image_transformed


def make_inference_torch(image_paths, model, gmm, scaler):
    """
    Perform inference on a set of images using a pre-trained model and Gaussian Mixture Model (GMM).
    """
    # Necessary transform to input the images to the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ColorCarsDataset(image_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    features = extract_features(model, data_loader)
    normalized_features = scaler.transform(features)
    labels = gmm.predict(normalized_features)

    return labels


def make_inference_onnx(image_paths, ort_session, gmm, scaler):
    """
    Perform inference on a set of images using a pre-trained onnx model and Gaussian Mixture Model (GMM).
    """
    # Necessary transform to input the images to the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Similar setup as before
    dataset = ColorCarsDataset(image_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Modificar la extracción de características para usar ONNX Runtime
    all_features = []
    for batch in data_loader:
        # ONNX Runtime espera numpy arrays
        batch_np = batch.numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: batch_np}
        ort_outs = ort_session.run(None, ort_inputs)
        pooled_outputs = np.mean(ort_outs[0], axis=(2, 3))
        all_features.append(pooled_outputs)

    # Resto del código similar
    features = np.concatenate(all_features, axis=0)
    normalized_features = scaler.transform(features)
    labels = gmm.predict(normalized_features)
    return labels
