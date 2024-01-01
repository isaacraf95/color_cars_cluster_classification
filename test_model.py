import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from utils import remove_specular_component, apply_threshold_segmentation, find_most_informative_region, \
    extract_roi, extract_features, get_random_image_paths
import cv2
import joblib
import time
from torch.utils.data import DataLoader, Dataset

# Load ResNet50 for feature extraction
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-2]))
model = model.eval()  # Poner el modelo en modo de evaluación

# Move data to mps
if torch.backends.mps.is_available():
    cpu_device = torch.device("mps")
    model.to(cpu_device)


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


def make_inference(image_paths, model, gmm, scaler):
    """
    Perform inference on a set of images using a pre-trained model and Gaussian Mixture Model (GMM).

    Args:
        image_paths (list of str): List of paths to the images for inference.
        model (torch.nn.Module): The pre-trained model used for feature extraction.
        gmm (sklearn.mixture.GaussianMixture): Trained Gaussian Mixture Model for clustering.
        scaler (sklearn.preprocessing.StandardScaler): Scaler used for normalizing features.

    Returns:
        numpy.ndarray: Predicted labels from the GMM for each image.

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

directory_path = ('./cars_images/')

# Get n random path images
image_paths = get_random_image_paths(directory_path, num_images=1000)

# Load model and scaler
gmm = joblib.load('./model/gmm_cars_v1.pkl')
scaler = joblib.load('./scaler/scaler_cars_v1.pkl')

# Latency Test
start_time = time.perf_counter()
labels = make_inference(image_paths, model, gmm, scaler)
end_time = time.perf_counter()
latency = end_time - start_time
print(f"Inference Latency: {latency} seconds")
print(labels)

# Throughput Test
total_time = end_time - start_time
throughput = len(image_paths) / total_time
print(f"Throughput: {throughput} imgs per second")


