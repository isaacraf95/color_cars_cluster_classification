import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from utils import remove_specular_component, apply_threshold_segmentation, find_most_informative_region, \
    extract_roi, extract_features, get_image_paths
from sklearn.preprocessing import StandardScaler
import cv2
from sklearn.mixture import GaussianMixture
import joblib

# Load ResNet50 architecture
model = models.resnet50(pretrained=True)
# Avgpool and fc are removed to be able to extract the feature vector
model = torch.nn.Sequential(*(list(model.children())[:-2]))
# Set model to eval
model = model.eval()

# In my case use my MAC power for this step
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    model.to(mps_device)


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


# Necessary transform to input the images to the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


image_paths = get_image_paths('./cars_images/')

# Load images
dataset = ColorCarsDataset(image_paths, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Feature extractions
features = extract_features(model, data_loader)

# Normalize the characteristics before applying the cluster
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Definition of the number of components
n_components = 6

# GMM model definition
gmm = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=0)

# Fit the model
gmm.fit(normalized_features)

# Save the model
model_path = './model/gmm_cars_v1.pkl'
model_directory = os.path.dirname(model_path)

# Create the directory if it does not exist
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Save GMM model
scaler_path = './scaler/scaler_cars_v1.pkl'
scaler_directory = os.path.dirname(scaler_path)

# Create the directory if it does not exist
if not os.path.exists(scaler_directory):
    os.makedirs(scaler_directory)

# Save scaler
joblib.dump(scaler, scaler_path)
