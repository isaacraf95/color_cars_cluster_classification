import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from utils import remove_specular_component, apply_threshold_segmentation, find_most_informative_region, \
    extract_roi, get_random_image_paths
import cv2
import joblib
import time
from torch.utils.data import DataLoader, Dataset

# Load ONNX Model
onnx_model_path = './mode/resnet50_feature_extractor.onnx'
ort_session = ort.InferenceSession(onnx_model_path)


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

def make_inference_onnx(image_paths, ort_session, gmm, scaler):
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

directory_path = ('./cars_images/')

# Get n random path images
image_paths = get_random_image_paths(directory_path, num_images=1000)

# Load model and scaler
gmm = joblib.load('./model/gmm_cars_v1.pkl')
scaler = joblib.load('./scaler/scaler_cars_v1.pkl')

# Latency Test
start_time = time.perf_counter()
labels = make_inference_onnx(image_paths, ort_session, gmm, scaler)
print(labels)
end_time = time.perf_counter()
latency = end_time - start_time
print(f"Latency ONNX: {latency} seconds")

# Throughput Test
total_time = end_time - start_time
throughput = len(image_paths) / total_time
print(f"Throughput: {throughput} imgs per second")
