import torch
import torchvision.models as models
from utils import get_random_image_paths, make_inference_torch
import joblib
import time

# Load ResNet50 for feature extraction
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-2]))
model = model.eval()  # Poner el modelo en modo de evaluación

# Move data to mps
if torch.backends.mps.is_available():
    cpu_device = torch.device("mps")
    model.to(cpu_device)

directory_path = ('./cars_images/')

# Get n random path images
image_paths = get_random_image_paths(directory_path, num_images=1)

# Load model and scaler
gmm = joblib.load('./model/gmm_cars_v1.pkl')
scaler = joblib.load('./scaler/scaler_cars_v1.pkl')

# Latency Test
start_time = time.perf_counter()
labels = make_inference_torch(image_paths, model, gmm, scaler)
end_time = time.perf_counter()
latency = end_time - start_time
print(f"Inference Latency: {latency} seconds")
print(labels)

# Throughput Test
total_time = end_time - start_time
throughput = len(image_paths) / total_time
print(f"Throughput: {throughput} imgs per second")
