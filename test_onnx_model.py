import onnxruntime as ort
from utils import get_random_image_paths, make_inference_onnx
import joblib
import time

# Load ONNX Model
onnx_model_path = './model/resnet50_feature_extractor.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

directory_path = ('./cars_images/')

# Get n random path images
image_paths = get_random_image_paths(directory_path, num_images=1)

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

# Throughput Test
total_time = end_time - start_time
throughput = len(image_paths) / total_time
print(f"Throughput: {throughput} imgs per second")
