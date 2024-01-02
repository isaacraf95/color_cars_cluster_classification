from utils import get_random_image_paths
import os
import requests

# URL API
url = 'http://localhost/predict/'

directory_path = './cars_images/'

# Get n random path images
image_paths = get_random_image_paths(directory_path, num_images=10)

# Create list to save the responses
responses = []

# Send images to API
for filename in image_paths:
    file_path = os.path.join(filename)

    # Open image and send to the API URL
    with open(file_path, 'rb') as file:
        files = {'files': (filename, file, 'multipart/form-data')}
        response = requests.post(url, files=files)
        responses.append(response.json())

# Print responses
for response in responses:
    print(response)
