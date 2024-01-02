# Use an official Python image as a base image
FROM python:3.8

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-dev

# Sets a working directory in the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for temporary images
RUN mkdir /app/temp_images
# Copies the model files and other necessary files into the container.
COPY ./model /app/model
COPY ./scaler /app/scaler
COPY ./utils.py /app/utils.py
COPY ./main_api.py /app/main_api.py

# The command to run the FastAPI application
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "80"]
