# 🚗 Color Cars Cluster Classification

This repository contains the scripts and resources needed for vehicle color classification using machine learning models. 🚀

## 📁 Repository Structure

- `utils.py`: Auxiliary functions used by other scripts. 🔧
- `train_model.py`: Script to train the GMM model. Includes the entire pipeline from feature extraction. 🧠
- `test_model.py`:  Script to test the model using the Torch pipeline. 🔍
- `test_onnx_model.py`: Script to test the model using the ONNX pipeline.🤖
- `test_preprocessing_steps.ipynb`: Notebook to test preprocessing steps based on different papers. 📊
- `Algotive_Challenge_Report.pdf`: Report of the results obtained from the project. 📄
- `model/`: Folder containing the trained models (the models must be downloaded; the link is located inside the folder). 📥
- `scaler/`: Folder containing the scalers used in the model. ⚖️
- `Dockerfile`: Dockerfile file to create the container image. 🐳
- `requirements.txt`: Dependencies required for the project. 📋
- `test_api.py`: Script to test the API after running the container. 🧪

## 🚀 How to Use

To use this project, first clone the repository and download the necessary models from the links provided in the `model/` and `scaler/` folders.

### 🐳 Build and Run the Container.

Use the following commands to build and run the Docker container:

```bash
docker build -t cars_classification_api .
docker run -p 80:80 cars_classification_api
```

#### 🧪 Testing the API
Once the container is running, you can test the API using the test_api.py script.

##### Contribute 🤝

If you have ideas to improve the API or find a bug, feel free to open an Issue or make a Pull Request.
