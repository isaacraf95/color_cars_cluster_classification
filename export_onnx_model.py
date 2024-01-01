import torch.onnx
import torchvision.models as models

# Load Model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-2]))
model.eval()

# Create a dummy tensor
input_tensor = torch.randn(1, 3, 224, 224)

# Set model name
output_onnx_file = './model/resnet50_feature_extractor.onnx'

# Export model
torch.onnx.export(model, input_tensor, output_onnx_file, export_params=True, opset_version=11, do_constant_folding=True,
                  input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'},
                                                                                'output': {0: 'batch_size'}})
