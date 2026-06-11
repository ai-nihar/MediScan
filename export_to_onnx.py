import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import os

class DiseaseModel(nn.Module):
    def __init__(self):
        super(DiseaseModel, self).__init__()
        # Initialize backbone structure
        self.base = mobilenet_v2(weights=None)
        
        # Modify the classifier head to match our trained PyTorch architecture
        num_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base(x)

def export_model(pth_path, onnx_path):
    # Ensure directory exists
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # Load model
    model = DiseaseModel()
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()
    
    # Dummy input representing a batch of 1 RGB image of size 224x224
    dummy = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX format
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11
    )
    print(f"Exported: {onnx_path}")

if __name__ == '__main__':
    export_model('ml_models/pneumonia/pneumonia_model.pth', 'ml_models/pneumonia/pneumonia_model.onnx')
    export_model('ml_models/retinopathy/retinopathy_model.pth', 'ml_models/retinopathy/retinopathy_model.onnx')
    export_model('ml_models/skin_cancer/skin_cancer_model.pth', 'ml_models/skin_cancer/skin_cancer_model.onnx')
