import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from django.conf import settings
from .preprocess import preprocess_image, get_class_label

class DiseaseModel(nn.Module):
    def __init__(self):
        super(DiseaseModel, self).__init__()
        # Initialize backbone structure (no need to download pre-trained weights since we load our state dict)
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


class ModelPredictor:
    def __init__(self):
        # Cache loaded models in a dict
        self.models = {}
        
        # MODEL_PATHS dict pointing to each .pth file
        self.MODEL_PATHS = {
            'pneumonia': os.path.join(settings.BASE_DIR, 'ml_models', 'pneumonia', 'pneumonia_model.pth'),
            'retinopathy': os.path.join(settings.BASE_DIR, 'ml_models', 'retinopathy', 'retinopathy_model.pth'),
            'skin_cancer': os.path.join(settings.BASE_DIR, 'ml_models', 'skin_cancer', 'skin_cancer_model.pth'),
        }

    def _get_model(self, disease_type, device):
        if disease_type not in self.models:
            model_path = self.MODEL_PATHS.get(disease_type)
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file for {disease_type} not found at {model_path}. "
                    f"Please run the corresponding training script in ml_models/{disease_type}/train.py first."
                )
            
            # Instantiate and load model
            model = DiseaseModel()
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            self.models[disease_type] = model
        return self.models[disease_type]

    def predict(self, image_path, disease_type):
        """
        Run model inference, return detailed analysis metadata.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model (lazy loading from cache)
        model = self._get_model(disease_type, device)
        
        # Call preprocess_image from preprocess.py (returns PyTorch tensor)
        processed_tensor = preprocess_image(image_path).to(device)
        
        # Run prediction (inference mode)
        with torch.no_grad():
            output = model(processed_tensor)
            confidence = float(output.item())  # Sigmoid output is shape [1, 1], item() gets the float
        
        # Threshold binary prediction
        result_index = 1 if confidence >= 0.5 else 0
        result_label = get_class_label(disease_type, result_index)
        
        # Standardize confidence reporting relative to the classified output
        confidence_pct = confidence * 100 if result_index == 1 else (1.0 - confidence) * 100
            
        return {
            'result': result_label,
            'confidence': round(confidence_pct, 2),
            'is_positive': result_index == 1,
            'raw_score': confidence
        }

# Create a singleton instance
predictor = ModelPredictor()

# Backwards compatibility wrapper for views/APIs
def predict_disease(disease_type, image_path):
    res = predictor.predict(image_path, disease_type)
    return res['result'], res['confidence']


