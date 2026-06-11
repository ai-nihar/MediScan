import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from django.conf import settings
from .preprocess import get_class_label

class ModelPredictor:
    def __init__(self):
        # Cache loaded ONNX sessions in a dict
        self.sessions = {}
        
        # MODEL_PATHS dict pointing to each .onnx file
        self.MODEL_PATHS = {
            'pneumonia': os.path.join(settings.BASE_DIR, 'ml_models', 'pneumonia', 'pneumonia_model.onnx'),
            'retinopathy': os.path.join(settings.BASE_DIR, 'ml_models', 'retinopathy', 'retinopathy_model.onnx'),
            'skin_cancer': os.path.join(settings.BASE_DIR, 'ml_models', 'skin_cancer', 'skin_cancer_model.onnx'),
        }

    def _get_session(self, disease_type):
        if disease_type not in self.sessions:
            model_path = self.MODEL_PATHS.get(disease_type)
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"ONNX Model file for {disease_type} not found at {model_path}. "
                    f"Please run the export script export_to_onnx.py first."
                )
            
            # Load ONNX Inference Session using CPU provider
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.sessions[disease_type] = session
        return self.sessions[disease_type]

    def _preprocess(self, image_path, target_size=(224, 224)):
        # 1. Open image with Pillow and convert to RGB (handles grayscale/RGBA)
        img = Image.open(image_path).convert('RGB')
        
        # 2. Resize to 224x224
        img = img.resize(target_size)
        
        # 3. Convert to float32 numpy array and scale to [0.0, 1.0] (matching ToTensor())
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # 4. Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # 5. Transpose from (H, W, C) to (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # 6. Add batch dimension: (1, C, H, W)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path, disease_type):
        """
        Run ONNX model inference, return detailed analysis metadata.
        """
        # Load session (lazy loading from cache)
        session = self._get_session(disease_type)
        
        # Preprocess the image to a normalized numpy array
        img_array = self._preprocess(image_path)
        
        # Run inference
        output = session.run(None, {'input': img_array})[0]
        
        # Note: The PyTorch MobileNetV2 classifier head we exported already ends with
        # a Sigmoid activation layer, so the ONNX output[0][0] is already the 
        # final sigmoid probability in the range [0, 1].
        confidence = float(output[0][0])
        
        # Threshold binary prediction at 0.5
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

# Create the singleton instance
predictor = ModelPredictor()

# Backwards compatibility wrapper for views/APIs
def predict_disease(disease_type, image_path):
    res = predictor.predict(image_path, disease_type)
    return res['result'], res['confidence']
