import os
import sys
import tempfile
import numpy as np
from PIL import Image

# Initialize Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mediscan.settings')
import django
try:
    django.setup()
    print("[OK] Django imported and settings are valid.")
except Exception as e:
    print(f"[ERROR] Django initialization failed: {e}")
    sys.exit(1)

from django.conf import settings
from predictor.utils.predict import predictor, DiseaseModel
import torch

def test_pipeline():
    print("\n--- Pipeline Verification Start ---")
    
    # 1. Check all 3 model files exist
    print("\n1. Checking model files on disk:")
    model_paths = predictor.MODEL_PATHS
    all_exist = True
    for disease, path in model_paths.items():
        if os.path.exists(path):
            print(f"  [OK] {disease.capitalize()} model file found at: {path}")
        else:
            print(f"  [ERROR] {disease.capitalize()} model file NOT found at: {path}")
            all_exist = False
            
    if not all_exist:
        print("\n[WARN] Some model files are missing. Make sure you train them first or place them in their folders.")

    # 2 & 3. Load each model and print its structure/summary
    print("\n2. Loading models and running dummy inference:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using inference device: {device}")
    
    # Create a dummy 224x224 image file to test prediction preprocessing & pipeline
    dummy_img_path = os.path.join(tempfile.gettempdir(), 'dummy_clinical_scan.png')
    dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(dummy_array)
    img.save(dummy_img_path)
    
    for disease in ['pneumonia', 'retinopathy', 'skin_cancer']:
        try:
            # Lazy load model (loads from state dict .pth)
            model = predictor._get_model(disease, device)
            print(f"\n  [ {disease.capitalize()} Model Loaded ]")
            print("  Structure:")
            print(f"    - Backbone: MobileNetV2")
            print(f"    - Classifier: {model.base.classifier}")
            
            # Run test prediction using predictor pipeline
            res = predictor.predict(dummy_img_path, disease)
            print(f"  [OK] [{disease.capitalize()}] loaded and working. Test result: {res}")
        except Exception as e:
            print(f"  [ERROR] [{disease.capitalize()}] failed: {e}")
            
    # Clean up dummy image
    if os.path.exists(dummy_img_path):
        os.remove(dummy_img_path)
        
    print("\n--- Pipeline Verification Complete ---")

if __name__ == '__main__':
    test_pipeline()
