import torch
import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Open an image, convert to RGB, resize, normalize, and add batch dimension using PyTorch.
    """
    # 1. Opens image using PIL (Pillow)
    img = Image.open(image_path)
    
    # 2. Converts to RGB (handles grayscale X-rays too)
    img = img.convert('RGB')
    
    # 3. Transforms image: resize, convert to tensor (0-1), and normalize matching MobileNetV2 standards
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img)
    
    # 4. Adds batch dimension: shape [1, 3, 224, 224]
    img_tensor = img_tensor.unsqueeze(0)
    
    # 5. Returns the preprocessed tensor
    return img_tensor

def get_class_label(disease_type, prediction_index):
    """
    Return human-readable label based on disease category and predicted class index.
    """
    idx = int(prediction_index)
    
    if disease_type == 'pneumonia':
        return 'Normal' if idx == 0 else 'Pneumonia Detected'
    elif disease_type == 'retinopathy':
        return 'No Diabetic Retinopathy' if idx == 0 else 'Diabetic Retinopathy Detected'
    elif disease_type == 'skin_cancer':
        return 'Benign' if idx == 0 else 'Malignant (Melanoma)'
    else:
        raise ValueError(f"Unknown disease type: {disease_type}")
