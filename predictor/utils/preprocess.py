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
