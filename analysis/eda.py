import os
import pandas as pd
import plotly.express as px
import plotly.offline as opy

base_dir = r"C:\Study\PN\python medical project\mediscan\datasets"

def get_pneumonia_chart():
    """Generate bar chart for Pneumonia Chest X-Ray class splits."""
    train_normal = len(os.listdir(os.path.join(base_dir, "chest_xray", "train", "NORMAL"))) if os.path.exists(os.path.join(base_dir, "chest_xray", "train", "NORMAL")) else 0
    train_pneumonia = len(os.listdir(os.path.join(base_dir, "chest_xray", "train", "PNEUMONIA"))) if os.path.exists(os.path.join(base_dir, "chest_xray", "train", "PNEUMONIA")) else 0
    test_normal = len(os.listdir(os.path.join(base_dir, "chest_xray", "test", "NORMAL"))) if os.path.exists(os.path.join(base_dir, "chest_xray", "test", "NORMAL")) else 0
    test_pneumonia = len(os.listdir(os.path.join(base_dir, "chest_xray", "test", "PNEUMONIA"))) if os.path.exists(os.path.join(base_dir, "chest_xray", "test", "PNEUMONIA")) else 0
    
    data = {
        'Class': ['Normal', 'Pneumonia', 'Normal', 'Pneumonia'],
        'Count': [train_normal, train_pneumonia, test_normal, test_pneumonia],
        'Split': ['Train', 'Train', 'Test', 'Test']
    }
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, x='Class', y='Count', color='Split', barmode='group',
        title='Chest X-Ray Split Distribution',
        color_discrete_sequence=['#00d4ff', '#7f8fa4'],
        template='plotly_dark'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="DM Sans, sans-serif", color="#ffffff"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return opy.plot(fig, auto_open=False, output_type='div')

def get_retinopathy_chart():
    """Generate pie chart for Diabetic Retinopathy severity."""
    csv_path = os.path.join(base_dir, "retinopathy", "train.csv")
    if not os.path.exists(csv_path):
        return "<p class='no-data'>No Retinopathy Data Found</p>"
    
    try:
        df = pd.read_csv(csv_path)
        mapping = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'}
        df['Severity'] = df['diagnosis'].map(mapping)
        df_counts = df['Severity'].value_counts().reset_index()
        df_counts.columns = ['Severity', 'Count']
        
        fig = px.pie(
            df_counts, names='Severity', values='Count',
            title='Retinopathy Severity splits',
            color_discrete_sequence=px.colors.sequential.Cyan,
            template='plotly_dark'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="DM Sans, sans-serif", color="#ffffff"),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return opy.plot(fig, auto_open=False, output_type='div')
    except Exception as e:
        return f"<p class='error-msg'>Error loading retinopathy chart: {str(e)}</p>"

def get_skin_cancer_chart():
    """Generate horizontal bar chart for Skin Cancer lesion types."""
    csv_path = os.path.join(base_dir, "skin_cancer", "HAM10000_metadata.csv")
    if not os.path.exists(csv_path):
        return "<p class='no-data'>No Skin Cancer Data Found</p>"
    
    try:
        df = pd.read_csv(csv_path)
        mapping = {
            'mel': 'Melanoma (Malignant)',
            'nv': 'Melanocytic Nevi',
            'bkl': 'Benign Keratosis',
            'df': 'Dermatofibroma',
            'vasc': 'Vascular Lesions',
            'bcc': 'Basal Cell Carcinoma',
            'akiec': 'Actinic Keratoses'
        }
        df['Diagnosis Type'] = df['dx'].map(mapping).fillna(df['dx'])
        df_counts = df['Diagnosis Type'].value_counts().reset_index()
        df_counts.columns = ['Diagnosis Type', 'Count']
        
        fig = px.bar(
            df_counts, x='Count', y='Diagnosis Type', orientation='h',
            title='Skin Cancer Lesion Categories (HAM10000)',
            color_discrete_sequence=['#00d4ff'],
            template='plotly_dark'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="DM Sans, sans-serif", color="#ffffff"),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return opy.plot(fig, auto_open=False, output_type='div')
    except Exception as e:
        return f"<p class='error-msg'>Error loading skin cancer chart: {str(e)}</p>"
