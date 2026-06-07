from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .eda import get_pneumonia_chart, get_retinopathy_chart, get_skin_cancer_chart
from predictor.models import Prediction, DatasetStats

@login_required(login_url='/accounts/login/')
def dashboard_view(request):
    """
    Render the analytics and EDA dashboard with active dataset distributions 
    and personal prediction logs.
    """
    # Pre-populate dataset metrics if database table is empty
    if not DatasetStats.objects.exists():
        DatasetStats.objects.create(
            disease_type='pneumonia',
            total_images=5856,
            positive_cases=4273,
            negative_cases=1583,
            model_accuracy=92.5
        )
        DatasetStats.objects.create(
            disease_type='retinopathy',
            total_images=3662,
            positive_cases=1805,
            negative_cases=1857,
            model_accuracy=88.0
        )
        DatasetStats.objects.create(
            disease_type='skin_cancer',
            total_images=10015,
            positive_cases=1113,
            negative_cases=8902,
            model_accuracy=85.3
        )

    # Load interactive Plotly charts
    pneumonia_div = get_pneumonia_chart()
    retinopathy_div = get_retinopathy_chart()
    skin_cancer_div = get_skin_cancer_chart()
    
    # Get user prediction history
    history = Prediction.objects.filter(user=request.user)[:10]
    
    # Get general dataset stats
    stats = DatasetStats.objects.all()
    
    context = {
        'pneumonia_chart': pneumonia_div,
        'retinopathy_chart': retinopathy_div,
        'skin_cancer_chart': skin_cancer_div,
        'history': history,
        'stats': stats
    }
    return render(request, 'analysis/dashboard.html', context)
