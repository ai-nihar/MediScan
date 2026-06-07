from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.views.generic import TemplateView, DetailView, ListView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.utils import timezone
import datetime
from .models import Prediction, DatasetStats
from .utils.predict import predictor
from .utils.prevention import get_prevention_tips

class DashboardView(LoginRequiredMixin, TemplateView):
    """
    Renders the clinician dashboard showing recent predictions, 
    general system metrics, and dataset stats.
    """
    template_name = 'predictor/dashboard.html'
    login_url = '/accounts/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        # User predictions queries
        user_preds = Prediction.objects.filter(user=user)
        context['recent_predictions'] = user_preds.order_by('-created_at')[:5]
        context['total_predictions'] = user_preds.count()
        
        # 1. Diseases detected (positive cases)
        neg_classes = ["Normal", "No Diabetic Retinopathy", "Benign", "Error", "Processing"]
        context['diseases_detected'] = user_preds.exclude(result__in=neg_classes).count()
        
        # 2. Scans performed in the last 7 days
        seven_days_ago = timezone.now() - datetime.timedelta(days=7)
        context['scans_this_week'] = user_preds.filter(created_at__gte=seven_days_ago).count()
        
        # 3. Account creation date
        context['account_created'] = user.date_joined
        
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
            
        context['stats'] = DatasetStats.objects.all()
        return context


class UploadView(LoginRequiredMixin, View):
    """
    Handles rendering the upload diagnostic scan form (GET) 
    and processing the uploaded scan via the PyTorch model (POST).
    """
    login_url = '/accounts/login/'

    def get(self, request):
        return render(request, 'predictor/upload.html')

    def post(self, request):
        disease_type = request.POST.get('disease_type')
        image_file = request.FILES.get('image')

        if not disease_type or not image_file:
            return render(request, 'predictor/upload.html', {
                'error': 'Please provide both a disease type and an image.'
            })

        # Save placeholder record to get image file path on disk
        prediction = Prediction(
            user=request.user,
            disease_type=disease_type,
            image=image_file,
            result='Processing',
            confidence=0.0,
            prevention_tips=''
        )
        prediction.save()

        try:
            # Get detailed prediction dictionary from PyTorch ModelPredictor
            res = predictor.predict(prediction.image.path, disease_type)
            result = res['result']
            confidence = res['confidence']
            is_positive = res['is_positive']

            # Retrieve prevention tips dictionary based on positive/negative status
            tips_dict = get_prevention_tips(disease_type, is_positive)

            # Format the dictionary into a clean, structured bullet-point layout
            formatted_tips = f"Severity Level: {tips_dict['severity']}\n\nImmediate Steps:\n"
            for step in tips_dict['immediate_steps']:
                formatted_tips += f"- {step}\n"
            formatted_tips += "\nLifestyle Recommendations:\n"
            for item in tips_dict['lifestyle']:
                formatted_tips += f"- {item}\n"
            formatted_tips += f"\nMedications:\n- {tips_dict['medications']}\n"
            formatted_tips += f"\nFollow-up:\n- {tips_dict['follow_up']}\n"
            formatted_tips += f"\nDisclaimer: {tips_dict['disclaimer']}"

            # Update prediction record
            prediction.result = result
            prediction.confidence = confidence
            prediction.prevention_tips = formatted_tips
            prediction.save()
        except Exception as e:
            prediction.result = 'Error'
            prediction.prevention_tips = f"Error during model processing: {str(e)}"
            prediction.save()

        return redirect('predictor:result', pk=prediction.pk)


class ResultView(LoginRequiredMixin, DetailView):
    """
    Renders the detailed diagnostic results page for a single Prediction.
    """
    model = Prediction
    template_name = 'predictor/result.html'
    context_object_name = 'prediction'
    login_url = '/accounts/login/'

    def get_queryset(self):
        # Users can only view their own predictions
        return Prediction.objects.filter(user=self.request.user)


class HistoryView(LoginRequiredMixin, ListView):
    """
    Displays the list of all predictions made by the authenticated user, paginated by 10.
    """
    model = Prediction
    template_name = 'predictor/history.html'
    context_object_name = 'predictions'
    paginate_by = 10
    login_url = '/accounts/login/'

    def get_queryset(self):
        return Prediction.objects.filter(user=self.request.user).order_by('-created_at')
