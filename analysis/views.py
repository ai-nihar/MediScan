import json
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from .eda import (
    get_dataset_overview,
    get_class_distribution,
    get_model_metrics,
    get_training_history,
    get_prediction_stats
)
from predictor.models import DatasetStats, Prediction

class DashboardView(LoginRequiredMixin, TemplateView):
    """
    Renders the interactive EDA and analytics dashboard.
    """
    template_name = 'analysis/dashboard.html'
    login_url = '/accounts/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user

        # Ensure DatasetStats exist
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

        # Call all eda.py functions
        dataset_overview = get_dataset_overview()
        class_distribution = get_class_distribution()
        model_metrics = get_model_metrics()
        prediction_stats = get_prediction_stats(user)

        # Retrieve training histories for all 3 diseases to allow dynamic JS switching
        history_data = {
            'pneumonia': get_training_history('pneumonia'),
            'retinopathy': get_training_history('retinopathy'),
            'skin_cancer': get_training_history('skin_cancer')
        }

        # Inject as JSON-serialized strings so JavaScript can read them directly
        context['dataset_overview_json'] = json.dumps(dataset_overview)
        context['class_distribution_json'] = json.dumps(class_distribution)
        context['model_metrics_json'] = json.dumps(model_metrics)
        context['history_data_json'] = json.dumps(history_data)
        context['prediction_stats_json'] = json.dumps(prediction_stats)

        # Non-serialized stats for simple django loops if needed
        context['stats'] = DatasetStats.objects.all()
        history = Prediction.objects.filter(user=user).order_by('-created_at')
        context['history_count'] = history.count()
        context['history'] = history[:10]

        return context
