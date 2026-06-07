from django.urls import path
from .views import UploadView, ResultView, DashboardView, HistoryView

app_name = 'predictor'

urlpatterns = [
    path('', UploadView.as_view(), name='upload'),
    path('result/<int:pk>/', ResultView.as_view(), name='result'),
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
    path('history/', HistoryView.as_view(), name='history'),
]
