from django.urls import path
from .views import (
    PredictionListCreateAPIView,
    PredictionDetailAPIView,
    DatasetStatsListAPIView
)

app_name = 'api'

urlpatterns = [
    path('predictions/', PredictionListCreateAPIView.as_view(), name='prediction_list_create'),
    path('predictions/<int:pk>/', PredictionDetailAPIView.as_view(), name='prediction_detail'),
    path('stats/', DatasetStatsListAPIView.as_view(), name='stats_list'),
]
