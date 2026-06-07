from django.urls import path
from .views import upload_view, result_view

app_name = 'predictor'

urlpatterns = [
    path('', upload_view, name='upload'),
    path('result/<int:pk>/', result_view, name='result'),
]
