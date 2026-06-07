from django.urls import path, include
from django.http import JsonResponse
from rest_framework.routers import DefaultRouter
from .views import PredictionViewSet, UserViewSet, DatasetStatsViewSet

app_name = 'api'

router = DefaultRouter()
router.register('predictions', PredictionViewSet, basename='prediction')
router.register('users', UserViewSet, basename='user')
router.register('stats', DatasetStatsViewSet, basename='stats')

urlpatterns = [
    path('', include(router.urls)),
    path('health/', lambda r: JsonResponse({'status': 'ok', 'version': '1.0'})),
]
