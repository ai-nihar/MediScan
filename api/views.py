from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import PermissionDenied
from django.contrib.auth import get_user_model
from predictor.models import Prediction, DatasetStats
from .serializers import (
    PredictionSerializer, 
    PredictionCreateSerializer, 
    UserSerializer, 
    DatasetStatsSerializer
)
from predictor.utils.predict import predictor
from predictor.utils.prevention import get_prevention_tips

User = get_user_model()

class PredictionViewSet(viewsets.ModelViewSet):
    """
    ViewSet for handling Prediction CRUD.
    List and retrieve only return predictions belonging to the authenticated user.
    On creation, triggers model inference and formats prevention guidelines.
    """
    permission_classes = [IsAuthenticated]
    filterset_fields = ['disease_type', 'created_at']

    def get_queryset(self):
        queryset = Prediction.objects.filter(user=self.request.user)
        
        # Add manual query parameter filtering to support requests
        # even if django_filters is not configured in DRF.
        disease_type = self.request.query_params.get('disease_type')
        if disease_type:
            queryset = queryset.filter(disease_type=disease_type)
            
        created_at = self.request.query_params.get('created_at')
        if created_at:
            # Match date component
            queryset = queryset.filter(created_at__date=created_at)
            
        return queryset

    def get_serializer_class(self):
        if self.action == 'create':
            return PredictionCreateSerializer
        return PredictionSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Save record with authenticated user and placeholder defaults to get the image path on disk
        prediction = serializer.save(
            user=self.request.user,
            result='Processing',
            confidence=0.0,
            prevention_tips=''
        )
        
        try:
            # Trigger PyTorch model prediction
            res = predictor.predict(prediction.image.path, prediction.disease_type)
            result = res['result']
            confidence = res['confidence']
            is_positive = res['is_positive']
            
            # Retrieve guidelines and format them into structured text
            tips_dict = get_prevention_tips(prediction.disease_type, is_positive)
            formatted_tips = f"Severity Level: {tips_dict['severity']}\n\nImmediate Steps:\n"
            for step in tips_dict['immediate_steps']:
                formatted_tips += f"- {step}\n"
            formatted_tips += "\nLifestyle Recommendations:\n"
            for item in tips_dict['lifestyle']:
                formatted_tips += f"- {item}\n"
            formatted_tips += f"\nMedications:\n- {tips_dict['medications']}\n"
            formatted_tips += f"\nFollow-up:\n- {tips_dict['follow_up']}\n"
            formatted_tips += f"\nDisclaimer: {tips_dict['disclaimer']}"
            
            # Update prediction instance with output results
            prediction.result = result
            prediction.confidence = confidence
            prediction.prevention_tips = formatted_tips
            prediction.save()
        except FileNotFoundError as e:
            prediction.result = 'Error'
            prediction.prevention_tips = 'Model file not found. Please ensure the trained model is present in ml_models/. Contact the administrator.'
            prediction.processing_error = str(e)
            prediction.save()
        except Exception as e:
            prediction.result = 'Error'
            prediction.prevention_tips = 'An unexpected error occurred during analysis. Please try again.'
            prediction.processing_error = str(e)
            prediction.save()
            
        # Serialize the updated prediction using the full PredictionSerializer
        response_serializer = PredictionSerializer(prediction, context={'request': request})
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for User Profiles.
    Only allows users to retrieve their own profile.
    """
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # Limits list results to the authenticated user only
        return User.objects.filter(id=self.request.user.id)

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance != request.user:
            raise PermissionDenied("You can only retrieve your own profile.")
        return super().retrieve(request, *args, **kwargs)


class DatasetStatsViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for Dataset Statistics metrics.
    """
    queryset = DatasetStats.objects.all()
    serializer_class = DatasetStatsSerializer
    permission_classes = [IsAuthenticated]
