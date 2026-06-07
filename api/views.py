from rest_framework import generics
from rest_framework.permissions import IsAuthenticated, AllowAny
from predictor.models import Prediction, DatasetStats
from .serializers import PredictionSerializer, DatasetStatsSerializer
from .permissions import IsOwner
from predictor.utils.predict import predict_disease
from predictor.utils.prevention import get_prevention_tips

class PredictionListCreateAPIView(generics.ListCreateAPIView):
    serializer_class = PredictionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Limit predictions returned to those belonging to the authenticated user."""
        return Prediction.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        """Save the prediction record, trigger neural model inference, and save output."""
        prediction = serializer.save(user=self.request.user)
        
        # Run neural network inference on the newly uploaded image file
        try:
            result, confidence = predict_disease(
                prediction.disease_type,
                prediction.image.path
            )
            tips = get_prevention_tips(prediction.disease_type, result)
            
            prediction.result = result
            prediction.confidence = confidence
            prediction.prevention_tips = tips
            prediction.save()
        except Exception as e:
            prediction.result = 'Error'
            prediction.prevention_tips = f"Error during model processing: {str(e)}"
            prediction.save()

class PredictionDetailAPIView(generics.RetrieveDestroyAPIView):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    permission_classes = [IsAuthenticated, IsOwner]

class DatasetStatsListAPIView(generics.ListAPIView):
    queryset = DatasetStats.objects.all()
    serializer_class = DatasetStatsSerializer
    permission_classes = [AllowAny]
