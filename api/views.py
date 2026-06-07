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
            from predictor.utils.predict import predictor
            from predictor.utils.prevention import get_prevention_tips
            
            # Get detailed prediction dictionary
            res = predictor.predict(prediction.image.path, prediction.disease_type)
            result = res['result']
            confidence = res['confidence']
            is_positive = res['is_positive']
            
            # Retrieve prevention tips dictionary based on classification status
            tips_dict = get_prevention_tips(prediction.disease_type, is_positive)
            
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

class PredictionDetailAPIView(generics.RetrieveDestroyAPIView):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    permission_classes = [IsAuthenticated, IsOwner]

class DatasetStatsListAPIView(generics.ListAPIView):
    queryset = DatasetStats.objects.all()
    serializer_class = DatasetStatsSerializer
    permission_classes = [AllowAny]
