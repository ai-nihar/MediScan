from rest_framework import serializers
from predictor.models import Prediction, DatasetStats

class PredictionSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source='user.username')
    result = serializers.ReadOnlyField()
    confidence = serializers.ReadOnlyField()
    prevention_tips = serializers.ReadOnlyField()
    created_at = serializers.ReadOnlyField()

    class Meta:
        model = Prediction
        fields = [
            'id',
            'user',
            'disease_type',
            'image',
            'result',
            'confidence',
            'prevention_tips',
            'created_at'
        ]

class DatasetStatsSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetStats
        fields = [
            'disease_type',
            'total_images',
            'positive_cases',
            'negative_cases',
            'model_accuracy',
            'last_updated'
        ]
