from rest_framework import serializers
from django.contrib.auth import get_user_model
from predictor.models import Prediction, DatasetStats

User = get_user_model()

class PredictionSerializer(serializers.ModelSerializer):
    user_name = serializers.SerializerMethodField()

    class Meta:
        model = Prediction
        fields = [
            'id',
            'user',
            'user_name',
            'disease_type',
            'image',
            'result',
            'confidence',
            'prevention_tips',
            'processing_error',
            'created_at'
        ]
        read_only_fields = ['id', 'user', 'result', 'confidence', 'prevention_tips', 'processing_error', 'created_at']

    def get_user_name(self, obj):
        return obj.user.username


class PredictionCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['disease_type', 'image']


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'date_joined']
        read_only_fields = ['id', 'username', 'email', 'date_joined']


class DatasetStatsSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetStats
        fields = '__all__'
