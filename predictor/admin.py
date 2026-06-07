from django.contrib import admin
from .models import Prediction, DatasetStats

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('user', 'disease_type', 'result', 'confidence', 'created_at')
    list_filter = ('disease_type', 'result', 'created_at')
    search_fields = ('user__username', 'result', 'prevention_tips')
    readonly_fields = ('created_at',)

@admin.register(DatasetStats)
class DatasetStatsAdmin(admin.ModelAdmin):
    list_display = ('disease_type', 'total_images', 'positive_cases', 'negative_cases', 'model_accuracy', 'last_updated')
    search_fields = ('disease_type',)
    readonly_fields = ('last_updated',)

