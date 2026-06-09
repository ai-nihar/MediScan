from django.db import models
from django.conf import settings

DISEASE_CHOICES = [
    ('pneumonia', 'Pneumonia'),
    ('retinopathy', 'Diabetic Retinopathy'),
    ('skin_cancer', 'Skin Cancer'),
]

class Prediction(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='predictions'
    )
    disease_type = models.CharField(max_length=50, choices=DISEASE_CHOICES)
    image = models.ImageField(upload_to='predictions/')
    result = models.CharField(max_length=100)
    confidence = models.FloatField()
    prevention_tips = models.TextField()
    processing_error = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.get_disease_type_display()} - {self.result}"

class DatasetStats(models.Model):
    disease_type = models.CharField(max_length=50, unique=True)
    total_images = models.IntegerField()
    positive_cases = models.IntegerField()
    negative_cases = models.IntegerField()
    model_accuracy = models.FloatField()
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.disease_type} Stats"

