from django.core.management.base import BaseCommand
from predictor.models import DatasetStats

class Command(BaseCommand):
    help = 'Seeds dataset statistics and accuracy metrics for the medical screening models.'

    def handle(self, *args, **options):
        stats_data = [
            {
                'disease_type': 'pneumonia',
                'total_images': 5216,
                'positive_cases': 3875,
                'negative_cases': 1341,
                'model_accuracy': 93.4
            },
            {
                'disease_type': 'retinopathy',
                'total_images': 3662,
                'positive_cases': 1805,
                'negative_cases': 1857,
                'model_accuracy': 89.1
            },
            {
                'disease_type': 'skin_cancer',
                'total_images': 10015,
                'positive_cases': 1113,
                'negative_cases': 8902,
                'model_accuracy': 87.6
            }
        ]

        for item in stats_data:
            DatasetStats.objects.update_or_create(
                disease_type=item['disease_type'],
                defaults={
                    'total_images': item['total_images'],
                    'positive_cases': item['positive_cases'],
                    'negative_cases': item['negative_cases'],
                    'model_accuracy': item['model_accuracy']
                }
            )

        self.stdout.write(self.style.SUCCESS('Stats seeded successfully'))
