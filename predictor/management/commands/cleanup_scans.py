import os
from django.core.management.base import BaseCommand
from django.utils import timezone
import datetime
from predictor.models import Prediction

class Command(BaseCommand):
    help = 'Deletes physical files of scan predictions older than a specified number of days, leaving database metadata intact.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Purge scan files older than this number of days (default: 30)'
        )

    def handle(self, *args, **options):
        days = options['days']
        cutoff_date = timezone.now() - datetime.timedelta(days=days)
        
        # Query predictions with image files older than cutoff
        old_predictions = Prediction.objects.filter(
            created_at__lt=cutoff_date
        ).exclude(image='')

        count = 0
        total_space_freed = 0

        for pred in old_predictions:
            if pred.image and os.path.exists(pred.image.path):
                try:
                    # Get file size to track space gains
                    file_size = os.path.getsize(pred.image.path)
                    total_space_freed += file_size
                    
                    # Delete the file from the filesystem
                    os.remove(pred.image.path)
                    
                    # Set the image field to empty string in the DB
                    pred.image = ''
                    pred.save(update_fields=['image'])
                    count += 1
                except Exception as e:
                    self.stderr.write(f"Error deleting scan file for prediction ID {pred.id}: {e}")

        # Convert bytes to megabytes
        space_mb = round(total_space_freed / (1024 * 1024), 2)
        
        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully deleted {count} clinical scan images older than {days} days. "
                f"Freed {space_mb} MB of server storage."
            )
        )
