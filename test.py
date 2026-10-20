import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pulse_project.settings')
django.setup()

from django.db import models
from core.models import Interaction

print(Interaction.objects.count())
print(Interaction.objects.values('kind').annotate(n=models.Count('kind')))