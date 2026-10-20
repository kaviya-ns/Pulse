import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pulse_project.settings')
django.setup()

from django.contrib.auth.models import User
from core.models import Interaction

user = User.objects.get(username='reader_0')
topic_counts = (
    Interaction.objects.filter(user=user)
    .values('post__topic')
    .distinct()
)
print("reader_0 interacted with topics:", [t['post__topic'] for t in topic_counts])