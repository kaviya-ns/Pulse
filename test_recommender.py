import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pulse_project.settings')
django.setup()

from django.contrib.auth.models import User
from core.models import Post
from recommend.recommender import recommend_content_based, recommend_collaborative

user = User.objects.filter(username__startswith='reader_').first()
print(f"Recommending for {user.username}\n")

print("Content-based:")
for post_id, score in recommend_content_based(user.id, top_n=5):
    post = Post.objects.get(id=post_id)
    print(f"  {post.title[:50]:50} ({post.topic}) score={score:.3f}")

print("\nCollaborative filtering:")
for post_id, score in recommend_collaborative(user.id, top_n=5):
    post = Post.objects.get(id=post_id)
    print(f"  {post.title[:50]:50} ({post.topic}) score={score:.3f}")