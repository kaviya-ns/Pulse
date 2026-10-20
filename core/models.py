from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User


class Post(models.Model):
    """A single 'document' — this is both an IR document (indexed,
    retrieved, ranked) AND a node in your network graph (cited/linked
    by other posts)."""
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='posts')
    title = models.CharField(max_length=300)
    content = models.TextField()
    topic = models.CharField(max_length=50, blank=True) 
    created_at = models.DateTimeField(auto_now_add=True)

    # Precomputed fields, filled in later by the IR pipeline
    quality_score = models.FloatField(default=0.0)          # feeds Bianconi-Barabási fitness
    pagerank_score = models.FloatField(default=0.0)
    quality_pagerank_score = models.FloatField(default=0.0)

    def __str__(self):
        return self.title


class Link(models.Model):
    """A directed edge: source_post cites/links to target_post.
    This table IS your graph's edge list — NetworkX reads it for
    the network-science side, IR link-analysis reads it too."""
    source_post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='outgoing_links')
    target_post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='incoming_links')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('source_post', 'target_post')


class Interaction(models.Model):
    """User-post interactions — your collaborative filtering data."""
    KIND_CHOICES = [('view', 'View'), ('like', 'Like'), ('share', 'Share')]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    post = models.ForeignKey(Post, on_delete=models.CASCADE)
    kind = models.CharField(max_length=10, choices=KIND_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)