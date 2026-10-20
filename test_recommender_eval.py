import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pulse_project.settings')
django.setup()

from collections import Counter
from django.contrib.auth.models import User
from core.models import Post, Interaction
from recommend.recommender import recommend_content_based, recommend_collaborative

KIND_WEIGHTS = {'view': 1, 'like': 3, 'share': 5}


def dominant_topic(user_id):
    """Proxy ground truth: the topic this user engaged with most, weighted
    by interaction strength (a 'share' counts more than a 'view')."""
    interactions = Interaction.objects.filter(user_id=user_id).select_related('post')
    weighted = Counter()
    for i in interactions:
        weighted[i.post.topic] += KIND_WEIGHTS[i.kind]
    return weighted.most_common(1)[0][0] if weighted else None


def precision_at_k(recommendations, target_topic):
    if not recommendations:
        return None
    hits = sum(1 for post_id, _ in recommendations if Post.objects.get(id=post_id).topic == target_topic)
    return hits / len(recommendations)


users = User.objects.filter(username__startswith='reader_')

content_scores, collab_scores = [], []
for user in users:
    target = dominant_topic(user.id)
    if target is None:
        continue

    cb = recommend_content_based(user.id, top_n=5)
    cf = recommend_collaborative(user.id, top_n=5)

    p_cb = precision_at_k(cb, target)
    p_cf = precision_at_k(cf, target)

    if p_cb is not None:
        content_scores.append(p_cb)
    if p_cf is not None:
        collab_scores.append(p_cf)

print(f"Content-based  precision@5: {sum(content_scores)/len(content_scores):.3f}  (n={len(content_scores)})")
print(f"Collaborative  precision@5: {sum(collab_scores)/len(collab_scores):.3f}  (n={len(collab_scores)})")
print(f"Random baseline expected:   {1/6:.3f}  (1 in 6 topics)")