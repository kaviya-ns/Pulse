import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pulse_project.settings')
django.setup()

from django.contrib.auth.models import User
from core.models import Post
from retrieval.classification import train_quality_classifier, quality_to_class, extract_keywords
from retrieval.clustering import run_clustering, top_terms_per_cluster
from recommend.recommender import recommend_content_based, recommend_collaborative

print("=== Classification ===")
result = train_quality_classifier()
clf, vectorizer, scaler = result['classifier'], result['vectorizer'], result['scaler']

# 9 & 10: pick one real high-quality and one low-quality post, predict manually
high_post = Post.objects.filter(quality_score__gt=0.66).first()
low_post = Post.objects.filter(quality_score__lt=0.33).first()

def predict_class(post):
    from scipy.sparse import hstack, csr_matrix
    import numpy as np
    text = f"{post.title} {post.content}"
    X_tfidf = vectorizer.transform([text])
    words = post.content.split()
    numeric = np.array([[len(words), len(set(words))/len(words) if words else 0,
                          np.mean([len(w) for w in words]) if words else 0]])
    X_numeric = scaler.transform(numeric)
    X = hstack([X_tfidf, csr_matrix(X_numeric)])
    return clf.predict(X)[0]

print(f"#9  High-quality post ({high_post.quality_score:.2f}) predicted:", predict_class(high_post))
print(f"#10 Low-quality post ({low_post.quality_score:.2f}) predicted:", predict_class(low_post))

# 11: reproducibility
acc1 = train_quality_classifier()['accuracy']
acc2 = train_quality_classifier()['accuracy']
print(f"#11 Accuracy run1={acc1:.4f} run2={acc2:.4f} match={acc1 == acc2}")

print("\n=== Information Extraction ===")
python_post = Post.objects.filter(topic='python programming').order_by('-quality_score').first()
kws = extract_keywords(f"{python_post.title} {python_post.content}", vectorizer)
print("#12 Python post keywords:", kws)

short_post = min(Post.objects.all(), key=lambda p: len(set(p.content.split())))
kws_short = extract_keywords(f"{short_post.title} {short_post.content}", vectorizer, top_n=15)
print(f"#13 Short post ({len(set(short_post.content.split()))} unique words) keyword count:", len(kws_short))

print("\n=== Clustering ===")
c1 = run_clustering()
print(f"#14 n_clusters={c1['n_clusters']} ARI={c1['ari']:.3f}")
print("#15 Top terms per cluster:")
for cid, terms in top_terms_per_cluster(c1).items():
    print(f"    {cid}: {terms}")
c2 = run_clustering()
print(f"#16 ARI run1={c1['ari']:.4f} run2={c2['ari']:.4f} match={c1['ari'] == c2['ari']}")

print("\n=== Recommender ===")
readers = list(User.objects.filter(username__startswith='reader_'))
user = readers[0]
print(f"#17/18 Content-based for {user.username}:", recommend_content_based(user.id, top_n=5))

# 19: user with zero interactions (create a throwaway one)
fresh_user, _ = User.objects.get_or_create(username='test_fresh_user')
print("#19 recommend_content_based (no interactions):", recommend_content_based(fresh_user.id, top_n=5))

# 20: user not in interaction matrix at all -> same fresh_user works here too
print("#20 recommend_collaborative (no interactions):", recommend_collaborative(fresh_user.id, top_n=5))