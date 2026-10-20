"""
Requirement 3 (Clustering).

Unsupervised topic clustering on post content using KMeans over TF-IDF
vectors. Validated against the existing `topic` field as ground truth
using Adjusted Rand Index (ARI) -- the standard metric for comparing
a clustering result to known labels (0 = random, 1 = perfect match).
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from core.models import Post


def run_clustering(n_clusters=None, max_features=300):
    """Clusters posts by content alone (no topic field used as input),
    then compares the resulting clusters against the true topic labels."""
    posts = list(Post.objects.all())
    texts = [f"{p.title} {p.content}" for p in posts]
    true_labels = [p.topic for p in posts]

    if n_clusters is None:
        n_clusters = len(set(true_labels))

    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(X)

    ari = adjusted_rand_score(true_labels, predicted_labels)
    silhouette = silhouette_score(X, predicted_labels)

    return {
        'model': kmeans,
        'vectorizer': vectorizer,
        'predicted_labels': predicted_labels,
        'true_labels': true_labels,
        'ari': ari,
        'silhouette': silhouette,
        'n_clusters': n_clusters,
    }


def top_terms_per_cluster(result, top_n=8):
    """Top TF-IDF terms per cluster centroid -- used to label/interpret
    each cluster (e.g. confirming cluster 2 'looks like' python posts)."""
    vectorizer = result['vectorizer']
    kmeans = result['model']
    feature_names = vectorizer.get_feature_names_out()
    centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    output = {}
    for i in range(result['n_clusters']):
        output[i] = [feature_names[idx] for idx in centroids[i, :top_n]]
    return output