"""
Requirement 5 (Classification) + Requirement 8 (Information Extraction).

Classification: predicts a quality CLASS (low/medium/high) from content
features alone -- a real supervised classifier, trained/tested on a split,
evaluated with accuracy. This validates whether "quality" is actually
learnable from content, independent of the ground-truth quality_score
used to drive graph growth (kept separate on purpose, see note below).

Information Extraction: pulls out the top keywords per post using
TF-IDF term weights -- the standard, defensible IE technique for
keyword/keyphrase extraction from short text.
"""
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from core.models import Post


def quality_to_class(score, low_thresh=0.33, high_thresh=0.66):
    if score < low_thresh:
        return 'low'
    elif score < high_thresh:
        return 'medium'
    return 'high'


def extract_numeric_features(posts):
    """Explicit structural features. TF-IDF normalizes away document
    length (it's a per-document distribution, not a raw count), so
    these numeric features are what actually let the classifier see
    the length/depth signal that distinguishes quality levels."""
    features = []
    for p in posts:
        words = p.content.split()
        word_count = len(words)
        unique_ratio = len(set(words)) / word_count if word_count else 0
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        features.append([word_count, unique_ratio, avg_word_len])
    return np.array(features)


def train_quality_classifier():
    """Trains Logistic Regression to predict quality class from content
    features (TF-IDF + explicit structural features combined)."""
    posts = list(Post.objects.all())
    texts = [f"{p.title} {p.content}" for p in posts]
    labels = [quality_to_class(p.quality_score) for p in posts]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
    X_tfidf = vectorizer.fit_transform(texts)

    numeric = extract_numeric_features(posts)
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(numeric)

    X = hstack([X_tfidf, csr_matrix(X_numeric)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=42, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        'classifier': clf,
        'vectorizer': vectorizer,
        'scaler': scaler,
        'accuracy': accuracy,
        'report': report,
    }


def extract_keywords(text, vectorizer, top_n=5):
    """Information Extraction: top-N TF-IDF-weighted terms for one document.
    Reuses a fitted vectorizer's vocabulary/IDF weights for consistency."""
    tfidf_vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_vec.toarray().flatten()
    top_indices = scores.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_indices if scores[i] > 0]


def run_extraction_on_all_posts(vectorizer, top_n=5):
    """Extracts keywords for every post -- this is what you'd display
    as 'tags' in a UI, or use as a lightweight index summary."""
    results = []
    for post in Post.objects.all():
        text = f"{post.title} {post.content}"
        keywords = extract_keywords(text, vectorizer, top_n=top_n)
        results.append((post, keywords))
    return results
