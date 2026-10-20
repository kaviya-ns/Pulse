"""
Recommender System.

Two approaches, as required:
- Content-based: recommend posts similar (via TF-IDF cosine similarity)
  to posts the user has already engaged with positively.
- Collaborative filtering: recommend posts liked/shared by *other* users
  with similar interaction patterns (user-based CF via cosine similarity
  on the user-item interaction matrix).
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.models import Post, Interaction

KIND_WEIGHTS = {'view': 1, 'like': 3, 'share': 5}


def build_interaction_matrix():
    """User x Post matrix, weighted by interaction strength (view < like < share).
    This is the shared input for collaborative filtering."""
    interactions = Interaction.objects.select_related('user', 'post').all()
    rows = [
        {
            'user': i.user_id,
            'post': i.post_id,
            'weight': KIND_WEIGHTS[i.kind],
        }
        for i in interactions
    ]
    df = pd.DataFrame(rows)
    matrix = df.pivot_table(
        index='user', columns='post', values='weight', aggfunc='sum', fill_value=0
    )
    return matrix


def recommend_content_based(user_id, top_n=5, max_features=300):
    """Builds a 'taste profile' as the average TF-IDF vector of posts the
    user has liked/shared, then ranks all other posts by cosine similarity
    to that profile."""
    posts = list(Post.objects.all())
    post_ids = [p.id for p in posts]
    texts = [f"{p.title} {p.content}" for p in posts]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(texts)
    id_to_row = {pid: i for i, pid in enumerate(post_ids)}

    liked_post_ids = list(
        Interaction.objects.filter(user_id=user_id, kind__in=['like', 'share'])
        .values_list('post_id', flat=True)
    )
    if not liked_post_ids:
        return []

    liked_rows = [id_to_row[pid] for pid in liked_post_ids if pid in id_to_row]
    profile_vector = X[liked_rows].mean(axis=0)
    profile_vector = np.asarray(profile_vector)

    similarities = cosine_similarity(profile_vector, X).flatten()

    seen_post_ids = set(
        Interaction.objects.filter(user_id=user_id).values_list('post_id', flat=True)
    )
    ranked = sorted(
        ((post_ids[i], similarities[i]) for i in range(len(post_ids))
         if post_ids[i] not in seen_post_ids),
        key=lambda x: x[1], reverse=True,
    )
    return ranked[:top_n]


def recommend_collaborative(user_id, top_n=5, k_neighbors=5):
    """User-based collaborative filtering: finds the k most similar users
    by interaction pattern (cosine similarity over the interaction matrix),
    then recommends posts they engaged with that this user hasn't seen."""
    matrix = build_interaction_matrix()
    if user_id not in matrix.index:
        return []

    similarities = cosine_similarity(matrix)
    sim_df = pd.DataFrame(similarities, index=matrix.index, columns=matrix.index)

    neighbor_scores = sim_df[user_id].drop(user_id).sort_values(ascending=False)
    neighbors = neighbor_scores.head(k_neighbors).index

    seen_posts = set(matrix.columns[matrix.loc[user_id] > 0])

    scores = matrix.loc[neighbors].multiply(neighbor_scores[neighbors], axis=0).sum(axis=0)
    scores = scores.drop(labels=[p for p in seen_posts if p in scores.index])

    ranked = scores.sort_values(ascending=False).head(top_n)
    return list(ranked.items())