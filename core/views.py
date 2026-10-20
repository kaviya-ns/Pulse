from django.shortcuts import render, get_object_or_404
from django.contrib.auth.models import User

from core.models import Post
from retrieval.ir_engine import IREngine
from retrieval.classification import extract_keywords, train_quality_classifier
from recommend.recommender import recommend_content_based, recommend_collaborative

# Trained once per process -- retraining per request would be slow and
# pointless since the underlying data doesn't change between requests.
_classifier_result = None


def get_classifier_result():
    global _classifier_result
    if _classifier_result is None:
        _classifier_result = train_quality_classifier()
    return _classifier_result


def home(request):
    return render(request, 'core/home.html')


def search(request):
    query = request.GET.get('q', '')
    refined = request.GET.get('refined') == 'on'
    results = []

    if query:
        engine = IREngine()
        if refined:
            hits = engine.vector_search_refined(query, top_k=10)
        else:
            hits = engine.vector_search(query, top_k=10)
        results = [{'post': post, 'score': round(score, 3)} for post, score in hits]

    return render(request, 'core/search.html', {
        'query': query,
        'refined': refined,
        'results': results,
    })


def post_detail(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    result = get_classifier_result()

    keywords = extract_keywords(f"{post.title} {post.content}", result['vectorizer'])

    return render(request, 'core/post_detail.html', {
        'post': post,
        'keywords': keywords,
    })


def recommendations(request):
    readers = User.objects.filter(username__startswith='reader_')
    selected_id = request.GET.get('user_id')
    content_recs, collab_recs = [], []

    if selected_id:
        content_hits = recommend_content_based(int(selected_id), top_n=5)
        collab_hits = recommend_collaborative(int(selected_id), top_n=5)
        content_recs = [{'post': Post.objects.get(id=pid), 'score': round(score, 3)} for pid, score in content_hits]
        collab_recs = [{'post': Post.objects.get(id=pid), 'score': round(score, 3)} for pid, score in collab_hits]

    return render(request, 'core/recommendations.html', {
        'readers': readers,
        'selected_id': selected_id,
        'content_recs': content_recs,
        'collab_recs': collab_recs,
    })