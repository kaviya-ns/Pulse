"""
Evaluation of retrieval models: Precision, Recall, Average Precision (AP),
and nDCG -- both set-based (Boolean/Whoosh) and ranked (Vector, BM25).

Ground truth: a post is "relevant" to a query if the query is (one of)
the topic keywords for that post's topic.
"""
import math

from core.models import Post
from retrieval.ir_engine import IREngine
from retrieval.whoosh_index import whoosh_search

TOPIC_QUERIES = {
    'python trending guide': 'python programming',
    'galaxy popular review': 'space astronomy',
    'recipe best tips': 'cooking recipes',
    'stock latest analysis': 'finance stocks',
    'football trending update': 'sports football',
    'fitness popular guide': 'health fitness',
}


def get_relevant_ids(topic):
    return set(Post.objects.filter(topic=topic).values_list('id', flat=True))


def precision(retrieved_ids, relevant_ids):
    if not retrieved_ids:
        return 0.0
    return len(retrieved_ids & relevant_ids) / len(retrieved_ids)


def recall(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    return len(retrieved_ids & relevant_ids) / len(relevant_ids)


def f1_score(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision(ranked_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    hits = 0
    precisions = []
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / len(relevant_ids) if precisions else 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k=10):
    def dcg(ids):
        return sum(
            (1.0 / math.log2(i + 2)) for i, doc_id in enumerate(ids[:k]) if doc_id in relevant_ids
        )
    actual_dcg = dcg(ranked_ids)
    ideal_dcg = dcg(list(relevant_ids)[:k])
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_all_models(top_k=10):
    engine = IREngine()
    results = {'whoosh_and': [], 'whoosh_or': [], 'vector': [], 'bm25': []}

    for query_term, topic in TOPIC_QUERIES.items():
        relevant_ids = get_relevant_ids(topic)
        if not relevant_ids:
            continue

        whoosh_and_hits = whoosh_search(query_term, limit=top_k, use_or=False)
        retrieved_and = {p.id for p, _ in whoosh_and_hits}
        p_and = precision(retrieved_and, relevant_ids)
        r_and = recall(retrieved_and, relevant_ids)
        results['whoosh_and'].append({'query': query_term, 'precision': p_and, 'recall': r_and, 'f1': f1_score(p_and, r_and)})

        whoosh_or_hits = whoosh_search(query_term, limit=top_k, use_or=True)
        retrieved_or = {p.id for p, _ in whoosh_or_hits}
        p_or = precision(retrieved_or, relevant_ids)
        r_or = recall(retrieved_or, relevant_ids)
        results['whoosh_or'].append({'query': query_term, 'precision': p_or, 'recall': r_or, 'f1': f1_score(p_or, r_or)})

        vector_hits = engine.vector_search(query_term, top_k=top_k)
        ranked_ids = [p.id for p, _ in vector_hits]
        results['vector'].append({
            'query': query_term,
            'AP': average_precision(ranked_ids, relevant_ids),
            'nDCG@10': ndcg_at_k(ranked_ids, relevant_ids, k=top_k),
        })

        bm25_hits = engine.probabilistic_search(query_term, top_k=top_k)
        ranked_ids_bm25 = [p.id for p, _ in bm25_hits]
        results['bm25'].append({
            'query': query_term,
            'AP': average_precision(ranked_ids_bm25, relevant_ids),
            'nDCG@10': ndcg_at_k(ranked_ids_bm25, relevant_ids, k=top_k),
        })

    return results


def summarize(results):
    summary = {}
    for model, rows in results.items():
        if not rows:
            continue
        keys = [k for k in rows[0].keys() if k != 'query']
        summary[model] = {k: sum(row[k] for row in rows) / len(rows) for k in keys}
    return summary