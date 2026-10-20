import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pulse_project.settings')
django.setup()

from core.models import Post
from retrieval.ir_engine import IREngine
from retrieval.evaluation import get_relevant_ids, average_precision, ndcg_at_k, TOPIC_QUERIES

engine = IREngine()

rows = []
for query_term, topic in TOPIC_QUERIES.items():
    relevant_ids = get_relevant_ids(topic)
    if not relevant_ids:
        continue

    # unrefined
    hits = engine.vector_search(query_term, top_k=10)
    ranked_ids = [p.id for p, _ in hits]
    ap_plain = average_precision(ranked_ids, relevant_ids)
    ndcg_plain = ndcg_at_k(ranked_ids, relevant_ids, k=10)

    # refined
    hits_r = engine.vector_search_refined(query_term, top_k=10)
    ranked_ids_r = [p.id for p, _ in hits_r]
    ap_refined = average_precision(ranked_ids_r, relevant_ids)
    ndcg_refined = ndcg_at_k(ranked_ids_r, relevant_ids, k=10)

    rows.append((query_term, ap_plain, ap_refined, ndcg_plain, ndcg_refined))

print(f"{'query':25} {'AP':>8} {'AP(refined)':>12} {'nDCG':>8} {'nDCG(refined)':>14}")
for query_term, ap_p, ap_r, ndcg_p, ndcg_r in rows:
    print(f"{query_term:25} {ap_p:8.3f} {ap_r:12.3f} {ndcg_p:8.3f} {ndcg_r:14.3f}")

avg = lambda i: sum(r[i] for r in rows) / len(rows)
print("\nAverages:")
print(f"AP:   {avg(1):.3f} -> {avg(2):.3f}")
print(f"nDCG: {avg(3):.3f} -> {avg(4):.3f}")