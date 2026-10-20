import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pulse_project.settings')
django.setup()

from retrieval.link_analysis import build_graph_from_db, compare_pagerank_vs_quality, save_scores_to_db

G = build_graph_from_db()
print(G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

results = compare_pagerank_vs_quality(G)
for k, v in results.items():
    print(k, ":", round(v, 4))

save_scores_to_db(G)
print("Scores saved to DB.")