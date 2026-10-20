import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pulse_project.settings')
django.setup()

from retrieval.clustering import run_clustering, top_terms_per_cluster

result = run_clustering()
print("n_clusters:", result['n_clusters'])
print("Adjusted Rand Index:", result['ari'])
print("Silhouette score:", result['silhouette'])

print("\nTop terms per cluster:")
for cluster_id, terms in top_terms_per_cluster(result).items():
    print(f"  Cluster {cluster_id}: {terms}")