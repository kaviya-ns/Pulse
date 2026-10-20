"""
Link-based retrieval: standard PageRank vs Quality-weighted PageRank.
"""
import networkx as nx
from scipy.stats import spearmanr

from core.models import Post, Link


def build_graph_from_db():
    G = nx.DiGraph()
    posts = list(Post.objects.all())
    for p in posts:
        G.add_node(p.id, quality=p.quality_score)

    for link in Link.objects.all().select_related('source_post', 'target_post'):
        G.add_edge(link.source_post_id, link.target_post_id)

    return G


def compute_standard_pagerank(G):
    return nx.pagerank(G)


def compute_quality_pagerank(G, alpha=0.85):
    qualities = nx.get_node_attributes(G, 'quality')
    total = sum(qualities.values())
    personalization = {node: q / total for node, q in qualities.items()} if total else None
    return nx.pagerank(G, alpha=alpha, personalization=personalization)


def compare_pagerank_vs_quality(G):
    qualities = nx.get_node_attributes(G, 'quality')
    nodes = list(G.nodes())
    quality_vals = [qualities[n] for n in nodes]

    standard_pr = compute_standard_pagerank(G)
    quality_pr = compute_quality_pagerank(G)

    standard_vals = [standard_pr[n] for n in nodes]
    quality_vals_pr = [quality_pr[n] for n in nodes]

    standard_corr, _ = spearmanr(standard_vals, quality_vals)
    quality_corr, _ = spearmanr(quality_vals_pr, quality_vals)

    return {
        'standard_pagerank_vs_quality_correlation': standard_corr,
        'quality_pagerank_vs_quality_correlation': quality_corr,
        'improvement': quality_corr - standard_corr,
    }


def save_scores_to_db(G):
    standard_pr = compute_standard_pagerank(G)
    quality_pr = compute_quality_pagerank(G)

    posts = {p.id: p for p in Post.objects.filter(id__in=G.nodes())}
    to_update = []
    for node in G.nodes():
        post = posts[node]
        post.pagerank_score = standard_pr[node]
        post.quality_pagerank_score = quality_pr[node]
        to_update.append(post)

    Post.objects.bulk_update(to_update, ['pagerank_score', 'quality_pagerank_score'])