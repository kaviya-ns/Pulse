import os
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup

from django.conf import settings

from core.models import Post

INDEX_DIR = os.path.join(settings.BASE_DIR, 'whoosh_index')

SCHEMA = Schema(
    id=ID(stored=True, unique=True),
    title=TEXT(stored=True),
    content=TEXT(stored=True),
)


def build_index():
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    ix = index.create_in(INDEX_DIR, SCHEMA)
    writer = ix.writer()
    for post in Post.objects.all():
        writer.add_document(id=str(post.id), title=post.title, content=post.content)
    writer.commit()
    return ix


def get_index():
    if index.exists_in(INDEX_DIR):
        return index.open_dir(INDEX_DIR)
    return build_index()

def whoosh_search(query_string, limit=10, use_or=False):
    """Boolean/structured search.
    use_or=False (default): AND semantics -- every term must appear
        (Whoosh's own default grouping).
    use_or=True: OR semantics -- any term may appear, ranked by how
        many/how well terms match.
    Returns [(Post, score), ...] ranked by Whoosh's relevance scoring.
    """
    ix = get_index()
    with ix.searcher() as searcher:
        group = OrGroup if use_or else None
        parser = QueryParser("content", ix.schema, group=group) if group else QueryParser("content", ix.schema)
        query = parser.parse(query_string)
        whoosh_results = searcher.search(query, limit=limit)

        post_ids = [int(r['id']) for r in whoosh_results]
        scores = {int(r['id']): r.score for r in whoosh_results}

        posts_by_id = Post.objects.in_bulk(post_ids)
        return [(posts_by_id[pid], scores[pid]) for pid in post_ids if pid in posts_by_id]