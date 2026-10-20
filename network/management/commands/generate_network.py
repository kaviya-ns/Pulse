import random

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from faker import Faker

from core.models import Post, Link
from network.graph_generators import generate_ba_graph, generate_ws_graph, generate_bb_graph

fake = Faker()
rng = random.Random(42)

MODEL_MAP = {
    'ba': generate_ba_graph,
    'ws': generate_ws_graph,
    'bb': generate_bb_graph,
}

TOPIC_KEYWORDS = {
    'python programming': ['python', 'django', 'code', 'function', 'variable', 'programming', 'developer', 'software', 'debugging', 'framework'],
    'space astronomy': ['galaxy', 'planet', 'telescope', 'nasa', 'orbit', 'astronomy', 'star', 'universe', 'satellite', 'cosmos'],
    'cooking recipes': ['recipe', 'kitchen', 'ingredient', 'bake', 'flavor', 'cooking', 'chef', 'dish', 'spice', 'oven'],
    'finance stocks': ['stock', 'market', 'investment', 'trading', 'portfolio', 'finance', 'shares', 'economy', 'dividend', 'broker'],
    'sports football': ['football', 'match', 'goal', 'stadium', 'team', 'player', 'coach', 'league', 'tournament', 'score'],
    'health fitness': ['fitness', 'workout', 'exercise', 'health', 'nutrition', 'gym', 'training', 'diet', 'muscle', 'wellness'],
}
TOPICS = list(TOPIC_KEYWORDS.keys())

SHARED_KEYWORDS = ['trending', 'review', 'guide', 'update', 'popular', 'best', 'latest', 'analysis', 'tips', 'community']

def generate_topical_text(topic, quality, min_sentences=3, max_sentences=12):
    """Higher quality -> longer, more topic-keyword-dense text, on average.
    Noise is added so word count correlates with quality without exactly
    reconstructing the generation formula."""
    keywords = TOPIC_KEYWORDS[topic]

    # mean length scales with quality; actual length is sampled around it
    mean_sentences = min_sentences + quality * (max_sentences - min_sentences)
    n_sentences = round(rng.gauss(mean_sentences, sigma=1.5))
    n_sentences = max(min_sentences, min(max_sentences, n_sentences))

    # mean keyword hit-rate scales with quality; actual rate is jittered
    mean_prob = 0.2 + quality * 0.65
    topic_word_prob = mean_prob + rng.uniform(-0.1, 0.1)
    topic_word_prob = max(0.05, min(0.95, topic_word_prob))

    sentences = []
    for _ in range(n_sentences):
        base = fake.sentence(nb_words=8).rstrip('.')
        kw = rng.choice(keywords) if rng.random() < topic_word_prob else rng.choice(SHARED_KEYWORDS)
        sentences.append(f"{base} {kw}.")
    return ' '.join(sentences)


class Command(BaseCommand):
    help = "Generate a synthetic Pulse graph (BA / WS / BB) and populate Post + Link tables."

    def add_arguments(self, parser):
        parser.add_argument('--model', choices=MODEL_MAP.keys(), default='ba')
        parser.add_argument('--n', type=int, default=200)
        parser.add_argument('--clear', action='store_true')

    def handle(self, *args, **options):
        model_name = options['model']
        n = options['n']

        if options['clear']:
            Link.objects.all().delete()
            Post.objects.all().delete()
            self.stdout.write("Cleared existing Post/Link rows.")

        author, _ = User.objects.get_or_create(username='pulse_bot', defaults={'email': 'bot@pulse.local'})

        self.stdout.write(f"Generating {model_name.upper()} graph with n={n} ...")
        graph_fn = MODEL_MAP[model_name]
        G = graph_fn(n=n)

        node_to_post = {}
        posts_to_create = []
        for node in G.nodes():
            fitness = G.nodes[node].get('fitness')
            quality_seed = fitness if fitness is not None else rng.uniform(0.1, 1.0)
            topic = rng.choice(TOPICS)
            posts_to_create.append(Post(
                author=author,
                title=fake.sentence(nb_words=6).rstrip('.') + f" ({topic.split()[0]})",
                content=generate_topical_text(topic,quality_seed),
                topic=topic,
                quality_score=quality_seed,
            ))
        created_posts = Post.objects.bulk_create(posts_to_create)

        for node, post in zip(G.nodes(), created_posts):
            node_to_post[node] = post

        links_to_create = []
        for source, target in G.edges():
            links_to_create.append(Link(
                source_post=node_to_post[source],
                target_post=node_to_post[target],
            ))
        Link.objects.bulk_create(links_to_create, ignore_conflicts=True)

        self.stdout.write(self.style.SUCCESS(
            f"Created {len(created_posts)} posts and {len(links_to_create)} links "
            f"using the {model_name.upper()} model."
        ))