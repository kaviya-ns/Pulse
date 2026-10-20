"""
Populates Interaction rows for Recommender System.

Creates synthetic reader users, each with a hidden preferred topic, then
generates interactions biased toward that topic (content-based signal)
and toward higher quality_score posts (realistic engagement bias). This
gives both collaborative filtering (shared behavior across users) and
content-based filtering (topic affinity) real signal to work with.
"""
import random

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from core.models import Post, Interaction

KIND_WEIGHTS = {'view': 0.6, 'like': 0.3, 'share': 0.1}


class Command(BaseCommand):
    help = "Generate synthetic reader users and their Interaction rows."

    def add_arguments(self, parser):
        parser.add_argument('--n-users', type=int, default=30)
        parser.add_argument('--interactions-per-user', type=int, default=15)
        parser.add_argument('--clear', action='store_true')

    def handle(self, *args, **options):
        n_users = options['n_users']
        n_per_user = options['interactions_per_user']

        if options['clear']:
            Interaction.objects.all().delete()
            User.objects.exclude(username='pulse_bot').delete()
            self.stdout.write("Cleared existing Interactions and reader users.")

        posts = list(Post.objects.all())
        topics = list({p.topic for p in posts})

        readers = []
        for i in range(n_users):
            preferred_topic = random.choice(topics)
            user, _ = User.objects.get_or_create(
                username=f'reader_{i}',
                defaults={'email': f'reader_{i}@pulse.local'}
            )
            readers.append((user, preferred_topic))

        interactions_to_create = []
        for user, preferred_topic in readers:
            # 70% of a reader's interactions come from their preferred topic,
            # 30% spill into other topics (nobody reads only one thing)
            same_topic_posts = [p for p in posts if p.topic == preferred_topic]
            other_posts = [p for p in posts if p.topic != preferred_topic]

            n_same = round(n_per_user * 0.7)
            n_other = n_per_user - n_same

            chosen = (
                random.choices(same_topic_posts, k=min(n_same, len(same_topic_posts)))
                + random.choices(other_posts, k=min(n_other, len(other_posts)))
            )

            for post in chosen:
                # higher quality_score -> more likely to escalate view -> like/share
                if post.quality_score > 0.7:
                    kind = random.choices(
                        ['view', 'like', 'share'], weights=[0.3, 0.4, 0.3]
                    )[0]
                elif post.quality_score > 0.4:
                    kind = random.choices(
                        ['view', 'like', 'share'], weights=[0.5, 0.35, 0.15]
                    )[0]
                else:
                    kind = random.choices(
                        ['view', 'like', 'share'], weights=[0.8, 0.15, 0.05]
                    )[0]

                interactions_to_create.append(Interaction(user=user, post=post, kind=kind))

        Interaction.objects.bulk_create(interactions_to_create)

        self.stdout.write(self.style.SUCCESS(
            f"Created {len(readers)} reader users and "
            f"{len(interactions_to_create)} interactions."
        ))