from django.core.management.base import BaseCommand
from retrieval.whoosh_index import build_index


class Command(BaseCommand):
    help = "Builds/rebuilds the Whoosh search index from current Post rows."

    def handle(self, *args, **options):
        build_index()
        self.stdout.write(self.style.SUCCESS("Whoosh index built successfully."))