from django.contrib import admin
from .models import Post, Link, Interaction

admin.site.register(Post)
admin.site.register(Link)
admin.site.register(Interaction)