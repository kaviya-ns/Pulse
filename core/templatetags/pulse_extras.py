import re

from django import template
from django.utils.html import escape
from django.utils.safestring import mark_safe

register = template.Library()


@register.filter
def topic_title(topic):
    """Human-readable label for synthetic posts (topic is the real subject)."""
    if not topic:
        return "Article"
    return topic.title()


@register.filter
def search_snippet(text, max_chars=200):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return f"{cut}…"


@register.filter
def highlight_query(text, query):
    """Bold query terms in snippets (presentation only)."""
    if not text:
        return ""
    safe = escape(text)
    if not query:
        return safe
    for word in query.split():
        word = word.strip()
        if len(word) < 2:
            continue
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        safe = pattern.sub(lambda m: f"<mark>{m.group()}</mark>", safe)
    return mark_safe(safe)
