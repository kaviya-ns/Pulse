import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pulse_project.settings')
django.setup()

from retrieval.classification import train_quality_classifier, run_extraction_on_all_posts

result = train_quality_classifier()
print("Accuracy:", result['accuracy'])
print(result['report'])

for post, keywords in run_extraction_on_all_posts(result['vectorizer'])[:10]:
    print(post.title, "->", keywords)