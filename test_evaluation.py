import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pulse_project.settings')
django.setup()

from retrieval.evaluation import evaluate_all_models, summarize

results = evaluate_all_models(top_k=10)

print("--- Per-query results ---")
for model, rows in results.items():
    print(f"\n{model}:")
    for row in rows:
        print(" ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

print("\n=== SUMMARY (averaged across queries) ===")
summary = summarize(results)
for model, metrics in summary.items():
    print(model, ":", {k: round(v, 3) for k, v in metrics.items()})