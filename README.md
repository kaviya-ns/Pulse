# Pulse

Search and discovery built around **content quality**, not engagement or popularity. Pulse combines information retrieval (TF-IDF vector search, BM25, boolean retrieval), quality-aware network models, classification, clustering, and hybrid recommendations behind a simple search-engine UI.

![Pulse search engine home](./pulse%20search%20engine.PNG)

![Pulse recommendations](./pulse%202.PNG)

## Features

- **Home + search UI** — Google-style landing at `/` and ranked results at `/search`
- **Query refinement** — optional query expansion for better recall on related terms
- **Post pages** — topic-focused article view with extracted keywords
- **Recommendations** — content-based and collaborative filtering for demo reader profiles
- **Synthetic corpus** — topical posts on six domains (programming, space, cooking, finance, sports, health) with quality scores and citation graph

## Requirements

- Python 3.10+
- Dependencies (install in a virtual environment): Django, scikit-learn, rank-bm25, NetworkX, Faker, NLTK, pandas, Whoosh (see `pip install` below)

## Setup

```powershell
cd C:\Users\kavya\Desktop\Pulse

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install django scikit-learn rank-bm25 networkx faker nltk pandas whoosh scipy

python manage.py migrate
python manage.py generate_network --model ba --n 200
python manage.py generate_interactions
python manage.py build_search_index
```

## Run

```powershell
.\venv\Scripts\Activate.ps1
python manage.py runserver
```

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:8000/ | Home / search box |
| http://127.0.0.1:8000/search?q=python | Search results |
| http://127.0.0.1:8000/recommendations/ | Personalized recommendations demo |
| http://127.0.0.1:8000/admin/ | Django admin (optional) |

## Try these searches

- `python`, `django`, `coding` (with **Refined search** for expansion)
- `galaxy`, `cosmos`, `astronomy`
- `recipe`, `cooking`, `meal`
- `stock`, `investment`
- `football`, `goal`
- `fitness`, `workout`

## Backend tests

```powershell
python test_all_backend.py
```

## Project layout

| Path | Role |
|------|------|
| `core/` | Django app, models, web views and templates |
| `retrieval/` | IR engine, classification, clustering, Whoosh index |
| `network/` | Graph generation and interaction data |
| `recommend/` | Content-based and collaborative recommenders |
| `pulse_project/` | Django settings and root URLs |
