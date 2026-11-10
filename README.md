# Pulse

**A search and recommendation engine that ranks social content by quality, not popularity.**

![Python](https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54) ![Django](https://img.shields.io/badge/Django-092E20?logo=django&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) ![NLTK](https://img.shields.io/badge/NLTK-154F5B) ![Bootstrap](https://img.shields.io/badge/Bootstrap-7952B3?logo=bootstrap&logoColor=white)

> **Highlights:** query refinement lifts nDCG from **0.61 → 0.99** on off-vocabulary queries · content-based recommender reaches **Precision@5 = 0.92** (vs 0.17 random baseline)

Pulse is a Django-based platform that goes beyond keyword matching to
retrieve, classify, cluster, and recommend social content based on
**content quality** rather than raw popularity. It combines classic
Information Retrieval techniques with supervised and unsupervised
machine learning over a corpus of synthetic social posts.

![Pulse search engine home](./pulse%20search%20engine.PNG)

![Pulse recommendations](./pulse%202.PNG)

## Features

### 1. Search (Boolean / Vector / Probabilistic Retrieval)
Full-text search over post content using TF-IDF vector space search,
with Boolean and probabilistic ranking models also implemented in
`retrieval/ir_engine.py`.

### 2. Query Refinement
Synonym-based query expansion (WordNet + a curated domain synonym
dictionary) that helps retrieve relevant results even when a query
uses different vocabulary than the documents. Original query terms
are weighted above expansion terms (Rocchio-style) so expansion only
helps, rather than diluting, exact-match queries.

### 3. Classification & Information Extraction
A Logistic Regression classifier predicts a post's quality class
(low / medium / high) from TF-IDF features plus explicit structural
features (word count, vocabulary richness, average word length).
TF-IDF-weighted keyword extraction surfaces the top terms per post.

### 4. Clustering
Unsupervised KMeans clustering of posts by content, validated against
the ground-truth topic labels using Adjusted Rand Index.

### 5. Recommender System
Two complementary recommendation strategies over a synthetic
user-post interaction log:
- **Content-based** — recommends posts similar to ones a user has
  previously liked/shared, via TF-IDF cosine similarity.
- **Collaborative filtering** — recommends posts liked by other users
  with similar interaction patterns, via a user-item interaction
  matrix and cosine similarity.

## Tech Stack

- **Backend:** Django
- **ML / NLP:** scikit-learn, scipy, nltk, numpy, pandas
- **Frontend:** Django templates + Bootstrap (server-rendered, no
  separate frontend build)

## Project Structure

```
Pulse/
├── pulse_project/        # Django project settings & root urls
├── core/                 # Post, User(reader), Interaction models + views/urls
│   └── management/commands/
│       └── generate_interactions.py
├── retrieval/             # IR engine, query refinement, classification,
│   │                       clustering
│   ├── ir_engine.py
│   ├── query_refinement.py
│   ├── classification.py
│   ├── clustering.py
│   └── evaluation.py
├── recommend/             # Recommendation logic
│   └── recommender.py
└── requirements.txt

```
## Setup

```powershell
git clone https://github.com/kaviya-ns/Pulse.git
cd Pulse

python -m venv venv
.\venv\Scripts\Activate.ps1      # macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

# PostgreSQL connection (create the database and user first)
$env:PULSE_DB_PASSWORD = "your-db-password"   # macOS/Linux: export PULSE_DB_PASSWORD=...
# optional: PULSE_DB_NAME, PULSE_DB_USER, PULSE_DB_HOST, PULSE_DB_PORT

python manage.py migrate

# generate synthetic post data (see management commands for options)
python manage.py generate_network --n 200

# generate synthetic reader users + interactions
python manage.py generate_interactions --n-users 30 --interactions-per-user 15

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

## Evaluation Summary

| Component | Metric | Result |
|---|---|---|
| Classification | Accuracy | 0.62 |
| Query refinement (off-vocab queries) | Mean Average Precision | 0.178 → 0.297 |
| Query refinement (off-vocab queries) | nDCG | 0.611 → 0.989 |
| Clustering | Adjusted Rand Index | 0.608 |
| Recommender (content-based) | Precision@5 | 0.920 |
| Recommender (collaborative) | Precision@5 | 0.480 |
| Recommender (random baseline) | Precision@5 | 0.167 |
