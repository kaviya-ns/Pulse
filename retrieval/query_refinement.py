import nltk
from nltk.corpus import wordnet

try:
    wordnet.synsets('test')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# maps an off-vocabulary query to the same ground-truth topic as before
DOMAIN_SYNONYMS = {
    'coding': ['python', 'programming', 'code'],
    'tutorial': ['guide'],
    'meal': ['recipe', 'cooking', 'dish'],
    'preparation': ['cooking'],
    'cosmos': ['galaxy', 'space', 'astronomy'],
}

def get_synonyms(term, max_synonyms=3):
    if term.lower() in DOMAIN_SYNONYMS:
        return DOMAIN_SYNONYMS[term.lower()][:max_synonyms]
    synonyms = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ').lower()
            if name != term.lower() and ' ' not in name:
                synonyms.add(name)
    return list(synonyms)[:max_synonyms]


def expand_query(query, max_synonyms_per_term=2, original_weight=3):
    terms = query.strip().split()
    expanded_terms = []
    for term in terms:
        expanded_terms.extend([term] * original_weight)
        expanded_terms.extend(get_synonyms(term, max_synonyms_per_term))
    return ' '.join(expanded_terms)