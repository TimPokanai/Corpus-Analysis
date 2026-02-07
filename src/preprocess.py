from typing import List
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Downloads for nltk word_tokenize() and list of stopwords
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.add("said")  # Add "said" to filter out common BBC reporting word
STOP_WORDS.add("would")  # Add "would" to filter out common BBC reporting word
STEM = PorterStemmer()

def preprocess_document(text: str, use_stemming: bool = False) -> List[str]:
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in STOP_WORDS]

    if use_stemming:
        tokens = [STEM.stem(t) for t in tokens]

    return tokens

def preprocess_corpus(documents: List[str], use_stemming: bool = False) -> List[List[str]]:
    return [preprocess_document(document, use_stemming=use_stemming) for document in documents]

# We pass the boolean arg use_stemming downstream through function calls 
# to control whether stemming is applied for experimentation

def preprocess_documents_by_category(
        grouped_docs: dict[str, List[str]], use_stemming: bool = False
    ) -> dict[str, List[List[str]]]:
    
    processed = {}

    for category, documents in grouped_docs.items():
        processed[category] = preprocess_corpus(documents, use_stemming=use_stemming)

    return processed
