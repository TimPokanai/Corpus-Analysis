import re
from typing import List
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEM = PorterStemmer()

def preprocess_document(text: str, use_stemming: bool = False) -> List[str]:
    
    # Lowercase the text
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Keeping alphabetic tokens
    tokens = [t for t in tokens if t.isalpha()]

    # Removing stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS]

    # Optional stemming
    if use_stemming:
        tokens = [STEM.stem(t) for t in tokens]

    return tokens

def preprocess_corpus(documents: List[str], use_stemming: bool = False) -> List[List[str]]:
    return [
        preprocess_document(document, use_stemming=use_stemming)
        for document in documents
    ]

def preprocess_documents_by_category(
        grouped_docs: dict[str, List[str]], use_stemming: bool = False
        ) -> dict[str, List[List[str]]]:
    
    processed = {}

    for category, documents in grouped_docs.items():
        processed[category] = preprocess_corpus(documents, use_stemming=use_stemming)

    return processed
