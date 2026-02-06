from typing import List, Tuple
from collections import Counter

from scipy.sparse import csr_matrix

def build_vocabulary(documents: List[List[str]]) -> dict:
    vocab = {}
    for document in documents:
        for token in document:
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab

def build_count_or_binary_bow(documents: List[List[str]], binary: bool = False, vocab: dict = None) -> Tuple[csr_matrix, dict]:
    if vocab is None:
        vocab = build_vocabulary(documents)

    rows = []
    cols = []
    data = []
    
    for document_id, document in enumerate(documents):
        counts = Counter(document)
        for token, count in counts.items():
            if token in vocab:
                rows.append(document_id)
                cols.append(vocab[token])
                data.append(1 if binary else count)

    sparse_matrix = csr_matrix(
        (data, (rows, cols)),
        shape = (len(documents), len(vocab)),
        dtype=float
    )

    return sparse_matrix, vocab

# Helper function to build bag of words in either count or binary variant
def build_bow(
    documents: List[List[str]],
    variant: str = "count",
    vocab: dict = None
):
    if variant == "count":
        return build_count_or_binary_bow(documents, binary = False, vocab=vocab)
    elif variant == "binary":
        return build_count_or_binary_bow(documents, binary = True, vocab=vocab)
    else:
        raise ValueError(f"Unknown BoW variant: {variant}")
