from typing import List, Tuple
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

def build_vocabulary(documents: List[List[str]]) -> dict:
    vocab = {}
    for document in documents:
        for token in document:
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab

def build_count_or_binary_bow(documents: List[List[str]], binary: bool = False) -> Tuple[csr_matrix, dict]:
    vocab = build_vocabulary(documents)

    rows = []
    cols = []
    data = []
    
    for document_id, document in enumerate(documents):
        counts = Counter(document)
        for token, count in counts.items():
            rows.append(document_id)
            cols.append(vocab[token])
            data.append(1 if binary else count)

    sparse_matrix = csr_matrix(
        (data, (rows, cols)),
        shape = (len(documents), len(vocab)),
        dtype=float
    )

    return sparse_matrix, vocab

def build_bow(
    documents: List[List[str]],
    variant: str = "count"
):
    if variant == "count":
        return build_count_or_binary_bow(documents, binary = False)
    elif variant == "binary":
        return build_count_or_binary_bow(documents, binary = True)
    else:
        raise ValueError(f"Unknown BoW variant: {variant}")
