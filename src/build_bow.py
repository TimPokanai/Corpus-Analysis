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

