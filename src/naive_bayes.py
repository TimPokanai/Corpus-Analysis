import numpy as np
from scipy.sparse import csr_matrix

def compute_log_likelihood_ratio_multiclass(
    bows_by_category: dict[str, csr_matrix], target_category: str, alpha: float = 1.0):
    # Get counts for target category we want to classify against all others
    X_c = bows_by_category[target_category]
    counts_c = np.asarray(X_c.sum(axis=0)).flatten()

    # Aggregate counts for all other classes
    other_counts = None
    for cat, X in bows_by_category.items():
        if cat == target_category:
            continue
        summed = np.asarray(X.sum(axis=0)).flatten()
        if other_counts is None:
            other_counts = summed
        else:
            other_counts += summed

    V = len(counts_c)

    # Apply add-one smoothing
    P_wc = (counts_c + alpha) / (counts_c.sum() + alpha * V)
    P_wC0 = (other_counts + alpha) / (other_counts.sum() + alpha * V)

    return np.log(P_wc) - np.log(P_wC0)

def top_k_words(
    llr: np.ndarray,
    vocab: dict,
    k: int = 10,
):
    inv_vocab = {idx: word for word, idx in vocab.items()}

    top_indices = np.argsort(llr)[::-1][:k]

    return [(inv_vocab[i], llr[i]) for i in top_indices]
