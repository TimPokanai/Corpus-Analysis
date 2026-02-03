import numpy as np
from scipy.sparse import csr_matrix

def train_naive_bayes(
    X: csr_matrix,
    alpha: float = 1.0,
):
    # Sum counts over all documents
    word_counts = np.asarray(X.sum(axis=0)).flatten()

    total_count = word_counts.sum()
    vocab_size = len(word_counts)

    # Apply Laplace smoothing
    prob_w_given_c = (word_counts + alpha) / (
        total_count + alpha * vocab_size
    )

    log_prob_w_given_c = np.log(prob_w_given_c)

    return log_prob_w_given_c


def compute_class_prior(
    num_docs_in_class: int, total_docs: int):
    return np.log(num_docs_in_class / total_docs)

def train_all_categories(
    bows_by_category: dict[str, csr_matrix],
    alpha: float = 1.0,
):
    model = {}
    total_docs = sum(X.shape[0] for X in bows_by_category.values())

    for category, X in bows_by_category.items():
        log_likelihoods = train_naive_bayes(X, alpha=alpha)
        log_prior = compute_class_prior(X.shape[0], total_docs)

        model[category] = {
            "log_likelihoods": log_likelihoods,
            "log_prior": log_prior,
        }

    return model

def compute_log_likelihood_ratio(
    model: dict, category_a: str, category_b: str):
    ll_a = model[category_a]["log_likelihoods"]
    ll_b = model[category_b]["log_likelihoods"]

    return ll_a - ll_b

def compute_log_likelihood_ratio_multiclass(
    bows_by_category: dict[str, csr_matrix], target_category: str, alpha: float = 1.0):
    # Target class counts
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

    top_indices = np.argsort(np.abs(llr))[::-1][:k]

    return [(inv_vocab[i], llr[i]) for i in top_indices]
