from load_data import load_raw_csv, validate_dataframe, get_documents_by_category
from preprocess import preprocess_corpus
from build_bow import build_bow
from naive_bayes import (
    compute_log_likelihood_ratio_multiclass,
    top_k_words
)

import numpy as np


def test_naive_bayes_pipeline():
    print("=== Loading and validating data ===")
    df = load_raw_csv()
    df = validate_dataframe(df)

    grouped_docs = get_documents_by_category(df)

    # Limit to a few categories for faster testing
    categories = list(grouped_docs.keys())
    print(f"Categories: {categories}")

    print("\n=== Preprocessing documents ===")
    processed_docs = {}
    for cat in categories:
        processed_docs[cat] = preprocess_corpus(grouped_docs[cat])

    print("\n=== Building BoW representations ===")
    bows = {}
    vocab = None

    # First pass: build unified vocabulary across all categories
    all_docs = []
    for docs in processed_docs.values():
        all_docs.extend(docs)
    
    # Build vocabulary from all documents
    _, vocab = build_bow(all_docs, variant="count")

    # Second pass: build BoW for each category using the unified vocabulary
    for cat, docs in processed_docs.items():
        X, _ = build_bow(docs, variant="count", vocab=vocab)
        bows[cat] = X
        print(f"{cat}: BoW shape = {X.shape}")

    vocab_size = len(vocab)
    print(f"\nVocabulary size: {vocab_size}")

    # Sanity check: all matrices must share the same vocab size
    for cat, X in bows.items():
        assert X.shape[1] == vocab_size, "Vocabulary size mismatch!"

    print("\n=== Testing log-likelihood ratio computation ===")
    target_category = categories[0]
    llr = compute_log_likelihood_ratio_multiclass(
        bows_by_category=bows,
        target_category=target_category
    )

    print(f"LLR vector shape for '{target_category}': {llr.shape}")

    # Basic numerical sanity checks
    assert llr.shape[0] == vocab_size
    assert np.all(np.isfinite(llr)), "LLR contains NaN or inf values"

    print("LLR sanity checks passed.")

    print("\n=== Extracting top informative words ===")
    top_words = top_k_words(llr, vocab, k=10)

    print(f"Top 10 words for category '{target_category}':")
    for word, score in top_words:
        print(f"{word:<15} {score:.4f}")

    # Validate ordering
    scores = [score for _, score in top_words]
    assert scores == sorted(scores, reverse=True), "Top words not sorted!"

    print("\nTop-word extraction test passed.")

    print("\n=== All Naive Bayes tests PASSED ===")


if __name__ == "__main__":
    test_naive_bayes_pipeline()
