from load_data import (
    load_raw_csv,
    validate_dataframe,
    get_documents_by_category,
)
from preprocess import preprocess_documents_by_category
from build_bow import build_bow


def test_bow_variants():
    # Load and validate data
    df = validate_dataframe(load_raw_csv())
    grouped_docs = get_documents_by_category(df)

    # Pick one category to inspect (keeps output readable)
    category = "sport"
    raw_docs = grouped_docs[category]

    # Preprocess
    processed_docs = preprocess_documents_by_category(
        {category: raw_docs},
        use_stemming=True,
    )[category]

    print(f"\nTesting category: {category}")
    print(f"Number of documents: {len(processed_docs)}")

    # -------------------------
    # Count BoW
    # -------------------------
    X_count, vocab_count = build_bow(processed_docs, variant="count")
    print("\nCount BoW")
    print("Shape:", X_count.shape)
    print("Sample vocab items:", list(vocab_count.items())[:10])
    print("First document vector (non-zero):")
    print(X_count[9].nonzero(), X_count[9].data)

    # -------------------------
    # Binary BoW
    # -------------------------
    X_binary, vocab_binary = build_bow(processed_docs, variant="binary")
    print("\nBinary BoW")
    print("Shape:", X_binary.shape)
    print("First document vector (non-zero):")
    print(X_binary[9].nonzero(), X_binary[9].data)


if __name__ == "__main__":
    test_bow_variants()
