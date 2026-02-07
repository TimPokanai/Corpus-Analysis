import numpy as np
from typing import Dict, Tuple

from load_data import load_raw_csv, validate_dataframe, get_documents_by_category, compute_dataset_stats
from preprocess import preprocess_documents_by_category
from build_bow import build_bow
from naive_bayes import compute_log_likelihood_ratio_multiclass, top_k_words
from topic_modelling import *

def print_dataset_info(df, processed_docs: Dict[str, list]) -> None:
    stats = compute_dataset_stats(df)
    print("Total documents: {}\n".format(stats["total_documents"]))
    print("Category         Docs   AvgTokens")
    for cat, count in stats["documents_per_category"].items():
        docs = processed_docs.get(cat, [])
        avg_tokens = (sum(len(doc) for doc in docs) / len(docs)) if docs else 0
        print(f"  {cat:<14} {count:>4}   {avg_tokens:>8.1f}")

def analyze_naive_bayes_configuration(
    processed_docs: Dict[str, list],
    bow_variant: str,
    use_stemming: bool,
) -> None:
    config_name = f"{bow_variant.upper()} BoW (Stemming: {use_stemming})"
    print(f"\n{config_name}")
    print("-" * 100)

    bows = {}
    vocab = None

    all_docs = []
    for docs in processed_docs.values():
        all_docs.extend(docs)

    _, vocab = build_bow(all_docs, variant=bow_variant)

    for cat, docs in processed_docs.items():
        X, _ = build_bow(docs, variant=bow_variant, vocab=vocab)
        bows[cat] = X

    categories = list(bows.keys())

    # Here we compute the top 10 words for each category and print them to console in tabular format
    top_words_by_cat = {}
    for target_cat in categories:
        llr = compute_log_likelihood_ratio_multiclass(bows, target_cat)
        top_words_by_cat[target_cat] = top_k_words(llr, vocab, k=10)

    column_width = 20
    header = "".join([f"{cat:<{column_width}}" for cat in categories])
    print(header)

    for row_idx in range(10):
        row_cells = []
        for cat in categories:
            word, score = top_words_by_cat[cat][row_idx]
            word = word[:12]
            row_cells.append(f"{word} {score:.2f}")
        print("".join([f"{cell:<{column_width}}" for cell in row_cells]))

def analyze_lda_configuration(
    processed_docs: Dict[str, list],
    use_stemming: bool,
    bow_variant: str,
    num_topics: int = 5,
) -> Tuple[Dict, Dict]:
    config_name = (
        f"LDA (topics: {num_topics}, stemming: {use_stemming}, bow: {bow_variant})"
    )
    print(f"\n--- Configuration: {config_name} ---\n")

    # Flatten documents for LDA
    all_docs = [doc for docs in processed_docs.values() for doc in docs]

    # Build dictionary and corpus
    dictionary, corpus = build_dictionary_and_corpus(all_docs, bow_variant=bow_variant)
    print(f"Dictionary size: {len(dictionary)}")
    print(f"Corpus size: {len(corpus)}")

    # Train LDA model
    lda = train_lda_model(corpus, dictionary, num_topics=num_topics)

    # Get top terms per topic
    topics = get_top_terms_per_topic(lda, num_terms=25)

    print(f"\nTop Terms per Topic (Top 25):")
    print("-" * 75)
    column_width = 18
    terms_per_row = 5
    for topic_id, terms in topics.items():
        print(f"Topic {topic_id}:")
        formatted_terms = [f"{word[:12]} {prob:.3f}" for word, prob in terms]
        for i in range(0, len(formatted_terms), terms_per_row):
            row = formatted_terms[i:i + terms_per_row]
            print("  " + "".join([f"{cell:<{column_width}}" for cell in row]))

    # Compute average topic distribution by category
    category_docs = {
        cat: [doc for docs in [processed_docs[cat]] for doc in docs]
        for cat in processed_docs.keys()
    }
    category_topic_avgs = compute_average_topic_distribution_by_category(
        category_docs, lda, bow_variant=bow_variant
    )

    print(f"\nTop Topics per Category:")
    print("-" * 75)
    for cat, avg_dist in category_topic_avgs.items():
        top_topics = get_top_topics_for_category(avg_dist, top_k=3)
        topics_str = ", ".join([f"T{tid}({prob:.3f})" for tid, prob in top_topics])
        print(f"{cat:<15} {topics_str}")

    return category_topic_avgs, topics

def main():
    df = load_raw_csv()
    df = validate_dataframe(df)

    grouped_docs = get_documents_by_category(df)

    # Preprocess with and without stemming
    processed_without_stem = preprocess_documents_by_category(
        grouped_docs, use_stemming=False
    )
    print_dataset_info(df, processed_without_stem)
    processed_with_stem = preprocess_documents_by_category(
        grouped_docs, use_stemming=True
    )

    print("\nNaive Bayes Analysis\n")
    print("COUNT BOW:")
    analyze_naive_bayes_configuration(processed_without_stem, "count", False)
    analyze_naive_bayes_configuration(processed_with_stem, "count", True)
    print("\nBINARY BOW:")
    analyze_naive_bayes_configuration(processed_without_stem, "binary", False)
    analyze_naive_bayes_configuration(processed_with_stem, "binary", True)

    print("\nLDA Topic Modeling\n")
    print("Count BoW - Without Stemming:")
    lda_avgs_no_stem_count, _ = analyze_lda_configuration(
        processed_without_stem, use_stemming=False, bow_variant="count", num_topics=5
    )
    print("\nCount BoW - With Stemming:")
    lda_avgs_with_stem_count, _ = analyze_lda_configuration(
        processed_with_stem, use_stemming=True, bow_variant="count", num_topics=5
    )
    print("\nBinary BoW - Without Stemming:")
    lda_avgs_no_stem_binary, _ = analyze_lda_configuration(
        processed_without_stem, use_stemming=False, bow_variant="binary", num_topics=5
    )
    print("\nBinary BoW - With Stemming:")
    lda_avgs_with_stem_binary, _ = analyze_lda_configuration(
        processed_with_stem, use_stemming=True, bow_variant="binary", num_topics=5
    )

if __name__ == "__main__":
    main()
