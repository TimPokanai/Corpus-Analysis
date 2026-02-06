from typing import List, Dict, Tuple
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from build_bow import build_bow, build_vocabulary

def build_dictionary_and_corpus(documents: List[List[str]]) -> Tuple[Dictionary, List[List[Tuple[int, int]]]]:
    # Build vocabulary and sparse matrix using my BoW implementation
    vocab = build_vocabulary(documents)
    bow_matrix, _ = build_bow(documents, variant="count", vocab=vocab)
    
    # Create gensim Dictionary from vocabulary mapping
    id2word = {word_id: word for word, word_id in vocab.items()}
    dictionary = Dictionary()
    dictionary.id2word = id2word
    dictionary.token2id = {v: k for k, v in id2word.items()}
    
    # Convert sparse matrix to gensim corpus format
    corpus = []
    bow_csr = bow_matrix.tocsr()
    for doc_id in range(bow_matrix.shape[0]):
        row = bow_csr.getrow(doc_id)
        words, counts = row.nonzero()[1], row.data
        corpus.append(list(zip(words, (int(c) for c in counts))))
    
    return dictionary, corpus

def train_lda_model(corpus, dictionary, num_topics: int = 10, passes: int = 10, random_state: int = 42) -> LdaModel:
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state
    )
    return lda

def get_top_terms_per_topic(lda_model: LdaModel, num_terms: int = 25) -> Dict[int, List[Tuple[str, float]]]:
    topics = {}
    for topic_id in range(lda_model.num_topics):
        topics[topic_id] = lda_model.show_topic(
            topic_id,
            topn=num_terms
        )
    return topics

def get_document_topic_distributions(lda_model: LdaModel, corpus) -> List[np.ndarray]:
    doc_topics = []
    for bow in corpus:
        topic_dist = lda_model.get_document_topics(
            bow,
            minimum_probability=0.0
        )
        doc_topics.append(
            np.array([prob for _, prob in topic_dist])
        )
    return doc_topics

def compute_average_topic_distribution_by_category(
    category_documents: Dict[str, List[List[str]]],
    lda_model: LdaModel) -> Dict[str, np.ndarray]:
    category_topic_avgs = {}

    for category, docs in category_documents.items():
        # Build corpus using custom BoW implementation
        vocab = build_vocabulary(docs)
        bow_matrix, _ = build_bow(docs, variant="count", vocab=vocab)
        
        # Convert sparse matrix to gensim corpus format
        corpus = []
        bow_csr = bow_matrix.tocsr()
        for doc_id in range(bow_matrix.shape[0]):
            row = bow_csr.getrow(doc_id)
            words, counts = row.nonzero()[1], row.data
            corpus.append(list(zip(words, (int(c) for c in counts))))
        
        doc_topic_dists = get_document_topic_distributions(
            lda_model, corpus
        )
        avg_distribution = np.mean(doc_topic_dists, axis=0)
        category_topic_avgs[category] = avg_distribution

    return category_topic_avgs

def get_top_topics_for_category(category_topic_avg: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    topic_scores = list(enumerate(category_topic_avg))
    topic_scores.sort(key=lambda x: x[1], reverse=True)
    return topic_scores[:top_k]
