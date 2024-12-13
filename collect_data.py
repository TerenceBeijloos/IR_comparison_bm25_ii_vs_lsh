import cProfile
import pstats
from sklearn.metrics import ndcg_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm # print progress
import matplotlib.pyplot as plt
import math
import os

from inverted_index import *
from lexer import *
from bm25 import BM25
from dataset import load_dataset

INDEX_PATH = 'ii.pickle'
CORPUS_PATH = "./dataset/nfcorpus/corpus.jsonl"
QUERIES_PATH = "./dataset/nfcorpus/queries.jsonl"
QRELS_PATH = "dataset/nfcorpus/qrels/train.tsv"

TOP_K = 25 #2 5 10 25

profiler = cProfile.Profile()

# Score a single query
def score_query(bm25, lexer, inverted_index, query, ground_truth_answers):
    if len(ground_truth_answers) == 0:
        return None, [], []

    query_tokens = lexer.tokenize(query)
    
    profiler.enable()
    ranked_docs = bm25.rank(query_tokens)[:TOP_K]
    profiler.disable()
    
    ranked_docs_dict = {doc_id: score for doc_id, score in ranked_docs}
    ground_truth_dict = {doc_id: 1 for doc_id in ground_truth_answers.keys()}

    # Create relevance lists
    y_true = [ground_truth_dict.get(doc_id, 0) for doc_id in ranked_docs_dict.keys()]
    y_scores = [ranked_docs_dict[doc_id] for doc_id in ranked_docs_dict.keys()]

    # Ensure the lists are valid for NDCG computation
    if len(y_true) <= 1 or len(y_scores) <= 1:
        return None, [], []

    ndcg = ndcg_score([y_true], [y_scores])
    return ndcg, y_true, y_scores


# Create or load inverted index
#@profile
def load_ii(corpus):
    if os.path.exists(INDEX_PATH):
        return Inverted_index().from_pickle(INDEX_PATH)
    
    ii = Inverted_index()
    ii.create(corpus)
    ii.pickle(INDEX_PATH)

    return ii


# Run the evaluation
#@profile
def run(ii, corpus, queries, qrels):
    bm25 = BM25(ii)
    lexer = Lexer()

    ndcg_scores = []
    precision_scores = []
    recall_scores = []

    for x in tqdm(range(len(queries))):
        query = queries["text"][x]
        query_id = queries["_id"][x]
        ground_truth_answers = qrels.get(query_id, {})

        ndcg, y_true, y_scores = score_query(bm25, lexer, ii, query, ground_truth_answers)
        if ndcg is not None:
            ndcg_scores.append(ndcg)

            # Compute Precision and Recall
            precision = math.fsum(y_true)/TOP_K
            recall = math.fsum(y_true)/len(ground_truth_answers)

            precision_scores.append(precision)
            recall_scores.append(recall)

    # Print mean metrics
    print(f"Mean NDCG: {np.mean(ndcg_scores)}")
    print(f"Mean Precision: {np.mean(precision_scores)}")
    print(f"Mean Recall: {np.mean(recall_scores)}")

    # Visualize Precision@k and Recall@k
    # visualize_precision_recall(precision_scores, recall_scores)


# Visualization function
def visualize_precision_recall(precision_scores, recall_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(precision_scores) + 1), precision_scores, label='Precision@k')
    plt.plot(range(1, len(recall_scores) + 1), recall_scores, label='Recall@k')

    plt.xlabel("Queries (k)")
    plt.ylabel("Score")
    plt.title("Precision@k and Recall@k")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    corpus, queries, qrels = load_dataset(CORPUS_PATH, QUERIES_PATH, QRELS_PATH)
    ii = load_ii(corpus)
    
    run(ii, corpus, queries, qrels)
    
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(10)
