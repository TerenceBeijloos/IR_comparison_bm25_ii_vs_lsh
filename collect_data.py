from create_lsh_index import create_lii as create_lsh_index
from sklearn.metrics import ndcg_score
from inverted_index import InvertedIndex
from lsh_index import LshInvertedIndex
from lexer import Lexer
from bm25 import BM25
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cProfile
import pstats
import os.path
import math

TOP_K = 25 #2 5 10 25

profiler = cProfile.Profile()

# Load dataset
#@profile
def load_dataset():
    qrels_file = "dataset/nfcorpus/qrels/train.tsv"

    qrels_df = pd.read_csv(qrels_file, sep='\t', names=['query_id', 'doc_id', 'score'])

    qrels = {}
    for _, row in qrels_df.iterrows():
        query_id = row['query_id']
        doc_id = row['doc_id']
        score = row['score']

        # Build the nested dictionary
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][doc_id] = score
            
    corpus_file = "./dataset/nfcorpus/corpus.jsonl"
    queries_file = "./dataset/nfcorpus/queries.jsonl"
    
    corpus = pd.read_json(corpus_file, lines=True)
    queries = pd.read_json(queries_file, lines=True)
    
    return corpus, queries, qrels


# Score a single query
def score_query(bm25, lexer, query, ground_truth_answers):
    if len(ground_truth_answers) == 0:
        return None, [], []

    query_tokens = lexer.tokenize(query)
    profiler.enable()
    ranked_docs = bm25.rank(query_tokens)[:TOP_K]
    profiler.disable()

    ranked_docs_dict = {doc_id: score for doc_id, score in ranked_docs}
    ground_truth_dict = {doc_id: 1 for doc_id in ground_truth_answers.keys()}

    # Create binary relevance lists
    y_true = [ground_truth_dict.get(doc_id, 0) for doc_id in ranked_docs_dict.keys()]
    y_scores = [ranked_docs_dict[doc_id] for doc_id in ranked_docs_dict.keys()]

    # Ensure the lists are valid for NDCG computation
    if len(y_true) <= 1 or len(y_scores) <= 1:
        return None, [], []

    ndcg = ndcg_score([y_true], [y_scores])
    return ndcg, y_true, y_scores


# Create or load inverted index
#@profile
def load_lii(corpus, pickle_file_name):
    if os.path.exists(pickle_file_name):
        return LshInvertedIndex().from_pickle(pickle_file_name)
    
    lii = create_lsh_index(corpus, pickle_file_name)
    lii.pickle(pickle_file_name)

    return lii

def load_ii(corpus, pickle_file_name):
    if os.path.exists(pickle_file_name):
        return InvertedIndex().from_pickle(pickle_file_name)
    
    ii = InvertedIndex()
    ii.create(corpus)
    ii.pickle(pickle_file_name)

    return ii

# Run the evaluation
#@profile
def run(ii, queries, qrels):
    bm25 = BM25(ii)
    lexer = Lexer()

    ndcg_scores = []
    precision_scores = []
    recall_scores = []

    print("Running system evaluation...")

    for x in tqdm(range(len(queries))):
        query = queries["text"][x]
        query_id = queries["_id"][x]
        ground_truth_answers = qrels.get(query_id, {})

        ndcg, y_true, _ = score_query(bm25, lexer, query, ground_truth_answers)
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


# Main entry point
if __name__ == "__main__":
    corpus, queries, qrels = load_dataset()
    lii = load_lii(corpus, "lii.pickle")
    run(lii, queries, qrels)

    # Analyze stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(10)
