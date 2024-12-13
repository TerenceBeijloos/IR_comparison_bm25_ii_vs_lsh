import cProfile
import pstats
from sklearn.metrics import ndcg_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import os

from inverted_index import *
from lexer import *
from bm25 import BM25
from dataset import load_dataset

# Change to your java path
os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-23"
# Change to your index location, relative paths did not work
INDEX_PATH = 'C:/dev/RU/IR/IR_group_18/index'
CORPUS_PATH = "./dataset/nfcorpus/corpus.jsonl"
QUERIES_PATH = "./dataset/nfcorpus/queries.jsonl"
QRELS_PATH = "dataset/nfcorpus/qrels/train.tsv"

import pyterrier as pt
import re

TOP_K = 25 #2 5 10 25

profiler = cProfile.Profile()

# Remove special characters since pyterrier cant deal with them
def preprocess_query(query):
    query = re.sub(r"[^\w\s]", "", query)
    query = " ".join(query.split())

    return query.lower()

# Score a single query
def score_query(bm25, lexer, inverted_index, query, ground_truth_answers):
    if len(ground_truth_answers) == 0:
        return None, [], []

    query = preprocess_query(query)
    profiler.enable()
    ranked_docs = bm25.search(query)
    profiler.disable()
    
    ranked_docs_dict = {row['docno']:1 for _, row in ranked_docs.iterrows()}
    ground_truth_dict = {doc_id: 1 for doc_id in ground_truth_answers.keys()}

    # Create relevance lists
    y_true = [ground_truth_dict.get(doc_id, 0) for doc_id in ranked_docs_dict.keys()]
    y_scores = [ranked_docs_dict[doc_id] for doc_id in ranked_docs_dict.keys()]

    # Ensure the lists are valid for NDCG computation
    if len(y_true) <= 1 or len(y_scores) <= 1:
        return None, [], []

    ndcg = ndcg_score([y_true], [y_scores])
    return ndcg, y_true, y_scores

# Helper function for pyterrier
def corpus_to_dict(corpus):
    lexer = Lexer()
    result = []
    for _, row in corpus.iterrows():
        result.append({"docno":row["_id"],"text":lexer.tokenize(row["text"])})
        
    return result

# Create or load inverted index
# @profile
def load_ii(corpus):
    corpus_dict = corpus_to_dict(corpus)
    if not os.path.exists(INDEX_PATH+'/data.properties'):
        indexer = pt.index.IterDictIndexer(INDEX_PATH)
        indexref = indexer.index(corpus_dict, fields=('docno', 'text'))
        index = pt.IndexFactory.of(indexref)
    else:
        index = pt.IndexFactory.of(INDEX_PATH)

    return index

# Run the evaluation
# @profile
def run(bm25, corpus, queries, qrels):
    lexer = Lexer()

    ndcg_scores = []
    precision_scores = []
    recall_scores = []

    for x in tqdm(range(len(queries))):
        query = queries["text"][x]
        query_id = queries["_id"][x]
        ground_truth_answers = qrels.get(query_id, {})

        ndcg, y_true, _ = score_query(bm25, lexer, ii, query, ground_truth_answers)
        if ndcg is not None:
            ndcg_scores.append(ndcg)

            # Compute Precision and Recall
            precision = math.fsum(y_true)/TOP_K
            recall = math.fsum(y_true)/len(ground_truth_answers)

            precision_scores.append(precision)
            recall_scores.append(recall)

    # Print results
    print(f"Mean NDCG: {np.mean(ndcg_scores)}")
    print(f"Mean Precision: {np.mean(precision_scores)}")
    print(f"Mean Recall: {np.mean(recall_scores)}")

    # visualize_precision_recall(precision_scores, recall_scores)


# Visualization function for precision and recall
def visualize_precision_recall(precision_scores, recall_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(precision_scores) + 1), precision_scores, label='Precision')
    plt.plot(range(1, len(recall_scores) + 1), recall_scores, label='Recall')

    plt.xlabel("Queries (k)")
    plt.ylabel("Score")
    plt.title("Precision and Recall")
    plt.legend()
    plt.grid()
    plt.show()

# Load pyterrier bm25 model
# @profile
def load_bm25(ii):
    return pt.terrier.Retriever(ii, wmodel="BM25",num_results=TOP_K)

# Main entry point
if __name__ == "__main__":
    corpus, queries, qrels = load_dataset(CORPUS_PATH, QUERIES_PATH, QRELS_PATH)
    ii = load_ii(corpus)
    bm25 = load_bm25(ii)
    run(bm25, corpus, queries, qrels)

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(10)
