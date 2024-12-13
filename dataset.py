import pandas as pd

# Load dataset given source paths
# @profile
def load_dataset(CORPUS_PATH, QUERIES_PATH, QRELS_PATH):
    qrels_df = pd.read_csv(QRELS_PATH, sep='\t', names=['query_id', 'doc_id', 'score'])

    qrels = {}
    for _, row in qrels_df.iterrows():
        query_id = row['query_id']
        doc_id = row['doc_id']
        score = row['score']

        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][doc_id] = score
    
    corpus = pd.read_json(CORPUS_PATH, lines=True)
    queries = pd.read_json(QUERIES_PATH, lines=True)
    
    return corpus, queries, qrels