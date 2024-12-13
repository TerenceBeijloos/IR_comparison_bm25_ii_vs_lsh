from lexer import Lexer
from inverted_index import InvertedIndex
from lsh_index import LshInvertedIndex
from tqdm import tqdm
import pickle

def update_doc_info(lii, ii):
    lii.total_docs = ii.number_of_documents
    lii.total_doc_length = ii.total_doc_length
    lii.average_doc_length = ii.average_doc_length

def get_unique_buckets():
    with open("buckets.pickle", "rb") as file:
        buckets = pickle.load(file)
        
    result = []
    for bucket in buckets:
        if bucket not in result:
            result.append(bucket)
            
    return result
    
def create_lii(corpus, lii_file_name):
    print("Creating LSH Inverted Index...")

    lexer = Lexer()
    lii = LshInvertedIndex()
    ii = InvertedIndex().from_pickle("ii.pickle")
    update_doc_info(lii, ii)

    # Setup corpus
    doc_data = {row["_id"]: row["text"] for _, row in corpus.iterrows()}

    # Filter out duplicate buckets 
    unique_buckets = get_unique_buckets()

    # Populate LSH Inverted Index
    for bucket in tqdm(unique_buckets):
        for doc_ref in bucket.doc_refs:
            tokens = lexer.tokenize(doc_data[doc_ref.doc_id])
            doc_ref.doc_len = len(tokens)
            for token in tokens:
                doc_ref.freq[token] = ii._store[token][doc_ref.doc_id].frequency
                lii.insert_term(token, bucket)
    
    # Write new LSH inverted index to pickle file to be loaded next time
    with open(lii_file_name,"wb") as f:
        pickle.dump(lii,f)

    return lii