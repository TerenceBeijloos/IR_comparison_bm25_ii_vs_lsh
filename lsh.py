from random import shuffle
from lsh_index import *
from lexer import Lexer
from tqdm import tqdm
import pandas as pd
import pickle

# Creates the vocabulary for LSH by shingling 'texts'
def createVocab(texts, shingle_size = 2):
    vocab = set()
    print("Creating vocab...")
    for text in tqdm(texts):
        for i in range(len(text) - shingle_size+1):
            vocab.add(text[i:i + shingle_size])
    return vocab

# Creates a one-hot encoded value for each document in 'texts' compared to 'vocab'
def createOneHotEncoding(texts, vocab):
    encoded_texts = {}
    print("Creating one-hot encoding...")
    for doc_id, text in tqdm(texts.items()):
        encoded_texts[doc_id] = ([1 if x in text else 0 for x in vocab])
    return encoded_texts

# Builds dense vectors for each one_hot encoded array using MinHashing
def createSignatures(one_hot_array, vocab_len, signature_size):
    hash_vectors = buildMinHashVectors(signature_size, vocab_len)
    signatures = {}
    print("Creating signatures...")
    for doc_id, arr in tqdm(one_hot_array.items()):
        signature = []
        for vector in hash_vectors:
            for i in range(1, vocab_len+1):
                index = vector.index(i)
                signature_value = arr[index]
                if signature_value == 1:
                    signature.append(index)
                    break
        signatures[doc_id] = signature
    return signatures

def buildMinHashVectors(n_bits, vocab_len):
    hashes = []
    print("Building minhash vectors...")
    for _ in tqdm(range(n_bits)):
        hash_ex = list(range(1, vocab_len+1))
        shuffle(hash_ex)
        hashes.append(hash_ex)
    return hashes

# Creates LSH buckets, adding documents to buckets when a band (part of the signature) matches
def createBuckets(signatures, band_size):
    buckets = {}
    for doc_id, sig in signatures.items():
        assert len(sig) % band_size == 0
        sig_bands = splitArray(sig, len(sig) / band_size)
        for band in sig_bands:
            key = ','.join(str(x) for x in band)
            if key in buckets:
                buckets[key].append(doc_id)
            else:
                buckets[key] = [doc_id]
    return buckets

# Util functions
def splitArray(arr, chunks):
    splitArr = []
    lenArr = len(arr)
    step = int(lenArr / chunks)
    for i in range(0, lenArr, step):
        x = i
        splitArr.append(arr[x:x+step])
    return splitArr

def get_unique_buckets(buckets):
    result = []
    for bucket in buckets.values():
        if bucket not in result:
            result.append(bucket)
            
    return result

def runLSH(signature_len, amount_bands):

    # Setup corpus
    corpus_file = "./dataset/nfcorpus/corpus.jsonl"
    corpus = pd.read_json(corpus_file, lines=True)
    doc_data = {row["_id"]: row["text"] for _, row in corpus.iterrows()}

    # Run LSH steps
    vocab = createVocab(doc_data.values())
    one_hot_encoded = createOneHotEncoding(doc_data, vocab)
    signatures = createSignatures(one_hot_encoded, len(vocab), signature_len)

    buckets = createBuckets(signatures, signature_len/amount_bands)
    unique_buckets = get_unique_buckets(buckets)

    # Remap the corpus documents back unto the buckets of doc ids
    document_buckets = [{doc_id: doc_data[doc_id] for doc_id in bucket} for bucket in unique_buckets]

    print("Creating unique buckets...")
    lexer = Lexer()

    typed_buckets: List[Bucket] = []

    for bucket in tqdm(document_buckets):
        typed_bucket = []
        for doc_id, doc in bucket.items():
            doc_len = len(lexer.tokenize(doc))
            typed_bucket.append(DocReference(doc_id, doc_len))
        typed_buckets.append(Bucket(typed_bucket))

    return typed_buckets

def writeToPickleFile(buckets, filename): 
    with open(filename, 'wb') as index_file:
        pickle.dump(buckets, index_file)

# Main entry point
if __name__ == "__main__":

    # Create buckets using LSH
    buckets = runLSH(128, 8)

    # Write buckets to pickle file to load when creating the LSH Inverted Index
    writeToPickleFile(buckets, "buckets.pickle")