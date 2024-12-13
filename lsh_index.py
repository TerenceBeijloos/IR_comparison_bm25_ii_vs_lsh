from collections.abc import MutableMapping
from typing import List
import pickle

KeyType = str
DocIdType = str
FreqType = int

class DocReference:
    def __init__(self, doc_id: DocIdType, doc_len: int):
        self.doc_id = doc_id
        self.doc_len = doc_len
        self.freq = {}
    
    def __repr__(self):
        return f"{self.doc_id}"

class Bucket:
    def __init__(self, doc_refs: List[DocReference]):
        self.doc_refs = doc_refs

    def __hash__(self):
        return hash("".join(str(doc_ref) for doc_ref in self.doc_refs))

    def __repr__(self):
        return f"{self.doc_refs}"
    
class LshInvertedIndex(MutableMapping):
    def __init__(self):
        self._store = {}  # Dictionary to store terms and their postings lists
        self.total_docs = 0  # Total amount of documents in index
        self.total_doc_length = 0 # Total number of tokens across all documents
        self.average_doc_length = 0  # Average document length across the corpus

    def __getitem__(self, key: KeyType) :
        return self._store[key]
    
    def __setitem__(self, key: KeyType, bucket: Bucket):
        if key not in self._store:
            self._store[key] = set()
        self._store[key].add(bucket)
    
    def __delitem__(self, key: KeyType):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def insert_term(self, key: KeyType, bucket: Bucket):
        if key not in self._store:
            self._store[key] = set()

        if bucket not in self._store[key]:
            self._store[key].add(bucket)

    def pickle(self, file_name):
        with open(file_name,"wb") as f:
            pickle.dump(self,f)
            
    def from_pickle(self,file_name):
        with open(file_name,"rb") as f:
            return pickle.load(f)

    def __repr__(self):
        return "\n".join(f"{term}: {buckets}" for term, buckets in self._store.items())