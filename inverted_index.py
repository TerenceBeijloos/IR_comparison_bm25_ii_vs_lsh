from collections.abc import MutableMapping
from lexer import *
from datasets import *
import pickle

KeyType = str

class Posting:
    def __init__(self, doc_id, doc_length, f=1):
        self.doc_id = doc_id
        self.frequency = f
        self.doc_length = doc_length

    def __lt__(self, other):
        return self.docId < other.docId

    def __repr__(self):
        return f"({self.docId}, freq: {self.frequency})"

class Inverted_index(MutableMapping):
    def __init__(self):
        self._store = {}  # Dictionary to store terms and their postings lists
        self.total_doc_length = 0  # Total number of tokens across all documents
        self.average_doc_length = 0  # Average document length across the corpus
        self.number_of_documents = 0

    # Returns postings given a term
    def __getitem__(self, key: KeyType):
        return self._store[key]

    # Set postings for a term
    def __setitem__(self, key: KeyType, value):
        if key not in self._store:
            self._store[key] = {}

        self._store[key] = value
        
    # Delete term from the inverted index
    def __delitem__(self, key: KeyType):
        del self._store[key]

    # Iterator for the inverted index
    def __iter__(self):
        return iter(self._store)

    # Returns the number of terms in the inverted index
    def __len__(self):
        return len(self._store)

    # Creates the inverted index
    # Corpus is expected to be a datasource which provides iterrows() which returns for each element _, dictionary
    # Dictionary should provide the source id given source_key and the data should be available via data_key
    def create(self,corpus,source_key="_id",data_key="text"):
        lexer = Lexer()
        doc_count = 0
        for _, row in corpus.iterrows():
            doc_count += 1
            doc_id = row[source_key]
            tokens = lexer.tokenize(row[data_key])
            for token in tokens:
                self.insert_term(token, doc_id, len(tokens))
                
        self.average_doc_length = self.total_doc_length / doc_count
        self.number_of_documents = doc_count

    # Updates document information, function is used internally
    def update_doc_info(self, token_count):
        self.total_doc_length += token_count

    # Inserts a term into the inverted index, function is used internally
    def insert_term(self, term: KeyType, doc_id, doc_length):
        posting_map = self._store.get(term)
        if term not in self._store:
            posting_map = {}
            self._store[term] = posting_map

        posting = posting_map.get(doc_id)
        if posting is None:
            posting = Posting(doc_id,doc_length,0)
            self._store[term][doc_id] = posting
        
        posting.frequency += 1
        self.update_doc_info(1)

    # Returns a string representation of the inverted index, listing each term and its postings list.
    def __str__(self):
        output = []
        for term, postings in self._store.items():
            output.append(f"{term}: {list(postings)}")
        return "\n".join(output)

    # Write the index to a file for inspection purposes
    def to_file(self, filename):
        # utf-8 is needed since some Greek symbols are used
        with open(filename, "w", encoding="utf-8") as f:
            for term, postings in self._store.items():
                f.write(f"{term}")
                for name, p in postings.items():
                    f.write(f" {p.doc_id} {p.doc_length} {p.frequency}")
                f.write("\n")

    # Stores the file in binary format so it can be loaded again with from_pickle so the index only has to be created once
    def pickle(self, file_name):
        with open(file_name,"wb") as f:
            pickle.dump(self,f)
            
    # Returns a Inverted_index from a binary file created by the method pickle
    def from_pickle(self,file_name):
        with open(file_name,"rb") as f:
            return pickle.load(f)

        
