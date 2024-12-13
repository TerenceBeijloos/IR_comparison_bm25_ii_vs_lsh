from sortedcontainers import SortedList
from collections.abc import MutableMapping
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

class InvertedIndex(MutableMapping):
    def __init__(self):
        self._store = {}  # Dictionary to store terms and their postings lists
        self.total_doc_length = 0  # Total number of tokens across all documents
        self.average_doc_length = 0  # Average document length across the corpus
        self.number_of_documents = 0

    def __getitem__(self, key: KeyType):
        return self._store[key]

    def __setitem__(self, key: KeyType, value: Posting):
        if key not in self._store:
            self._store[key] = SortedList()
        self._store[key].add(value)

    def __delitem__(self, key: KeyType):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def update_doc_info(self, token_count):
        self.total_doc_length += token_count

    def insert_term(self, term: KeyType, doc_id, doc_length):
        """
        Inserts a term into the inverted index for a given document ID.
        
        :param key: The term to be indexed.
        :param doc_id: The document ID in which the term appears.
        """
        posting_map = self._store.get(term)
        if term not in self._store:
            posting_map = {}
            self._store[term] = posting_map

        # Check if the document ID already exists in the postings list
        posting = posting_map.get(doc_id)
        if posting is None:
            posting = Posting(doc_id,doc_length,0)
            self._store[term][doc_id] = posting
        
        posting.frequency += 1
        self.update_doc_info(1)

    def __str__(self):
        """
        Returns a string representation of the inverted index, listing each term
        and its postings list.
        """
        output = []
        for term, postings in self._store.items():
            output.append(f"{term}: {list(postings)}")
        return "\n".join(output)

    def to_file(self, filename):
        # utf-8 is needed since some Greek symbols are used
        with open(filename, "w", encoding="utf-8") as f:
            for term, postings in self._store.items():
                f.write(f"{term}")
                for _, p in postings.items():
                    f.write(f" {p.doc_id} {p.doc_length} {p.frequency}")
                f.write("\n")

    def pickle(self, file_name):
        with open(file_name,"wb") as f:
            pickle.dump(self,f)
            
    def from_pickle(self,file_name):
        with open(file_name,"rb") as f:
            return pickle.load(f)

        
