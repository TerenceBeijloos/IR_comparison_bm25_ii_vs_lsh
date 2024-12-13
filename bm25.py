import math
from inverted_index import Inverted_index

class BM25:
  
    def __init__(self, inverted_index : Inverted_index, k1=1.5, b=0.75):
        self.inverted_index = inverted_index
        self.k1 = k1
        self.b = b
        self.N = inverted_index.number_of_documents
        self.avg_doc_length = inverted_index.average_doc_length

    # Compute the score given a term and a document id
    def compute_score(self, term, doc_id):
        postings = self.inverted_index.get(term)
        if postings == None:
            return 0
        
        posting = postings.get(doc_id)
        if posting == None:
            return 0

        doc_freq = len(self.inverted_index[term])
        term_freq = posting.frequency
        doc_length = posting.doc_length
        
        if doc_freq > 0:
            idf = math.log(1 + (self.N - doc_freq + 0.5) / (doc_freq + 0.5))  # idf - Inverse Document Frequency
        else:
            idf = 0
        
        # BM25 score
        num = term_freq * (self.k1 + 1)
        den = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
        
        if den != 0:
            return idf * (num / den)
        else:
            return 0

    # Return a list of tuples
    # where the first element of the tuple is the source id and the second is the score or a 1 if binary is set to True
    def rank(self, query_terms, binary=True):
        scores = {}

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            term_docs = self.inverted_index[term] 
            
            for doc_id, term_doc in term_docs.items():
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += self.compute_score(term, doc_id)
        
        # Sort documents based on score
        result = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if len(result) == 0:
            return []
        
        if binary:
            result = [(key, 1) for key, value in result]
        else:
            result = [(key, value) for key, value in result]
        
        return result