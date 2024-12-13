#updated BM25
import math

class BM25:
  
    def __init__(self, lii, k1=1.5, b=0.75):
        self.lii = lii
        self.k1 = k1
        self.b = b
        self.N = lii.total_docs  # Total number of documents
        self.avg_doc_length = lii.average_doc_length   # Average document length
        if self.avg_doc_length <= 0:
            self.avg_doc_length = 1

    def compute_score(self, term, doc_ref):
        doc_id = doc_ref.doc_id
        if term in self.lii:
            doc_freq = len(self.lii[term])
        else:
            doc_freq = 0

        term_freq = doc_ref.freq.get(term,0.5)
        
        doc_length = doc_ref.doc_len
        
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
        
    def concat(self, term, lii):
        result = set()
        for bucket in lii._store[term]:
            for doc in bucket.doc_refs:
                result.add(doc)
            
        return result
        
    def rank(self, query_terms):
        scores = {}

        for term in query_terms:
            if term not in self.lii:
                continue
            term_docs = self.concat(term,self.lii)
            
            for doc_ref in term_docs:
                if doc_ref.doc_id not in scores:
                    scores[doc_ref.doc_id] = 1
                scores[doc_ref.doc_id] += self.compute_score(term, doc_ref)
        
        result = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if len(result) == 0:
            return []
        
        result = [(key, 1) for key, _ in result]

        return result