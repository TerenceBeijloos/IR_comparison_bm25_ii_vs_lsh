## Branches Overview

This repository contains two key branches, each focused on distinct aspects of our information retrieval (IR) project:

### `bm25_ii_vs_pyterrier`

This branch compares our custom IR system with PyTerrier. Our IR system implements the BM25 algorithm for ranking and evaluates performance using:
- **Speed**: Query execution time.
- **Memory Usage**: Resource consumption during indexing and retrieval.
- **Accuracy**: Relevance of retrieved documents, measured by Normalized Discounted Cumulative Gain (NDCG).

The comparison focuses exclusively on the Normal Inverted Index and its performance relative to PyTerrier.

### `lsh`

This branch addresses data collection and preprocessing for the IR system using Locality-Sensitive Hashing (LSH). Key aspects include:
- **Document Bucketing**: Grouping similar documents into buckets.
- **Index Construction**: Building inverted indexes within buckets for efficient retrieval.

This branch highlights the scalability of LSH for handling large datasets.

## Context and Insights

This project explores indexing trade-offs in IR systems using the NFCorpus dataset. The report provides further details on experimental design, implementation, and evaluation.

---

For additional details, refer to the corresponding Python files and the project report.
