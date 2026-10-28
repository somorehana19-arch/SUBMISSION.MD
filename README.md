task 1 Manual RAG on IACS No. 47

Overview
This project compares BM25 keyword search and embedding search using pages 4–8 of the IACS No. 47 Steel Shipbuilding Quality 2025 standard. It retrieves answers without using an LLM.

How to Run
```bash
pip install pypdf rank-bm25 sentence-transformers numpy
python examples/Solution.py
```

Run from the `task-1-basic-rag` folder. An internet connection is needed on the first run to download the embedding model.

Results
Both methods retrieved relevant pages for all development and test questions. They also correctly identified the two test questions that could not be answered from the supplied pages.

BM25 uses keyword matching, while embeddings search for text with similar meaning. Both performed equally in this evaluation, so there isn't enough evidence to say which method is better.

The main limitation is answer selection. Sometimes the system picks a sentence that doesn't directly answer the question, and extracting tables from the PDF can affect the results.
