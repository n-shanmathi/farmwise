# Farmwise - Geography Resolver

## Overview

The Geography Resolver converts a natural-language location query into a canonical Tamil Nadu geographic location.

It supports:

- State
- District
- Taluk
- Block
- Village

The resolver uses:

1. Ministral 3 8B for location extraction
2. Amazon Titan Text Embeddings V2 for semantic retrieval
3. NumPy vector index for similarity search
4. Reranking using name, place type and hierarchy information
5. Ambiguity detection to avoid incorrect geographic resolution

---

## Project Structure

```text
Geography_Model/
│
├── api/
│   └── main.py
│
├── scripts/
│   └── geography_resolver.py
│
├── output/
│   ├── geography_numpy_index.npz
│   └── geography_numpy_metadata.csv
│
├── requirements.txt
├── README.md
└── .gitignore