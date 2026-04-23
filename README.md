# MA5770 Project: Johnson-Lindenstrauss Lemma & Locality-Sensitive Hashing for Near-Duplicate Detection

## Overview
This project explores the application of the **Johnson–Lindenstrauss (JL) Lemma** and **Locality-Sensitive Hashing (LSH)** for scalable **near-duplicate detection** and **high-dimensional similarity search** across multiple modalities (text, images, speech).

The core idea is to combine:
- Dimensionality reduction (JL Lemma)
- Approximate similarity search (LSH)

to overcome the curse of dimensionality.

---

## Key Concepts

### Johnson–Lindenstrauss Lemma
Projects high-dimensional data into a lower-dimensional space while approximately preserving distances:

k = O(log(n) / ε²)

### Locality-Sensitive Hashing (LSH)
Uses hash functions that map similar items to the same buckets with high probability, enabling fast approximate nearest neighbor search.

---

## Features
- Multi-modal experiments (NLP, CV, Speech)
- JL-based dimensionality reduction
- Custom LSH (random hyperplane hashing)
- Graph-based clustering via LSH
- Evaluation metrics: ARI, NMI, Purity, Silhouette
- Near-duplicate dataset generation
- Scalability and memory benchmarking

---

## Project Structure

```
MA5770_PROJECT/
│
├── wikipedia_outputs/
│   ├── articles_sample.csv
│   ├── clustering_results.csv
│   ├── duplicates.csv
│   ├── final_summary.csv
│   ├── pair_discovery_results.csv
│   ├── qualitative_examples.csv
│   ├── retrieval_results.csv
│   ├── scaling_results.csv
│   ├── storage_summary.csv
│
├── feature_utils.py
├── jl_projection.py
├── lsh_core.py
├── metrics_utils.py
│
├── MA5770_Expt1.ipynb
├── MA5770_Expt2.ipynb
├── MA5770_Expt3.ipynb
├── MA5770_Expt4.ipynb
├── MA5770_Expt5.ipynb
├── MA5770_Expt6.ipynb
├── MA5770_Final_Expt.ipynb
│
├── reproducibility_demo.py
└── README.md
```

---

## Installation

pip install numpy pandas scikit-learn matplotlib tqdm
pip install tensorflow tensorflow-datasets
pip install hdbscan librosa
pip install datasets sentence-transformers

---

## How to Run

Run notebooks:
jupyter notebook MA5770_Expt1.ipynb

Run full pipeline:
python reproducibility_demo.py

---

## Experimental Highlights
- JL reduces dimensionality significantly with minimal distortion
- LSH enables scalable approximate search
- Graph-based clustering improves grouping of near-duplicates
- Traditional clustering struggles in high dimensions

---

## Methodology
1. Feature extraction (TF-IDF / image / audio)
2. Normalize vectors
3. Apply JL projection
4. Perform clustering and retrieval
5. Apply LSH for scalability
6. Evaluate performance

---

## References
- Johnson & Lindenstrauss (1984)
- Indyk & Motwani (1998)

---

## License
MIT License

---

## Author
Puneet
