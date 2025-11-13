# Master Thesis – Risk Disclosure Text Mining (FinBERT & Sentence-LDA)

This repository contains the complete code used in my master's thesis on risk disclosure text analysis of FinTech firms.  
The project has three main components:

- `00_preprocessing/` – extract risk-factor sections, clean titles/table-of-contents, and split into sentences.
- `01_finbert/` – FinBERT-based embeddings, vector deduplication, UMAP + HDBSCAN clustering, and cluster-level summaries.
- `02_sentence_lda/` – sentence-level LDA pipeline including preprocessing, phrase mining, model selection, training (K=92), and topic interpretation.

> **Note:** All large proprietary datasets (full risk disclosures, Parquet/CSV files) are excluded from this repository.  
> Only code and minimal configuration files are provided.

## Folder structure

- `00_preprocessing/`
  - `00_collect_raw_data.py`
  - `01_extract_risk_txt.py`
  - `02_clean_titles.py`
  - `02_remove_toc_page_numbers.py`
  - `03_split_sentences.py`

- `01_finbert/`
  - `00_compare_bert_models.py`
  - `01_split_semicolons.py`
  - `02_compute_sentence_embeddings.py`
  - `03_dedup_(sim_threshold=0.97)_and_remove_boilerplate.py`
  - `04_umap_reduction.py`
  - `05_hdbscan_cluster_generation.py`
  - `06_auto_merge_small_clusters.py`
  - `07_apply_merge_mapping.py`
  - `08_cluster_size_and_centroid_recalculation.py`
  - `09_cluster_visualization_and_similarity.py`
  - `10_cluster_level_summaries_for_naming.py`

- `02_sentence_lda/`
  - `00_preprocess_tokens.py`
  - `00_stopwords/` (stopword and phrase lists)
  - `01_phrase_mining.py`
  - `02_apply_all_stopwords.py`
  - `03_clean_final_corpus_and_descriptive_stats.py`
  - `04_lda_model_selection.py`
  - `05_lda_train_final_K92.py`
  - `06_lda_topic_interpretation.py`
  - `07_tf_idf.py`
  - `08_topic_labeling.py`

## Environment

- Python 3.9+
- Main libraries: `pandas`, `numpy`, `tqdm`, `tomotopy`, `gensim`, `sentence-transformers`, `transformers`, `umap-learn`, `hdbscan`, `matplotlib`, `seaborn`, `scikit-learn`

A full `requirements.txt` can be added later if needed.
