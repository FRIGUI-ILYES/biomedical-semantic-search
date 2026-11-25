# Biomedical Semantic Search on NFCorpus

**Does semantic search really beat keyword search on medical questions?**
This project compares **one model from each modern retrieval family** on
[NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/): everyday nutrition
questions from NutritionFacts.org, matched against PubMed scientific abstracts.

Everything starts from the **raw** data and lives in a single notebook,
[`NFCorpus_Semantic_IR.ipynb`](NFCorpus_Semantic_IR.ipynb), which is both the report and the slide deck.

---

## The problem

Keyword search scores **words, not meaning**. A user asks *"What should I eat to avoid a heart attack?"*,
but the relevant abstract talks about *"myocardial infarction"* and *"dietary intervention"*.
In NFCorpus, **half** of the relevant (question, document) pairs share **no** content word at all.
That vocabulary gap is exactly what semantic models claim to close.

## Models compared

All models are used **zero-shot** (no fine-tuning, no tuning on the test set).

| Family | Model | Size | Library |
|---|---|---|---|
| Lexical | BM25 | — | `bm25s` |
| Cross-encoder reranker (on BM25 top-100) | `BAAI/bge-reranker-v2-m3` | 568M | sentence-transformers |
| Dense bi-encoder | `BAAI/bge-base-en-v1.5` | 110M | sentence-transformers |
| Late interaction | `answerdotai/answerai-colbert-small-v1` | 33M | PyLate |
| Learned sparse | `naver/splade-v3` | 110M | sentence-transformers |
| LLM embedder | `Qwen/Qwen3-Embedding-0.6B` | 0.6B | sentence-transformers |

## Results

Test split, 323 questions, `titles` queries (evaluated with `ranx`):

| Model | nDCG@10 | Recall@100 | MRR@10 | Search time / question |
|---|---:|---:|---:|---:|
| **BGE** | **0.375** | 0.337 | 0.565 | 2.7 ms |
| ColBERT | 0.373 | 0.316 | **0.584** | 6.9 ms |
| Qwen3 | 0.370 | **0.338** | 0.571 | 128 ms |
| SPLADE | 0.363 | 0.299 | 0.574 | 3.7 ms |
| Reranker (BM25 → bge-reranker) | 0.328 | 0.244 | 0.548 | 2,134 ms |
| BM25 | 0.317 | 0.244 | 0.522 | 1.3 ms |

**Key findings**

- The 4 semantic first-stage models beat BM25 by **+15% to +18%** nDCG@10 (paired t-test, Bonferroni, p ≤ 2e-5).
- Between themselves they are a **statistical tie** (paired bootstrap, 95% CI ≈ ±0.035). A 110M model (BGE) is as good as a 0.6B LLM embedder, and about 47× faster to search.
- The cross-encoder reranker is **not** significantly better than BM25: it can only reorder BM25's candidates, and most relevant documents share no word with the question.
- Longer questions help (`all` query type): BGE reaches **0.460** nDCG@10.
- 17% of questions are failed by all six models, and 72–76% of each model's top 10 is unlabelled, so all scores are lower bounds.

## Notebook structure

1. **Introduction & problem**: keyword vs meaning, short history of IR, timeline, glossary
2. **State of the art**: retrieve-then-rerank pipeline, one key paper per family, benchmarks (BEIR, MTEB)
3. **Data & preparation**: BEIR format, 5 query types, data leakage, cleaning, tokenization, splits
4. **Methodology & code**: metrics, shared evaluation harness, each model in chronological order, cost
5. **Results**: leaderboard, per-query-type heatmap, win/tie/loss, quality vs cost, live demo
6. **Interpretation & critical analysis**: vocabulary gap, reranker ceiling, failed questions, confidence intervals
7. **Conclusion**: answer, lessons, which model to use, limits, perspectives (RRF fusion)
8. **References**

## Repository layout

```
NFCorpus_Semantic_IR.ipynb   the whole project (report + slides)
requirements.txt             pinned, tested versions
nfcorpus/                    original NFCorpus release (raw/ is the source we clean)
data/processed/              cleaned corpus, queries and qrels in BEIR format
data/runs/<model>/           cached top-100 runs per query type + timing stats
```

## Run it

Tested on Windows 11, Python 3.13, RTX 4060 Laptop (8 GB VRAM).

```bash
python -m venv venv
venv/Scripts/activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name nlp-project --display-name "Python (NLP Project)"
huggingface-cli login            # naver/splade-v3 is gated on Hugging Face
```

Then open the notebook and run all cells. Model runs are **cached** in `data/runs/`, so the
notebook renders in minutes; set `FORCE_RECOMPUTE = True` in the Section 4 config (or delete a
model's folder) to recompute. A full recompute takes about 40 minutes, mostly the reranker.

Headless run:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=nlp-project NFCorpus_Semantic_IR.ipynb
```

As slides (RISE-compatible metadata on every cell):

```bash
jupyter nbconvert --to slides NFCorpus_Semantic_IR.ipynb
```

## Dataset

**NFCorpus**: Boteva, V., Gholipour, D., Sokolov, A., & Riezler, S. (2016).
*A Full-Text Learning to Rank Dataset for Medical Information Retrieval.* ECIR 2016.
Official page: https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/ (BibTeX in `nfcorpus/ecir2016.bib`).

The full list of 24 references (BM25, BEIR, MTEB, ColBERT, SPLADE, BGE, Qwen3 Embedding, …) is in Section 8 of the notebook.

## Author

Ilyes Frigui
