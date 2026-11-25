# Audit report: NFCorpus Semantic IR project (final version)

**Date:** 2026-09-27 (second audit, after Sections 6 and 7 were written)
**Scope:** `NFCorpus_Semantic_IR.ipynb` (206 cells, 121 code cells executed in order, no error outputs), the raw and original NFCorpus files, `data/processed/`, the 30 runs in `data/runs/`, `requirements.txt` and `CLAUDE.md`.
**Method:** I read every cell and its outputs, and compared Sections 0–5 with the first audit (only the ColBERT fix and the Contents table changed). I rechecked the key numbers of Sections 6–7 with independent scripts, and ran three side experiments in the scratchpad: real Reciprocal Rank Fusion, paired bootstrap intervals for every model pair, and the reranker applied to BGE's top-100. The notebook itself was not modified.

---

## 1. Verdict

The project is **complete and in good shape**. All 8 sections required by the professor are present, the notebook runs top to bottom without errors, and every number I rechecked matches its output. Sections 6 and 7 are strong: each claim is measured on the runs, and the takeaways are computed, not typed.

The remaining issues are about **how some results are argued**, not about the numbers themselves. Four are worth fixing before submission:
1. **The "statistically tied" argument relies on overlapping confidence intervals** (6.5, 6.9, 7.1). That reasoning would also "tie" BM25 with BGE, whose intervals overlap too. The conclusion is still right, but it should rest on the paired tests (A1).
2. **The fusion "oracle" (0.454) overstates what fusion gives.** Real RRF of the four neural models reaches 0.384, and adding BM25 lowers it (A2).
3. **6.3 attributes ColBERT's tiny `all` gain to its 32-token question limit.** Our own earlier run at 128 tokens gained no more (+0.007), so this cause is not shown (A3).
4. **7.4 proposes reranking a semantic top-100.** A test shows that reranking BGE's list lowers it (0.375 → 0.341). This supports the 6.2 explanation but undercuts the future-work row (A4).

### Status after the fixes (2026-09-27)

The notebook was edited in place and re-executed from top to bottom: 121 code cells in order, no errors. All takeaways stay computed.

| Finding | Status | What changed |
|---|---|---|
| A1 tie argued from overlapping intervals | **Fixed** | 6.5 computes paired bootstrap differences: every neural − BM25 lower bound is ≥ +0.030, and all 6 neural − neural differences include 0. 6.9 ⑤ and 7.1 cite the paired tests. |
| A2 fusion oracle | **Fixed** | 6.7 measures real RRF (0.384) and calls 0.454 an upper bound. The 7.4 row is updated. |
| A3 ColBERT cause on `all` | **Fixed** | 6.3 and 6.9 ③ call the 32-token limit a possible reason. The 7.4 row now reads "test whether…". |
| A4 reranking a dense top-100 | **Fixed (wording)** | The 7.4 row says it must be tested. The 7.3 reranker line is updated. The BGE-list experiment was not added to the notebook. |
| A5 limits table | **Fixed** | Qwen3 instruction wording; duplicate twins (113 test questions, computed); rows added for truncated abstracts (9%) and hyphenated BM25 tokens (7,157, e.g. `plant-based`). |
| A6 ColBERT in 7.3 | **Fixed** | "Same quality as BGE with a very small model". |
| B1 `titles` vs `all` | **Fixed** | 5.2 and 5.6 takeaways state that on `all` BM25 (0.397) is ahead of the reranker (0.384) and ColBERT (0.380). 2.7 now says "Can improve the order of a first-stage list". |
| B2 per-type significance | **Fixed (wording)** | 5.2 says the per-type ranking of neural models is descriptive. |
| B3 cell 127 reproducibility | **Fixed** | Tries the local cache, downloads otherwise. |
| B5 p-value definition | **Fixed** | |
| B6 references | **Mostly fixed** | [24] moved to Section 8; timeline label "BGE reranker [23]". [7]/[8] left uncited on purpose (kept in the list, as CLAUDE.md records). |
| B7 Section 1 | **Fixed** | One line on how the question is answered. |
| B8 minor items | **Partly** | `requirements.txt` has the CUDA extra index. Copyright 572/589 and heading styles left as is (cosmetic). |
| B9 variable reuse | **Fixed** | `q_overlap`, `failed_all`. |
| B4 hard-coded tables | Not changed | Values verified correct; they only matter if a length setting changes. |

---

## 2. Compliance with the professor's requirements

| Requirement | Status |
|---|---|
| Single notebook with integrated code | OK |
| 1 Introduction + problématique | OK (short; see B7) |
| 2 Bref état de l'art + références | OK |
| 3 Données + préparation | OK, strong |
| 4 Méthodologie + code | OK |
| 5 Résultats | OK (B1, B2 still open) |
| 6 Interprétation + analyse critique | OK, new; see A1–A5 |
| 7 Conclusion | OK, new; see A1, A2, A6 |
| 8 Références | OK (minor: B6) |
| Dataset link / source | OK (title slide, Section 3, Section 8) |
| Contents table | Lists sections 1–8 with working anchors |
| Oral (5 min) | Out of scope; slide metadata is on every cell |

---

## 3. Findings on Sections 6 and 7 (new)

### A1. "Statistically tied" is argued from overlapping intervals (6.5, 6.9, 7.1). Medium.
- 6.5 says the neural models' intervals "overlap almost completely", so we cannot tell which is best. 7.1 then says the four are "statistically tied (±0.035, Section 6.5)".
- **Why this is weak:** overlapping per-model intervals do not show that two models are equal. The per-model intervals of BM25 [0.284, 0.352] and BGE [0.340, 0.412] also overlap, yet the paired difference BGE − BM25 is [+0.038, +0.079], clearly above zero. A reader could turn the same argument against the main result.
- **The right evidence already exists or is cheap:** the paired t-tests in 5.1 (BGE vs Qwen3 p = 0.58, SPLADE 0.15, ColBERT 0.77) and the paired bootstrap already computed for BGE − Qwen3. I checked the other pairs with the same paired bootstrap (seed 42, 2,000 resamples): BGE − ColBERT [−0.014, +0.019], BGE − SPLADE [−0.005, +0.029], Qwen3 − SPLADE [−0.008, +0.024]. All include 0, so the tie holds. ColBERT − SPLADE [−0.001, +0.021] is borderline.
- **Fix:** base the tie on the paired tests. In 7.1, cite "paired tests, Section 5.1" instead of "±0.035". Keep ±0.035 as the precision of each score, which is what it is.

### A2. The fusion oracle overstates complementarity (6.7, 7.4). Medium.
- "Best first-stage model per question" = 0.454 is an **upper bound**, and an optimistic one: taking the maximum of five noisy per-question scores is always higher than any single model, even for models that do not complement each other.
- **Real fusion, measured** (ranx `fuse(method="rrf")`, titles):

  | RRF of | nDCG@10 |
  |---|---|
  | BGE (best single model) | 0.375 |
  | BGE + ColBERT | 0.381 |
  | BGE + Qwen3 + SPLADE + ColBERT | **0.384** |
  | BM25 + the four neural models | 0.377 |
  | BM25 + BGE | 0.355 (worse than BGE alone) |

- **Fix:** in 6.7, call 0.454 an upper bound, not an expected result. In the 7.4 row "Combine models", replace "Higher nDCG@10" with a realistic expectation (about +0.01 in our data), and note that adding BM25 does not help.

### A3. The cause given for ColBERT's small `all` gain is not supported (6.3, 6.9 ③, 7.4). Medium.
- 6.3 says ColBERT gains only +0.008 **because** it reads 32 question tokens, and calls this "a limit of the model's configuration". 7.4 proposes "Longer ColBERT questions → fair comparison on long questions".
- **Our own data contradicts the causal link:** the earlier run with 128 question tokens (before the fix) gained **+0.007** on `all`, no more than at 32. Reading more of the question did not help. That run is confounded by the `[MASK]` padding, so neither setting is a clean test.
- The sentence "cut to little more than their title" is also imprecise: 32 tokens usually hold the title plus the start of the description.
- **Fix:** present the 32-token limit as a *possible* cause ("ColBERT reads only the first 32 tokens, the length it was trained with, so most of the page is ignored"), without "because". In 7.4, keep the idea but change the expected effect to "test whether longer questions help ColBERT at all".

### A4. The reranker conclusion in 6.2 is right, but the evidence is indirect, and 7.4 expects a gain the data does not show. Medium.
- 6.2 concludes that "the list is not the main problem" because a perfect reordering of BM25's 100 candidates would reach 0.546. The ceiling alone is weak evidence: every first-stage list has a high ceiling (BGE's top-100 would reach 0.662).
- **Direct test (side experiment):** reranking **BGE's** top-100 instead of BM25's gives **0.341**. That is better than BM25 + reranker (0.328) but **worse than BGE alone (0.375)**. With better candidates, the cross-encoder still lowers the ranking. This **supports** 6.2's conclusion that the reranker's judgment disagrees with the labels, and it is stronger evidence than the ceiling.
- **It contradicts 7.4**, whose row "Rerank a semantic top-100 instead of BM25's" expects "more relevant candidates for the reranker" to help. Measured on our data, it does not beat the first-stage model.
- **Fix:** optionally add the BGE-list result to 6.2 as the direct test. In 7.4, change the expected effect to "better candidates alone are not enough (a test gives 0.341 < 0.375); a reranker would need medical or in-domain tuning", or drop the row.

### A5. Limits table (6.8): three small corrections. Low.
- "Zero-shot, nothing tuned: Qwen3 instruction and BM25 k1/b **left at defaults**". The Qwen3 instruction was written by us, not a default. Suggested wording: "BM25 k1/b at literature defaults; Qwen3 instruction written once, not tuned".
- "Duplicate abstracts kept: both copies can take two top-10 slots". The measured problem is different: the two copies are **judged differently** in 40 of 41 groups (113 of 323 test questions affected). An unlabelled twin of a relevant document is then counted as wrong. The effect is small (4–9 such cases per model in 3,230 top-10 slots), so "small effect" stays true.
- Two limits are missing: **document truncation** (9% of abstracts are cut at 512 tokens for BGE, SPLADE, ColBERT and the reranker) and **BM25 tokenization of hyphenated words** (`b-12` ≠ `b12`), which is still the open question from Section 3.

### A6. Conclusion (7.x): small points. Low.
- 7.3 recommends BGE. That is reasonable, but ColBERT is equal on quality (0.373, best MRR@10 0.584, 6.9 ms), and the table lists only "very small model" for it. Consider one clause saying ColBERT is an equally good choice when the index size is acceptable.
- 7.2 point 4 and 7.3 are consistent with 5.5. No issue with the numbers.

### Verified as correct in Sections 6–7
- 6.1: every model's score on low-overlap questions is about half its high-overlap score (ratios 0.46–0.53). The reranker equals BM25 on low overlap (0.222 vs 0.223). The takeaway is accurate.
- 6.2: ceiling 0.546, level-2 (119 questions) and level-1 (310 questions) tables, and the 95% level-1 share.
- 6.4: 54 questions failed by all models. I also checked that **81% of them are topic pages** (vs 50% of the other questions), which supports "one- or two-word topic names". This is worth adding as evidence.
- 6.5: bootstrap ±0.035; BGE − Qwen3 in [−0.012, +0.022].
- 6.6: 72–76% of each top 10 unlabelled.
- 6.7: Spearman correlations 0.80–0.94; oracle 0.454 (the value is correct; only its reading is the issue, A2).
- 7.1: +15% to +18% over BM25, p ≤ 2e-6, reranker not significant.

---

## 4. Findings carried over from the first audit (Sections 1–5)

Resolved since the first audit:
- **ColBERT query length** (was M1): fixed; ColBERT now scores 0.373 on `titles`.
- **Missing Sections 6 and 7:** written.
- **[15] RRF uncited:** now cited in 6.7 and 7.4.
- **Level-1 labels and duplicates not discussed:** covered in 6.2, 6.6 and 6.8 (see A5 for the duplicate wording).

Still open:
- **B1. Section 5 generalises from `titles` to all query types (was M2). Medium.** On `all`, BM25 (0.397) beats ColBERT (0.380) and the reranker (0.384): the reranker makes its own BM25 list **worse** there. Still true but still unqualified: 5.6 ① "Every neural first-stage model beats BM25", 5.6 ③ "the reranker barely improves on BM25" and 2.7 "Improves any first-stage model". Adding "on `titles`" to 5.6 ① and ③, and one sentence in 6.3 about the reranker on `all`, would close it.
- **B2. No significance test per query type (was M3). Low–medium.** "BGE best on 4 of 5 types, Qwen3 on vid-desc" rests on 102–144 questions with gaps of 0.002–0.02. After A1, a clause such as "descriptive; the differences are within the uncertainty of 6.5" would be enough.
- **B3. Reproducibility of cell 127 (was D1). Medium.** The SPLADE illustration still loads with `local_files_only=True`. On a machine that has `data/runs/` but not the Hugging Face cache, the notebook stops at this cell. A `try/except` with a short note avoids that.
- **B4. Hard-coded numbers (was D4). Low.** The 4.1 truncation table and the `CUT_ALL` dictionary in 6.3 are typed by hand (values checked and correct). They will only go stale if a length setting changes again.
- **B5. p-value definition in the glossary (was D3). Low.** Suggested: "probability of a difference at least this large *if the two models were in fact equal*; small (< 0.01 here) = unlikely to be chance".
- **B6. References (was m1–m3). Low.** [24] ranx still has its anchor in 4.1 (cell 104), not in Section 8, although CLAUDE.md says Section 8 holds [1]–[24]. [7] Contriever and [8] E5 are listed but never cited. In the timeline, "BGE-M3 [23]" sits in the *Reranking* lane (it is the embedding model the reranker is built on). [15] is ★ under "Reranking & fusion", which now fits since fusion is discussed in 6.7.
- **B7. Section 1 does not state the evaluation plan (was m8). Low.** One line ("answered with nDCG@10 on 323 test questions and paired significance tests") would link the question to Sections 4–7.
- **B8. Other minor items (were m4–m7, m9).** Copyright counts 572 vs 589 not explained. Heading styles differ (Section 3 "Part A–E" vs 4.x/5.x/6.x/7.x). `requirements.txt` needs the CUDA index for torch. 4 test titles equal a train title (irrelevant for zero-shot).
- **B9. Code hygiene (new). Low.** Section 6 reuses names defined earlier: `failed` (the list of failed checks in Part D becomes the "failed by all six" Series in 6.4), and `overlap` (A11 list → 6.1 Series). The notebook is right when run top to bottom, but re-running the Part E summary cell after Section 6 would print a wrong "checks passed" count. Renaming the Section 6 variables (e.g. `failed_all`, `q_overlap`) removes the trap.

---

## 5. Side experiments (scratchpad only, not in the notebook)

| Experiment (titles, test) | Result | Used in |
|---|---|---|
| Paired bootstrap BGE − BM25 | [+0.038, +0.079] | A1 |
| Per-model bootstrap BM25 / BGE | [0.284, 0.352] / [0.340, 0.412] (they overlap) | A1 |
| Paired bootstrap among neural models | all intervals contain 0 (ColBERT − SPLADE borderline) | A1 |
| RRF of the 4 neural models | 0.384 (vs 0.375 best single, 0.454 oracle) | A2 |
| RRF BM25 + BGE | 0.355 | A2 |
| ColBERT at 128 question tokens (first audit) | `all` gain +0.007 | A3 |
| Reranker on BGE's top-100 | 0.341 (BGE alone 0.375; BM25 + reranker 0.328) | A4 |
| Perfect reordering of BGE's top-100 | 0.662 (BM25's: 0.546) | A4 |
| Failed questions that are topic pages | 81% (vs 50%) | 6.4 check |

---

## 6. Suggested final touches, in order of value

1. A1: base "statistically tied" on the paired tests (6.5, 6.9 ⑤, 7.1). Wording only.
2. A2: call the 0.454 oracle an upper bound; adjust the 7.4 fusion row.
3. A3: remove the causal "because" for ColBERT on `all`; adjust the 7.4 ColBERT row.
3b. A4: adjust the 7.4 row "Rerank a semantic top-100" (measured 0.341 < 0.375).
4. B1: add "on `titles`" to 5.6 ① and ③, and one sentence on `all` (BM25 ahead of ColBERT and the reranker).
5. B3: make cell 127 robust.
6. A4, A5, B2, B5, B6, B9: small wording and hygiene fixes.

None of these changes the main answer of the project: on `titles`, all four semantic first-stage models beat BM25 significantly (+15% to +18% nDCG@10), and they cannot be separated from each other.

---

## 7. Third check (2026-09-27, after the slide restructuring)

**What changed since the fixes:** 9 continuation sub-slides were added. Each starts with a short `### x.y · …` title and moves a chart to its own slide (2.9, A4, A5, A8, C, D, 4.6, 4.9, 5.1). "How every model is run" became "4.2b". `styled()` now left-aligns text columns. No analysis code or number changed.

**State of the notebook:** 215 cells; 121 code cells executed in order, no errors. Every cell has slide metadata: 9 `slide` (the title and the 8 sections), 80 `subslide`, 42 `fragment`, 8 `skip`, 76 `-`. No decorative emojis. The Contents table lists sections 1–8. The saved runs are the ones verified in this audit (ColBERT recomputed at 32 query tokens; the other models unchanged).

**Remaining points (all low, none affects a result):**
- Section 2 still calls the reranker "most accurate" (2.7 text and table, 2.9 cards and takeaway), without qualification. Our own data disagrees, which 6.2 explains. Suggested: "most accurate in published benchmarks".
- 4.1: "These are the models' own limits … not tuned choices" is not quite true for ColBERT's 512 document tokens, which we chose (the model was trained with 300).
- "4.2b" is the only lettered sub-section number in Sections 4–7; "4.2 · How every model is run (continued)" would match the other continuation slides.
- The glossary has no entries for terms used in Sections 5–6: *confidence interval / bootstrap*, *paired test*, *Reciprocal Rank Fusion*.
- Unchanged from before: the 4.1 truncation table and `CUT_ALL` are typed by hand (correct); [7] and [8] are uncited by choice; the reranker-on-BGE result (0.341) is only in this audit.

**Verdict:** ready to submit. The deliverable meets all written requirements, and the numbers and conclusions are consistent from Section 3 to Section 7.

---

## 8. Readability pass (2026-09-29)

Text only; no code, run or number changed.
- **Pipeline made visible:** a breadcrumb under every section header (① Problem → ② State of the art → ③ Data → ④ Models → ⑤ Results → ⑥ Analysis → ⑦ Conclusion, current step in bold) and a new slide **1.4 Our pipeline** (step → what we do → what comes out → section).
- **Charts:** every chart slide now says how to read it (axes, colours, what "good" looks like).
- **Order stated** at the start of Part A, Part B and Section 4; bridge lines at the end of Part E and in A12.
- **Plainer wording:** nDCG@10 definition ("counts fully at rank 1, half at rank 3, less than a third at rank 10"), the 5.1 significance test, cosine, SPLADE and BM25 formulas each get one "In words" / example line.
- **Glossary:** + truncation, cosine similarity, paired test, bootstrap / confidence interval, fusion (RRF) (closes a point of §7).
- **Fixed (§7):** 4.2 no longer says the input limits are "not tuned choices": ColBERT's 512 document tokens and Qwen3's 2,560 are ours.
- The other §7 points were already fixed before this pass: no "4.2b" left, and the reranker is "most precise in published benchmarks".

---

## 9. Section 3 condensed, timeline, fonts (2026-09-29)

- The `Pipeline: ① … ⑦` lines and the 1.4 plan slide were removed at the user's request.
- **Section 3: 82 → 33 cells**, in the order the user asked for:
  - loading, and why only 3,633 documents are used;
  - data leakage;
  - cleaning, with one before/after example;
  - tokenization, stopwords, and stemming vs lemmatization (Snowball stemming is the one we use);
  - the final files;
  - the train/dev/test split;
  - the summary.
- **Removed from the notebook:** everything that compared our data with the pre-tokenized files, the vocabulary-gap slide, the validation checks, the manual review and the exploratory charts.
- 3.1 now gives each BEIR file its purpose and why it has that format. 3.2 says which page fields build each query type, and why there are five.
- **Wrong example fixed:** Snowball stems *dietary* to `dietari`, not `diet`.
- The timeline shows only our six models, at the year their family appeared.
- **Fonts:** Segoe UI in the charts, plus a CSS block for markdown text and tables.
- Short comments were added where each model is loaded. "Not tuned" is now explained in 4.4 and in the glossary.
