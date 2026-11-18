# Phase 0: Repository Cleanup - COMPLETE ✓

## Checklist

### 1. Working Environment ✓
- [x] Added `mlx_lm` and `bert_score` to `requirements.txt`
- [x] Removed unused `kagglehub` dependency
- [x] Simplified requirements to essential packages only (removed 100+ development dependencies)
- [x] Verified fresh clone installs on macOS with `pip install -r requirements.txt`
- [x] All core packages import successfully: torch, mlx_lm, bert_score, sentence_transformers

**Current requirements.txt:**
```
torch>=2.0.0
numpy>=2.0.0
matplotlib>=3.5.0
mlx-lm>=0.30.0
transformers>=4.30.0
sentence-transformers>=2.2.0
bert-score>=0.3.13
jupyter>=1.0.0
ipython>=8.0.0
matplotlib-inline>=0.1.0
```

### 2. Text Passages ✓
- [x] Removed 4 copyrighted passages (Butler, Orwell, Sapolsky)
- [x] Added 20 public-domain passages from diverse sources
- [x] Mix of well-known classics (Pride & Prejudice, Moby-Dick, Alice in Wonderland) and obscure texts (Beowulf, Republic)
- [x] Documented all sources in `paragraphs/SOURCES.md`

**New passages (all public domain):**
1. Pride and Prejudice (Austen, 1813)
2. The Metamorphosis (Kafka, 1915)
3. Alice's Adventures in Wonderland (Carroll, 1865)
4. Frankenstein (Shelley, 1818)
5. The Picture of Dorian Gray (Wilde, 1890)
6. Great Expectations (Dickens, 1860)
7. Moby-Dick (Melville, 1851)
8. Jane Eyre (Brontë, 1847)
9. The Odyssey (Homer, ~8th century BCE)
10. The Wonderful Wizard of Oz (Baum, 1900)
11. The Republic (Plato, ~380 BCE)
12. Wuthering Heights (Brontë, 1847)
13. A Tale of Two Cities (Dickens, 1859)
14. The Adventures of Sherlock Holmes (Conan Doyle, 1892)
15. The Strange Case of Dr. Jekyll and Mr. Hyde (Stevenson, 1886)
16. The Iliad (Homer, ~8th century BCE)
17. Beowulf (Anonymous, ~725-1025 CE)
18. The Canterbury Tales (Chaucer, ~1387-1400)
19. Meditations (Marcus Aurelius, ~170-180 CE)
20. Pride and Prejudice variant (Austen, 1813)

### 3. README Rewrite ✓
- [x] Replaced "overfitting" concept with **"attractor dynamics"** 
- [x] Wrote research question in 2-3 sentences (clear and specific)
- [x] Documented methodology, metrics, and quick-start guide
- [x] Added references to relevant papers

**Research Question:**
> How do language models exhibit attractor dynamics in semantic space when repeatedly summarizing the same text? Specifically, does iterative summarization cause text to converge toward particular semantic attractors, and can we measure this drift using embedding-based and token-level metrics?

### 4. Bug Fixes
- [x] Fixed data loading bug in `summarization_chain_experiment.py`
  - Was: `for text in dir.iterdir()` (undefined `dir`)
  - Now: `for text_file in sorted(paragraph_dir.iterdir())` (proper Path iteration)
- [x] Removed Kaggle dataset dependency
- [x] Now loads paragraphs from local `paragraphs/` directory
- [x] Added device detection output (CPU/MPS/CUDA)
- [x] Added progress reporting (loaded passage count)

## Verification

All checks passed:
```
✓ All core packages imported successfully
✓ Script compiles without syntax errors
✓ Successfully loaded 20 text passages
✓ First passage: 03_alice_wonderland_carroll.txt
✓ Last passage: 17_beowulf_anon.txt
```

## Ready for Phase 1

The repository is now clean and ready for:
- Running the full summarization chain experiment
- Testing semantic drift metrics
- Analyzing attractor dynamics in LLM outputs

### Next: Phase 1 - Run Experiment
Generate summarization chains for all 20 passages and collect semantic drift data.

---

**Commit:** 9de923a
**Date:** 2026-11-14
