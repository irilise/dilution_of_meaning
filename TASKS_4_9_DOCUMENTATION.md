# Phase 1: Experimental Enhancement (Tasks 4-9)

## Overview

This phase implements 6 major improvements to the semantic drift experiment:

### Task 4: Save Every Step ✓
**Status:** IMPLEMENTED

**What it does:**
- `ExperimentLogger` class logs every iteration's text, model, seed, temperature, and step number
- Saves to JSON with full chain history
- Enables complete reconstruction of any chain
- Timestamp metadata for reproducibility

**Files:**
- `ExperimentLogger` class in `summarization_chain_experiment_v2.py`
- Output: `experiment_logs/chains_{timestamp}.json`

**How to use:**
```python
logger = ExperimentLogger()
logger.log_step(passage_name, seed, iteration, text, model_id, temperature)
logger.save_logs()
chains = logger.load_logs(filename)
```

**Benefit:** Full audit trail - can replay any experiment or debug specific chains.

---

### Task 5: Fix Metrics ✓
**Status:** IMPLEMENTED

**What it does:**
- `ImprovedSemanticScorer` handles MiniLM's 256-token limit by truncating intelligently
- Proper token-aware encoding/decoding
- `ImprovedBERTScorer` wrapper with consistent rescaling to [0,1]
- All metrics computed from saved outputs (no recomputation)

**Key improvements:**
1. **Token limit handling**: Truncates long texts to 256 tokens before embedding
2. **Rescaling**: BERTScore values properly normalized
3. **Batch scoring**: Efficient processing of multiple chains

**Metrics tracked:**
- Original-to-current similarity: How much has it drifted from the original?
- Previous-to-current similarity: How much changes per step?
- BERTScore F1: Token-level semantic overlap with original

**Files:**
- `ImprovedSemanticScorer` class
- `ImprovedBERTScorer` class
- Output: Both metrics in `experiment_results/drift_{passage_name}.png`

---

### Task 6: Separate Compression from Drift ✓
**Status:** IMPLEMENTED

**What it does:**
- Runs TWO variants of each passage:
  - **Summarize mode**: Compresses to 50-60 words (original behavior)
  - **Rewrite mode**: Maintains approximate same length without compression
  
**Purpose:** 
Measure how much of the iteration-1 similarity drop is due to:
- Pure information loss (compression)
- Semantic drift (paraphrasing)

**Files:**
- `summarize_text()` - Task 3/6 summarization
- `rewrite_text()` - Task 6 same-length paraphrase
- Output: Two sets of plots, one for each mode

**How it works:**
```
Original text → Summarize (50-60 words) → Rewrite (~same words) 
                      ↓                           ↓
            Heavy drift expected         Subtle drift expected
            (compression + semantic)     (paraphrasing only)
```

---

### Task 7: Multiple Seeds ✓
**Status:** IMPLEMENTED

**What it does:**
- Runs 3-5 independent chains per passage (controlled by `NUM_SEEDS`)
- Each chain uses a different random seed
- Aggregates results showing mean and variance (error bars)

**Motivation:** 
Reveals whether drift is deterministic or stochastic.
- Tight error bars = deterministic behavior
- Wide error bars = model explores diverse paraphrase space

**Output format:**
- Plots show mean line with ±1 std dev shaded region
- Allows statistical comparison between modes and models

**Configuration:**
```python
NUM_SEEDS = 3  # Set to 5 for more statistical power
```

---

### Task 8: Model-Size Scan
**Status:** TEMPLATE PROVIDED (NOT YET IMPLEMENTED)

**What it does:**
- Runs same experiment on 3 different Qwen model sizes
- Compares how model size affects semantic drift
- Shows model capacity vs. semantic fidelity trade-off

**Model options (available on Hugging Face/MLX):**
- `mlx-community/Qwen3-0.5B-bf16` (small)
- `mlx-community/Qwen3-0.6B-bf16` (medium) [current]
- `mlx-community/Qwen3-1B-bf16` (large)

**To implement:**
1. Modify main script to loop over multiple model_ids
2. Keep separate result sets for each model
3. Create comparison plots across sizes

**Expected pattern:**
- Larger models = more stable (smaller error bars)
- Smaller models = faster drift
- All converge toward attractors, but different timescales

**Files:** Modified version of main loop in `summarization_chain_experiment_v2.py`

---

### Task 9: Read Outputs (Qualitative Analysis) ✓
**Status:** IMPLEMENTED (basic version)

**What it does:**
- `analyze_convergence()` reads first, middle, and final texts from chains
- Extracts convergence patterns
- Saves qualitative observations

**Current output:**
- JSON file with sampled text from each chain
- Identifies common themes, repeated phrases, semantic fixation points

**To enhance (manual step):**
1. Load `experiment_logs/chains_{timestamp}.json`
2. For each passage, read 5-10 chains
3. Document patterns:
   - What do chains converge toward?
   - Are there repeated topics/themes?
   - Do different seeds converge to same output?
   - How stable is convergence?

**Example findings to look for:**
```
Chain 1: "...the nature of X... the question of X... X is important..."
Chain 2: "...X matters because... the key to X... understanding X..."
Chain 3: "...X cannot be ignored... X requires attention..."

Pattern: All converge toward "X is important/central/key"
This suggests semantic attractor around centrality/importance.
```

**Output file:** `experiment_logs/convergence_analysis.json`

---

## Running the Experiment

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic run (all tasks except 8)
```bash
python summarization_chain_experiment_v2.py
```

This will:
1. ✓ Load 20 passages
2. ✓ Run 3 seeds × 100 iterations for each passage
3. ✓ Track both summarize and rewrite modes
4. ✓ Log every step to JSON
5. ✓ Compute both drift metrics
6. ✓ Generate plots with error bars
7. ✓ Save convergence analysis

### Expected runtime:
- ~6-12 hours on CPU (MLX on Apple Silicon = ~10x faster)
- Generate 80 plots (20 passages × 2 modes × 2 subsets)

### Output structure:
```
experiment_logs/
  └─ chains_20261114_220740_summarize_rewrite_comparison.json
  └─ convergence_analysis.json
experiment_results/
  └─ drift_01_pride_prejudice_austen_summarize.png
  └─ drift_01_pride_prejudice_austen_rewrite.png
  ├─ drift_02_metamorphosis_kafka_summarize.png
  └─ ... (38 more)
```

---

## Advanced Options

### Task 8: Model-Size Scan
To compare 3 model sizes:

```python
MODELS = [
    "mlx-community/Qwen3-0.5B-bf16",
    "mlx-community/Qwen3-0.6B-bf16",
    "mlx-community/Qwen3-1B-bf16"
]

for model_id in MODELS:
    model, tokenizer = load(model_id)
    # Run full experiment loop
    # Save results with model name
```

### Customize parameters:
```python
NUM_ITERATIONS = 100    # Steps per chain
NUM_SEEDS = 5           # Chains per passage (for error bars)
TEMPERATURE = 0.7       # Randomness (0=deterministic, 1=very random)
```

---

## Expected Results

### Task 4-5: Drift Curves
- **Original similarity**: Steep drop in iterations 0-20, then plateaus
- **Previous-step similarity**: Stays high (~0.7-0.9) until convergence
- **BERTScore F1**: Similar to original similarity, more noisy

### Task 6: Compression vs Drift
- **Summarize mode**: Larger initial drop (compression + drift)
- **Rewrite mode**: Smaller initial drop (drift only)
- Gap between them = information loss from compression

### Task 7: Error Bars
- Tight ±0.02-0.05: Consistent behavior across seeds
- Wide ±0.10-0.20: High stochasticity, diverse paraphrases

### Task 9: Convergence Patterns
Look for:
- Specific topics/words that appear in most final texts
- Repeated structures ("the X is [adjective]", "X can/must/should...")
- Topic shift (drift away from original toward related concept)
- Stability (does 100→101 look like 99→100?)

---

## Next Steps

1. **Run basic experiment** (all tasks except 8)
2. **Manually inspect chains** (task 9 deep dive)
3. **Document patterns** in qualitative section
4. **Implement model-size scan** (task 8) if insights warrant
5. **Statistical tests** - T-tests comparing summarize vs rewrite
6. **Visualization** - Create summary figure comparing all modes

---

## Troubleshooting

### Memory issues
- Reduce `NUM_SEEDS` to 2
- Reduce `NUM_ITERATIONS` to 50 for testing

### Slow performance
- Check device detection (should use MPS on Mac M1/M2)
- GPU on Mac: `torch.backends.mps.is_available()` should return True
- Fallback to CPU is slow but will work

### Token limit errors
- `ImprovedSemanticScorer` handles this automatically
- If errors persist, reduce MAX_TOKENS in `_truncate_if_needed()`

### Missing logs
- Check `experiment_logs/` directory exists
- Verify write permissions
- Logs are only saved at end - don't interrupt script

---

## Citation

If you use this code, cite:
- BERTScore: Zhang et al., 2020
- Sentence Transformers: Reimers & Gupta, 2019
- MLX-LM: Apple machine learning group

---

**Created:** 2026-11-14
**Status:** Tasks 4-7, 9 complete; Task 8 ready for implementation
