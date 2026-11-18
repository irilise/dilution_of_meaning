# Phase 1: Tasks 4-9 Implementation Checklist

## Task 4: Save Every Step ✅
**Implementation Status:** COMPLETE

### Code Components:
- [x] `ExperimentLogger` class with full chain history
- [x] `log_step()` method for recording each iteration
- [x] `save_logs()` method for JSON serialization
- [x] `load_logs()` for reconstruction
- [x] `get_chain()` for retrieving specific chains

### Features:
- [x] Logs: iteration, text, model, seed, temperature, mode
- [x] Timestamp metadata for reproducibility
- [x] JSON format for easy parsing
- [x] Supports multiple seeds per passage
- [x] Chain reconstruction capability

### Output:
- File: `experiment_logs/chains_{timestamp}_*.json`
- Format: JSON with full text at each step
- Can rebuild any chain: `logger.get_chain(passage, seed)`

---

## Task 5: Fix Metrics ✅
**Implementation Status:** COMPLETE

### Code Components:
- [x] `ImprovedSemanticScorer` class
- [x] `ImprovedBERTScorer` wrapper
- [x] Token limit handling (256-token MiniLM)
- [x] Intelligent text truncation
- [x] Batch scoring support
- [x] Consistent rescaling to [0,1]

### Metrics Tracked:
1. **Original-to-current similarity**: How much drift from original?
   - Computed via Sentence Transformers embedding
   - Handles texts up to 500+ words (truncates to 256 tokens)
   
2. **Previous-to-current similarity**: Change per step?
   - Measures local stability vs. global divergence
   - Helps identify when convergence happens
   
3. **BERTScore F1**: Token-level semantic overlap
   - Complements embedding-based similarity
   - Catches cases where embeddings might fail

### Computation:
- [x] All metrics from saved chain texts (no recomputation)
- [x] Batch operations for efficiency
- [x] Error handling for edge cases
- [x] Proper rescaling and normalization

### Output:
- File: `experiment_results/drift_{passage_name}.png`
- Includes: Mean ± std dev error bars across seeds
- Both original and previous-step similarity curves

---

## Task 6: Separate Compression from Drift ✅
**Implementation Status:** COMPLETE

### Code Components:
- [x] `summarize_text()` - Task 3/6 summarization to 50-60 words
- [x] `rewrite_text()` - Same-length paraphrase without compression
- [x] Mode parameter in `run_chain()`
- [x] Separate tracking for each mode

### How It Works:
```
Original text (e.g., 300 words)
    ↓
[Mode 1: SUMMARIZE]     [Mode 2: REWRITE]
    ↓                           ↓
50-60 words             ~300 words (paraphrased)
Compression + Drift     Drift only
    ↓                           ↓
Larger initial drop     Smaller initial drop
Heavy semantic loss     Minimal semantic loss
```

### Analysis:
- Gap = Information loss from compression
- Summarize mode convergence point ≠ Rewrite mode
- Different attractor basins for each mode

### Output:
- File: `experiment_results/drift_{passage}_summarize.png`
- File: `experiment_results/drift_{passage}_rewrite.png`
- One can compare the curves to quantify compression effects

---

## Task 7: Multiple Seeds ✅
**Implementation Status:** COMPLETE

### Code Components:
- [x] `NUM_SEEDS` configuration (default: 3)
- [x] Loop over seeds in `run_chain()`
- [x] Different seed per iteration: `seed + iteration`
- [x] Independent chain tracking
- [x] Mean and std computation

### Statistics:
- [x] Collect N chains per passage
- [x] Compute mean at each iteration
- [x] Compute standard deviation
- [x] Plot with error bars (shaded regions)

### Interpretation:
- **Tight error bars** (±0.02-0.05)
  - Deterministic behavior
  - Model consistently reaches same output
  
- **Wide error bars** (±0.10-0.20)
  - Stochastic behavior
  - High temperature explores diverse paraphrases

### Output:
- All plots include error bars (±1 std dev)
- Shows variance across 3-5 chains
- Enables statistical comparison between modes

---

## Task 8: Model-Size Scan 🚀
**Implementation Status:** READY TO RUN

### Code Components:
- [x] `task_8_model_scan.py` - Complete implementation
- [x] `run_model_comparison()` - Main execution loop
- [x] `plot_model_comparison()` - Multi-model visualization
- [x] `save_model_comparison_stats()` - Summary statistics

### Models:
```
mlx-community/Qwen3-0.5B-bf16   (small)
mlx-community/Qwen3-0.6B-bf16   (medium) [current default]
mlx-community/Qwen3-1B-bf16     (large)
```

### Expected Results:
- Larger models = more stable (smaller variance)
- Larger models = slower initial drift
- All converge to attractors (but at different speeds)

### Usage:
```bash
python task_8_model_scan.py
# Compares 3 model sizes on 5 sample passages
# Generates: model_comparison_all_metrics.png
#           model_comparison_iteration_snapshots.png
#           model_comparison_stats.json
```

### Output:
- File: `experiment_results/model_comparison_all_metrics.png`
- File: `experiment_results/model_comparison_iteration_snapshots.png`
- File: `experiment_results/model_comparison_stats.json`

### Runtime:
- ~3-6 hours (5 passages × 3 models × 3 seeds)
- Can reduce with `SAMPLE_PASSAGES` parameter

---

## Task 9: Read Outputs & Qualitative Analysis ✅
**Implementation Status:** READY FOR MANUAL REVIEW

### Code Components:
- [x] `analyze_convergence()` - Automated extraction
- [x] JSON output of first/middle/final texts
- [x] Chain-by-chain text sampling

### Automated Analysis:
- Extracts first 100 chars from iterations 0, 50, 100
- Saves to `experiment_logs/convergence_analysis.json`
- Groups by chain for easy comparison

### Manual Analysis Steps (you do this):

1. **Load chains**:
   ```python
   import json
   with open('experiment_logs/chains_*.json') as f:
       chains = json.load(f)
   ```

2. **For each passage, examine 5-10 chains**:
   - Read the progression: iteration 0 → 50 → 100
   - Look for repeated phrases
   - Track semantic shifts

3. **Document patterns**:
   - What topics/words appear in final outputs?
   - Do different seeds converge to same phrase?
   - How stable is convergence (101 → 102)?

4. **Write findings**:
   - Page of examples with annotations
   - Identified attractor regions
   - Model behavior patterns

### Example Findings Format:
```
PASSAGE: Pride and Prejudice excerpt
CONVERGENCE PATTERN: "the importance of [social aspect]"

Chain 1 (iter 0-50-100):
  0: "It is a truth universally acknowledged..."
  50: "Society deems important the state of matrimony..."
  100: "The importance of matrimony in society matters greatly."

Chain 2 (iter 0-50-100):
  0: "It is a truth universally acknowledged..."
  50: "One must acknowledge social importance of marriage..."
  100: "Marriage holds importance for societal function."

Pattern: All chains converge on "importance + matrimony/marriage"
This represents a semantic attractor in the embedding space.
```

### Output File:
- File: `experiment_logs/convergence_analysis.json`
- Also: Manual document with qualitative findings

---

## Files Created

### Main Implementation:
- `summarization_chain_experiment_v2.py` - Full v2 with tasks 4-7, 9
- `task_8_model_scan.py` - Model comparison (task 8)
- `validate_tasks.py` - Validation tests

### Documentation:
- `TASKS_4_9_DOCUMENTATION.md` - Detailed guide for all tasks
- `PHASE_1_CHECKLIST.md` - This file

### Key Classes:
```
ExperimentLogger           # Task 4: Logging
ImprovedSemanticScorer     # Task 5: Semantic metrics
ImprovedBERTScorer         # Task 5: BERT metrics
(Various functions)        # Task 6: Mode separation
(run_chain, plot_results)  # Task 7: Multiple seeds
(task_8_model_scan.py)     # Task 8: Model comparison
(analyze_convergence)      # Task 9: Qualitative analysis
```

---

## Running the Experiment

### Minimal test (1 passage, 10 iterations):
```bash
# Edit summarization_chain_experiment_v2.py:
# NUM_ITERATIONS = 10
# Then modify paragraph_dir loading to get just 1 file
python summarization_chain_experiment_v2.py
```

### Full experiment (tasks 4-7, 9):
```bash
python summarization_chain_experiment_v2.py
# Runtime: 6-12 hours depending on device
# Outputs: experiment_logs/ and experiment_results/
```

### Model comparison (task 8):
```bash
python task_8_model_scan.py
# Runtime: 3-6 hours (5 passages × 3 models)
# Outputs: model_comparison_*.png files
```

### Validation:
```bash
python validate_tasks.py
# Quick sanity checks before running full experiment
```

---

## Expected Outputs

### Logs:
```
experiment_logs/
  ├─ chains_20261114_220000_summarize_rewrite_comparison.json
  ├─ chains_20261114_220000_*summarize*.json
  ├─ convergence_analysis.json
  └─ [model_comparison_logs]/
      ├─ chains_*.json
      └─ ...
```

### Plots:
```
experiment_results/
  ├─ drift_01_pride_prejudice_austen_summarize.png
  ├─ drift_01_pride_prejudice_austen_rewrite.png
  ├─ drift_02_metamorphosis_kafka_summarize.png
  ├─ ... (total: 40 plots)
  ├─ model_comparison_all_metrics.png
  └─ model_comparison_iteration_snapshots.png
```

### Statistics:
```
experiment_results/
  └─ model_comparison_stats.json
```

---

## Key Insights to Look For

1. **Attractor Dynamics**: Do chains converge to specific topics?
2. **Model Size Effect**: Does larger = more stable?
3. **Compression Cost**: How much of iteration-1 drop is compression?
4. **Seed Variance**: Deterministic or stochastic behavior?
5. **Convergence Rate**: How many iterations to plateau?

---

## Next Steps After Running

1. **Analyze logs**: Load chains and manually inspect convergence
2. **Statistical tests**: Compare summarize vs rewrite with t-tests
3. **Visualization**: Create summary figure for paper
4. **Writing**: Document patterns in qualitative section
5. **Model scan**: Run task 8 if insights warrant

---

**Created:** 2026-11-14
**Author:** Copilot
**Status:** All tasks implemented, ready for execution
