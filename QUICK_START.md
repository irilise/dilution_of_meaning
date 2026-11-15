# Quick Start Guide: Running Tasks 4-9

## TL;DR

```bash
# 1. Validate everything is working
python validate_tasks.py

# 2. Run the full experiment (tasks 4-7, 9)
# Runtime: 6-12 hours
python summarization_chain_experiment_v2.py

# 3. (Optional) Compare model sizes (task 8)
# Runtime: 3-6 hours
python task_8_model_scan.py
```

---

## What Each Command Does

### `python validate_tasks.py`
**Purpose:** Quick sanity checks (5 minutes)

Tests:
- ✓ Logging system works
- ✓ Semantic scorer handles long texts
- ✓ BERT scorer produces valid scores
- ✓ All 20 passages load correctly
- ✓ Output directories exist

**Exit code:** 0 = ready to run, 1 = fix issues first

---

### `python summarization_chain_experiment_v2.py`
**Purpose:** Full experiment with all tasks except model scan

What it does:
1. **Task 4**: Logs every step to JSON
   - Output: `experiment_logs/chains_*.json`
   
2. **Task 5**: Computes 3 metrics (original sim, previous sim, BERT F1)
   - Handles MiniLM's 256-token limit automatically
   - Both metrics in every output
   
3. **Task 6**: Runs two modes for each passage:
   - Summarize (50-60 words)
   - Rewrite (same length, no compression)
   - Separate curves to show compression effect
   
4. **Task 7**: Multiple seeds (default: 3)
   - 3 chains per passage per mode
   - Error bars on plots show variance
   
5. **Task 9**: Analyzes convergence patterns
   - Output: `experiment_logs/convergence_analysis.json`
   - Ready for manual review

**Output structure:**
```
experiment_logs/
  └─ chains_20261114_*.json           (full chain history)
  └─ convergence_analysis.json         (sampled texts from chains)
  
experiment_results/
  └─ drift_*.png                       (40 plots, one per passage per mode)
```

**Runtime:** ~6-12 hours (depending on CPU/MPS/GPU)
- MPS (Apple Silicon): ~6 hours
- CPU: ~12+ hours
- CUDA GPU: ~2-3 hours

**Monitor progress:**
- Script prints passage name as it processes
- Check `experiment_results/` for growing number of PNG files

---

### `python task_8_model_scan.py`
**Purpose:** Compare drift across 3 Qwen model sizes (task 8)

Compares:
- Qwen3-0.5B (small)
- Qwen3-0.6B (medium, current default)
- Qwen3-1B (large)

What it does:
1. Loads 5 sample passages (to keep runtime reasonable)
2. Runs 3 seeds per passage per model
3. Generates comparison plots
4. Saves statistics

**Output:**
```
experiment_results/
  ├─ model_comparison_all_metrics.png        (4 subplots)
  └─ model_comparison_iteration_snapshots.png (bar chart)
  
experiment_results/
  └─ model_comparison_stats.json
```

**Runtime:** ~3-6 hours (smaller than full experiment)

**To use different passage count:**
Edit the function call at bottom:
```python
run_model_comparison(
    sample_passages=10  # Change this number
)
```

---

## Configuration Options

### Main experiment configuration:
```python
# In summarization_chain_experiment_v2.py, find:
NUM_ITERATIONS = 100    # Steps per chain (0-100)
NUM_SEEDS = 3           # Chains per passage (for error bars)
```

### Temperature (randomness):
```python
# In summarize_text() or rewrite_text():
temperature=0.7         # 0=deterministic, 1=very random
```

### For quick testing (before full run):
```bash
# Edit bottom of summarization_chain_experiment_v2.py:
NUM_ITERATIONS = 10         # Reduce to 10 steps
# Then modify paragraph loading to take only first passage:
texts = texts[:1]
text_names = text_names[:1]
```

---

## Interpreting Results

### Plots (`.png` files)

Each passage generates plots showing:

**Subplot 1: Similarity to Original**
- Y-axis: Goes from 1.0 (identical) to ~0.2 (very different)
- Steep drop in first 20 iterations = compression
- Plateau after = convergence reached

**Subplot 2: Similarity to Previous Step**
- Shows iteration-by-iteration change
- Stays high (~0.8-0.9) if changes are small
- Drops when text suddenly changes

**Subplot 3: BERTScore F1**
- Similar to original similarity
- More noisy but complements embedding metric

**Subplot 4: Comparison**
- All three metrics on one plot
- Shows correlation between them

**Error bars:** ±1 std dev across 3 seeds
- Tight = deterministic
- Wide = stochastic

### Comparing modes:

Look at pairs:
- `drift_01_..._summarize.png` vs `drift_01_..._rewrite.png`
- Summarize should drop faster (compression)
- Gap ≈ information loss from compression

### Model comparison plots:

**All Metrics**: Shows same 4 subplots but 3 lines (one per model)
- Compare curves across sizes
- Larger models typically drift slower

**Iteration Snapshots**: Bar chart at selected iterations
- Shows final similarity at iterations [0, 10, 25, 50, 99]
- Model sizes compared side-by-side

### JSON logs (`.json` files)

```python
import json

# Load chains
with open('experiment_logs/chains_*.json') as f:
    chains = json.load(f)

# Access a specific chain
key = 'passage_name_seed0'
chain = chains[key]

# Structure:
# chain = {
#   'passage': 'filename',
#   'seed': 0,
#   'model': 'model-id',
#   'temperature': 0.7,
#   'steps': [
#     {'iteration': 0, 'text': '...', 'mode': 'summarize'},
#     {'iteration': 1, 'text': '...', 'mode': 'summarize'},
#     ...
#   ]
# }

# Get full text at iteration 50
text_iter_50 = chain['steps'][50]['text']
```

---

## Troubleshooting

### Script fails with "No such file"
- Make sure you're in the right directory: `cd dilution_of_meaning`
- Check paragraphs/ has 20 .txt files

### Memory error
- Reduce `NUM_SEEDS` to 2
- Reduce `NUM_ITERATIONS` to 50

### Very slow performance
- Check device: Script prints at start
- MPS should be used on Mac (not CPU)
- If stuck on CPU, check: `torch.backends.mps.is_available()`
- Last resort: GPU if available

### BERT scorer error
- Usually means OOM (out of memory)
- Reduce batch size by using fewer seeds

### Plots not generating
- Check `experiment_results/` directory exists
- Check disk space (plots are ~1MB each)
- Look for Python errors in console

### "Module not found"
- Make sure you installed: `pip install -r requirements.txt`
- Double-check: `python -c "import mlx_lm; print('OK')"`

---

## What to do after running

### 1. Quick inspection (5 min)
```bash
ls -lh experiment_results/*.png | head -10
ls -lh experiment_logs/*.json
```

Should see ~40 PNG files and chains JSON file.

### 2. Spot check one plot
Open `experiment_results/drift_01_*.png` in your image viewer.
- Do error bars make sense?
- Does similarity drop and plateau?
- Compare summarize vs rewrite side-by-side

### 3. Analyze convergence (manual, 30-60 min)
```python
import json

with open('experiment_logs/chains_*.json') as f:
    chains = json.load(f)

# Pick a passage
for key in list(chains.keys())[:5]:
    chain = chains[key]
    # Read iterations 0, 50, 99
    for step in [chain['steps'][i] for i in [0, 50, 99]]:
        print(f"Iter {step['iteration']}: {step['text'][:80]}...\n")
```

### 4. Write qualitative findings
Document:
- What topics do chains converge toward?
- How stable is convergence?
- Differences between modes?
- Differences between seeds?

---

## Next Phase (After Tasks 4-9)

Once complete, you can:
1. Run statistical tests (t-test: summarize vs rewrite)
2. Identify semantic attractors
3. Test with different temperatures
4. Analyze longer chains (500+ iterations)
5. Compare with other models (GPT, Llama, etc.)
6. Write up findings for paper

---

## Example Output

**Progress during run:**
```
Using device: mps
Loading model: mlx-community/Qwen3-0.6B-bf16
Loaded 20 passages

Running chains...

Processing: 01_pride_prejudice_austen_summarize
  Chain 1/3... ✓
  Chain 2/3... ✓
  Chain 3/3... ✓

Processing: 01_pride_prejudice_austen_rewrite
  Chain 1/3... ✓
  Chain 2/3... ✓
  Chain 3/3... ✓

Processing: 02_metamorphosis_kafka_summarize
  ...

✓ Saved chains to experiment_logs/chains_20261114_220740_...json
Generating plots...
✓ Saved plot to experiment_results/drift_01_pride_prejudice_austen_summarize.png
✓ Saved plot to experiment_results/drift_01_pride_prejudice_austen_rewrite.png
...

✓ Experiment complete!
Logs saved to: experiment_logs/
Plots saved to: experiment_results/
```

---

## Help!

If something breaks:
1. Check error message (scroll up in terminal)
2. Look at Troubleshooting section above
3. Review TASKS_4_9_DOCUMENTATION.md for details
4. Check code comments in summarization_chain_experiment_v2.py

---

**Ready to run!**

```bash
python summarization_chain_experiment_v2.py
```

Good luck! 🚀
