import torch
import matplotlib.pyplot as plt
from mlx_lm import load, generate
from mlx_lm import sample_utils
from pathlib import Path
from bert_score import BERTScorer
import numpy as np
from sentence_transformers import SentenceTransformer, util
import json
import random
from datetime import datetime
from typing import List, Dict, Tuple
import re


class ExperimentLogger:
    """Logs every iteration step to enable chain reconstruction."""
    
    def __init__(self, output_dir: Path = Path('experiment_logs')):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.chains = {}
    
    def log_step(self, passage_name: str, seed: int, iteration: int, 
                 text: str, model_id: str, temperature: float, mode: str = 'summarize'):
        """
        Log a single iteration step.
        
        Args:
            passage_name: Name of the source passage
            seed: Random seed used
            iteration: Step number (0-indexed)
            text: The generated text at this step
            model_id: Model identifier
            temperature: Temperature parameter used
            mode: 'summarize' or 'rewrite' (task 6)
        """
        chain_key = f"{passage_name}_seed{seed}"
        if chain_key not in self.chains:
            self.chains[chain_key] = {
                'passage': passage_name,
                'seed': seed,
                'model': model_id,
                'temperature': temperature,
                'steps': []
            }
        
        self.chains[chain_key]['steps'].append({
            'iteration': iteration,
            'text': text,
            'mode': mode,
            'timestamp': datetime.now().isoformat()
        })
    
    def save_logs(self, suffix: str = ''):
        """Save all logged chains to JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f'chains_{timestamp}{suffix}.json'
        with open(filename, 'w') as f:
            json.dump(self.chains, f, indent=2)
        print(f"✓ Saved chains to {filename}")
        return filename
    
    def load_logs(self, filename: Path) -> Dict:
        """Load previously saved chains."""
        with open(filename, 'r') as f:
            self.chains = json.load(f)
        return self.chains
    
    def get_chain(self, passage_name: str, seed: int) -> List[str]:
        """Reconstruct a single chain from logs."""
        chain_key = f"{passage_name}_seed{seed}"
        if chain_key in self.chains:
            return [step['text'] for step in self.chains[chain_key]['steps']]
        return []


class ImprovedSemanticScorer:
    """Enhanced semantic scorer with token limit handling and rescaling."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None, max_tokens=256):
        """
        Initialize with token limit awareness.
        MiniLM has a 256-token limit.
        """
        if device is None:
            self.device = torch.device("cpu")
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
        else:
            self.device = device
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.max_tokens = max_tokens
        self.tokenizer = self.model.tokenizer
    
    def _truncate_if_needed(self, text: str) -> str:
        """Truncate text to max_tokens to avoid MiniLM limit."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_tokens:
            # Decode back to approximate byte position
            truncated = self.tokenizer.decode(tokens[:self.max_tokens], skip_special_tokens=True)
            return truncated
        return text
    
    def score(self, source: str, target: str) -> float:
        """Calculate cosine similarity with token limit handling."""
        source = self._truncate_if_needed(source)
        target = self._truncate_if_needed(target)
        
        embeddings = self.model.encode([source, target], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return similarity.item()
    
    def score_batch(self, sources: List[str], targets: List[str]) -> List[float]:
        """Batch scoring with token limit handling."""
        if len(sources) != len(targets):
            raise ValueError("Source and Target lists must be the same length.")
        
        sources = [self._truncate_if_needed(s) for s in sources]
        targets = [self._truncate_if_needed(t) for t in targets]
        
        emb1 = self.model.encode(sources, convert_to_tensor=True)
        emb2 = self.model.encode(targets, convert_to_tensor=True)
        cosine_scores = util.cos_sim(emb1, emb2)
        return torch.diagonal(cosine_scores).tolist()


class ImprovedBERTScorer:
    """Wrapper around BERTScore with proper rescaling."""
    
    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.scorer = BERTScorer(lang="en", device=self.device)
    
    def score(self, ref: str, cand: str) -> Tuple[float, float, float]:
        """
        Score candidate against reference.
        Returns rescaled scores in [0, 1] range.
        """
        P, R, F1 = self.scorer.score([cand], [ref])
        # Convert to Python floats and handle any scaling
        precision = float(P.mean().item())
        recall = float(R.mean().item())
        f1 = float(F1.mean().item())
        return precision, recall, f1
    
    def score_batch(self, refs: List[str], cands: List[str]) -> Tuple[List[float], List[float], List[float]]:
        """Batch scoring."""
        P, R, F1 = self.scorer.score(cands, refs)
        return (
            [float(p) for p in P],
            [float(r) for r in R],
            [float(f) for f in F1]
        )


def _generate(model, tokenizer, prompt: str, temperature: float, max_tokens: int, seed: int = None):
    """Generate with optional seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    sampler = sample_utils.make_sampler(temp=temperature)
    logits_processors = sample_utils.make_logits_processors()
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        sampler=sampler,
        logits_processors=logits_processors,
        max_tokens=max_tokens
    )


def summarize_text(text: str, tokenizer, model, temperature: float = 0.7, seed: int = None) -> str:
    """Task 3/6: Summarize text to 50-60 words."""
    tokenized = tokenizer.apply_chat_template(
        [
            {'role': 'system', 'content':
                'Summarize the provided text in 50 to 60 words.'},
            {'role': 'user', 'content': text},
        ],
        add_generation_prompt=True,
        enable_thinking=False
    )
    return _generate(
        model=model,
        tokenizer=tokenizer,
        prompt=tokenized,
        temperature=temperature,
        max_tokens=200,  # Reduced from 1000 to avoid bloat
        seed=seed
    )


def rewrite_text(text: str, tokenizer, model, temperature: float = 0.7, seed: int = None) -> str:
    """Task 6: Rewrite text to same approximate length (not compress)."""
    # Estimate word count
    word_count = len(text.split())
    
    tokenized = tokenizer.apply_chat_template(
        [
            {'role': 'system', 'content':
                f'Rewrite the provided text in approximately {word_count} words, changing the wording but preserving the meaning.'},
            {'role': 'user', 'content': text},
        ],
        add_generation_prompt=True,
        enable_thinking=False
    )
    return _generate(
        model=model,
        tokenizer=tokenizer,
        prompt=tokenized,
        temperature=temperature,
        max_tokens=300,
        seed=seed
    )


def run_chain(passage_text: str, passage_name: str, model, tokenizer,
              num_iterations: int = 100, num_seeds: int = 1,
              semantic_scorer: ImprovedSemanticScorer = None,
              bert_scorer: ImprovedBERTScorer = None,
              logger: ExperimentLogger = None,
              mode: str = 'summarize') -> Dict:
    """
    Run a single chain with multiple seeds.
    
    Args:
        passage_text: The original text
        passage_name: Name of passage for logging
        model: Language model
        tokenizer: Model tokenizer
        num_iterations: How many steps
        num_seeds: Number of independent chains (task 7)
        semantic_scorer: Scorer for embedding similarity
        bert_scorer: Scorer for BERTScore
        logger: Logger for saving steps
        mode: 'summarize' or 'rewrite' (task 6)
    
    Returns:
        Dictionary with results for this passage
    """
    results = {
        'passage_name': passage_name,
        'chains': []
    }
    
    for seed in range(num_seeds):
        print(f"  Chain {seed + 1}/{num_seeds}...", end='', flush=True)
        
        chain_data = {
            'seed': seed,
            'texts': [passage_text],
            'original_to_current': [1.0],  # Similarity to original
            'previous_to_current': [1.0],  # Similarity to previous step
            'bert_f1': [1.0],
        }
        
        current_text = passage_text
        
        # Log initial state
        if logger:
            logger.log_step(passage_name, seed, 0, passage_text, 
                          'original', 0.0, mode=mode)
        
        for iteration in range(num_iterations):
            # Task 4: Generate text and log every step
            if mode == 'summarize':
                next_text = summarize_text(current_text, tokenizer, model, 
                                          temperature=0.7, seed=seed + iteration)
            elif mode == 'rewrite':
                next_text = rewrite_text(current_text, tokenizer, model,
                                        temperature=0.7, seed=seed + iteration)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Clean up output (remove partial sentences at end)
            next_text = next_text.strip()
            
            # Task 5: Compute both similarity metrics
            original_sim = semantic_scorer.score(passage_text, next_text) if semantic_scorer else 0.0
            previous_sim = semantic_scorer.score(current_text, next_text) if semantic_scorer else 0.0
            
            _, _, f1 = bert_scorer.score(passage_text, next_text) if bert_scorer else (0.0, 0.0, 0.0)
            
            chain_data['texts'].append(next_text)
            chain_data['original_to_current'].append(original_sim)
            chain_data['previous_to_current'].append(previous_sim)
            chain_data['bert_f1'].append(f1)
            
            # Task 4: Log step
            if logger:
                logger.log_step(passage_name, seed, iteration + 1, next_text,
                              'model', 0.7, mode=mode)
            
            current_text = next_text
        
        results['chains'].append(chain_data)
        print(" ✓")
    
    return results


def plot_results(all_results: List[Dict], output_dir: Path = Path('experiment_results')):
    """
    Create publication-quality plots showing drift patterns.
    Task 7: Plot with means and error bars across seeds.
    """
    output_dir.mkdir(exist_ok=True)
    
    # Aggregate results across passages and seeds
    for result in all_results:
        passage_name = result['passage_name']
        chains = result['chains']
        num_iters = len(chains[0]['texts']) - 1
        
        # Task 5: Extract both metrics
        original_sims = np.array([c['original_to_current'] for c in chains])
        previous_sims = np.array([c['previous_to_current'] for c in chains])
        bert_f1s = np.array([c['bert_f1'] for c in chains])
        
        # Task 7: Compute means and std
        orig_mean = original_sims.mean(axis=0)
        orig_std = original_sims.std(axis=0)
        prev_mean = previous_sims.mean(axis=0)
        prev_std = previous_sims.std(axis=0)
        bert_mean = bert_f1s.mean(axis=0)
        bert_std = bert_f1s.std(axis=0)
        
        iterations = np.arange(len(orig_mean))
        
        # Create figure with subplots for this passage
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Semantic Drift: {passage_name}', fontsize=14, fontweight='bold')
        
        # Plot 1: Similarity to Original (with error bars)
        ax = axes[0, 0]
        ax.plot(iterations, orig_mean, 'b-', linewidth=2, label='Mean')
        ax.fill_between(iterations, orig_mean - orig_std, orig_mean + orig_std, 
                        alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Similarity to Original Text')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Similarity to Previous Step (with error bars)
        ax = axes[0, 1]
        ax.plot(iterations, prev_mean, 'g-', linewidth=2, label='Mean')
        ax.fill_between(iterations, prev_mean - prev_std, prev_mean + prev_std,
                        alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Similarity to Previous Step')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: BERTScore F1 (with error bars)
        ax = axes[1, 0]
        ax.plot(iterations, bert_mean, 'r-', linewidth=2, label='Mean')
        ax.fill_between(iterations, bert_mean - bert_std, bert_mean + bert_std,
                        alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('BERTScore F1')
        ax.set_title('BERTScore F1 to Original')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: All metrics on same plot for comparison
        ax = axes[1, 1]
        ax.plot(iterations, orig_mean, 'b-', linewidth=2, label='Orig Similarity', alpha=0.7)
        ax.plot(iterations, prev_mean, 'g-', linewidth=2, label='Prev Similarity', alpha=0.7)
        ax.plot(iterations, bert_mean, 'r-', linewidth=2, label='BERTScore F1', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.set_title('All Metrics Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_name = re.sub(r'[^\w\-_]', '_', passage_name)
        filepath = output_dir / f'drift_{safe_name}.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {filepath}")
        plt.close()


def analyze_convergence(logger: ExperimentLogger, num_chains_to_read: int = 5):
    """
    Task 9: Manually read outputs and identify convergence patterns.
    Returns qualitative analysis.
    """
    analysis = []
    chain_keys = list(logger.chains.keys())[:num_chains_to_read]
    
    for chain_key in chain_keys:
        chain = logger.chains[chain_key]
        steps = chain['steps']
        
        # Get first, middle, and final texts
        first = steps[0]['text']
        middle = steps[len(steps)//2]['text']
        final = steps[-1]['text']
        
        analysis.append({
            'chain': chain_key,
            'first_100_chars': first[:100],
            'middle_100_chars': middle[:100],
            'final_100_chars': final[:100],
            'total_steps': len(steps)
        })
    
    return analysis


if __name__ == "__main__":
    # Configuration
    paragraph_dir = Path('paragraphs')
    NUM_ITERATIONS = 100
    NUM_SEEDS = 3  # Task 7: Multiple seeds
    
    # Device setup
    device = "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Initialize loggers and scorers
    logger = ExperimentLogger()
    semantic_scorer = ImprovedSemanticScorer(device=device)
    bert_scorer = ImprovedBERTScorer(device=device)
    
    # Load model
    model_id = "mlx-community/Qwen3-0.6B-bf16"
    print(f"Loading model: {model_id}")
    model, tokenizer = load(model_id)
    
    # Load passages
    print("Loading passages...")
    texts = []
    text_names = []
    for text_file in sorted(paragraph_dir.iterdir()):
        if text_file.is_file() and text_file.suffix == '.txt' and text_file.name != ".DS_Store":
            with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read().strip())
                text_names.append(text_file.name)
    
    print(f"Loaded {len(texts)} passages\n")
    
    # Run experiment
    all_results = []
    print("Running chains...")
    for passage_text, passage_name in zip(texts, text_names):
        print(f"\nProcessing: {passage_name}")
        
        # Task 6: Run both summarization and rewrite variants
        result_summarize = run_chain(
            passage_text, f"{passage_name}_summarize", model, tokenizer,
            num_iterations=NUM_ITERATIONS,
            num_seeds=NUM_SEEDS,
            semantic_scorer=semantic_scorer,
            bert_scorer=bert_scorer,
            logger=logger,
            mode='summarize'
        )
        all_results.append(result_summarize)
        
        # Also run rewrite for comparison
        result_rewrite = run_chain(
            passage_text, f"{passage_name}_rewrite", model, tokenizer,
            num_iterations=NUM_ITERATIONS,
            num_seeds=NUM_SEEDS,
            semantic_scorer=semantic_scorer,
            bert_scorer=bert_scorer,
            logger=logger,
            mode='rewrite'
        )
        all_results.append(result_rewrite)
    
    # Task 4: Save all logs
    print("\n" + "="*60)
    logger.save_logs(suffix='_summarize_rewrite_comparison')
    
    # Task 5: Generate plots from saved outputs
    print("Generating plots...")
    plot_results(all_results)
    
    # Task 9: Analyze convergence patterns
    print("\nAnalyzing convergence patterns...")
    convergence_analysis = analyze_convergence(logger, num_chains_to_read=5)
    
    # Save analysis
    analysis_file = Path('experiment_logs') / 'convergence_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(convergence_analysis, f, indent=2)
    print(f"✓ Saved convergence analysis to {analysis_file}")
    
    print("\n" + "="*60)
    print("✓ Experiment complete!")
    print(f"Logs saved to: experiment_logs/")
    print(f"Plots saved to: experiment_results/")
    print(f"Analysis saved to: {analysis_file}")
