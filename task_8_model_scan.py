#!/usr/bin/env python3
"""
Task 8: Model-Size Scan
Run the same experiment across 3 different Qwen model sizes.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np
from summarization_chain_experiment_v2 import (
    run_chain, ImprovedSemanticScorer, ImprovedBERTScorer, ExperimentLogger, load
)


def run_model_comparison(paragraph_dir: Path = Path('paragraphs'),
                        num_iterations: int = 100,
                        num_seeds: int = 3,
                        sample_passages: int = 5):
    """
    Run the same summarization chains on 3 different Qwen model sizes.
    Uses a sample of passages to reduce computation time.
    
    Args:
        paragraph_dir: Directory with passage files
        num_iterations: Steps per chain
        num_seeds: Chains per passage per model
        sample_passages: Number of passages to use (for faster testing)
    """
    
    # Model options (increasing size)
    models = [
        "mlx-community/Qwen3-0.5B-bf16",   # Small
        "mlx-community/Qwen3-0.6B-bf16",   # Medium
        "mlx-community/Qwen3-1B-bf16",     # Large
    ]
    
    device = "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    
    print(f"Device: {device}")
    print(f"Models to test: {models}")
    print(f"Configuration: {num_seeds} seeds × {num_iterations} iterations")
    print()
    
    # Load passages (sample subset for faster turnaround)
    text_files = sorted(
        [f for f in paragraph_dir.iterdir() 
         if f.is_file() and f.suffix == '.txt' and f.name != ".DS_Store"]
    )[:sample_passages]
    
    texts = []
    text_names = []
    for text_file in text_files:
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            texts.append(f.read().strip())
            text_names.append(text_file.name)
    
    print(f"Using {len(texts)} passages (sampled from {len(list(paragraph_dir.glob('*.txt')))})\n")
    
    # Initialize scorers (reuse across models)
    semantic_scorer = ImprovedSemanticScorer(device=device)
    bert_scorer = ImprovedBERTScorer(device=device)
    
    # Store results by model
    model_results = {}
    
    for model_id in models:
        print(f"\n{'='*70}")
        print(f"Loading model: {model_id}")
        print(f"{'='*70}")
        
        model, tokenizer = load(model_id)
        logger = ExperimentLogger(output_dir=Path('experiment_logs') / model_id.split('/')[-1])
        
        results = []
        
        for passage_text, passage_name in zip(texts, text_names):
            print(f"\nProcessing: {passage_name}")
            
            result = run_chain(
                passage_text, passage_name, model, tokenizer,
                num_iterations=num_iterations,
                num_seeds=num_seeds,
                semantic_scorer=semantic_scorer,
                bert_scorer=bert_scorer,
                logger=logger,
                mode='summarize'
            )
            results.append(result)
        
        # Save logs for this model
        logger.save_logs(suffix=f'_{model_id.split("/")[-1]}')
        model_results[model_id] = results
    
    # Create comparison plots
    print(f"\n{'='*70}")
    print("Generating model comparison plots...")
    print(f"{'='*70}\n")
    
    plot_model_comparison(model_results, models)
    
    # Save summary statistics
    save_model_comparison_stats(model_results, models, text_names)


def plot_model_comparison(model_results: dict, models: list, output_dir: Path = Path('experiment_results')):
    """
    Create plots comparing drift across model sizes.
    """
    output_dir.mkdir(exist_ok=True)
    
    # Aggregate metrics by model
    model_stats = {}
    for model_id, results in model_results.items():
        model_name = model_id.split('/')[-1].replace('-bf16', '')
        
        # Collect all original-to-current similarities
        all_original_sims = []
        all_previous_sims = []
        all_bert_f1s = []
        
        for result in results:
            for chain in result['chains']:
                all_original_sims.append(chain['original_to_current'])
                all_previous_sims.append(chain['previous_to_current'])
                all_bert_f1s.append(chain['bert_f1'])
        
        all_original_sims = np.array(all_original_sims)
        all_previous_sims = np.array(all_previous_sims)
        all_bert_f1s = np.array(all_bert_f1s)
        
        model_stats[model_name] = {
            'original_sims': all_original_sims,
            'previous_sims': all_previous_sims,
            'bert_f1s': all_bert_f1s,
            'orig_mean': all_original_sims.mean(axis=0),
            'orig_std': all_original_sims.std(axis=0),
            'prev_mean': all_previous_sims.mean(axis=0),
            'prev_std': all_previous_sims.std(axis=0),
            'bert_mean': all_bert_f1s.mean(axis=0),
            'bert_std': all_bert_f1s.std(axis=0),
        }
    
    # Determine iteration count
    num_iters = len(next(iter(model_stats.values()))['orig_mean'])
    iterations = np.arange(num_iters)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Size Comparison: Semantic Drift Across Qwen Sizes', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    
    # Plot 1: Original Similarity
    ax = axes[0, 0]
    for (model_name, stats), color in zip(model_stats.items(), colors):
        ax.plot(iterations, stats['orig_mean'], linewidth=2.5, color=color, label=model_name)
        ax.fill_between(iterations, 
                       stats['orig_mean'] - stats['orig_std'],
                       stats['orig_mean'] + stats['orig_std'],
                       alpha=0.2, color=color)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Similarity to Original Text', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Previous-Step Similarity
    ax = axes[0, 1]
    for (model_name, stats), color in zip(model_stats.items(), colors):
        ax.plot(iterations, stats['prev_mean'], linewidth=2.5, color=color, label=model_name)
        ax.fill_between(iterations,
                       stats['prev_mean'] - stats['prev_std'],
                       stats['prev_mean'] + stats['prev_std'],
                       alpha=0.2, color=color)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Similarity to Previous Step', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: BERTScore F1
    ax = axes[1, 0]
    for (model_name, stats), color in zip(model_stats.items(), colors):
        ax.plot(iterations, stats['bert_mean'], linewidth=2.5, color=color, label=model_name)
        ax.fill_between(iterations,
                       stats['bert_mean'] - stats['bert_std'],
                       stats['bert_mean'] + stats['bert_std'],
                       alpha=0.2, color=color)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('BERTScore F1', fontsize=12)
    ax.set_title('BERTScore F1 to Original', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Rate of Drift (derivative)
    ax = axes[1, 1]
    for (model_name, stats), color in zip(model_stats.items(), colors):
        # Compute iteration-to-iteration change
        drift_rate = np.diff(stats['orig_mean'])
        ax.plot(range(len(drift_rate)), drift_rate, linewidth=2.5, color=color, label=model_name)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Change in Similarity (per iteration)', fontsize=12)
    ax.set_title('Rate of Semantic Drift', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = output_dir / 'model_comparison_all_metrics.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {filepath}")
    plt.close()
    
    # Additional plot: Error bar comparison at specific iterations
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iterations_to_plot = [0, 10, 25, 50, 99]
    x_positions = np.arange(len(model_stats))
    width = 0.15
    
    for idx, iteration in enumerate(iterations_to_plot):
        means = [stats['orig_mean'][iteration] for stats in model_stats.values()]
        stds = [stats['orig_std'][iteration] for stats in model_stats.values()]
        x_pos = x_positions + (idx - len(iterations_to_plot)//2) * width
        ax.bar(x_pos, means, width, yerr=stds, capsize=5, 
              label=f'Iteration {iteration}', alpha=0.8)
    
    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('Similarity to Original', fontsize=12)
    ax.set_title('Model Size Effect on Semantic Drift (Selected Iterations)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_stats.keys())
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = output_dir / 'model_comparison_iteration_snapshots.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved iteration snapshot plot to {filepath}")
    plt.close()


def save_model_comparison_stats(model_results: dict, models: list, text_names: list):
    """Save summary statistics for model comparison."""
    stats_file = Path('experiment_results') / 'model_comparison_stats.json'
    
    summary = {
        'models': models,
        'passages': text_names,
        'metrics': {}
    }
    
    for model_id, results in model_results.items():
        model_name = model_id.split('/')[-1].replace('-bf16', '')
        
        model_metrics = []
        for result in results:
            for chain in result['chains']:
                model_metrics.append({
                    'passage': result['passage_name'],
                    'final_original_sim': float(chain['original_to_current'][-1]),
                    'final_bert_f1': float(chain['bert_f1'][-1]),
                    'initial_drop': float(chain['original_to_current'][0] - chain['original_to_current'][1]),
                })
        
        summary['metrics'][model_name] = {
            'avg_final_similarity': float(np.mean([m['final_original_sim'] for m in model_metrics])),
            'std_final_similarity': float(np.std([m['final_original_sim'] for m in model_metrics])),
            'avg_initial_drop': float(np.mean([m['initial_drop'] for m in model_metrics])),
        }
    
    with open(stats_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved statistics to {stats_file}\n")
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    for model_name, metrics in summary['metrics'].items():
        print(f"\n{model_name}:")
        print(f"  Final similarity: {metrics['avg_final_similarity']:.4f} ± {metrics['std_final_similarity']:.4f}")
        print(f"  Initial drop: {metrics['avg_initial_drop']:.4f}")


if __name__ == "__main__":
    # Configuration
    NUM_ITERATIONS = 100
    NUM_SEEDS = 3
    SAMPLE_PASSAGES = 5  # Use subset for faster comparison
    
    run_model_comparison(
        num_iterations=NUM_ITERATIONS,
        num_seeds=NUM_SEEDS,
        sample_passages=SAMPLE_PASSAGES
    )
    
    print("\n" + "="*70)
    print("✓ Model comparison complete!")
    print("="*70)
