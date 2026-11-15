# Dilution of Meaning: Semantic Drift in Iterative LLM Summarization

## Research Question

How do language models exhibit attractor dynamics in semantic space when repeatedly summarizing the same text? Specifically, does iterative summarization cause text to converge toward particular semantic attractors, and can we measure this drift using embedding-based and token-level metrics?

## Overview

This project investigates whether language models exhibit "attractor dynamics"—a tendency to converge toward particular outputs or semantic regions—when iteratively summarizing or paraphrasing text. By repeatedly running a text through a summarization pipeline, we can observe:

1. **Semantic Drift**: Gradual loss of fidelity to the original meaning
2. **Convergence Patterns**: Whether the model converges to stable outputs or exhibits chaotic behavior
3. **Memorization Effects**: How much well-known passages behave differently from obscure texts

This has implications for understanding LLM reliability in multi-turn conversations, chain-of-thought reasoning, and cumulative error propagation.

## Methodology

- **Model**: Qwen3-0.6B (lightweight, runs on Apple Silicon)
- **Iterations**: 100 rounds of summarization per text
- **Input Texts**: 20 passages from public-domain literature (mix of famous and obscure)
- **Metrics**:
  - **BERTScore F1**: Semantic precision/recall relative to original text
  - **Cosine Similarity**: Embedding-based similarity using Sentence Transformers

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the experiment
python summarization_chain_experiment.py

# View results
# Results are plotted and saved as graphs
```

## Directory Structure

```
dilution_of_meaning/
├── paragraphs/              # 20 public-domain test passages
│   └── SOURCES.md          # Attribution and licensing info
├── summarization_chain_experiment.py
├── requirements.txt
└── README.md
```

## References

- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)
- [The Telephone Game for Local LLMs: Semantic Drift](https://joshua8.ai/llm-telephone-game-semantic-drift/)
- [Attractor Dynamics in Neural Language Models](https://www.semanticscholar.org/paper/Drift-No-More-Context-Equilibria-in-Multi-Turn-LLM-Dongre-Rossi/7af5369308e2ee98df1e458f3e040ce674559f1d)
