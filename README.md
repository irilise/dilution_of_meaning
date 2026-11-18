# Dilution of Meaning: Semantic Drift in Iterative LLM Summarization

## Research Question

When viewing LLM to LLM text chains as dynamical systems, whether as a result of agentic loops or the propagation of LLM-generated content on the internet, how do attractor dynamics apply?

We seek to identify any attractors and their strength, in addition to making attempts at predicting possible dangers or demerits of these attractors. Are they harmless, or might they introduce ethicial concerns such as biases, or even cause degeneration of model performance similar to model collapse?
## Overview

We first reproduce the experiment of Perez et. al., 2025 "When LLM's play the telephone game," by creating chains of repeated rephrasing and measuring toxicity, positivity, difficulty, and length. As an extension, we also use metrics like bertscore, cosine similarity, and calibrated cosine similarity. 

We then extend Perez' experiment with a different LLM chain, which seeks to emulate what happens with question answer chains, modeling the propagation of LLM-generated responses to questions across the internet. 

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
