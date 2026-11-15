#!/usr/bin/env python3
"""
Validation script for Tasks 4-9
Quick sanity checks without running the full experiment.
"""

import torch
import json
from pathlib import Path
from summarization_chain_experiment_v2 import (
    ExperimentLogger, ImprovedSemanticScorer, ImprovedBERTScorer
)


def test_logger():
    """Test Task 4: Logging functionality."""
    print("\n" + "="*70)
    print("TEST 1: Logging (Task 4)")
    print("="*70)
    
    logger = ExperimentLogger(output_dir=Path('/tmp/test_logs'))
    
    # Simulate logging some steps
    passage = "test_passage.txt"
    for seed in range(2):
        for iteration in range(5):
            text = f"This is iteration {iteration} with seed {seed}."
            logger.log_step(passage, seed, iteration, text, "test-model", 0.7)
    
    # Test retrieval
    chain_0 = logger.get_chain(passage, 0)
    chain_1 = logger.get_chain(passage, 1)
    
    print(f"✓ Logged 10 steps (2 seeds × 5 iterations)")
    print(f"✓ Retrieved chain 0 length: {len(chain_0)}")
    print(f"✓ Retrieved chain 1 length: {len(chain_1)}")
    
    # Save and load
    saved_file = logger.save_logs(suffix='_test')
    print(f"✓ Saved to {saved_file}")
    
    logger2 = ExperimentLogger()
    chains = logger2.load_logs(saved_file)
    print(f"✓ Loaded {len(chains)} chains from file")
    
    # Verify we can reconstruct
    assert len(chains[f"{passage}_seed0"]['steps']) == 5
    print(f"✓ Reconstructed chain has correct number of steps")
    
    return True


def test_semantic_scorer():
    """Test Task 5: Semantic scorer with token limits."""
    print("\n" + "="*70)
    print("TEST 2: Improved Semantic Scorer (Task 5)")
    print("="*70)
    
    device = "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    scorer = ImprovedSemanticScorer(device=device)
    print(f"✓ Initialized scorer on {device}")
    
    # Test 1: Simple similarity
    text1 = "The cat sat on the mat."
    text2 = "A cat was sitting on a mat."
    sim = scorer.score(text1, text2)
    print(f"✓ Similarity score: {sim:.4f} (expected ~0.9)")
    assert 0.7 < sim < 1.0, f"Unexpected similarity: {sim}"
    
    # Test 2: Very different texts
    text3 = "The quick brown fox."
    sim2 = scorer.score(text1, text3)
    print(f"✓ Different text similarity: {sim2:.4f} (expected ~0.5)")
    assert 0.3 < sim2 < 0.7, f"Unexpected similarity: {sim2}"
    
    # Test 3: Long text (test token limit handling)
    long_text = " ".join(["word"] * 500)  # 500 words
    try:
        sim3 = scorer.score(text1, long_text)
        print(f"✓ Handled 500-word text without error")
    except Exception as e:
        print(f"✗ Failed on long text: {e}")
        return False
    
    # Test 4: Batch scoring
    texts1 = ["cat", "dog", "bird"]
    texts2 = ["feline", "canine", "avian"]
    batch_sims = scorer.score_batch(texts1, texts2)
    print(f"✓ Batch scoring: {len(batch_sims)} scores computed")
    assert len(batch_sims) == 3, "Wrong batch size"
    
    return True


def test_bert_scorer():
    """Test Task 5: Improved BERT scorer."""
    print("\n" + "="*70)
    print("TEST 3: Improved BERT Scorer (Task 5)")
    print("="*70)
    
    device = "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    scorer = ImprovedBERTScorer(device=device)
    print(f"✓ Initialized BERT scorer on {device}")
    
    # Test scoring
    ref = "The cat is on the mat."
    cand = "A cat sits on a mat."
    
    try:
        p, r, f1 = scorer.score(ref, cand)
        print(f"✓ Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
        assert 0 <= p <= 1 and 0 <= r <= 1 and 0 <= f1 <= 1, "Scores out of range"
        print(f"✓ All scores in valid [0,1] range")
    except Exception as e:
        print(f"✗ BERT scoring failed: {e}")
        return False
    
    return True


def test_passages_loaded():
    """Test that passages exist and are readable."""
    print("\n" + "="*70)
    print("TEST 4: Passages (Task 2)")
    print("="*70)
    
    paragraph_dir = Path('paragraphs')
    
    text_files = list(
        f for f in paragraph_dir.iterdir()
        if f.is_file() and f.suffix == '.txt' and f.name != ".DS_Store"
    )
    
    print(f"✓ Found {len(text_files)} passages")
    assert len(text_files) >= 20, f"Expected 20+ passages, found {len(text_files)}"
    
    # Test loading
    total_words = 0
    for text_file in text_files[:3]:  # Check first 3
        with open(text_file, 'r') as f:
            text = f.read().strip()
            words = len(text.split())
            total_words += words
            print(f"  - {text_file.name}: {words} words")
    
    avg_words = total_words // 3
    print(f"✓ Average passage: {avg_words} words")
    assert avg_words > 50, f"Passages too short: {avg_words}"
    
    return True


def test_configuration():
    """Test that all configuration options work."""
    print("\n" + "="*70)
    print("TEST 5: Configuration")
    print("="*70)
    
    # Test device detection
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ Apple Silicon detected (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"✓ CUDA GPU detected")
    else:
        print(f"✓ Using CPU fallback")
    
    # Test output directories exist
    log_dir = Path('experiment_logs')
    result_dir = Path('experiment_results')
    
    log_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    
    print(f"✓ Output directories ready:")
    print(f"  - {log_dir}")
    print(f"  - {result_dir}")
    
    return True


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "VALIDATION TESTS: TASKS 4-9" + " "*26 + "║")
    print("╚" + "="*68 + "╝")
    
    tests = [
        ("Logging (Task 4)", test_logger),
        ("Semantic Scorer (Task 5)", test_semantic_scorer),
        ("BERT Scorer (Task 5)", test_bert_scorer),
        ("Passages (Task 2)", test_passages_loaded),
        ("Configuration", test_configuration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for test_name, passed_flag in results:
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All systems ready! Run with:")
        print("  python summarization_chain_experiment_v2.py")
    else:
        print("\n✗ Some tests failed. Check errors above.")
    
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
