# PubMed Alignment - Quick Reference

## 🚀 Quick Start (Most Common Usage)

```bash
# Recommended: Hybrid strategy (best balance)
python align_pubmed_optimized.py \
    --abstracts abstracts.txt \
    --dataset your_hf_dataset \
    --text-column abstract \
    --id-column pmid \
    --mesh-ids-column mesh_ids \
    --strategy hybrid \
    --threshold 0.7 \
    --output matches.json
```

## 📊 When to Use Which Script

| Script | Best For | Speed | Accuracy |
|--------|----------|-------|----------|
| `align_pubmed_optimized.py` | Most cases | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ |
| `align_pubmed_embeddings.py` | Semantic matching | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| `benchmark_alignment.py` | Testing strategies | - | - |
| `analyze_matches.py` | Result analysis | - | - |

## 🎯 Strategy Selection

```bash
# Exact duplicates only → FASTEST
--strategy exact_hash

# High similarity (>0.7) → FAST + ACCURATE
--strategy lsh --threshold 0.7

# Fuzzy matching → SLOWER + FLEXIBLE
--strategy tfidf_blocking --threshold 0.6

# Unknown data quality → RECOMMENDED
--strategy hybrid --threshold 0.7

# Best semantic accuracy → GPU RECOMMENDED
python align_pubmed_embeddings.py --model all-MiniLM-L6-v2 --gpu
```

## 💡 Common Commands

### Test First (10k records)
```bash
python align_pubmed_optimized.py \
    --abstracts abstracts.txt \
    --dataset your_dataset \
    --text-column abstract \
    --id-column pmid \
    --mesh-ids-column mesh_ids \
    --strategy hybrid \
    --debug
```

### Full Run with Progress
```bash
python align_pubmed_optimized.py \
    --abstracts abstracts.txt \
    --dataset your_dataset \
    --text-column abstract \
    --id-column pmid \
    --mesh-ids-column mesh_ids \
    --strategy hybrid \
    --threshold 0.7 \
    --output matches.json \
    --log-level INFO
```

### Embeddings with GPU
```bash
python align_pubmed_embeddings.py \
    --abstracts abstracts.txt \
    --dataset your_dataset \
    --text-column abstract \
    --id-column pmid \
    --mesh-ids-column mesh_ids \
    --model allenai-specter \
    --gpu \
    --threshold 0.75 \
    --cache embeddings_cache.pkl
```

### Benchmark Strategies
```bash
python benchmark_alignment.py \
    --abstracts abstracts.txt \
    --dataset your_dataset \
    --text-column abstract \
    --id-column pmid \
    --mesh-ids-column mesh_ids \
    --sample-size 1000
```

### Analyze Results
```bash
python analyze_matches.py matches.json --report report.txt --plot plot.png
```

## ⚙️ Key Parameters

| Parameter | Default | Description | Tuning Guide |
|-----------|---------|-------------|--------------|
| `--threshold` | 0.7 | Similarity cutoff | 0.9=strict, 0.6=loose |
| `--strategy` | hybrid | Matching method | See table above |
| `--debug` | False | Limit to 10k | Always test first |
| `--batch-size` | 32 | Processing batch | Increase with GPU |
| `--gpu` | False | Use GPU | For embeddings only |

## 🔧 Troubleshooting Quick Fixes

```bash
# Too few matches
--threshold 0.6  # Lower threshold
--strategy tfidf_blocking  # More flexible

# Too many false positives
--threshold 0.8  # Raise threshold
--strategy lsh  # More strict

# Out of memory
--strategy exact_hash  # Minimal memory
--max-abstracts 500000  # Process in chunks

# Too slow
--strategy exact_hash  # Fastest
--gpu  # For embeddings
```

## 📈 Expected Performance (10M records)

| Strategy | Time | Memory | Match Rate |
|----------|------|--------|------------|
| exact_hash | 10 sec | 2 GB | High (exact) |
| lsh | 14 hrs | 8 GB | 90-95% |
| hybrid | 6 hrs | 12 GB | 95-99% |
| embeddings | 3-8 hrs | 15 GB | 95-99% |

## 🎓 Best Practices

1. **Always test first**: Use `--debug` mode
2. **Start with hybrid**: Best default strategy
3. **Check results**: Use `analyze_matches.py`
4. **Tune threshold**: Start at 0.7, adjust based on results
5. **Use GPU**: For embeddings, 10-50× speedup

## 📦 Installation

```bash
# Optimized version
pip install datasets nltk numpy

# Embeddings version
pip install sentence-transformers faiss-cpu datasets
# OR with GPU:
pip install sentence-transformers faiss-gpu datasets

# Analysis tools
pip install numpy matplotlib pandas psutil
```

## 📚 Full Documentation

- **Complete Guide**: `README_ALIGNMENT.md`
- **Technical Details**: `OPTIMIZATION_GUIDE.md`
- **Original Script**: `align_pubmed.py` (legacy)

## 🆘 Need Help?

1. Check if file exists: `ls -lh abstracts.txt`
2. Verify dataset loads: `python -c "from datasets import load_dataset; ds = load_dataset('your_dataset', streaming=True)"`
3. Test with debug mode: `--debug --log-level DEBUG`
4. Run benchmark: `python benchmark_alignment.py ...`

---

**TL;DR**: Start with `python align_pubmed_optimized.py --strategy hybrid --debug`, check results, then run full dataset.