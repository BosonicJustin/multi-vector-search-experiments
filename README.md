# ColBERT: Multi-Vector Dense Retrieval

A from-scratch implementation of [ColBERT](https://arxiv.org/abs/2004.12832) (Contextualized Late Interaction over BERT) trained and evaluated on the MS MARCO passage ranking dataset.

## Results (Step 200k)

| Method              | MRR@10 (ours) | MRR@10 (paper) |
|---------------------|---------------|----------------|
| ColBERT Re-ranking  | **0.3506**    | 0.3490         |
| ColBERT End-to-End  | **0.6477**    | 0.3600         |

## Project Structure

```
colbert/
├── training.py              # Main training loop (AMP, checkpointing, TensorBoard)
├── colbert_encoders.py      # Dual-encoder: query & document encoding with BERT
├── loss_fn.py               # Contrastive triplet loss via cross-entropy
├── scoring.py               # MaxSim late-interaction scoring
├── validation.py            # Re-ranking evaluation during training
├── ms_marco_data_loader.py  # Memory-efficient streaming dataset via byte offsets
├── preprocess.py            # MS MARCO preprocessing + BM25 hard negative mining
├── eval/
│   ├── run_eval.py          # Full evaluation: re-ranking & end-to-end with FAISS
│   └── evaluation.ipynb     # Results visualization & comparison with baselines
├── data/                    # MS MARCO dataset files
├── checkpoints/             # Saved model checkpoints
├── eval_results/            # Evaluation result JSONs
└── runs/                    # TensorBoard logs
```

## How It Works

ColBERT encodes queries and documents into **per-token embeddings** (multi-vector representations), then scores relevance using **MaxSim**: each query token finds its best-matching document token, and these maximum similarities are summed.

- **Query encoder**: Pads to 32 tokens with `[MASK]`, projects to 128-dim, L2-normalizes
- **Document encoder**: Up to 180 tokens, filters punctuation embeddings, projects to 128-dim, L2-normalizes
- **Scoring**: `score = sum over q_i of max over d_j of (q_i . d_j)`

## Quick Start

### Prerequisites

```
torch
transformers
datasets
bm25s
faiss-cpu  # or faiss-gpu
numpy
tensorboard
tqdm
```

### Preprocessing

Download MS MARCO and generate training triplets with BM25 hard negatives:

```bash
python preprocess.py --splits train validation
```

### Training

```bash
python training.py \
    --max_steps 200000 \
    --batch_size 32 \
    --lr 3e-6 \
    --model_name bert-base-uncased \
    --search_dim 128
```

Key flags:
- `--resume` to continue from a checkpoint
- `--overfit_batches N` for debugging on N fixed batches
- `--save_every 10000` checkpoint interval
- `--val_every 10000` / `--val_every_early 1000` validation intervals

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs/
```

### Evaluation

```bash
# Re-ranking (ColBERT re-scores top-1000 BM25 candidates)
python eval/run_eval.py --step 200000 --mode rerank

# End-to-end (FAISS-based retrieval from full corpus)
python eval/run_eval.py --step 200000 --mode e2e --max_passages 1000000

# Both
python eval/run_eval.py --step 200000 --mode all
```

## Training Details

- **Base model**: `bert-base-uncased`
- **Optimizer**: Adam, lr=3e-6
- **Mixed precision**: Enabled (AMP with GradScaler)
- **Loss**: Cross-entropy over (positive_score, negative_score) pairs
- **Hard negatives**: BM25-mined from top-10 non-relevant passages
- **Data loading**: Lazy byte-offset reading for memory efficiency on 28GB+ triplet files
