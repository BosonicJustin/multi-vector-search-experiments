import argparse
import json
import os
import time
import glob
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from colbert_encoders import ColbertEncoder
from ms_marco_data_loader import MSMarcoDataset, collate_fn
from loss_fn import MultivectorLoss
from validation import validate, K_VALUES


def save_checkpoint(query_encoder, doc_encoder, optimizer, scaler, step, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save({
        "step": step,
        "epoch": epoch,
        "query_encoder": query_encoder.state_dict(),
        "doc_encoder": doc_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }, path)
    print(f"Saved checkpoint to {path}")


def load_latest_checkpoint(query_encoder, doc_encoder, optimizer, scaler, checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        return 0, 0

    path = files[-1]
    print(f"Resuming from {path}")
    ckpt = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    query_encoder.load_state_dict(ckpt["query_encoder"])
    doc_encoder.load_state_dict(ckpt["doc_encoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"], ckpt["epoch"]


def save_logs(log, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_log.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def load_logs(output_dir):
    path = os.path.join(output_dir, "training_log.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"train_loss": [], "val_metrics": []}


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: max_steps={args.max_steps}, batch_size={args.batch_size}, lr={args.lr}, "
          f"model={args.model_name}, search_dim={args.search_dim}")

    query_encoder = ColbertEncoder(encoder_branch="query", model_name=args.model_name, search_dim=args.search_dim)
    doc_encoder = ColbertEncoder(encoder_branch="document", model_name=args.model_name, search_dim=args.search_dim)
    loss_fn = MultivectorLoss()

    params = list(query_encoder.parameters()) + list(doc_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    start_step, start_epoch = 0, 0
    if args.resume:
        start_step, start_epoch = load_latest_checkpoint(query_encoder, doc_encoder, optimizer, scaler, args.checkpoint_dir)
        print(f"Resumed at step {start_step}, epoch {start_epoch}")

    print("Loading datasets...")
    train_dataset = MSMarcoDataset(split="train")
    val_dataset = MSMarcoDataset(split="validation")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=(device == "cuda"), persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=4, pin_memory=(device == "cuda"), persistent_workers=True)

    if args.overfit_batches:
        fixed_batches = []
        for batch in train_loader:
            fixed_batches.append(batch)
            if len(fixed_batches) >= args.overfit_batches:
                break
        print(f"Overfit mode: looping over {len(fixed_batches)} fixed batches")

    log = load_logs(args.output_dir) if args.resume else {"train_loss": [], "val_metrics": []}
    writer = SummaryWriter(log_dir=args.tb_dir)

    step = start_step
    running_loss = 0.0
    epoch = start_epoch
    start_time = time.time()

    while step < args.max_steps:
        query_encoder.train()
        doc_encoder.train()
        epoch += 1

        for batch in (fixed_batches if args.overfit_batches else train_loader):
            if step >= args.max_steps:
                break

            queries = batch["queries"]
            positives = batch["positives"]
            negative_lists = batch["negative_lists"]

            with torch.amp.autocast(device, enabled=(device == "cuda")):
                query_embs, query_mask = query_encoder(queries)
                pos_embs, pos_mask = doc_encoder(positives)

                num_negs = [len(negs) for negs in negative_lists]
                all_negatives = [neg for negs in negative_lists for neg in negs]
                neg_embs_flat, neg_mask_flat = doc_encoder(all_negatives)

                neg_embs = torch.nn.utils.rnn.pad_sequence(
                    torch.split(neg_embs_flat, num_negs), batch_first=True)
                neg_mask = torch.nn.utils.rnn.pad_sequence(
                    torch.split(neg_mask_flat, num_negs), batch_first=True)

                loss = loss_fn(query_embs, pos_embs, pos_mask, neg_embs, neg_mask)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            running_loss += loss.item()

            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                elapsed = time.time() - start_time
                print(f"Step {step}/{args.max_steps} (epoch {epoch}) | "
                      f"Loss {avg_loss:.4f} | {elapsed:.1f}s elapsed")
                log["train_loss"].append({"step": step, "epoch": epoch, "loss": avg_loss})
                writer.add_scalar("train/loss", avg_loss, step)
                running_loss = 0.0
                save_logs(log, args.output_dir)

            if step % args.val_every == 0:
                print("Running validation...")
                val_start = time.time()
                metrics = validate(query_encoder, doc_encoder, val_loader, device, args.max_val_batches)
                val_time = time.time() - val_start

                print(f"\n--- Validation at step {step} ({val_time:.1f}s) ---")
                print(f"  Val Loss={metrics['loss']:.4f}")
                for k in K_VALUES:
                    print(
                        f"  @{k:>3d}  "
                        f"Recall={metrics[f'recall@{k}']:.4f}  "
                        f"Precision={metrics[f'precision@{k}']:.4f}  "
                        f"NDCG={metrics[f'ndcg@{k}']:.4f}"
                    )
                print()

                for key, val in metrics.items():
                    writer.add_scalar(f"val/{key}", val, step)
                log["val_metrics"].append({"step": step, "epoch": epoch, **metrics})
                save_logs(log, args.output_dir)

                query_encoder.train()
                doc_encoder.train()

            if step % args.save_every == 0:
                save_checkpoint(query_encoder, doc_encoder, optimizer, scaler, step, epoch, args.checkpoint_dir)

    save_checkpoint(query_encoder, doc_encoder, optimizer, scaler, step, epoch, args.checkpoint_dir)
    save_logs(log, args.output_dir)
    writer.close()
    print(f"Training complete. {step} steps in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--search_dim", type=int, default=128)
    parser.add_argument("--val_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--max_val_batches", type=int, default=50)
    parser.add_argument("--tb_dir", type=str, default="runs")
    parser.add_argument("--overfit_batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())
