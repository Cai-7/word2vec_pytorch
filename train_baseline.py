"""Train NNLM / RNNLM baseline models and export word embeddings.

These are the neural language-model baselines compared against word2vec in
Mikolov et al. "Efficient Estimation of Word Representations in Vector Space".

Checkpointing (aligned with word2vec.py style)
----------------------------------------------
- Mid-training: every ``--checkpoint-interval`` steps
  -> checkpoints_*/checkpoint_XXXXXXXX.pt
- Always also write ``latest.pt`` (easy resume)
- End of each epoch: epoch snapshot + embedding export
- Resume: ``--resume latest`` or ``--resume checkpoint_00001234.pt``

Examples
--------
# Align with CBOW 3 epochs on text8 (emb/hidden=300)
# PowerShell: one line, or multi-line with backtick `
python train_baseline.py --model nnlm --input text8/text8 --output nnlm_text8.txt --emb-dim 300 --hidden 300 --context-size 5 --epochs 3 --batch-size 256 --min-count 5 --checkpoint-dir checkpoints_nnlm --checkpoint-interval 2000

python train_baseline.py --model rnnlm --input text8/text8 --output rnnlm_text8.txt --emb-dim 300 --hidden 300 --seq-len 35 --epochs 3 --batch-size 20 --min-count 5 --checkpoint-dir checkpoints_rnnlm --checkpoint-interval 500

# Resume after interrupt
python train_baseline.py --model nnlm --input text8/text8 --output nnlm_text8.txt --epochs 3 --checkpoint-dir checkpoints_nnlm --resume latest

Evaluate:
    python word_analogy_eval.py --embedding-path nnlm_text8.txt
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_models import NNLM, RNNLM, save_embedding_matrix
from lm_data import LMCorpus, NNLMDataset, NNLMIterable, RNNLMStream, collate_nnlm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NNLM / RNNLM baselines and export word embeddings."
    )
    parser.add_argument("--model", choices=["nnlm", "rnnlm"], required=True,
                        help="Baseline model type.")
    parser.add_argument("--input", required=True, help="Training corpus path.")
    parser.add_argument("--output", required=True, help="Output embedding .txt path.")
    parser.add_argument("--checkpoint-dir", default="checkpoints_baseline",
                        help="Directory for model checkpoints.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save a checkpoint every N optimizer steps. "
             "Default: auto (similar scale to word2vec.py).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a checkpoint path, or 'latest' / 'latest.pt' under "
             "--checkpoint-dir. Use 'none' to disable.",
    )
    parser.add_argument("--emb-dim", type=int, default=300, help="Embedding dimension.")
    parser.add_argument("--hidden", type=int, default=300, help="Hidden layer size.")
    parser.add_argument("--context-size", type=int, default=5,
                        help="NNLM context window (previous words).")
    parser.add_argument("--seq-len", type=int, default=35,
                        help="RNNLM BPTT sequence length.")
    parser.add_argument("--rnn-type", choices=["rnn", "lstm", "gru"], default="rnn",
                        help="Recurrent cell for RNNLM (paper uses plain rnn).")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="NNLM mini-batch size / RNNLM number of parallel streams.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of data passes. Default 3 to align with a 3-epoch CBOW run.",
    )
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Initial learning rate (SGD).")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-vocab", type=int, default=None,
                        help="Optional hard cap on vocabulary size.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Optional cap on NNLM training examples (debug/smoke).")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Optional cap on batches per epoch (debug/smoke).")
    parser.add_argument("--stream", action="store_true",
                        help="Stream NNLM windows instead of materializing them "
                             "(recommended for large corpora).")
    parser.add_argument("--clip", type=float, default=5.0,
                        help="Gradient clipping max-norm (0 to disable).")
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument(
        "--export-every-checkpoint",
        action="store_true",
        help="Also rewrite --output embedding file at every mid-training checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0 ...")
    parser.add_argument("--tie-weights", action="store_true",
                        help="Tie input/output embeddings when dimensions allow.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, name: str, lr: float):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    return torch.optim.SGD(model.parameters(), lr=lr)


def resolve_checkpoint_path(checkpoint_dir: str, resume: Optional[str]) -> Optional[str]:
    if resume is None or str(resume).lower() in {"", "none", "null"}:
        return None
    if resume in {"latest", "latest.pt"}:
        path = os.path.join(checkpoint_dir, "latest.pt")
    elif os.path.isabs(resume) or os.path.exists(resume):
        path = resume
    else:
        path = os.path.join(checkpoint_dir, resume)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {resume} (resolved={path}, dir={checkpoint_dir})"
        )
    return path


def auto_checkpoint_interval(model_name: str, batch_size: int) -> int:
    """Pick a sensible default interval when user does not set one."""
    if model_name == "nnlm":
        # text8 ~17M tokens / batch 256 => ~66k steps/epoch; save ~ every 3%
        return max(500, 2000)
    # RNNLM: fewer steps (batch=streams, seq_len tokens each)
    return max(100, 500)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    epoch: int,
    step: int,
    loss: float,
    args,
    id2word: dict,
    also_latest: bool = True,
) -> None:
    os.makedirs(os.path.dirname(path) or args.checkpoint_dir or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_type": args.model,
        "emb_dimension": args.emb_dim,
        "hidden_size": args.hidden,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "initial_lr": args.lr,
        "min_count": args.min_count,
        "input_file_name": args.input,
        "output_file_name": args.output,
        "args": vars(args),
        "id2word": id2word,
    }
    torch.save(payload, path)
    print(f"Checkpoint saved: {path}  (epoch={epoch}, step={step}, loss={loss:.4f})")
    if also_latest:
        latest = os.path.join(args.checkpoint_dir, "latest.pt")
        torch.save(payload, latest)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    device: torch.device,
) -> Tuple[int, int]:
    """Restore weights/optimizer. Returns (start_epoch, global_step)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    epoch = int(ckpt.get("epoch", 0))
    step = int(ckpt.get("step", 0))
    print(f"Loaded checkpoint: {path} (epoch={epoch}, step={step})")
    return epoch, step


@torch.no_grad()
def export_embeddings(model: nn.Module, id2word: dict, path: str) -> None:
    emb = model.get_input_embeddings()
    save_embedding_matrix(emb, id2word, path)
    print(f"Embeddings exported: {path}  (vocab={len(id2word)}, dim={emb.shape[1]})")


def set_sgd_lr(optimizer, args, epoch: int) -> None:
    """Linear LR decay across epochs (same spirit as word2vec schedule)."""
    if args.optimizer != "sgd" or args.epochs <= 1:
        return
    # After finishing `epoch`, set lr for the next epoch.
    progress = epoch / float(args.epochs)
    new_lr = args.lr * max(1.0 - progress, 0.01)
    for g in optimizer.param_groups:
        g["lr"] = new_lr


def maybe_checkpoint(
    args,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    loss: float,
    id2word: dict,
    interval: int,
    force: bool = False,
    tag: str = "",
) -> None:
    if not force and (interval <= 0 or global_step % interval != 0):
        return
    name = f"checkpoint_{global_step:08d}.pt"
    if tag:
        name = f"{tag}_{name}"
    path = os.path.join(args.checkpoint_dir, name)
    save_checkpoint(
        path, model, optimizer, epoch, global_step, loss, args, id2word, also_latest=True
    )
    if args.export_every_checkpoint:
        export_embeddings(model, id2word, args.output)


def train_nnlm(args, corpus: LMCorpus, device: torch.device) -> nn.Module:
    # Auto-stream on large token counts to avoid multi-GB RAM spikes.
    use_stream = args.stream or corpus.token_count > 5_000_000
    if use_stream:
        print("[NNLM] using streaming window iterator")
        dataset = NNLMIterable(
            corpus,
            context_size=args.context_size,
            max_examples=args.max_examples,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_nnlm,
            drop_last=False,
        )
    else:
        dataset = NNLMDataset(
            corpus,
            context_size=args.context_size,
            max_examples=args.max_examples,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_nnlm,
            drop_last=False,
        )

    model = NNLM(
        vocab_size=corpus.vocab_size,
        emb_dimension=args.emb_dim,
        context_size=args.context_size,
        hidden_size=args.hidden,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.optimizer, args.lr)

    resume_path = resolve_checkpoint_path(args.checkpoint_dir, args.resume)
    start_epoch = 1
    global_step = 0
    if resume_path is not None:
        finished_epoch, global_step = load_checkpoint(
            resume_path, model, optimizer, device
        )
        start_epoch = finished_epoch + 1
        if start_epoch > args.epochs:
            print(
                f"Resume epoch ({finished_epoch}) already >= target epochs "
                f"({args.epochs}); exporting embeddings only."
            )
            export_embeddings(model, corpus.id2word, args.output)
            return model

    interval = args.checkpoint_interval
    if interval is None:
        interval = auto_checkpoint_interval("nnlm", args.batch_size)
    print(f"[NNLM] checkpoint_interval={interval} steps, dir={args.checkpoint_dir}")

    # Estimate batches/epoch for tqdm (streaming loaders have no __len__).
    if args.max_batches is not None:
        estimated_batches = args.max_batches
    elif hasattr(loader.dataset, "__len__"):
        estimated_batches = math.ceil(len(loader.dataset) / max(1, args.batch_size))
    else:
        # ~one prediction per token after context
        estimated_batches = max(
            1, int(corpus.token_count // max(1, args.batch_size))
        )

    last_loss = 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()
        pbar = tqdm(
            enumerate(loader, start=1),
            total=estimated_batches,
            desc=f"NNLM epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for batch_idx, (context, target) in pbar:
            if args.max_batches is not None and batch_idx > args.max_batches:
                break
            context = context.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(context)
            loss = criterion(logits, target)
            loss.backward()
            if args.clip and args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            bs = target.size(0)
            last_loss = loss.item()
            total_loss += last_loss * bs
            total_tokens += bs
            global_step += 1

            if batch_idx % args.log_interval == 0 or batch_idx == 1:
                avg = total_loss / max(1, total_tokens)
                ppl = math.exp(min(avg, 20))
                pbar.set_postfix(
                    loss=f"{avg:.4f}",
                    ppl=f"{ppl:.1f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.5f}",
                    step=global_step,
                    refresh=False,
                )

            maybe_checkpoint(
                args, model, optimizer, epoch, global_step, last_loss,
                corpus.id2word, interval, force=False,
            )

        pbar.close()
        avg_epoch = total_loss / max(1, total_tokens)
        # End-of-epoch snapshot (always)
        epoch_path = os.path.join(
            args.checkpoint_dir, f"nnlm_epoch{epoch:02d}_step{global_step:08d}.pt"
        )
        save_checkpoint(
            epoch_path, model, optimizer, epoch, global_step, avg_epoch,
            args, corpus.id2word, also_latest=True,
        )
        export_embeddings(model, corpus.id2word, args.output)
        set_sgd_lr(optimizer, args, epoch)
        print(
            f"[NNLM] finished epoch {epoch}/{args.epochs}  "
            f"avg_loss={avg_epoch:.4f}  ppl={math.exp(min(avg_epoch, 20)):.2f}  "
            f"time={time.time() - t0:.1f}s"
        )

    return model


def train_rnnlm(args, corpus: LMCorpus, device: torch.device) -> nn.Module:
    stream = RNNLMStream(
        corpus,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    # IterableDataset: one "epoch" = one pass over stream
    loader = DataLoader(stream, batch_size=None, num_workers=0)

    model = RNNLM(
        vocab_size=corpus.vocab_size,
        emb_dimension=args.emb_dim,
        hidden_size=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
        tie_weights=args.tie_weights,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.optimizer, args.lr)

    resume_path = resolve_checkpoint_path(args.checkpoint_dir, args.resume)
    start_epoch = 1
    global_step = 0
    if resume_path is not None:
        finished_epoch, global_step = load_checkpoint(
            resume_path, model, optimizer, device
        )
        start_epoch = finished_epoch + 1
        if start_epoch > args.epochs:
            print(
                f"Resume epoch ({finished_epoch}) already >= target epochs "
                f"({args.epochs}); exporting embeddings only."
            )
            export_embeddings(model, corpus.id2word, args.output)
            return model

    interval = args.checkpoint_interval
    if interval is None:
        interval = auto_checkpoint_interval("rnnlm", args.batch_size)
    print(
        f"[RNNLM/{args.rnn_type}] checkpoint_interval={interval} steps, "
        f"dir={args.checkpoint_dir}"
    )

    if args.max_batches is not None:
        estimated_batches = args.max_batches
    else:
        # tokens reshaped to [batch_size, T]; steps = (T-1) // seq_len
        tokens_per_stream = max(1, corpus.token_count // max(1, args.batch_size))
        estimated_batches = max(1, (tokens_per_stream - 1) // max(1, args.seq_len))

    last_loss = 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        hidden = model.init_hidden(args.batch_size, device)
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()

        pbar = tqdm(
            enumerate(loader, start=1),
            total=estimated_batches,
            desc=f"RNNLM/{args.rnn_type} epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for batch_idx, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)

            # Detach hidden to truncate BPTT across batches
            if isinstance(hidden, tuple):
                hidden = tuple(h.detach() for h in hidden)
            else:
                hidden = hidden.detach()

            optimizer.zero_grad()
            logits, hidden = model(x, hidden)
            # logits: [B, T, V], y: [B, T]
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            if args.clip and args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            n_tokens = y.numel()
            last_loss = loss.item()
            total_loss += last_loss * n_tokens
            total_tokens += n_tokens
            global_step += 1

            if batch_idx % args.log_interval == 0 or batch_idx == 1:
                avg = total_loss / max(1, total_tokens)
                ppl = math.exp(min(avg, 20))
                pbar.set_postfix(
                    loss=f"{avg:.4f}",
                    ppl=f"{ppl:.1f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.5f}",
                    step=global_step,
                    refresh=False,
                )

            maybe_checkpoint(
                args, model, optimizer, epoch, global_step, last_loss,
                corpus.id2word, interval, force=False,
            )

        pbar.close()
        avg_epoch = total_loss / max(1, total_tokens)
        epoch_path = os.path.join(
            args.checkpoint_dir,
            f"rnnlm_{args.rnn_type}_epoch{epoch:02d}_step{global_step:08d}.pt",
        )
        save_checkpoint(
            epoch_path, model, optimizer, epoch, global_step, avg_epoch,
            args, corpus.id2word, also_latest=True,
        )
        export_embeddings(model, corpus.id2word, args.output)
        set_sgd_lr(optimizer, args, epoch)
        print(
            f"[RNNLM/{args.rnn_type}] finished epoch {epoch}/{args.epochs}  "
            f"avg_loss={avg_epoch:.4f}  ppl={math.exp(min(avg_epoch, 20)):.2f}  "
            f"time={time.time() - t0:.1f}s"
        )

    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print(f"Baseline model : {args.model.upper()}"
          + (f" ({args.rnn_type})" if args.model == "rnnlm" else ""))
    print(f"Device         : {device}")
    print(f"Input corpus   : {args.input}")
    print(f"Output emb     : {args.output}")
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"emb_dim={args.emb_dim}  hidden={args.hidden}  "
          f"batch_size={args.batch_size}  epochs={args.epochs}  lr={args.lr}")
    print("=" * 60)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input corpus not found: {args.input}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    corpus = LMCorpus(
        file_name=args.input,
        min_count=args.min_count,
        max_vocab=args.max_vocab,
    )

    if args.model == "nnlm":
        if corpus.vocab_size > 100000:
            print(
                f"[Warn] vocab_size={corpus.vocab_size} is large for full-softmax NNLM. "
                "Consider --max-vocab 50000 or a higher --min-count."
            )
        model = train_nnlm(args, corpus, device)
    else:
        if corpus.vocab_size > 100000:
            print(
                f"[Warn] vocab_size={corpus.vocab_size} is large for full-softmax RNNLM. "
                "Consider --max-vocab 50000 or a higher --min-count."
            )
        model = train_rnnlm(args, corpus, device)

    export_embeddings(model, corpus.id2word, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
