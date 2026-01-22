import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from soil_minet.data import MicrobiomeDataset, build_vocab, load_samples, save_vocab
from soil_minet.model import GPTConfig, GPTModel


def collate_batch(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    output = {}
    for key in keys:
        output[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
    return output


def train(args: argparse.Namespace) -> None:
    samples = load_samples(args.data_path)
    vocab = build_vocab(samples)
    dataset = MicrobiomeDataset(samples, vocab, args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    config = GPTConfig(
        vocab_size=len(vocab.token_to_id),
        block_size=args.block_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPTModel(config)
    device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                labels=batch["labels"],
            )
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()
            if step % args.log_interval == 0:
                avg_loss = total_loss / args.log_interval
                print(f"Epoch {epoch + 1} Step {step} Loss {avg_loss:.4f}")
                total_loss = 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "gpt_model.pt")
    save_vocab(vocab, str(output_dir / "vocab.json"))
    config_path = output_dir / "config.json"
    config_payload = {
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_embd": config.n_embd,
        "dropout": config.dropout,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT-style model on microbiome abundance sequences.")
    parser.add_argument("--data-path", required=True, help="Path to .csv or .jsonl abundance file.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save model and vocab.")
    parser.add_argument("--block-size", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
