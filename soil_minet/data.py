import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]


def load_samples(path: str) -> List[Dict[str, float]]:
    """Load microbiome samples.

    Supported formats:
    - JSONL: each line is a dict of {genus: abundance}
    - CSV (wide): header columns are genera, optional first column sample_id
    """
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return _load_jsonl_samples(source)
    if source.suffix.lower() == ".csv":
        return _load_csv_samples(source)
    raise ValueError(f"Unsupported file format: {source.suffix}")


def _load_jsonl_samples(path: Path) -> List[Dict[str, float]]:
    samples: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sample = {k: float(v) for k, v in record.items() if float(v) > 0}
            samples.append(sample)
    return samples


def _load_csv_samples(path: Path) -> List[Dict[str, float]]:
    samples: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        genus_columns = header
        has_id = False
        if header and header[0].lower() in {"sample_id", "id", "sample"}:
            genus_columns = header[1:]
            has_id = True
        for row in reader:
            if not row:
                continue
            values = row[1:] if has_id else row
            sample = {
                genus: float(value)
                for genus, value in zip(genus_columns, values)
                if value and float(value) > 0
            }
            samples.append(sample)
    return samples


def build_vocab(samples: Iterable[Dict[str, float]]) -> Vocab:
    tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    genera = sorted({genus for sample in samples for genus in sample.keys()})
    tokens.extend(genera)
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token)


def save_vocab(vocab: Vocab, path: str) -> None:
    payload = {"token_to_id": vocab.token_to_id}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab(path: str) -> Vocab:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    token_to_id = {k: int(v) for k, v in payload["token_to_id"].items()}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token)


def encode_sample(
    sample: Dict[str, float],
    vocab: Vocab,
    max_tokens: int | None = None,
) -> Tuple[List[int], List[float]]:
    sorted_items = sorted(sample.items(), key=lambda item: item[1], reverse=True)
    token_ids = [vocab.bos_id]
    abundances = [0.0]
    for genus, abundance in sorted_items:
        token_ids.append(vocab.token_to_id.get(genus, vocab.unk_id))
        abundances.append(float(abundance))
    token_ids.append(vocab.eos_id)
    abundances.append(0.0)
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
        abundances = abundances[:max_tokens]
    return token_ids, abundances


def make_lm_features(
    token_ids: List[int],
    abundances: List[float],
    block_size: int,
    pad_id: int,
) -> Tuple[List[int], List[int], List[int], List[float]]:
    if len(token_ids) < 2:
        raise ValueError("Need at least 2 tokens for language modeling.")
    input_ids = token_ids[:-1]
    labels = token_ids[1:]
    abundance_values = abundances[:-1]
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > block_size:
        input_ids = input_ids[:block_size]
        labels = labels[:block_size]
        attention_mask = attention_mask[:block_size]
        abundance_values = abundance_values[:block_size]

    pad_len = block_size - len(input_ids)
    if pad_len > 0:
        input_ids.extend([pad_id] * pad_len)
        labels.extend([-100] * pad_len)
        attention_mask.extend([0] * pad_len)
        abundance_values.extend([0.0] * pad_len)

    position_ids = list(range(block_size))
    return input_ids, labels, attention_mask, position_ids, abundance_values


class MicrobiomeDataset:
    def __init__(self, samples: List[Dict[str, float]], vocab: Vocab, block_size: int) -> None:
        self.samples = samples
        self.vocab = vocab
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        token_ids, abundances = encode_sample(self.samples[index], self.vocab)
        input_ids, labels, attention_mask, position_ids, abundance_values = make_lm_features(
            token_ids, abundances, self.block_size, self.vocab.pad_id
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "abundance_values": abundance_values,
        }
