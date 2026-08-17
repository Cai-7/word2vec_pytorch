"""Language-model data pipeline for NNLM / RNNLM baselines.

Reuses the same vocabulary construction conventions as word2vec InputData
(min_count filtering + light token normalization) so baselines are comparable
on the same corpus and word list.
"""

from __future__ import annotations

import string
from collections import Counter
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class LMCorpus:
    """Build vocabulary and stream tokenized sentences from a corpus file.

    Each non-empty line is treated as one sentence of space-separated tokens.
    Very long single-line corpora (e.g. text8) are also supported.
    """

    def __init__(
        self,
        file_name: str,
        min_count: int = 5,
        max_vocab: Optional[int] = None,
        unk_token: str = "<unk>",
        eos_token: str = "<eos>",
    ):
        self.file_name = file_name
        self.min_count = min_count
        self.max_vocab = max_vocab
        self.unk_token = unk_token
        self.eos_token = eos_token

        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.word_frequency: Dict[int, int] = {}
        self.sentence_count = 0
        self.token_count = 0
        self._build_vocab()

    @staticmethod
    def _normalize_token(token: str) -> str:
        if not token:
            return ""
        return token.strip(string.punctuation + "“”‘’")

    def _iter_raw_tokens(self) -> Iterator[List[str]]:
        with open(self.file_name, encoding="utf-8") as handle:
            for line in handle:
                tokens = [
                    self._normalize_token(tok)
                    for tok in line.strip().split()
                ]
                tokens = [tok for tok in tokens if tok]
                if tokens:
                    yield tokens

    def _build_vocab(self) -> None:
        counter: Counter = Counter()
        sentence_count = 0
        token_count = 0
        for tokens in self._iter_raw_tokens():
            sentence_count += 1
            token_count += len(tokens)
            counter.update(tokens)

        # Keep frequent words; reserve special tokens at the front.
        items = [(w, c) for w, c in counter.items() if c >= self.min_count]
        items.sort(key=lambda x: (-x[1], x[0]))
        if self.max_vocab is not None and self.max_vocab > 0:
            # leave room for <unk> and <eos>
            items = items[: max(0, self.max_vocab - 2)]

        self.word2id = {self.unk_token: 0, self.eos_token: 1}
        self.id2word = {0: self.unk_token, 1: self.eos_token}
        self.word_frequency = {0: 0, 1: sentence_count}  # eos ~ once per sentence

        wid = 2
        kept_tokens = 0
        for word, count in items:
            if word in self.word2id:
                continue
            self.word2id[word] = wid
            self.id2word[wid] = word
            self.word_frequency[wid] = count
            kept_tokens += count
            wid += 1

        # Approximate unk frequency from discarded mass.
        total_raw = sum(counter.values())
        self.word_frequency[0] = max(0, total_raw - kept_tokens)
        self.sentence_count = sentence_count
        self.token_count = token_count
        self.vocab_size = len(self.word2id)

        print(f"[LMCorpus] file={self.file_name}")
        print(f"[LMCorpus] sentences={self.sentence_count}, raw_tokens={self.token_count}")
        print(f"[LMCorpus] vocab_size={self.vocab_size} (min_count={self.min_count})")

    @property
    def unk_id(self) -> int:
        return self.word2id[self.unk_token]

    @property
    def eos_id(self) -> int:
        return self.word2id[self.eos_token]

    def encode_tokens(self, tokens: Sequence[str], add_eos: bool = True) -> List[int]:
        ids = [self.word2id.get(tok, self.unk_id) for tok in tokens]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def iter_sentence_ids(self, add_eos: bool = True) -> Iterator[List[int]]:
        for tokens in self._iter_raw_tokens():
            yield self.encode_tokens(tokens, add_eos=add_eos)


class NNLMDataset(Dataset):
    """In-memory fixed-context windows for NNLM (good for small/medium corpora)."""

    def __init__(self, corpus: LMCorpus, context_size: int = 5, max_examples: Optional[int] = None):
        self.context_size = context_size
        self.contexts: List[List[int]] = []
        self.targets: List[int] = []

        for ids in corpus.iter_sentence_ids(add_eos=True):
            if len(ids) <= context_size:
                continue
            for i in range(context_size, len(ids)):
                self.contexts.append(ids[i - context_size:i])
                self.targets.append(ids[i])
                if max_examples is not None and len(self.targets) >= max_examples:
                    print(f"[NNLMDataset] capped at {max_examples} examples")
                    return

        print(f"[NNLMDataset] examples={len(self.targets)}, context_size={context_size}")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return (
            torch.tensor(self.contexts[index], dtype=torch.long),
            torch.tensor(self.targets[index], dtype=torch.long),
        )


class NNLMIterable(IterableDataset):
    """Streaming NNLM windows — preferred for large corpora (wiki / enwik9)."""

    def __init__(
        self,
        corpus: LMCorpus,
        context_size: int = 5,
        max_examples: Optional[int] = None,
    ):
        self.corpus = corpus
        self.context_size = context_size
        self.max_examples = max_examples

    def __iter__(self):
        produced = 0
        for ids in self.corpus.iter_sentence_ids(add_eos=True):
            if len(ids) <= self.context_size:
                continue
            for i in range(self.context_size, len(ids)):
                context = torch.tensor(ids[i - self.context_size:i], dtype=torch.long)
                target = torch.tensor(ids[i], dtype=torch.long)
                yield context, target
                produced += 1
                if self.max_examples is not None and produced >= self.max_examples:
                    return


class RNNLMStream(IterableDataset):
    """Stream fixed-length sequences for RNNLM BPTT training.

    Yields (input_ids, target_ids) pairs of shape [seq_len], where
    target_ids[t] = input_ids[t+1] shifted by one (classic LM objective).
    """

    def __init__(
        self,
        corpus: LMCorpus,
        seq_len: int = 35,
        batch_size: int = 20,
        max_batches: Optional[int] = None,
    ):
        self.corpus = corpus
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_batches = max_batches

    def _flatten_ids(self) -> np.ndarray:
        chunks: List[int] = []
        for ids in self.corpus.iter_sentence_ids(add_eos=True):
            chunks.extend(ids)
        if not chunks:
            raise RuntimeError("Corpus produced zero tokens after encoding.")
        return np.asarray(chunks, dtype=np.int64)

    def __iter__(self):
        # Materialize once per worker/epoch. For multi-worker DataLoader this is
        # fine on text8-scale data; for multi-GB dumps prefer single worker.
        ids = self._flatten_ids()
        # Drop remainder so we can reshape into [batch, n_steps]
        n_batch = self.batch_size
        usable = (len(ids) // n_batch) * n_batch
        ids = ids[:usable]
        if usable < n_batch * (self.seq_len + 1):
            raise RuntimeError(
                f"Not enough tokens ({len(ids)}) for batch_size={n_batch}, seq_len={self.seq_len}"
            )

        data = ids.reshape(n_batch, -1)  # [B, T]
        total_steps = data.shape[1] - 1
        n_batches = total_steps // self.seq_len
        if self.max_batches is not None:
            n_batches = min(n_batches, self.max_batches)

        for batch_idx in range(n_batches):
            start = batch_idx * self.seq_len
            end = start + self.seq_len
            # input: data[:, start:end], target: data[:, start+1:end+1]
            x = torch.from_numpy(data[:, start:end].copy())
            y = torch.from_numpy(data[:, start + 1:end + 1].copy())
            yield x, y


def collate_nnlm(batch):
    contexts = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    return contexts, targets
