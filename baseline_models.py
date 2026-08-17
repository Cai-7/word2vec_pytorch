"""Baseline neural language models used in the original word2vec papers.

NNLM  : Feed-forward Neural Network Language Model (Bengio et al., 2003)
RNNLM : Recurrent Neural Network Language Model (Mikolov et al.)

Both models are trained as next-word predictors. After training, the input
word embedding matrix is exported in word2vec text format so it can be
evaluated with the same analogy script as CBOW / Skip-gram.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NNLM(nn.Module):
    """Feed-forward Neural Network Language Model (Bengio-style).

    Given the previous ``context_size`` words, predict the next word:

        emb  = [e(w_{t-n}), ..., e(w_{t-1})]   # concat projection
        h    = tanh(W_h * emb + b_h)
        y    = W_o * h + b_o                   # logits over vocabulary

    The learned input embeddings ``e(·)`` are used as word vectors.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dimension: int = 300,
        context_size: int = 5,
        hidden_size: int = 300,
        dropout: float = 0.0,
        tie_weights: bool = False,
    ):
        super().__init__()
        if context_size < 1:
            raise ValueError("context_size must be >= 1")

        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.context_size = context_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(vocab_size, emb_dimension)
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(context_size * emb_dimension, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=True)

        if tie_weights:
            if hidden_size != emb_dimension:
                raise ValueError(
                    "tie_weights requires hidden_size == emb_dimension for NNLM"
                )
            # Direct projection path is not tied; only meaningful when dims match
            # for a reduced architecture. Keep option for experiments.
            pass

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5 / self.emb_dimension
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: LongTensor of shape [batch, context_size]

        Returns:
            logits: FloatTensor of shape [batch, vocab_size]
        """
        emb = self.embeddings(context)  # [B, C, D]
        emb = self.dropout(emb)
        flat = emb.reshape(emb.size(0), -1)  # [B, C*D]
        hidden = torch.tanh(self.hidden(flat))
        hidden = self.dropout(hidden)
        logits = self.output(hidden)
        return logits

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embeddings.weight.data


class RNNLM(nn.Module):
    """Recurrent Neural Network Language Model (Elman / Mikolov-style).

    At each time step:

        x_t = e(w_t)
        h_t = f(W_x x_t + W_h h_{t-1} + b)
        y_t = W_o h_t + c

    Supports plain RNN (paper default) and LSTM (more stable training).
    Word vectors are taken from the input embedding matrix.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dimension: int = 300,
        hidden_size: int = 300,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "rnn",
        tie_weights: bool = False,
    ):
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in {"rnn", "lstm", "gru"}:
            raise ValueError("rnn_type must be one of: rnn, lstm, gru")

        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        self.embeddings = nn.Embedding(vocab_size, emb_dimension)
        dropout_between = dropout if num_layers > 1 else 0.0
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                emb_dimension,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout_between,
                batch_first=True,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                emb_dimension,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout_between,
                batch_first=True,
            )
        else:
            self.rnn = nn.RNN(
                emb_dimension,
                hidden_size,
                num_layers=num_layers,
                nonlinearity="tanh",
                dropout=dropout_between,
                batch_first=True,
            )

        self.drop = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

        if tie_weights:
            if hidden_size != emb_dimension:
                raise ValueError(
                    "tie_weights requires hidden_size == emb_dimension for RNNLM"
                )
            self.output.weight = self.embeddings.weight

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5 / self.emb_dimension
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        if self.output.weight is not self.embeddings.weight:
            nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Create a zero initial hidden state."""
        weight = next(self.parameters())
        if self.rnn_type == "lstm":
            return (
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            )
        return weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def forward(self, input_ids: torch.Tensor, hidden=None):
        """
        Args:
            input_ids: LongTensor [batch, seq_len]
            hidden: optional recurrent state

        Returns:
            logits: FloatTensor [batch, seq_len, vocab_size]
            hidden: updated recurrent state
        """
        emb = self.drop(self.embeddings(input_ids))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        logits = self.output(output)
        return logits, hidden

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embeddings.weight.data


def save_embedding_matrix(
    embedding: torch.Tensor,
    id2word: dict,
    file_name: str,
) -> None:
    """Save embeddings in word2vec plain-text format."""
    if embedding.is_cuda:
        embedding = embedding.cpu()
    vectors = embedding.detach().numpy()
    with open(file_name, "w", encoding="utf-8") as fout:
        fout.write("%d %d\n" % (len(id2word), vectors.shape[1]))
        for wid, word in id2word.items():
            row = vectors[wid]
            fout.write("%s %s\n" % (word, " ".join(str(float(x)) for x in row)))
