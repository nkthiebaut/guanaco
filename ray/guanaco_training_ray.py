import math

from datasets import load_dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from einops import rearrange, reduce
import lightning as L
from torch.cuda.amp import autocast
import ray.train.lightning
from ray.train.torch import TorchTrainer


class SelfAttention(nn.Module):
    def __init__(self, emb_dim: int = 64, head_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.head_dim = head_dim
        self.Wq = nn.Linear(emb_dim, head_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, head_dim, bias=False)
        self.Wv = nn.Linear(emb_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, complex_rotations, mask):
        # Compute Queries, Keys, and Values from embeddings
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Apply RoPE to queries and keys
        Q = apply_rope(Q, complex_rotations)
        K = apply_rope(K, complex_rotations)

        attention = (Q @ K.mT / math.sqrt(self.head_dim)).to(x.device)

        scores = F.softmax(attention + mask, dim=-1)
        scores = self.dropout(scores)
        return scores @ V


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        # Note: explicit casting to fp32 to avoid numerical underflow
        x_fp32 = x.to(torch.float32)
        mean_square = reduce(x_fp32**2, "... d -> ... 1", "mean")
        inverse_rms = torch.rsqrt(mean_square + self.eps)
        inverse_rms = inverse_rms.type_as(x)  # For fp16 compatibility
        return self.weight * x * inverse_rms


class FeedForward(nn.Module):
    def __init__(self, emb_dims: int, hidden_dims: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(emb_dims, hidden_dims, bias=False)
        self.fc2 = nn.Linear(emb_dims, hidden_dims, bias=False)
        self.fc3 = nn.Linear(hidden_dims, emb_dims, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        gate = F.silu(self.fc1(x))  # Silu(x) = x * sigmoid(x)
        x = self.fc2(x)
        x = x * gate
        x = self.fc3(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, emb_dim=64, n_heads=4, dropout: float = 0.1):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.head_dim = emb_dim // n_heads

        self.att_norm = RMSNorm(emb_dim)
        self.heads = nn.ModuleList(
            [SelfAttention(emb_dim, self.head_dim, dropout) for _ in range(n_heads)]
        )
        self.projection = nn.Linear(emb_dim, emb_dim)

        self.ffn_norm = RMSNorm(emb_dim)
        self.feed_forward = FeedForward(emb_dim, 4 * emb_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, complex_rotations, mask):
        x_sa_norm = self.att_norm(x)
        x_self_attention = torch.cat(
            [h(x_sa_norm, complex_rotations, mask) for h in self.heads], dim=-1
        )
        x = x + x_self_attention

        x_ffn_norm = self.ffn_norm(x)
        x_ffn = self.feed_forward(x_ffn_norm)
        x = x + x_ffn

        x = self.dropout(x)

        x = self.projection(x)
        return x


def compute_complex_rotations(T, C):
    c_values = torch.arange(1, C / 2 + 1)
    thetas = 10000 ** (2 * (c_values - 1) / C)  # Shape (C/2,)
    timesteps = torch.arange(T)  # Shape (T,)
    # Angular frequencies for each (t, c) pairs
    omegas = torch.outer(timesteps, thetas)  # Shape (T, C/2)
    # Turn those into complex numbers
    z = torch.polar(torch.ones_like(omegas), omegas)
    return z


@autocast(enabled=False)
def apply_rope(q, complex_rotations):
    q_pairs = rearrange(q.to(torch.float32), "B T (C p) -> B T C p", p=2)
    q_complex = torch.view_as_complex(q_pairs)
    q_rotated = q_complex * complex_rotations
    q_rotated = torch.view_as_real(q_rotated)  # Back to real numbers
    q_rotated = rearrange(q_rotated, "B T C p -> B T (C p)")
    return q_rotated.type_as(q_pairs)


class Guanaco(nn.Module):
    def __init__(
        self, vocab_size=3, emb_dim=64, n_heads=4, n_layers=2, max_len=128, dropout=0.1
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.emb_dim = emb_dim
        self.max_len = max_len

        self.blocks = nn.ModuleList(
            [
                MultiHeadSelfAttentionBlock(emb_dim, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )

        self.output_norm = RMSNorm(emb_dim)
        self.output_layer = nn.Linear(emb_dim, vocab_size)

        complex_rotations = torch.view_as_real(
            compute_complex_rotations(max_len, emb_dim // n_heads)
        )
        self.register_buffer("complex_rotations", complex_rotations)

        mask = torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, token_ids):
        lookback = self.max_len
        token_ids = token_ids[:, -lookback:].to(torch.long)
        B, T = token_ids.shape

        complex_rotations = torch.view_as_complex(self.complex_rotations[:T, :])
        mask = self.mask[:T, :T]

        x = self.embeddings(token_ids)  # (B,T,C)

        for block in self.blocks:
            x = block(x, complex_rotations, mask)  # (B,T,C)
        x = self.output_norm(x)  # (B,T,C)
        logits = self.output_layer(x)  # (B,T,V)
        return logits


class GuanacoModule(L.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        return self.model(inputs)

    def training_step(self, tokens: torch.tensor) -> float:
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits = self.forward(inputs)

        logits = rearrange(logits, "B T C -> (B T) C")
        targets = rearrange(targets, "B T -> (B T)")

        loss = F.cross_entropy(logits, targets)
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, tokens: torch.tensor) -> float:
        loss = self.training_step(tokens)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def train_func():
    def tokenize(text: str) -> list[int]:
        """Converts a string to a bytes object using UTF-8 encoding."""
        return [code_unit for code_unit in text.encode("utf-8")]

    def detokenize(token_ids: list[int]) -> str:
        """Converts a bytes object to a string using UTF-8 encoding."""
        return bytes(token_ids).decode("utf-8", errors="replace")

    def collate_fn(batch, context_length=128):
        token_ids = [torch.tensor(tokenize(b["text"])[:context_length]) for b in batch]
        token_ids = pad_sequence(token_ids, batch_first=True)
        return token_ids

    dataset = load_dataset("roneneldan/TinyStories")
    dataset.set_format(type="torch", columns=["text"])

    batch_size = 64

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4 if torch.cuda.is_available() else 0,
        persistent_workers=bool(torch.cuda.is_available()),
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"Data sample from dataloader: {next(iter(train_dataloader))[0]}")

    model = GuanacoModule(
        Guanaco(vocab_size=256, emb_dim=512, n_heads=1, n_layers=1, max_len=128),
        learning_rate=1e-3,
    )

    trainer = L.Trainer(
        max_epochs=2,
        devices="auto",
        accelerator="cpu",
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=[ray.train.lightning.RayTrainReportCallback()],
        overfit_batches=0.01,
        enable_checkpointing=False,
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_dataloader, val_dataloader)

    def generate(model, x: str, n_tokens: int = 5, device=model.device):
        """Predict next token with greedy decoding."""
        x = torch.tensor(tokenize(x)).unsqueeze(0)
        x = x.to(device)
        model = model.to(device)

        for _ in range(n_tokens):
            pred = model(x)[:, -1, :]  # Logits of the next token prediction (B, V)
            next_tokens = pred.argmax(dim=-1)  # Next token_id with highest proba (B)
            next_tokens = rearrange(next_tokens, "B -> B 1")
            x = torch.cat((x, next_tokens), dim=1)
        return "".join(detokenize(x[0].tolist()))

    generate(model, "Once upon a time", n_tokens=200)


scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=False)

trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=ray.train.RunConfig(
        storage_path="gs://model-repo-msds-631-02/guanaco-runs"
    ),
)
result: ray.train.Result = trainer.fit()
