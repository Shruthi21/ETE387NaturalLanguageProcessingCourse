"""
Lab 2: GPT from Scratch
Complete implementation of all tasks.

Requirements:
    pip install torch numpy tiktoken

To run:
    python lab2_gpt.py

Note: For Tasks 2.10–2.12 you also need the file:
    gpt-2-pretrained.npz   (provided by your course)
"""

from dataclasses import dataclass
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════
# TASK 2.01 — Model Configuration
# ══════════════════════════════════════════════════════════
@dataclass
class Config:
    n_vocab: int = 50_257   # vocabulary size (BPE: 50256 tokens + 1 end-of-text)
    n_ctx:   int = 1_024    # context window — max tokens the model can attend to
    n_embd:  int = 768      # embedding dimension — size of each token vector
    n_head:  int = 12       # number of attention heads (768 / 12 = 64 dims/head)
    n_layer: int = 12       # number of stacked transformer blocks


# ══════════════════════════════════════════════════════════
# TASK 2.02 — GELU Activation Function
# ══════════════════════════════════════════════════════════
def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Approximated GELU activation (Page 1977 approximation, used in GPT-2).

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Key properties vs ReLU:
      - ReLU(x) = max(0, x)  — hard cutoff at 0, gradient = 0 for x < 0
      - GELU is smooth — small negative values are not fully suppressed
      - Minimum value ≈ -0.1702 at x ≈ -0.7511
      - Gradient exists everywhere (no dead neurons problem)
    """
    return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x**3)))


# ══════════════════════════════════════════════════════════
# TASK 2.03 — Feed-Forward Network (MLP) with shape annotations
# ══════════════════════════════════════════════════════════
class MLP(nn.Module):
    """
    Two-layer feed-forward network with GELU activation.
    Expands embedding 4x then compresses back — gives model capacity
    to learn complex non-linear transformations.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape
        # x:         [batch_size, seq_len, n_embd]       e.g. [4, 10, 768]

        x = self.c_fc(x)
        # c_fc = Linear(n_embd, n_embd*4)
        # x:         [batch_size, seq_len, n_embd*4]     e.g. [4, 10, 3072]

        x = gelu(x)
        # element-wise non-linearity — shape unchanged
        # x:         [batch_size, seq_len, n_embd*4]     e.g. [4, 10, 3072]

        x = self.c_proj(x)
        # c_proj = Linear(n_embd*4, n_embd)
        # x:         [batch_size, seq_len, n_embd]       e.g. [4, 10, 768]

        return x


# ══════════════════════════════════════════════════════════
# TASK 2.04 — Causal Mask
# ══════════════════════════════════════════════════════════
def make_causal_mask(n: int) -> torch.Tensor:
    """
    Create an upper-triangular matrix of -inf values.
    Used to prevent attending to future tokens.

    For n=4:
        [[  0, -inf, -inf, -inf],
         [  0,    0, -inf, -inf],
         [  0,    0,    0, -inf],
         [  0,    0,    0,    0]]

    When added to attention scores and softmax applied:
        softmax(-inf) = 0  → future tokens get zero attention weight
    """
    return torch.triu(torch.full((n, n), float("-inf")), diagonal=1)


# ══════════════════════════════════════════════════════════
# TASK 2.05 — Multi-Head Attention with shape annotations
# ══════════════════════════════════════════════════════════
class Attention(nn.Module):
    """
    Multi-head causal self-attention.

    Splits embeddings into n_head parallel attention heads,
    computes scaled dot-product attention in each head,
    then concatenates and projects back.
    """
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "n_embd must be divisible by n_head"
        self.n_head = config.n_head

        # Projects input to Q, K, V all at once (3x)
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Register causal mask as a buffer (not a parameter — not trained)
        # Moves to GPU automatically with model.to(device)
        self.register_buffer(
            "mask", make_causal_mask(config.n_ctx), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape
        # x:    [batch_size, seq_len, n_embd]

        head_embd = n_embd // self.n_head
        # head_embd = 768 // 12 = 64

        # ── Project to Q, K, V ──────────────────────────
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        # c_attn(x):  [batch_size, seq_len, n_embd*3]
        # after chunk: each of q, k, v is [batch_size, seq_len, n_embd]

        # ── Reshape into heads ───────────────────────────
        q = q.view(batch_size, seq_len, self.n_head, head_embd)
        k = k.view(batch_size, seq_len, self.n_head, head_embd)
        v = v.view(batch_size, seq_len, self.n_head, head_embd)
        # each: [batch_size, seq_len, n_head, head_embd]

        # ── Transpose: put n_head before seq_len ─────────
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        # each: [batch_size, n_head, seq_len, head_embd]

        # ── Scaled dot-product attention ──────────────────
        x = q @ k.transpose(-1, -2)
        # k.T:  [batch_size, n_head, head_embd, seq_len]
        # x:    [batch_size, n_head, seq_len,   seq_len]   ← attention scores

        x = x / head_embd ** 0.5
        # Scale by sqrt(head_embd) to stabilise gradients
        # x:    [batch_size, n_head, seq_len, seq_len]

        x = x + self.mask[:seq_len, :seq_len]
        # mask: [seq_len, seq_len] → BROADCAST to [batch_size, n_head, seq_len, seq_len]
        # Future positions become -inf

        x = torch.softmax(x, dim=-1)
        # x:    [batch_size, n_head, seq_len, seq_len]   (rows sum to 1)

        x = x @ v
        # v:    [batch_size, n_head, seq_len, head_embd]
        # x:    [batch_size, n_head, seq_len, head_embd]

        # ── Merge heads back ──────────────────────────────
        x = x.transpose(-2, -3).contiguous()
        # x:    [batch_size, seq_len, n_head, head_embd]

        x = x.view(batch_size, seq_len, n_embd)
        # x:    [batch_size, seq_len, n_embd]

        # ── Output projection ─────────────────────────────
        x = self.c_proj(x)
        # x:    [batch_size, seq_len, n_embd]

        return x


# ══════════════════════════════════════════════════════════
# TASK 2.06 — Layer Normalisation
# ══════════════════════════════════════════════════════════
class LayerNorm(nn.Module):
    """
    Layer normalisation — normalises across the embedding dimension.

    keepdim=True: keeps shape [batch, seq, 1] so subtraction broadcasts correctly.
    1e-05: prevents division by zero when variance is 0.
    g, b: learnable scale and shift parameters (gamma and beta).
    """
    def __init__(self, config: Config):
        super().__init__()
        self.g = nn.Parameter(torch.ones(config.n_embd))   # scale (gamma)
        self.b = nn.Parameter(torch.zeros(config.n_embd))  # shift (beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean     = x.mean(dim=-1, keepdim=True)
        # mean:     [batch_size, seq_len, 1]   ← keepdim keeps last dim as 1

        variance = x.var(unbiased=False, dim=-1, keepdim=True)
        # variance: [batch_size, seq_len, 1]

        # Normalise, then scale and shift
        return self.g * (x - mean) / torch.sqrt(variance + 1e-05) + self.b
        # output:   [batch_size, seq_len, n_embd]


# ══════════════════════════════════════════════════════════
# TASK 2.07 — Transformer Block (Pre-norm / Decoder block)
# ══════════════════════════════════════════════════════════
class Block(nn.Module):
    """
    GPT-2 Transformer decoder block — pre-norm architecture.

    Pre-norm:  x = x + sublayer(LayerNorm(x))
    Post-norm: x = LayerNorm(x + sublayer(x))   ← original Transformer

    Pre-norm benefits (Xiong et al. 2020):
      - More stable gradients during training
      - Does not require learning rate warmup
      - Easier to train deeper networks
    """
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = LayerNorm(config)    # norm before attention
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config)    # norm before MLP
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection around attention (with pre-norm)
        x = x + self.attn(self.ln_1(x))
        # Residual connection around MLP (with pre-norm)
        x = x + self.mlp(self.ln_2(x))
        return x


# ══════════════════════════════════════════════════════════
# TASK 2.08 — Full GPT-2 Model
# ══════════════════════════════════════════════════════════
def make_positions(n: int) -> torch.Tensor:
    """Returns a tensor [0, 1, 2, ..., n-1] of position indices."""
    return torch.arange(n, dtype=torch.long)


class Model(nn.Module):
    """
    Complete GPT-2 model.

    Components:
      wte   — token embedding table       [n_vocab, n_embd]
      wpe   — position embedding table    [n_ctx,   n_embd]
      h     — stack of 12 transformer blocks
      ln_f  — final layer norm
      lm_head — linear projection to vocab logits

    Buffers (register_buffer):
      - Not trained (no gradients)
      - Automatically move to GPU with model.to(device)
      - Saved with model state (if persistent=True)
      Benefits over plain tensors: device management is automatic.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.wte     = nn.Embedding(config.n_vocab, config.n_embd)  # token embeddings
        self.wpe     = nn.Embedding(config.n_ctx,   config.n_embd)  # position embeddings
        self.h       = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.ln_f    = LayerNorm(config)
        self.lm_head = nn.Linear(config.n_embd, config.n_vocab, bias=False)

        # Buffer: position indices [0, 1, ..., n_ctx-1]
        self.register_buffer("pos", make_positions(config.n_ctx), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        # x:      [batch_size, seq_len]          ← token IDs

        wte = self.wte(x)
        # wte:    [batch_size, seq_len, n_embd]  ← token embeddings

        wpe = self.wpe(self.pos[:seq_len])
        # pos:    [seq_len]
        # wpe:    [seq_len, n_embd]              ← position embeddings

        x = wte + wpe
        # broadcast: [batch_size, seq_len, n_embd]

        x = self.h(x)
        # 12 blocks, shape unchanged
        # x:      [batch_size, seq_len, n_embd]

        x = self.ln_f(x)
        # shape unchanged

        x = self.lm_head(x)
        # x:      [batch_size, seq_len, n_vocab] ← logits for every next token

        return x


# ══════════════════════════════════════════════════════════
# TASK 2.09 — Count Parameters + Weight Sharing
# ══════════════════════════════════════════════════════════
def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def apply_weight_sharing(model: Model) -> Model:
    """
    Share weights between token embedding and output linear layer.
    Both have shape [n_vocab, n_embd] — tying them saves ~38.6M parameters.
    Vaswani et al. (2017) introduced this technique.
    """
    model.lm_head.weight = model.wte.weight   # one line!
    return model


# ══════════════════════════════════════════════════════════
# TASK 2.10 — Load Pre-trained Weights
# ══════════════════════════════════════════════════════════
def copy_weights(source, target: torch.Tensor) -> None:
    """Copy a NumPy array into a PyTorch tensor in-place."""
    assert source.shape == target.shape, \
        f"Shape mismatch: {source.shape} vs {target.shape}"
    with torch.no_grad():
        target.copy_(torch.tensor(source, dtype=torch.float32))


def from_pretrained(npz_path: str = "gpt-2-pretrained.npz") -> Model:
    """
    Load GPT-2 pre-trained weights from the .npz archive.

    Actual key mapping (verified from npz file):
      npz key            shape           → PyTorch tensor
      ─────────────────────────────────────────────────────
      wte                (50257, 768)    → model.wte.weight
      wpe                (1024,  768)    → model.wpe.weight
      h{i}.ln_1.g        (768,)          → h[i].ln_1.g
      h{i}.ln_1.b        (768,)          → h[i].ln_1.b
      h{i}.ln_2.g        (768,)          → h[i].ln_2.g
      h{i}.ln_2.b        (768,)          → h[i].ln_2.b
      h{i}.attn.c_attn.w (768, 2304)    → h[i].attn.c_attn.weight  (TRANSPOSED)
      h{i}.attn.c_attn.b (2304,)        → h[i].attn.c_attn.bias
      h{i}.attn.c_proj.w (768, 768)     → h[i].attn.c_proj.weight  (TRANSPOSED)
      h{i}.attn.c_proj.b (768,)         → h[i].attn.c_proj.bias
      h{i}.mlp.c_fc.w    (768, 3072)    → h[i].mlp.c_fc.weight     (TRANSPOSED)
      h{i}.mlp.c_fc.b    (3072,)        → h[i].mlp.c_fc.bias
      h{i}.mlp.c_proj.w  (3072, 768)    → h[i].mlp.c_proj.weight   (TRANSPOSED)
      h{i}.mlp.c_proj.b  (768,)         → h[i].mlp.c_proj.bias
      ln_f.g             (768,)          → model.ln_f.g
      ln_f.b             (768,)          → model.ln_f.b

    Why transpose? PyTorch nn.Linear stores weights as [out, in],
    but the npz file stores them as [in, out]. So we call .T before copying.
    """
    import numpy as np

    model = Model(Config())
    pretrained = np.load(npz_path)

    # ── Token & position embeddings ──────────────────────
    # npz keys are just "wte" and "wpe" (no .weight suffix)
    copy_weights(pretrained["wte"], model.wte.weight)
    copy_weights(pretrained["wpe"], model.wpe.weight)

    # ── Transformer blocks ────────────────────────────────
    for i in range(model.config.n_layer):
        b = model.h[i]

        # Layer norms (g = gamma/scale, b = beta/shift)
        copy_weights(pretrained[f"h{i}.ln_1.g"], b.ln_1.g)
        copy_weights(pretrained[f"h{i}.ln_1.b"], b.ln_1.b)
        copy_weights(pretrained[f"h{i}.ln_2.g"], b.ln_2.g)
        copy_weights(pretrained[f"h{i}.ln_2.b"], b.ln_2.b)

        # Attention weights — .T because npz is [in, out], PyTorch wants [out, in]
        copy_weights(pretrained[f"h{i}.attn.c_attn.w"].T, b.attn.c_attn.weight)
        copy_weights(pretrained[f"h{i}.attn.c_attn.b"],   b.attn.c_attn.bias)
        copy_weights(pretrained[f"h{i}.attn.c_proj.w"].T, b.attn.c_proj.weight)
        copy_weights(pretrained[f"h{i}.attn.c_proj.b"],   b.attn.c_proj.bias)

        # MLP weights — same transposition needed
        copy_weights(pretrained[f"h{i}.mlp.c_fc.w"].T,   b.mlp.c_fc.weight)
        copy_weights(pretrained[f"h{i}.mlp.c_fc.b"],      b.mlp.c_fc.bias)
        copy_weights(pretrained[f"h{i}.mlp.c_proj.w"].T,  b.mlp.c_proj.weight)
        copy_weights(pretrained[f"h{i}.mlp.c_proj.b"],    b.mlp.c_proj.bias)

    # ── Final layer norm ──────────────────────────────────
    copy_weights(pretrained["ln_f.g"], model.ln_f.g)
    copy_weights(pretrained["ln_f.b"], model.ln_f.b)

    model.eval()
    print(f"✅ Loaded pre-trained weights from '{npz_path}'")
    return model


# ══════════════════════════════════════════════════════════
# TASK 2.11 — Text Generation (greedy + sampling + temperature + top-k)
# ══════════════════════════════════════════════════════════
def generate(
    model:        nn.Module,
    context:      torch.Tensor,
    context_size: int   = 1024,
    n_tokens:     int   = 20,
    temperature:  float = 1.0,
    top_k:        int   = None,
    greedy:       bool  = False,
) -> torch.Tensor:
    """
    Generate tokens autoregressively.

    Args:
        model:        GPT-2 model
        context:      Starting token IDs, shape [1, seq_len]
        context_size: Max context window (truncate if longer)
        n_tokens:     How many new tokens to generate
        temperature:  > 1 = more random, < 1 = more focused, 1 = default
        top_k:        If set, only sample from the top-k most likely tokens
        greedy:       If True, always pick the most likely token (ignores temp/top_k)

    Returns:
        context tensor with n_tokens appended, shape [1, seq_len + n_tokens]
    """
    for _ in range(n_tokens):
        # Truncate context to fit within context window
        context = context[:, -context_size:]

        with torch.no_grad():
            logits = model(context)[:, -1, :]
            # logits: [batch_size, n_vocab] — scores for next token

        if greedy:
            # Always pick the highest-scoring token
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # ── Temperature scaling ──────────────────────
            # Divide logits before softmax:
            #   temp < 1 → sharpen distribution (more confident)
            #   temp > 1 → flatten distribution (more random)
            logits = logits / temperature

            # ── Top-k filtering ───────────────────────────
            # Zero out all tokens except the top-k most likely
            if top_k is not None:
                top_values, _ = torch.topk(logits, top_k)
                threshold = top_values[:, -1, None]       # k-th highest value
                logits[logits < threshold] = float("-inf")

            # ── Sample from the distribution ─────────────
            probs    = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            # next_idx: [batch_size, 1]

        context = torch.cat([context, next_idx], dim=-1)

    return context


def generate_helper(text, model, tokenizer, context_size=1024, n_tokens=20,
                    temperature=1.0, top_k=None, greedy=False):
    """Convenience wrapper — handles encoding and decoding."""
    context = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
    context = generate(
        model, context,
        context_size=context_size,
        n_tokens=n_tokens,
        temperature=temperature,
        top_k=top_k,
        greedy=greedy,
    )
    return tokenizer.decode(context[0].tolist())


# ══════════════════════════════════════════════════════════
# TASK 2.12 — HellaSwag Evaluation
# ══════════════════════════════════════════════════════════
def evaluate_hellaswag(model: nn.Module, tokenizer, jsonl_path: str) -> float:
    """
    Evaluate model on HellaSwag benchmark.

    For each sample:
      - Encode the context (prefix)
      - For each of 4 possible endings, compute cross-entropy loss
        (how surprised the model is by that ending given the context)
      - Pick the ending with the LOWEST loss (= highest probability)
      - Compare to gold label

    Cross-entropy slicing explained:
        logits[0, -len(suffix)-1 : -1]  = model's predictions at each suffix position
        context[0, -len(suffix):]        = the actual suffix tokens

        Position i in the context predicts position i+1.
        So to score the suffix tokens, we use the logits
        from the positions just BEFORE each suffix token.

    Scores:
        Random baseline  : 25%   (4 choices)
        GPT-2 small      : ~29%
        Human performance: ~95%
    """
    model.eval()
    n_correct = 0
    n_total   = 0

    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            prefix = tokenizer.encode(sample["ctx"])
            ending_scores = []

            for i, ending in enumerate(sample["endings"]):
                # Add a space before each ending (matches natural text)
                suffix  = tokenizer.encode(" " + ending)
                context = torch.tensor([prefix + suffix], dtype=torch.long)

                with torch.no_grad():
                    logits = model(context)
                    # Score = cross-entropy of the suffix tokens
                    # Lower = model finds this ending more likely
                    ending_score = F.cross_entropy(
                        logits[0, -len(suffix) - 1 : -1],  # predictions before suffix
                        context[0, -len(suffix) :]          # actual suffix tokens
                    )
                ending_scores.append((ending_score, i))

            # Pick the ending with the lowest cross-entropy (most probable)
            predicted = min(ending_scores)[1]
            n_correct += int(predicted == sample["label"])
            n_total   += 1

    accuracy = n_correct / n_total
    print(f"HellaSwag accuracy: {accuracy:.2%}  ({n_correct}/{n_total})")
    print(f"  Random baseline:   25.00%")
    print(f"  Human performance: ~95.00%")
    return accuracy


# ══════════════════════════════════════════════════════════
# DEMO — run without pretrained weights
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 60)
    print("TASK 2.01 — Model Configuration")
    print("=" * 60)
    config = Config()
    print(f"  n_vocab : {config.n_vocab:,}  (BPE vocab + end-of-text token)")
    print(f"  n_ctx   : {config.n_ctx:,}   (max context window)")
    print(f"  n_embd  : {config.n_embd}   (embedding dimensions)")
    print(f"  n_head  : {config.n_head}      (attention heads)")
    print(f"  n_layer : {config.n_layer}      (transformer blocks)")

    print("\n" + "=" * 60)
    print("TASK 2.02 — GELU")
    print("=" * 60)
    x = torch.linspace(-2, 2, 100)
    y = gelu(x)
    min_val = y.min().item()
    min_x   = x[y.argmin()].item()
    print(f"  Minimum GELU value : {min_val:.4f}")
    print(f"  At input x         : {min_x:.4f}")
    print(f"  ReLU min           : 0.0  (hard cutoff)")
    print(f"  GELU min           : {min_val:.4f}  (allows small negatives)")

    print("\n" + "=" * 60)
    print("TASK 2.04 — Causal Mask")
    print("=" * 60)
    mask = make_causal_mask(4)
    print("  Causal mask (n=4):")
    print(mask)
    x_demo = torch.rand(1, 2, 3, 3)
    result = x_demo + mask[:3, :3]
    print(f"\n  x shape      : {list(x_demo.shape)}")
    print(f"  mask[:3,:3]  : {list(mask[:3,:3].shape)}")
    print(f"  result shape : {list(result.shape)}")
    print("  Broadcasting: [3,3] expands to [1,2,3,3] automatically")

    print("\n" + "=" * 60)
    print("TASK 2.09 — Parameter Count")
    print("=" * 60)
    model = Model(Config())
    n_before = count_parameters(model)
    print(f"  Parameters (before weight sharing) : {n_before:,}")
    print(f"  Radford et al. reported            : ~117,000,000  (wrong!)")
    print(f"  They missed the LayerNorm parameters")

    model = apply_weight_sharing(model)
    n_after = count_parameters(model)
    saved   = n_before - n_after
    print(f"\n  Parameters (after weight sharing)  : {n_after:,}")
    print(f"  Parameters saved by weight sharing : {saved:,}")
    print(f"  Reduction                          : {saved/n_before*100:.1f}%")

    print("\n" + "=" * 60)
    print("TASK 2.05 — Forward pass shape check")
    print("=" * 60)
    model.eval()
    dummy_input = torch.randint(0, config.n_vocab, (2, 10))
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Input  shape : {list(dummy_input.shape)}   (batch=2, seq_len=10)")
    print(f"  Output shape : {list(output.shape)}  (batch=2, seq_len=10, vocab=50257)")
    print("  ✅ Forward pass successful")

    print("\n" + "=" * 60)
    print("To use pre-trained model, run:")
    print("=" * 60)
    print("""
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    model = from_pretrained("gpt-2-pretrained.npz")

    # Greedy generation
    print(generate_helper("Linköping University is", model, tokenizer, greedy=True))

    # Sampling with temperature and top-k
    print(generate_helper("Linköping University is", model, tokenizer,
                           temperature=0.8, top_k=40))

    # HellaSwag evaluation
    evaluate_hellaswag(model, tokenizer, "hellaswag-mini.jsonl")
    """)
