
"""
Lab 1 — Part 2: Embeddings
Full implementation with detailed comments for all tasks 1.07–1.12.
Run this in a Jupyter environment where PyTorch is available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter


# ══════════════════════════════════════════════════════════
# TASK 1.07 — Bag-of-Words Classifier
# ══════════════════════════════════════════════════════════
"""
How the bag-of-words classifier works:

  Input: a batch of reviews, each represented as a list of token IDs
         Shape: (batch_size, seq_len)

  Step 1 — Embedding lookup:
    self.embedding(x) looks up a vector for every token ID.
    Output shape: (batch_size, seq_len, embedding_dim)

  Step 2 — Mean pooling (.mean(dim=-2)):
    Average all token vectors in the review into one vector.
    dim=-2 means "average along the sequence (token) dimension".
    Output shape: (batch_size, embedding_dim)
    This is the "bag" — word ORDER is lost, only presence/frequency matters.

  Step 3 — Linear classification:
    self.linear maps the embedding vector to class scores.
    Output shape: (batch_size, num_classes)

Why only ONE nn.Embedding when the diagram shows three?
  The diagram shows separate embedding layers for each word POSITION,
  but in a bag-of-words model ALL tokens share the same embedding table.
  One nn.Embedding holds all token vectors; each token ID indexes into it.

What does dim=-2 do?
  Tensors are indexed from the last dimension backwards with negative indices.
  Shape is (batch, seq_len, embed_dim) = dims 0, 1, 2 = dims 0, -2, -1.
  dim=-2 averages over the seq_len dimension → collapses sequences to one vector.
"""

class Classifier(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, num_classes: int):
        super().__init__()
        # Embedding table: num_embeddings rows, each embedding_dim wide
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Linear layer: maps embedding_dim → num_classes (one score per class)
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        # embedding(x): (batch_size, seq_len, embedding_dim)
        # .mean(dim=-2): (batch_size, embedding_dim)  ← bag-of-words step
        # linear(...): (batch_size, num_classes)
        return self.linear(self.embedding(x).mean(dim=-2))


# ══════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════

type Item = tuple[list[str], str]

class ReviewDataset(Dataset):
    def __init__(self, filename: str, label: int = 0) -> None:
        with open(filename) as f:
            tokenized_lines = [line.split() for line in f]
        # tokens[0] = category (camera/music)
        # tokens[1] = sentiment (neg/pos)
        # tokens[2:] = the review words
        self.items = [(tokens[2:], tokens[label]) for tokens in tokenized_lines]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Item:
        return self.items[idx]


# ══════════════════════════════════════════════════════════
# TASK 1.08 — Vectoriser
# ══════════════════════════════════════════════════════════
"""
Step 1 — How does unzipping work?
  dataset contains (review_tokens, label) pairs.
  zip(*dataset) transposes: groups all reviews together, all labels together.
  reviews = tuple of lists of strings   e.g. (['oh','man',...], ['great',...])
  labels  = tuple of strings            e.g. ('neg', 'pos', 'neg', ...)

Step 2 — Token-to-ID and label-to-ID mappings:
  Counter counts every token across all reviews.
  .most_common(n) returns (token, count) pairs sorted by frequency.
  If two tokens have the same count, Python's Counter breaks ties by
  insertion order (arbitrary, not guaranteed alphabetically).

  t2i reserves IDs 0 and 1 for [PAD] and [UNK],
  then assigns IDs 2..n to the n_vocab-2 most frequent tokens.

  l2i sorts the unique labels alphabetically:
  e.g. {'camera': 0, 'music': 1} or {'neg': 0, 'pos': 1}

Step 3 — __call__() implementation: see code below.
"""

class ReviewVectorizer:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, dataset: ReviewDataset, n_vocab: int = 1024) -> None:
        # Unzip (review, label) pairs into two separate tuples
        reviews, labels = zip(*dataset)
        # type(reviews) = tuple[list[str], ...]
        # type(labels)  = tuple[str, ...]

        # Count tokens and keep the most frequent (n_vocab - 2) of them
        counter = Counter(t for r in reviews for t in r)
        most_common = [t for t, _ in counter.most_common(n_vocab - 2)]

        # t2i: special tokens first, then most-common vocabulary
        self.t2i = {t: i for i, t in enumerate([self.PAD, self.UNK] + most_common)}
        # l2i: sorted unique labels → integers
        self.l2i = {l: i for i, l in enumerate(sorted(set(labels)))}

    def __call__(self, items: list[Item]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a batch of (review_tokens, label) pairs to tensors.

        Returns:
            X: LongTensor of shape (m, max_len) — padded token ID matrix
            y: LongTensor of shape (m,)          — label IDs
        """
        reviews, labels = zip(*items)

        # Convert each review to a list of token IDs
        # Unknown tokens → UNK id; known tokens → their id
        unk_id = self.t2i[self.UNK]
        pad_id = self.t2i[self.PAD]

        id_seqs = [
            [self.t2i.get(tok, unk_id) for tok in review]
            for review in reviews
        ]

        # Pad all sequences to the length of the longest review in this batch
        max_len = max(len(seq) for seq in id_seqs)
        padded = [
            seq + [pad_id] * (max_len - len(seq))
            for seq in id_seqs
        ]

        # Convert labels to IDs
        label_ids = [self.l2i[label] for label in labels]

        X = torch.tensor(padded,    dtype=torch.long)
        y = torch.tensor(label_ids, dtype=torch.long)
        return X, y


# ══════════════════════════════════════════════════════════
# TASK 1.09 — Training loop (with full comments + kwargs)
# ══════════════════════════════════════════════════════════

def train(
    train_file: str  = "reviews-train.txt",   # path to training data
    label:      int  = 0,                     # 0=category, 1=sentiment
    n_vocab:    int  = 1024,                  # vocabulary size
    embed_dim:  int  = 64,                    # embedding vector size
    lr:         float = 0.001,               # Adam learning rate
    batch_size: int  = 16,                   # reviews per gradient step
    n_epochs:   int  = 10,                   # passes over training data
):
    """
    Full training loop with line-by-line comments.

    Returns the trained (vectorizer, model) pair.
    """

    # ── Data ──────────────────────────────────────────────
    # Load the reviews from file; label selects category or sentiment
    dataset = ReviewDataset(train_file, label=label)

    # Build token→id and label→id mappings from training data
    vectorizer = ReviewVectorizer(dataset, n_vocab)

    # ── Model ─────────────────────────────────────────────
    # Create the bag-of-words classifier
    # num_embeddings = vocab size, num_classes = number of unique labels
    model = Classifier(n_vocab, embed_dim, len(vectorizer.l2i))

    # ── Optimiser ─────────────────────────────────────────
    # Adam adapts learning rate per parameter — works well out of the box
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── DataLoader ────────────────────────────────────────
    # Batches reviews, shuffles between epochs, uses vectorizer as collate_fn
    # collate_fn=vectorizer means each batch is passed through __call__()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,           # shuffle ensures different batch compositions per epoch
        collate_fn=vectorizer,  # converts raw (tokens, label) → (X tensor, y tensor)
    )

    # ── Training epochs ───────────────────────────────────
    for epoch in range(n_epochs):
        model.train()           # enable dropout/batchnorm training behaviour
        running_loss = 0.0

        for bx, by in data_loader:
            # bx: (batch_size, max_len)  token ID matrix
            # by: (batch_size,)          label IDs

            optimizer.zero_grad()           # clear gradients from last step

            output = model(bx)              # forward pass → (batch_size, num_classes)

            loss = F.cross_entropy(output, by)   # compare predictions to true labels

            loss.backward()                 # backprop: compute gradients

            optimizer.step()                # update weights using gradients

            running_loss += loss.item()     # accumulate scalar loss

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch:>2}, loss: {avg_loss:.4f}")

    return vectorizer, model


# ══════════════════════════════════════════════════════════
# TASK 1.10 — Training both classifiers
# ══════════════════════════════════════════════════════════
"""
Run this in your Jupyter notebook:

    torch.manual_seed(42)   # seed makes results reproducible across runs
    print("=== Category classifier ===")
    vec_cat, model_cat = train(train_file="reviews-train.txt", label=0)

    torch.manual_seed(42)
    print("\\n=== Sentiment classifier ===")
    vec_sent, model_sent = train(train_file="reviews-train.txt", label=1)

Seed purpose: torch.manual_seed(42) fixes the random number generator,
so weight initialisation and data shuffling are the same every run →
results are reproducible and comparable.

Which task is harder?
  Sentiment (pos/neg) usually gives HIGHER loss than category (camera/music).
  Sentiment is subtler — words like "good" can be positive or negative
  depending on context ("not good"). Category classification is easier
  because domain vocabulary (shutter, megapixel vs. album, track) is
  very distinct.
"""


# ══════════════════════════════════════════════════════════
# Save embeddings for Embedding Projector
# ══════════════════════════════════════════════════════════

def save_embeddings(
    vectorizer: ReviewVectorizer,
    model: Classifier,
    vectors_filename: str,
    metadata_filename: str,
) -> None:
    i2t = {i: t for t, i in vectorizer.t2i.items()}
    embeddings = model.embedding.weight.detach().numpy()
    items = [(i2t[i], e) for i, e in enumerate(embeddings)]
    with open(vectors_filename, "wt") as f1, open(metadata_filename, "wt") as f2:
        for w, e in items:
            print("\t".join("{:.5f}".format(x) for x in e), file=f1)
            print(w, file=f2)


# ══════════════════════════════════════════════════════════
# TASK 1.11 — Inspecting the embeddings
# ══════════════════════════════════════════════════════════
"""
How to use the Embedding Projector:
  1. Go to https://projector.tensorflow.org/
  2. Click "Load data" (top left)
  3. Upload vectors.tsv  as "Tensor"
  4. Upload metadata.tsv as "Metadata"

What to expect:

  Category classifier (camera vs. music):
    Two clearly separated clusters — camera words (lens, zoom, battery,
    megapixel) cluster together, music words (album, track, song, bass)
    cluster together. The separation should be very clean.

  Sentiment classifier (pos vs. neg):
    Clusters based on emotional valence — positive words (great, love,
    excellent, perfect) in one region; negative words (waste, terrible,
    broken, disappointed) in another. More overlap than category.

  repair vs. sturdy:
    In the CATEGORY model: both appear near "camera" vocabulary (repairs
    and sturdiness are common camera concerns) → CLOSE together.
    In the SENTIMENT model: "repair" suggests something broke → negative;
    "sturdy" suggests quality → positive → FARTHER apart.
    This shows the embeddings encode task-relevant relationships.

Dimensionality reduction:
  PCA:   Fast, global structure, good for seeing overall shape of clusters.
  T-SNE: Better at revealing local clusters, but can distort global distances.
  UMAP:  Best balance of local and global structure; often most interpretable.
"""


# ══════════════════════════════════════════════════════════
# TASK 1.12 — Kaiming initialisation for embedding layer
# ══════════════════════════════════════════════════════════
"""
PyTorch default for nn.Linear (from source):
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

This samples from a uniform distribution with bounds:
    ±sqrt(6 / (fan_in * (1 + a²)))

where fan_in = number of input features.

For an embedding layer, fan_in = embedding_dim.

To apply the same initialisation to the embedding layer:
"""

import math as _math

def make_classifier_kaiming(
    num_embeddings: int,
    embedding_dim:  int,
    num_classes:    int,
) -> Classifier:
    """
    Create a Classifier with Kaiming-uniform initialised embedding weights,
    matching PyTorch's default nn.Linear initialisation.
    """
    model = Classifier(num_embeddings, embedding_dim, num_classes)

    # Reproduce nn.Linear's reset_parameters():
    #   nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    nn.init.kaiming_uniform_(model.embedding.weight, a=_math.sqrt(5))

    return model


"""
Usage in Jupyter:

    torch.manual_seed(42)
    print("=== Default (normal) init ===")
    vec, model_default = train(label=0)

    torch.manual_seed(42)
    print("\\n=== Kaiming init ===")
    dataset = ReviewDataset("reviews-train.txt", label=0)
    vec2 = ReviewVectorizer(dataset, 1024)
    model_kaiming = make_classifier_kaiming(1024, 64, len(vec2.l2i))
    # ... plug into train() by replacing model = Classifier(...) with
    # model = make_classifier_kaiming(...)

Expected effect:
  Kaiming initialisation keeps gradient magnitudes stable across layers.
  You may see faster initial convergence (lower loss in early epochs)
  and sometimes a lower final loss. The embedding vector space may also
  appear more structured when visualised in the Embedding Projector.
"""

print("Lab 1 Part 2 loaded successfully.")
print("Run train() in your Jupyter environment to train the classifier.")
print("See task comments throughout this file for full explanations.")
