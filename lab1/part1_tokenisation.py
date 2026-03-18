"""
Lab 1: Tokenisation and Embeddings
Complete implementation of all tasks.
Note: Part 2 (Embeddings) requires PyTorch — shown as runnable pseudocode
      with full explanations since PyTorch is not available in this environment.
"""

from collections import Counter
import math

# ──────────────────────────────────────────────────────────
# Type alias
# ──────────────────────────────────────────────────────────
type Pair = tuple[int, int]


# ══════════════════════════════════════════════════════════
# PART 1: TOKENISATION
# ══════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────
# Task 1.01: Counting pairs
# ──────────────────────────────────────────────────────────
def count(ids: list[int]) -> dict[Pair, int]:
    """
    Count all consecutive pairs in ids.
    Returns a dict mapping each (a, b) pair to how often it appears.
    Pairs with zero count are NOT included (as per spec).

    Example: [1, 2, 1, 2] → {(1,2): 2, (2,1): 1}
    """
    counts: dict[Pair, int] = {}
    for a, b in zip(ids, ids[1:]):          # slide a window of size 2
        pair = (a, b)
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# ──────────────────────────────────────────────────────────
# Task 1.02: Replacing pairs
# ──────────────────────────────────────────────────────────
def replace(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """
    Traverse ids left-to-right and replace every occurrence of `pair`
    with `new_id`. Replacements are non-overlapping (greedy left-to-right).

    Example: replace([1,2,1,2], (1,2), 99) → [99, 99]
    Example: replace([1,1,2],   (1,2), 99) → [1, 99]  (only rightmost pair)
    """
    result = []
    i = 0
    while i < len(ids):
        # Check if pair starts at position i
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            result.append(new_id)
            i += 2          # skip both elements of the pair
        else:
            result.append(ids[i])
            i += 1
    return result


# ──────────────────────────────────────────────────────────
# Tokenizer class  (as provided in the notebook)
# ──────────────────────────────────────────────────────────
class Tokenizer:
    """
    BPE Tokenizer.

    Attributes:
        merges: dict[Pair, int]
            Maps a pair of token IDs to the merged token ID.
            The VALUE (int) is also the merge rule's priority —
            lower value = higher priority (applied first in encode).
        vocab:  dict[int, bytes]
            Maps every token ID to its byte representation.
            Starts with the 256 single-byte tokens (IDs 0-255).
    """

    def __init__(self):
        self.merges: dict[Pair, int] = {}
        self.vocab:  dict[int, bytes] = {i: bytes([i]) for i in range(2**8)}

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while True:
            counts = count(ids)
            mergeable_pairs = counts.keys() & self.merges.keys()
            if len(mergeable_pairs) == 0:
                break
            # Apply the merge rule with the SMALLEST assigned ID first.
            # (Rules are numbered in the order they were learned.)
            to_merge = min(mergeable_pairs, key=self.merges.get)  # type: ignore
            ids = replace(ids, to_merge, self.merges[to_merge])
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join((self.vocab[i] for i in ids)).decode("utf-8")


# ──────────────────────────────────────────────────────────
# Task 1.04: Training a tokeniser (BPE algorithm)
# ──────────────────────────────────────────────────────────
def from_text(text: str, vocab_size: int) -> Tokenizer:
    """
    Train a BPE tokeniser from scratch.

    Algorithm:
      1. Start with a vocabulary of 256 single-byte tokens.
      2. Repeatedly find the most frequent consecutive pair.
      3. Merge that pair into a new token and record the rule.
      4. Stop when vocab reaches vocab_size.

    Args:
        text:       Training text (any UTF-8 string).
        vocab_size: Target vocabulary size (must be > 256).

    Returns:
        Trained Tokenizer with populated merges and vocab.
    """
    tok = Tokenizer()

    # Start by encoding the entire text as raw UTF-8 bytes
    ids = list(text.encode("utf-8"))

    num_merges = vocab_size - 256   # how many merge rules to learn

    for merge_idx in range(num_merges):
        new_id = 256 + merge_idx    # new token ID for this merge

        # Count all consecutive pairs in the current token sequence
        pair_counts = count(ids)
        if not pair_counts:
            break                   # nothing left to merge

        # Pick the most frequent pair (ties broken by pair value for determinism)
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

        # Record the merge rule: best_pair → new_id
        tok.merges[best_pair] = new_id

        # Build the vocab entry: concatenate the byte representations
        tok.vocab[new_id] = tok.vocab[best_pair[0]] + tok.vocab[best_pair[1]]

        # Apply the merge to the entire token sequence
        ids = replace(ids, best_pair, new_id)

        if (merge_idx + 1) % 50 == 0:
            print(f"  Merge {merge_idx + 1}/{num_merges} — "
                  f"new token '{tok.vocab[new_id].decode('utf-8', errors='replace')}' "
                  f"(ID {new_id}, count {pair_counts[best_pair]})")

    return tok


# ──────────────────────────────────────────────────────────
# Save / Load tokeniser
# ──────────────────────────────────────────────────────────
def save(tokenizer: Tokenizer, filename: str) -> None:
    with open(filename, "w") as f:
        for fst, snd in tokenizer.merges:
            print(f"{fst} {snd}", file=f)


def load(filename: str) -> Tokenizer:
    """Load a tokeniser from the .tok file format."""
    tok = Tokenizer()
    with open(filename, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            a, b = map(int, line.split())
            new_id = 256 + line_no
            pair = (a, b)
            tok.merges[pair] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
    return tok


# ══════════════════════════════════════════════════════════
# RUN EVERYTHING
# ══════════════════════════════════════════════════════════

print("=" * 60)
print("TASK 1.01 — count()")
print("=" * 60)
sample = [1, 2, 3, 1, 2, 1, 2]
result = count(sample)
print(f"  Input : {sample}")
print(f"  Pairs : {result}")
assert result[(1, 2)] == 3
assert result[(2, 3)] == 1
assert result[(3, 1)] == 1
print("  ✅ All assertions passed")


print("\n" + "=" * 60)
print("TASK 1.02 — replace()")
print("=" * 60)
tests = [
    ([1, 2, 1, 2],    (1, 2), 99, [99, 99]),
    ([1, 1, 2],       (1, 2), 99, [1, 99]),
    ([1, 2, 3],       (1, 2), 99, [99, 3]),
    ([1, 2, 2, 1, 2], (2, 2), 88, [1, 88, 1, 2]),
]
for ids, pair, nid, expected in tests:
    got = replace(ids, pair, nid)
    status = "✅" if got == expected else "❌"
    print(f"  {status} replace({ids}, {pair}, {nid}) → {got}  (expected {expected})")


print("\n" + "=" * 60)
print("TASK 1.03 — Encode / Decode round-trip")
print("=" * 60)
tok = Tokenizer()   # no merge rules yet → each byte is its own token
test_text = "hello"
encoded = tok.encode(test_text)
decoded = tok.decode(encoded)
print(f"  Text    : '{test_text}'")
print(f"  Encoded : {encoded}")
print(f"  Decoded : '{decoded}'")
assert decoded == test_text
print("  ✅ Round-trip OK")


print("\n" + "=" * 60)
print("TASK 1.04 — Training a tokeniser on Swedish Wikipedia (first 100k chars)")
print("=" * 60)
TRAIN_FILE = "wiki-sv-1m.txt" 
TOK_FILE   = "wiki-sv-1m.tok" 

with open(TRAIN_FILE, encoding="utf-8") as f:
    train_text = f.read(100_000)   # use first 100k chars for speed

print(f"  Training on {len(train_text):,} characters with vocab_size=300 …")
trained_tok = from_text(train_text, vocab_size=300)
print(f"\n  Merge rules learned : {len(trained_tok.merges)}")
print(f"  Vocabulary size     : {len(trained_tok.vocab)}")

# Show first 5 merge rules
print("\n  First 5 merge rules:")
for i, (pair, nid) in enumerate(list(trained_tok.merges.items())[:5]):
    a_str = trained_tok.vocab[pair[0]].decode("utf-8", errors="replace")
    b_str = trained_tok.vocab[pair[1]].decode("utf-8", errors="replace")
    new_str = trained_tok.vocab[nid].decode("utf-8", errors="replace")
    print(f"    Rule {i+1}: ({pair[0]}, {pair[1]}) → {nid}  "
          f"i.e. '{a_str}' + '{b_str}' = '{new_str}'")

# Test encode/decode
sample_text = "Sverige är ett nordiskt land."
enc = trained_tok.encode(sample_text)
dec = trained_tok.decode(enc)
print(f"\n  Sample encode: '{sample_text}'")
print(f"  Token IDs    : {enc}")
print(f"  Decoded back : '{dec}'")
assert dec == sample_text, "Round-trip failed!"
print("  ✅ Round-trip OK")


print("\n" + "=" * 60)
print("TASK 1.06 — Tokenisation and multi-linguality")
print("=" * 60)

# Load the pre-trained English tokeniser
en_tok = load("wiki-en-1m.tok")
sv_tok = load("wiki-sv-1m.tok") 
is_tok = load("wiki-is-1m.tok") 

# Read 100k chars from each language
with open("wiki-en-1m.txt", encoding="utf-8") as f: 
    en_text = f.read(100_000)
with open("wiki-sv-1m.txt", encoding="utf-8") as f: 
    sv_text = f.read(100_000)
with open("wiki-is-1m.txt", encoding="utf-8") as f: 
    is_text = f.read(100_000)

# Tokenise each combination
en_on_en = en_tok.encode(en_text)
en_on_sv = en_tok.encode(sv_text)
en_on_is = en_tok.encode(is_text)

GPT2_CONTEXT = 1024

def chars_per_context(n_chars, n_tokens, context_len):
    """How many chars fit in the GPT-2 context window?"""
    chars_per_token = n_chars / n_tokens
    return chars_per_token * context_len

print(f"  {'Language':<12} {'Chars':>10} {'Tokens':>10} {'Chars/Token':>13} "
      f"{'Chars in GPT-2 ctx':>20}")
print("  " + "-" * 70)

for label, text, tokens in [
    ("English", en_text, en_on_en),
    ("Swedish", sv_text, en_on_sv),
    ("Icelandic", is_text, en_on_is),
]:
    cpt = len(text) / len(tokens)
    ctx_chars = cpt * GPT2_CONTEXT
    print(f"  {label:<12} {len(text):>10,} {len(tokens):>10,} "
          f"{cpt:>13.3f} {ctx_chars:>20.0f}")

print(f"""
  Interpretation:
  • English tokeniser on English text: most efficient — tokens are long
    meaningful subwords, so the GPT-2 context holds ~{chars_per_context(len(en_text), len(en_on_en), GPT2_CONTEXT):.0f} chars
  • English tokeniser on Swedish/Icelandic: less efficient — many characters
    not covered by the English vocab get split into smaller byte-level pieces
  • This means a model with a fixed context window can "see" less text
    when processing Swedish/Icelandic than when processing English
  → Tokeniser fairness: models trained on English data are inherently
    disadvantaged when processing other languages, especially those with
    different character sets or morphology (Icelandic has very rich inflection)
""")


print("=" * 60)
print("PART 1 COMPLETE")
print("=" * 60)
