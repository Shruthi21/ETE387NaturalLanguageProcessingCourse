"""
Statistical Bigram Language Model — using numpy (same logic as PyTorch version)
Based on Swedish first names dataset
"""

import math
import numpy as np

# ─────────────────────────────────────────────
# 1. Load the Data
# ─────────────────────────────────────────────
with open("names-train.txt", encoding="utf-8") as f:
    names = [line.rstrip() for line in f]

with open("names-test.txt", encoding="utf-8") as f:
    test_names = [line.rstrip() for line in f]

print(f"Training names : {len(names)}")
print(f"Test names     : {len(test_names)}")
print(f"First 5 names  : {names[:5]}")

# ─────────────────────────────────────────────
# TASK 1: What's in the data?
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("TASK 1: What's in the data?")
print("="*55)

for sample in ["anna", "lars", "maria"]:
    if sample in names:
        rank = names.index(sample) + 1
        print(f"  '{sample}' → rank #{rank} (1 = most frequent)")
    else:
        print(f"  '{sample}' is NOT in the dataset")

print(f"\n  Total unique names : {len(names)}")
print(f"  Most frequent      : {names[0]}")
print(f"  Rarest (last)      : {names[-1]}")

# ─────────────────────────────────────────────
# 2. Character-to-Index Mapping
# ─────────────────────────────────────────────
char2idx = {"$": 0}
for name in names:
    for char in name:
        if char not in char2idx:
            char2idx[char] = len(char2idx)

idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(char2idx)

# ─────────────────────────────────────────────
# TASK 2: Vocabulary
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("TASK 2: Vocabulary")
print("="*55)
print(f"  Vocabulary size : {vocab_size}")
chars_only = sorted([c for c in char2idx if c != "$"])
print(f"  Characters      : {'  '.join(chars_only)}")
print("  Includes Swedish-specific: å, ä, ö etc. — unlike English names.")

# ─────────────────────────────────────────────
# 3. Bigram Generator
# ─────────────────────────────────────────────
def bigrams(names):
    for name in names:
        for x, y in zip("$" + name, name + "$"):
            yield x, y

print("\n" + "="*55)
print("Example bigrams from first 2 names:")
print("="*55)
print(" ", [b for b in bigrams(names[:2])])

# ─────────────────────────────────────────────
# 4. Build the Bigram Model
# ─────────────────────────────────────────────
counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
for x, y in bigrams(names):
    counts[char2idx[x], char2idx[y]] += 1

row_sums = counts.sum(axis=1, keepdims=True)
model = counts / row_sums   # model[i][j] = P(char_j | char_i)

# ─────────────────────────────────────────────
# TASK 3: Inspecting the model
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("TASK 3: Inspecting the model")
print("="*55)

c_idx = char2idx['c']
top_after_c = sorted([(idx2char[j], model[c_idx, j]) for j in range(vocab_size)], key=lambda x: -x[1])[:5]
print("  Top 5 most likely letters after 'c':")
for char, prob in top_after_c:
    print(f"    P({char} | c) = {prob:.4f}")

top_starts = sorted([(idx2char[j], model[0, j]) for j in range(1, vocab_size)], key=lambda x: -x[1])[:5]
print("\n  Top 5 most likely starting letters:")
for char, prob in top_starts:
    print(f"    P({char} | $) = {prob:.4f}")

top_ends = sorted([(idx2char[i], model[i, 0]) for i in range(1, vocab_size)], key=lambda x: -x[1])[:5]
print("\n  Top 5 most likely ending letters:")
for char, prob in top_ends:
    print(f"    P($ | {char}) = {prob:.4f}")

print("\n  Impossible name example: 'bxq' — bigram 'bx' never appears → P=0")

# ─────────────────────────────────────────────
# 5. Generate Names
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("Generating 10 names:")
print("="*55)
np.random.seed(42)
for _ in range(10):
    generated = "$"
    while True:
        prev_idx = char2idx[generated[-1]]
        probs = model[prev_idx]
        next_idx = np.random.choice(vocab_size, p=probs)
        next_char = idx2char[next_idx]
        if next_char == "$":
            break
        generated += next_char
    print(f"  {generated[1:]}")

# ─────────────────────────────────────────────
# TASK 4: Probability of a name
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("TASK 4: Probability of specific names")
print("="*55)

def name_probability(name, model, char2idx):
    log_prob = 0.0
    sequence = "$" + name + "$"
    for x, y in zip(sequence, sequence[1:]):
        if x not in char2idx or y not in char2idx:
            return 0.0
        p = model[char2idx[x], char2idx[y]]
        if p == 0:
            return 0.0
        log_prob += math.log(p)
    return math.exp(log_prob)

for word in ["anna", "lars", "maria", "s", "zxqw"]:
    prob = name_probability(word, model, char2idx)
    print(f"  P(generate '{word}') = {prob:.10f}")

# ─────────────────────────────────────────────
# 6. Perplexity on Test Set
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("Perplexity on Test Set")
print("="*55)
nlls = []
for prev_char, next_char in bigrams(test_names):
    p = model[char2idx[prev_char], char2idx[next_char]]
    nlls.append(-math.log(p))
ppl = math.exp(sum(nlls) / len(nlls))
print(f"  Perplexity = {ppl:.1f}")

# ─────────────────────────────────────────────
# TASK 5: Upper bound on perplexity
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("TASK 5: Upper bound on perplexity")
print("="*55)
print(f"  Lower bound : 1  (model perfectly certain at every step)")
print(f"  Upper bound : {vocab_size-1}  (uniform random over all {vocab_size-1} characters)")
print(f"  Our model   : {ppl:.1f}  (effectively choosing from ~{ppl:.0f} chars per step)")
