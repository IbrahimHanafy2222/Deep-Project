# Deep Learning for GEC — Complete Explainer
### LSTM · Bahdanau Attention · Transformer
*For Ibrahim & teammate — no prior background assumed*

---

## Table of Contents

1. [The Problem: Grammatical Error Correction](#1-the-problem)
2. [Sequence-to-Sequence Learning](#2-sequence-to-sequence-learning)
3. [Recurrent Neural Networks (RNNs)](#3-recurrent-neural-networks)
4. [Long Short-Term Memory (LSTM)](#4-long-short-term-memory-lstm)
5. [LSTM Encoder-Decoder (Phase 3)](#5-lstm-encoder-decoder-phase-3)
6. [The Bottleneck Problem](#6-the-bottleneck-problem)
7. [Bahdanau Attention (Phase 4)](#7-bahdanau-attention-phase-4)
8. [The Transformer (Phase 5)](#8-the-transformer-phase-5)
9. [Training Details Across All Phases](#9-training-details-across-all-phases)
10. [Inference: Greedy vs Beam Search](#10-inference-greedy-vs-beam-search)
11. [Metrics We Use](#11-metrics-we-use)
12. [How the Phases Compare](#12-how-the-phases-compare)

---

## 1. The Problem

**Grammatical Error Correction (GEC)** means taking a sentence that has grammar
mistakes and producing the corrected version.

```
Input  (corrupted): "She go to school yesterday and learning many thing."
Output (corrected):  "She went to school yesterday and learned many things."
```

The mistakes span many categories:
- Wrong verb tense ("go" → "went", "learning" → "learned")
- Missing plurals ("thing" → "things")
- Word choice errors

This is a **sequence-to-sequence** problem: we map a sequence of input tokens
to a sequence of output tokens. The output is not just a label — it is a whole
new sentence.

Our dataset comes from **C4_200M**, a large collection of English text scraped
from the web. Researchers created (corrupted, clean) pairs by introducing
realistic grammar errors into the clean text. We use 1,000,000 such pairs:
- **850,000** for training
- **50,000** for validation (checking performance during training)
- **100,000** for testing (final evaluation)

The text is first converted to subword tokens using a **16k ByteLevel BPE tokenizer**
trained in Phase 1. Every word is split into pieces: "learning" might become
["learn", "ing"]. This keeps the vocabulary manageable (16,000 possible tokens
instead of millions of English words).

---

## 2. Sequence-to-Sequence Learning

### What is a neural network?

A neural network is a function with many adjustable parameters (weights). You
feed it input numbers, it does lots of matrix multiplications and non-linear
transformations, and it produces output numbers. Training means finding the
weight values that make the output match the correct answer for many examples.

### The seq2seq idea

For translation/correction, input and output have different lengths. A simple
feed-forward network can't handle this — it needs fixed-size input and output.

The solution (Sutskever et al., 2014) is to split the network into two parts:

```
[encoder]  reads the entire input → produces a summary vector
[decoder]  reads the summary → generates the output one token at a time
```

The decoder is **autoregressive**: at each step it uses the token it generated
at the previous step as input, and produces a probability distribution over all
possible next tokens.

```
INPUT: "She go to school"

Encoder: reads all 4 tokens → summary vector C

Decoder step 1: C → probability over vocab → picks "She"     (outputs "She")
Decoder step 2: C + "She" → ... → picks "went"              (outputs "went")
Decoder step 3: C + "She went" → ... → picks "to"           (outputs "to")
...continues until it outputs </s> (end of sentence)
```

During **training** we use "teacher forcing": instead of feeding the decoder's
own previous output, we always feed the correct previous token. This makes
training faster and more stable.

During **inference** we feed the decoder's own output back in — the model must
be self-consistent.

---

## 3. Recurrent Neural Networks

### The core idea: hidden state

A standard neural network processes each token independently. An RNN processes
tokens in order and maintains a **hidden state** — a vector that summarises
"everything seen so far."

```
At each time step t:
    h_t = f(h_{t-1}, x_t)
```

Where:
- `x_t` is the current input token (embedded as a vector)
- `h_{t-1}` is the previous hidden state
- `h_t` is the new hidden state
- `f` is a learned function (the RNN cell)

After reading the whole input, `h_T` contains a compressed summary of the
entire sequence.

```
"She   go    to   school"
  ↓      ↓     ↓     ↓
 x_1   x_2   x_3   x_4

h_0 → [RNN] → h_1 → [RNN] → h_2 → [RNN] → h_3 → [RNN] → h_4
```

`h_4` is the encoder's final output — the "thought vector" summarising the input.

### The vanishing gradient problem

To train, we compute a loss (how wrong the output is) and propagate the error
backwards through the network to update weights. This is called **backpropagation
through time (BPTT)**.

The problem: for long sequences, the gradient signal has to travel backwards
through every time step. Each step multiplies the gradient by the RNN's
weight matrix. If those multiplications shrink the gradient slightly at each
step, after 50 steps the gradient has almost vanished — the early time steps
receive essentially zero training signal.

This means vanilla RNNs struggle to learn **long-range dependencies** like:

```
"The cats that the dog chased were scared."
                                   ↑
"were" depends on "cats" which is 7 words back.
```

**LSTMs** solve this problem. That's next.

---

## 4. Long Short-Term Memory (LSTM)

### The big idea: a separate memory lane

An LSTM cell has two vectors instead of one:
- `h_t` — the **hidden state** (fast, recent information, flows through the network)
- `c_t` — the **cell state** (long-term memory, has a direct path with less distortion)

The key insight is that information can be written to, read from, and erased from
the cell state through learned **gates**. Gates are numbers between 0 and 1 produced
by sigmoid functions: 0 means "block everything", 1 means "let everything through."

### The four computations inside one LSTM cell

At each time step, given input `x_t` and previous state `(h_{t-1}, c_{t-1})`:

#### Gate 1: Forget gate `f_t`
"How much of the old cell state should I keep?"

```
f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
```

Values near 0 → forget the past. Values near 1 → keep the past.

**Example:** When we start a new sentence, the forget gate resets the cell state
for irrelevant context from the previous sentence.

#### Gate 2: Input gate `i_t`
"How much of the new candidate information should I write in?"

```
i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)
```

#### New candidate values `g_t`
"What new information could be written into the cell state?"

```
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
```

`tanh` outputs values in [-1, +1], encoding the actual content to potentially store.

#### Update cell state
Combine old memory (filtered by forget gate) with new content (filtered by input gate):

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
```

`⊙` is element-wise multiplication.

#### Gate 3: Output gate `o_t`
"What part of the cell state should be exposed as the hidden state?"

```
o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(c_t)
```

### Why LSTMs avoid vanishing gradients

The cell state `c_t` is updated with **addition** rather than multiplication:
```
c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t
```
Gradients flow through the addition directly, unimpeded, unless the forget gate
actively closes (f_t ≈ 0). The model learns when to close the forget gate —
it only forgets when it should.

This "highway" for gradients lets the LSTM remember information across
100+ time steps.

### BiLSTM (Bidirectional LSTM)

In our encoder (Phase 3 and 4), we use a **Bidirectional LSTM**:

```
Forward LSTM:  reads left → right:  x_1, x_2, ..., x_T  → h_T_forward
Backward LSTM: reads right → left:  x_T, x_{T-1}, ..., x_1 → h_1_backward
```

We concatenate the two directions at each position:
```
h_t = [h_t_forward ; h_t_backward]   (dimension doubles: 256 → 512)
```

This gives each token's representation access to both past and future context.
For GEC, whether "go" is an error depends on both "She" (before) and "to school
yesterday" (after).

---

## 5. LSTM Encoder-Decoder (Phase 3)

This is the simplest seq2seq model. No attention. The encoder compresses the
entire input into a single fixed-size vector.

### Architecture

```
INPUT: "She go to school yesterday"

ENCODER (BiLSTM):
  - Embed each token → 256-dim vector
  - Feed through BiLSTM (hidden=256 each direction → 512 effective)
  - Final forward+backward states concatenated → projected to (h_0, c_0)
    h_0 = tanh(W_h · [h_T_fwd ; h_1_bwd])     ← 512-dim
    c_0 = tanh(W_c · [c_T_fwd ; c_1_bwd])

DECODER (1-layer LSTM, hidden=512):
  - Initialised with (h_0, c_0) from encoder
  - At each step t:
    input:   embedding of previous generated token
    output:  h_t → Linear(512 → 16000) → softmax → next token probability
```

### What the decoder sees

The decoder only has two ways to access the input:
1. The initial hidden state (h_0, c_0) — set once from the encoder
2. Its own generated tokens (via teacher forcing during training)

After step 1, the decoder never directly "looks at" the source again.
Everything it knows about the input is squeezed into h_0 and c_0 — 512 numbers.

### Training objective

Cross-entropy loss, computed over all generated tokens (PAD tokens ignored):

```
Loss = -Σ_t log P(correct_token_t | all previous tokens, source)
```

We want to maximise the probability the model assigns to the correct output.
Equivalently, minimise the negative log probability (cross-entropy).

### What this model does well and badly

**Well:** Short sentences where all relevant information fits in 512 numbers.
**Badly:** Long sentences. By the time the decoder generates token 30, the
initial state h_0 has faded and the model starts hallucinating.

---

## 6. The Bottleneck Problem

Imagine trying to describe a 50-word paragraph using only a 512-dimensional
vector, then having someone reconstruct the paragraph from just that vector.
Some information is always lost.

This is the encoder-decoder bottleneck:

```
"The bank says in a statement that more hikes will be needed,
 but this time around it omitted the word 'gradual' from its
 explanation on how it will approach future rate increases..."
                    ↓
              [512 numbers]
                    ↓
  "The bank says that there would be no more than stop, but
   I would need the 'Arganda' of its application process..."
```

The Phase 3 output for that sentence is completely wrong. The model lost crucial
information in the bottleneck.

The solution: let the decoder look at the encoder's **full sequence of hidden
states**, not just the final one. At each decoding step, the decoder can
"ask" which parts of the input are most relevant right now.

This is attention.

---

## 7. Bahdanau Attention (Phase 4)

Introduced by Bahdanau, Cho & Bengio (2014). The key insight: instead of
compressing the input into one vector, keep all the encoder's hidden states
and learn to selectively focus on them.

### Encoder output

The BiLSTM encoder now saves the full sequence of hidden states:
```
"She   go    to    school  yesterday"
  ↓      ↓     ↓     ↓        ↓
[h_1]  [h_2] [h_3] [h_4]   [h_5]     ← 5 vectors, each 512-dim
```

These are the **encoder annotations** — one vector per input token.

### Computing the attention score

At decoder step t, the decoder has a current hidden state `s_t`.
We compute a **score** for every encoder state `h_i`:

```
score(s_t, h_i) = v^T  · tanh( W_s · s_t  +  W_h · h_i )
                  ↑           ↑              ↑
               (scalar)    (project        (project
                           decoder)        encoder)
```

This is the "additive" or "concat" attention form. `v`, `W_s`, `W_h` are
learned parameters.

Intuitively: we project both the decoder state and each encoder state into a
shared space (dimension `attn_dim`), add them, apply `tanh`, and project down
to a scalar. Higher score = more relevant encoder state for this decoding step.

### Attention weights (alignment)

We apply softmax over all scores to get a probability distribution:

```
α_{t,i} = softmax_i( score(s_t, h_i) )    → sums to 1 over all i
```

These are the **attention weights**: α_{t,1}, α_{t,2}, ..., α_{t,T_src}.
They tell us how much decoder step t should "look at" each source position.

```
Decoder step 1 (generating "She"):
  α = [0.85, 0.05, 0.03, 0.04, 0.03]  ← most attention on "She"

Decoder step 2 (generating "went"):
  α = [0.10, 0.82, 0.03, 0.03, 0.02]  ← most attention on "go"
```

This is very interpretable! The model learned that correcting "go" → "went"
requires attending to the source token "go."

### Context vector

We take a weighted sum of the encoder states:

```
ctx_t = Σ_i  α_{t,i} · h_i
```

This is a `512-dim` vector — a blend of all encoder states, weighted by relevance.

### Decoder step with attention

The decoder now uses the context vector at every step:

```
input to LSTM:     [embedding(prev_token) ; ctx_t]   (concatenated)
output projection: Linear([LSTM_hidden ; ctx_t] → vocab)
```

The context is injected both as input and in the output — maximising the
chance the decoder uses the relevant source information.

### Why attention is better than the bottleneck

| | Phase 3 (no attention) | Phase 4 (Bahdanau) |
|---|---|---|
| Source access | Only h_0, c_0 (512 numbers, set once) | Full sequence (T × 512 numbers, refreshed each step) |
| Long sentences | Catastrophic forgetting | Can focus on distant tokens |
| Interpretability | Black box | Attention weights show alignment |
| Parameters added | 0 | ~W_s + W_h + v (≈ 0.5M extra) |

### PAD masking in attention

Source sentences in a batch have different lengths. Shorter ones are padded with
`<PAD>` tokens to match the longest. We must prevent the attention from attending
to these pad positions:

```python
scores = scores.masked_fill(src_mask == PAD, float("-inf"))
alpha  = softmax(scores)
```

After softmax, `-inf` becomes 0 — padding contributes nothing to the context.

---

## 8. The Transformer (Phase 5)

Introduced by Vaswani et al. (2017) in "Attention Is All You Need."

The big idea: **replace all recurrence with attention.** No LSTM cells, no
sequential processing. The model reads the entire sequence at once.

This gives two massive advantages:
1. **Parallelism**: all tokens are processed simultaneously → 10-100× faster training
2. **Direct connections**: token 1 can directly attend to token 50 without
   information passing through 49 intermediate states

### 8.1 Self-Attention

In the LSTM encoder, each token's representation was built by reading sequentially.
Self-attention builds each token's representation by directly comparing it to
every other token in the sequence — simultaneously.

#### Query, Key, Value

For each token i, we compute three vectors from its embedding `x_i`:
```
q_i = W_Q · x_i    (query:  "what am I looking for?")
k_i = W_K · x_i    (key:    "what do I contain?")
v_i = W_V · x_i    (value:  "what do I contribute if matched?")
```

The attention score between token i (query) and token j (key):
```
score(i, j) = (q_i · k_j) / sqrt(d_k)
```

Dividing by `sqrt(d_k)` prevents the dot products from growing too large in
high dimensions (which would push softmax into regions with tiny gradients).

The output for token i:
```
output_i = Σ_j  softmax_j(score(i, j)) · v_j
```

Token i's output is a weighted sum of all value vectors, where the weights are
determined by how similar each token's key is to i's query.

In matrix form (all tokens at once):
```
Attention(Q, K, V) = softmax( Q · K^T / sqrt(d_k) ) · V
```

This is **one big matrix multiplication** — runs in parallel for all tokens.

#### What does self-attention learn?

For GEC, consider "go" in "She go to school yesterday":
- "go" attends to "She" → learns it needs a third-person singular subject
- "go" attends to "yesterday" → learns the sentence is past tense
- Therefore "go" should become "went"

The model learns these relationships entirely from data. No hand-coded grammar rules.

### 8.2 Multi-Head Attention

Running attention once with full `d_model`-dimensional Q, K, V is suboptimal —
the single attention head must learn all relationships simultaneously.

Multi-head attention splits the representation into `h` heads:

```
head_i = Attention(Q · W_Q_i, K · W_K_i, V · W_V_i)
```

Each head uses a smaller dimension: `d_k = d_model / h = 256 / 8 = 32`.

The heads run in parallel and are concatenated:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
```

Different heads specialise in different relationships:
- Head 1: subject-verb agreement
- Head 2: noun-adjective agreement
- Head 3: coreferential links
- etc.

In our model: 8 heads, d_model=256, d_k=32 per head.

### 8.3 Positional Encoding

Self-attention has no concept of order — it treats the input as a set, not a
sequence. Swapping "She go" → "go She" would produce the same attention scores.

We inject position information by adding a **positional encoding** to each
embedding:
```
embedding_with_position = embedding(token_i) + PE(position_i)
```

We use sinusoidal encodings (fixed, not learned):
```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

Why sinusoidal? Each position gets a unique pattern across dimensions, and the
model can learn to distinguish relative positions because:
```
PE(pos + offset) = linear combination of PE(pos)
```

This means the model can easily compute "how far apart are these two positions?"
from the encodings alone.

### 8.4 Feed-Forward Sublayer

After each multi-head attention sublayer, there is a position-wise
**feed-forward network (FFN)**:

```
FFN(x) = ReLU(x · W_1 + b_1) · W_2 + b_2
```

- W_1: d_model → dim_ffn = 256 → 512
- W_2: dim_ffn → d_model = 512 → 256

This runs independently on each position (hence "position-wise"). It's like
a 2-layer MLP applied to each token's representation after attention.

The FFN adds capacity for the model to do computations that attention alone
can't (attention mixes information between tokens; FFN transforms each token's
representation in isolation).

### 8.5 Layer Normalisation and Residual Connections

Each sublayer (attention or FFN) is wrapped in:
```
output = LayerNorm(x + Sublayer(x))
```

**Residual connection** (`x + ...`): adds the input directly to the output.
- If the sublayer learns nothing, output ≡ input (no harm done)
- Prevents vanishing gradients in deep stacks (like the cell state in LSTM)

**Layer normalisation**: normalises each token's representation to have
mean=0, variance=1 (then scales and shifts with learned parameters).
- Stabilises training by preventing activations from exploding/vanishing
- Unlike BatchNorm, works on a single sequence (no batch statistics needed)

### 8.6 Transformer Encoder

One encoder layer = attention sublayer + FFN sublayer, both with residuals + LayerNorm:

```
x = LayerNorm(x + MultiHeadAttention(x, x, x))    ← self-attention
x = LayerNorm(x + FFN(x))
```

We stack 3 such layers. Each layer refines the representations with richer context.

```
Input: "She go to school yesterday"
  ↓ Embedding + Positional Encoding
  ↓ Encoder Layer 1  (learns surface patterns: spelling, adjacent words)
  ↓ Encoder Layer 2  (learns syntactic structure: subject-verb, prepositional phrases)
  ↓ Encoder Layer 3  (learns semantic context: tense consistency, agreement)
  ↓
Memory: 5 vectors of 256-dim, rich with contextual information
```

### 8.7 Transformer Decoder

The decoder is more complex — it does **three** computations per layer:

#### Self-attention with causal mask

The decoder attends to its own previously generated tokens. But we must prevent
it from seeing future tokens (that would be cheating during training):

```
Causal mask:
  Position 1 can attend to: [1]
  Position 2 can attend to: [1, 2]
  Position 3 can attend to: [1, 2, 3]
  ...
```

This is a triangular mask added before softmax:
```
scores = scores + mask
```
Where `mask[i,j] = 0 if j <= i, else -inf`.

#### Cross-attention (encoder-decoder attention)

This is the transformer's version of Bahdanau attention, but computed for
every decoder layer:

```
query  = decoder hidden state (at this layer)
key    = encoder memory
value  = encoder memory
```

Every decoder layer directly attends to the full encoder output. The model
doesn't rely on a single context vector — it re-queries the source at each
layer, with progressively more abstract decoder states.

#### Position-wise FFN (same as encoder)

### 8.8 Full Model

```
SOURCE: "She go to school"
  → Embed → Add PE → Encoder Stack (3 layers) → Memory M

TARGET (during training): "<s> She went to"
  → Embed → Add PE → Decoder Stack (3 layers):
    - Layer k receives: queries from below, keys/values from M
  → Linear(d_model → vocab) → Softmax → Next token probabilities
```

**Weight tying**: the output projection matrix (vocab × d_model) shares weights
with the embedding matrix. They encode the same thing from opposite directions:
embedding maps token_id → vector; output projection maps vector → token_id scores.
Tying reduces parameters by ~4M and often improves performance.

### 8.9 Why Transformer > LSTM+Attention for GEC

| Property | LSTM + Attention | Transformer |
|---|---|---|
| Path length between tokens | O(n) — through sequential hidden states | O(1) — direct attention |
| Parallelism in training | Sequential — step t waits for step t-1 | Fully parallel over T |
| Memory of long-range deps | Degrades with distance | Equal attention to all positions |
| Number of layers needed | Usually 2-4 | Usually 6 (paper) but 3 works for small models |
| Training speed (GPUs) | Slow (CUDA doesn't parallelize loops) | Fast (attention = matrix multiply) |

For GEC specifically, **long-range dependencies matter a lot**:
- "The cats that the dog chased **were** scared" — agreement across 6 tokens
- "She **has** been **going** to school for five years" — tense consistency across sentence
- Proper nouns at position 1 affecting verb form at position 15

The transformer handles these by construction. The LSTM has to learn to carry
that information across many sequential steps.

---

## 9. Training Details Across All Phases

### Tokenization

All phases use the same **ByteLevel BPE (Byte-Pair Encoding)** tokenizer with
16,000 subword tokens.

BPE works by merging the most frequent pairs of characters/subwords:
- Start: every character is its own token
- Merge "e" + "d" → "ed" (most frequent pair)
- Merge "t" + "he" → "the" (next most frequent)
- ...repeat 16,000 times

The result: common words are one token ("the" → "the"), rare/misspelled words
split into pieces ("Halfpint" → ["Half", "p", "int"]).

Special tokens: `<pad>=0, <unk>=1, <s>=2, </s>=3`.

### Embeddings

Each token ID is mapped to a `d_model`-dimensional dense vector via an embedding
table (a lookup matrix). The model learns these embeddings end-to-end.

For the transformer, we scale embeddings by `sqrt(d_model)` before adding
positional encoding. This makes the embedding magnitude comparable to the
positional encoding magnitude (otherwise the position signal would be swamped).

### Loss Function

**Cross-entropy loss** over all non-padding output tokens:
```
Loss = -Σ_t  log P(y_t | y_1, ..., y_{t-1}, source)
```

**Phase 5 adds label smoothing (ε=0.1)**: instead of training the model to
predict probability 1 for the correct token, we use a soft target:
```
target[correct] = 1 - ε + ε/V ≈ 0.9
target[others]  = ε/V ≈ 0.000006
```

This prevents overconfidence and improves generalisation.

### Optimiser: Adam

Adam maintains a running average of gradients (momentum) and a running average
of squared gradients (for adaptive step sizes per parameter):
```
m_t = β_1 · m_{t-1} + (1 - β_1) · g_t       (1st moment, β_1=0.9)
v_t = β_2 · v_{t-1} + (1 - β_2) · g_t²      (2nd moment, β_2=0.98)
θ_t = θ_{t-1} - lr · m̂_t / (sqrt(v̂_t) + ε)
```

Parameters with large gradients get smaller updates (stable);
parameters with small gradients get larger updates (don't get stuck).

### Learning Rate Schedules

**Phases 3 & 4: ReduceLROnPlateau**
Start at lr=1e-3. If validation loss doesn't improve for 2 epochs, halve lr.
Simple, works well for LSTM.

**Phase 5: Inverse-sqrt warmup (Vaswani et al. formula)**
```
lr = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))
```

- Steps 1–4000: LR ramps linearly from 0 to peak (~6e-4 for d_model=256)
- Steps 4000+: LR decays as 1/sqrt(step)

Why warmup for transformers? At the start, Adam's second-moment estimates
are unreliable (initialised at 0). A high LR with bad estimates → catastrophic
updates. Warmup gives Adam time to calibrate before the LR gets large.

### Gradient Clipping

We clip the global L2 norm of all gradients to 1.0:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

If the gradient vector's norm exceeds 1.0, it's scaled down proportionally.
This prevents "gradient explosion" — a single bad batch sending weights to ±∞.
Critical for LSTMs, still useful for transformers.

### Mixed Precision (AMP)

Computation in float16 (half precision), gradients accumulated in float32.
- ~2× training speed on modern GPUs (half the memory bandwidth)
- `GradScaler` multiplies the loss by a large scale factor before backward pass
  to prevent float16 underflow, then unscales before optimizer step

### Teacher Forcing (Phases 3 & 4)

During training, the decoder always receives the correct previous token.
This avoids "exposure bias" — errors compounding if the model generates a
wrong token early. In Phase 3, we linearly decay the teacher-forcing ratio
from 1.0 to 0.5 across epochs (mix of correct and self-generated inputs).

The transformer doesn't use teacher forcing in the same sense — it processes
all target positions in parallel using the causal mask.

---

## 10. Inference: Greedy vs Beam Search

### Greedy decoding

At each step, pick the highest-probability token:
```
y_t = argmax_v  P(v | y_1,...,y_{t-1}, source)
```

Fast (one forward pass per step), but can make locally good choices that
lead to globally bad sentences.

```
"She go to school yesterday"

Step 1: P("She")=0.9, P("I")=0.05, ...   → pick "She"
Step 2: P("went")=0.7, P("had")=0.2, ... → pick "went"
...
```

### Beam search

Maintain the top-k most probable partial sequences simultaneously (k = beam width).
At each step, expand each beam and keep only the top-k:

```
Beam = 2 example:
Initial: {["<s>"], score=0}

Step 1 expansions:
  ["<s>"] → ["<s>", "She"]  score=-0.1   (prob 0.90)
           → ["<s>", "The"]  score=-0.7   (prob 0.50)
  Top 2: [["<s>","She",-0.1], ["<s>","The",-0.7]]

Step 2 expansions:
  ["<s>","She"] → ["<s>","She","went"] score=-0.25
               → ["<s>","She","had"]  score=-0.90
  ["<s>","The"] → ["<s>","The","bank"] score=-1.2
               → ["<s>","The","cats"] score=-2.1
  Top 2: [["She","went",-0.25], ["She","had",-0.90]]
```

**Length normalisation**: longer sequences accumulate more negative log-probs,
so shorter beams get unfairly high scores. We normalise by length^α (α=0.7):
```
final_score = log_prob / (length ^ 0.7)
```

Beam search reliably improves GLEU by 1-3 points over greedy at inference.
The trade-off: it is `beam` times slower than greedy.

### For the transformer specifically

Unlike LSTM where we have a hidden state and just process one token per step,
the transformer must reprocess the entire prefix at each decoding step.
This is because there's no recurrent state to carry forward — the decoder
attends to the full token sequence.

We mitigate this by encoding the source once (`model.encode(src)`) and caching
the memory, then only running the decoder for each step.

Full KV-cache optimisation (as in GPT inference) could avoid reprocessing the
decoder prefix too, but is not implemented here for simplicity.

---

## 11. Metrics We Use

### GLEU (Generalized Language Evaluation Understanding)

Developed specifically for GEC evaluation. Like BLEU (n-gram precision) but
also penalises the model for copying errors from the source that appear in
the hypothesis.

Ranges from 0 (worst) to 1 (perfect). Our models score ~0.21–0.30.

### Exact Match (EM)

What fraction of predictions exactly equal the reference (character-for-character).
Even a single different character = 0. Very strict — most GEC systems score <5%.

### Token-level Precision, Recall, F0.5

Frame GEC as a token retrieval problem:
- **Precision**: of all tokens the model output, what fraction appear in the reference?
- **Recall**: of all tokens in the reference, what fraction did the model output?
- **F0.5**: harmonic mean weighting precision 2× more than recall.
  GEC convention: wrong "corrections" (false positives) are worse than missing
  a correction (false negative), so we favour precision.

```
F0.5 = (1 + 0.5²) × P × R / (0.5² × P + R)
     = 1.25 × P × R / (0.25P + R)
```

### Length-bucketed EM

Split test set by reference length:
- Short: ≤10 tokens
- Medium: 11–20 tokens
- Long: >20 tokens

Short sentences are easiest (less to correct, less to hallucinate).
The gap between LSTM and Transformer should be largest on long sentences,
because attention over the full sequence helps most there.

---

## 12. How the Phases Compare

### Architecture progression

```
Phase 3 (LSTM, no attention)
  - BiLSTM encoder → single vector → LSTM decoder
  - Information bottleneck hurts long sentences
  - ~19.5M parameters

Phase 4 (LSTM + Bahdanau attention)
  - BiLSTM encoder → all hidden states kept
  - Decoder queries all states at each step via learned alignment
  - ~20M parameters

Phase 5 (Transformer)
  - No recurrence: self-attention over full sequence
  - 3 encoder + 3 decoder layers, each with multi-head attention
  - Cross-attention in every decoder layer
  - ~13M parameters (weight tying saves ~4M)
  - Trains much faster (parallelism)
  - Better on long sentences (O(1) path length)
```

### Expected results (based on literature)

| Model | GLEU (greedy) | Advantage |
|-------|---------------|-----------|
| Phase 3 | ~0.21 | Baseline |
| Phase 4 | ~0.25 | +attention: better alignment |
| Phase 5 | ~0.28–0.32 | +transformer: parallelism + deep context |

Your actual numbers depend on the 1M training budget and number of epochs.
The relative ordering (P3 < P4 < P5) should hold.

### Key takeaway for the report

**Phase 3 → Phase 4**: Adding attention fixes the bottleneck. The model no
longer has to compress 50 tokens into 512 numbers. Alignment weights are
interpretable and show the model "knows" which source token to fix.

**Phase 4 → Phase 5**: Replacing recurrence with self-attention makes every
pairwise interaction direct and simultaneous. The transformer can model
arbitrarily complex dependencies without sequential information loss.
Training is also 5-10× faster on GPU, allowing more epochs in the same time.

---

*This document covers everything in the project from first principles.
If any part is unclear, the best follow-up is to look at the code in
notebooks 05, 06, and 07 — the math and the code are direct translations
of each other.*
