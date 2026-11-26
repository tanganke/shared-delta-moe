# Delta compression study

Which compression of the deltas keeps a delta MoE's accuracy, at what storage? `compress.py` holds every method tried,
simulated: each maps a dense $\Delta$ to its compressed-then-restored $\hat\Delta$ (dense, so the module and its
kernels are unchanged) and reports the storage the compressed form would take. Packed storage and kernels exist only
for the formats that are kept ([DARE](dare.md), [BitDelta](bitdelta.md)); the others are here with their results so
they need not be tried again.

## Usage

```python
from shared_delta_moe import compress as C

# x[e]: (T_e, H) MLP inputs the delta expert e should be calibrated on (its routed tokens or its domain's data);
# only the calibrated methods (GPTQ, SparseGPT, Wanda, importance DARE) need them
grams = C.calibration_grams(moe, x)
bits = C.compress_deltas(moe, C.rtn(2, group=128))          # no data needed
bits = C.compress_deltas(moe, C.gptq(2, group=128), grams)
```

`moe` is a `DeltaMoE` or a `HierDeltaMoE` (both store dense deltas the same way); `compress_deltas` replaces every
delta in place and returns the bits per entry. A method is a callable `(delta, gram) -> (approx, bits)`, so a new one
is a function.

## Setting

One projection of one expert: a delta $\Delta \in \mathbb{R}^{R \times C}$ (input dimension $C$; gate / up: $R=I$,
$C=H$; down: $R=H$, $C=I$) and the inputs $X \in \mathbb{R}^{T \times C}$ that reach it. Its contribution to the
output is $X\Delta^\top$, so what a compression should keep small is the output error, not the weight error:

$$
\mathcal{E}(\hat\Delta) = \|X\hat\Delta^\top - X\Delta^\top\|_F^2
= \sum_{i=1}^{R} (\hat\delta_i - \delta_i)^\top \mathbf{H}\, (\hat\delta_i - \delta_i),
\qquad \mathbf{H} = X^\top X,
$$

with $\delta_i$ the $i$-th row of $\Delta$. The rows separate, and $\mathbf{H}$ (the inputs' Gram, `calibration_grams`)
is all a calibrated method needs of the data. Uncalibrated methods minimise the weight error
$\|\hat\Delta - \Delta\|_F^2$ instead, i.e. assume $\mathbf{H} = \mathbf{I}$.

Storage is counted in bits per delta entry: dense bf16 is 16. For a Qwen3.5-4B layer (hidden 2560, inter 9216) one
delta is 141.6 MB dense, so $b$ bits per entry is $141.6\, b / 16$ MB.

## Methods

### Low-rank: truncated SVD (`svd`)

With $\Delta = U \Sigma V^\top$, keep the $r$ largest singular values: $\hat\Delta = U_r \Sigma_r V_r^\top$, the best
rank-$r$ approximation in Frobenius norm (Eckart–Young), error $\sum_{k>r} \sigma_k^2$. Storage: the two factors in
bf16, $16\, r (R + C) / (RC)$ bits per entry (rank 256 at $9216 \times 2560$: 2.0 bits). It pays only if the
spectrum decays fast, as for a LoRA delta; a full fine-tune's is flat (below).

### Scalar quantisation: round to nearest (`rtn`)

Split each row into groups of $g$ consecutive entries along the input dimension. A group $w \in \mathbb{R}^g$ gets an
asymmetric grid of $2^b$ levels spanning its range,

$$
z = \min_k w_k, \qquad s = \frac{\max_k w_k - \min_k w_k}{2^b - 1}, \qquad
\hat w_k = z + s \cdot \mathrm{clamp}\!\left(\mathrm{round}\frac{w_k - z}{s},\, 0,\, 2^b - 1\right),
$$

stored as $b$-bit codes plus $s$ and $z$ in fp16: $b + 32/g$ bits per entry (2 bits, $g=128$: 2.25). The rounding
error is at most $s/2$ per entry, about $s^2/12$ in mean square; a smaller group fits the grid to a narrower range at
more overhead. No data is needed, and decoding is a multiply-add per entry on a regular layout.

### GPTQ (`gptq`)

GPTQ (Frantar et al., 2023) quantises on the same grid but minimises $\mathcal{E}$ rather than the weight error. Per
row, quantise the columns one at a time; after column $j$ is rounded to $q_j$, the remaining (not yet quantised)
entries $F$ are moved to cancel the error it caused, which is the Optimal Brain Surgeon update for the quadratic
$(\hat\delta - \delta)^\top \mathbf{H} (\hat\delta - \delta)$:

$$
\delta_F \leftarrow \delta_F - \frac{\delta_j - q_j}{[\mathbf{H}_F^{-1}]_{jj}} \,[\mathbf{H}_F^{-1}]_{F,j},
$$

with $\mathbf{H}_F$ the Gram restricted to $j$ and the columns after it. Quantising in a fixed column order for every
row makes $\mathbf{H}_F^{-1}$ the same for all rows, and all of them come from one Cholesky factor: with
$\mathbf{H}^{-1} = U^\top U$ ($U$ upper triangular), $[\mathbf{H}_F^{-1}]_{F,j} / [\mathbf{H}_F^{-1}]_{jj} = U_{j,F} / U_{jj}$,
so the step is $e = (\delta_j - q_j)/U_{jj}$, $\delta_{j:} \leftarrow \delta_{j:} - e\, U_{j,j:}$. Updates are applied
lazily in blocks of 128 columns; $\mathbf{H}$ is damped by $1\%$ of its mean diagonal for a stable inverse. A group's
grid is fixed when its first column is reached, from the values updated so far (a plain min-max grid; the reference
implementation also widens it to include 0 and rounds the offset to a level). Storage is that of `rtn`.

### Residual binarisation (`residual_sign`)

BitDelta is one sign plane with a scale: for a fixed sign pattern $\mathrm{sign}(\delta)$, the scale minimising
$\|\delta - a\, \mathrm{sign}(\delta)\|^2$ is $a = \mathrm{mean}_k |\delta_k|$. $n$ planes binarise the residual
again:

$$
R_1 = \Delta,\qquad P_k = \mathrm{diag}(a_k)\,\mathrm{sign}(R_k),\ a_{k,i} = \mathrm{mean}_j |R_{k,ij}|,\qquad
R_{k+1} = R_k - P_k,\qquad \hat\Delta = \sum_{k=1}^n P_k,
$$

$n$ bits per entry plus one scale per row per plane. Each plane is a BitDelta sign plane, so the BitDelta kernel
would read it $n$ times.

### Sparsification: magnitude, DARE, importance DARE (`magnitude`, `dare`, `importance_dare`)

Keep a fraction $p$ of the entries, store their values (bf16) and where they are.

- **Magnitude** keeps the largest $|\Delta_{ij}|$ (TIES-style trimming); the mask must be stored (a bitmap, 1 bit
  per entry). It drops every small change, a biased error.
- **DARE** keeps each entry (or 1 x $B$ block) with probability $p$ and rescales it, $\hat\Delta_{ij} = \Delta_{ij}
  m_{ij} / p$, $m_{ij} \sim \mathrm{Bernoulli}(p)$. It is unbiased, $\mathbb{E}[\hat\Delta] = \Delta$, and its
  output error is zero-mean noise of variance $\frac{1-p}{p} \sum_j \Delta_{ij}^2 \|X_{:,j}\|^2$ per output row. The
  mask is regenerated from a seed, so storage is $16p$ plus the decode-time bitmap ($1/B$). (The variance is for
  single entries, $B = 1$; with blocks, the entries of a block share their mask and the terms become
  $\frac{1-p}{p} \sum_b \|X_{:,b}\, \delta_{i,b}\|^2$ over the blocks $b$ of row $i$.)
- **Importance DARE** keeps entry $(i,j)$ with its own probability $q_{ij}$, rescaled by $1/q_{ij}$ (still
  unbiased). The output-error variance $\sum_{ij} \Delta_{ij}^2 \|X_{:,j}\|^2 (1/q_{ij} - 1)$ under the budget
  $\sum_{ij} q_{ij} = pRC$ is minimised (Lagrange) by

  $$
  q_{ij} = \min\!\left(1,\ c\, s_{ij}\right), \qquad s_{ij} = |\Delta_{ij}|\, \|X_{:,j}\|,
  $$

  with $c$ set by the budget: sample in proportion to each entry's effect on the output. With 1 x $B$ blocks the same
  argument gives $s_{ib} = \|X_{:,b}\, \delta_{i,b}\| = (\delta_{i,b}^\top \mathbf{H}_{bb}\, \delta_{i,b})^{1/2}$,
  $\mathbf{H}_{bb}$ the block's diagonal block of the Gram. A decoder must regenerate the mask, so it needs $q$:
  `importance_dare` replaces $s$ by the outer product of its row and column sums,
  $s_{ib} \approx (\sum_{b'} s_{ib'})(\sum_{i'} s_{i'b}) / \sum_{i'b'} s_{i'b'}$, stored as two vectors
  ($R + C/B$ values): a heuristic, no longer optimal. `rank1=False` keeps $q$ per block (the optimum, at 16 bits per
  block of storage). `gamma` raises $s$ to a power (0.5 flattens the probabilities).

### Pruning with calibration: Wanda, SparseGPT (`wanda`, `sparsegpt`)

- **Wanda** (Sun et al., 2024): zeroing one entry changes output row $i$ by $\Delta_{ij} X_{:,j}$, of norm
  $|\Delta_{ij}|\, \|X_{:,j}\|$; Wanda keeps the $p$ largest such scores in each row (here, of the delta) and does
  not change the rest. $\|X_{:,j}\| = \sqrt{\mathbf{H}_{jj}}$, so only the Gram's diagonal is used.
- **SparseGPT** (Frantar & Alistarh, 2023) is GPTQ with "round to zero" for the pruned entries: per block of 128
  columns it prunes the entries of least Optimal Brain Surgeon saliency
  $\Delta_{ij}^2 / [\mathbf{H}_F^{-1}]_{jj} = \Delta_{ij}^2 / U_{jj}^2$ (the error that pruning entry $j$ alone, with
  the later entries compensating, adds to $\mathcal{E}$), then walks the columns as GPTQ does, pushing each pruned
  entry's error onto the later columns, so the kept entries are changed to compensate.

Both store the mask (1 bit per entry unstructured, $1/B$ for 1 x $B$ blocks) and $16p$ of values.

## Results

Qwen3.5-4B MoE (5 full-fine-tuned deltas: engineering, history, law, medical, generation; plus a zero expert, top-1;
ntu-projects `moe-indomain-acc` log [22]). Each expert is calibrated on its own domain's training data (96 samples,
up to 512 tokens, 8192 tokens per layer) with the inputs of the dense-delta model.

### Per-layer output error

Seven layers (0, 5, 10, 15, 20, 25, 31), each expert on its own domain's tokens, every projection compressed: the
share of the expert's effect lost,
$\|\mathrm{MLP}(X; W_b+\hat\Delta) - \mathrm{MLP}(X; W_b+\Delta)\|_F \,/\, \|\mathrm{MLP}(X; W_b+\Delta) - \mathrm{MLP}(X; W_b)\|_F$,
mean over layers and experts (lower is better):

| method | bits / entry | lost, mean | lost, medical |
|---|---:|---:|---:|
| BitDelta (row scales) | 1.00 | 0.36 | 0.38 |
| SVD rank 128 | 1.02 | 0.44 | 0.56 |
| SVD 64 + BitDelta on the residual | 1.51 | 0.23 | 0.30 |
| residual sign, 2 planes | 2.01 | 0.21 | 0.23 |
| SVD rank 256 | 2.04 | 0.36 | 0.47 |
| RTN 2 bit, group 128 | 2.25 | 0.18 | 0.23 |
| GPTQ 2 bit, group 128 | 2.25 | 0.085 | 0.12 |
| GPTQ 2 bit, group 64 | 2.50 | 0.075 | 0.10 |
| DARE 0.2, block 64 | 3.22 | 0.74 | 0.97 |
| RTN 3 bit, group 128 | 3.25 | 0.076 | 0.10 |
| GPTQ 3 bit, group 128 | 3.25 | 0.041 | 0.046 |
| RTN 4 bit, group 128 | 4.25 | 0.032 | 0.048 |
| SVD rank 512 | 4.09 | 0.26 | 0.36 |
| GPTQ 4 bit, group 128 | 4.25 | 0.026 | 0.030 |
| magnitude 0.2 | 6.40 | 0.52 | 0.50 |

- **Low rank does not work for full fine-tunes**: rank 512 (4 bits) still loses a quarter, more than 2-bit RTN. The
  deltas' spectra are flat.
- **The experts share little**: $\|\Delta_e - \bar\Delta\| / \|\Delta_e\|$ with $\bar\Delta$ the mean of the five is
  0.70–1.08, so a shared part plus small per-expert residuals is not available.
- **Calibration lowers the error** at the same storage (GPTQ vs RTN): by half at 2 and 3 bits, less at 4.
- This metric is unfair to DARE: its error is zero-mean noise, large per layer but averaging out downstream (DARE 0.2
  is close to lossless in accuracy, below). It ranks biased methods (quantisation, pruning) among themselves.

### End-to-end KL

Every layer's deltas compressed (each expert calibrated as above), then on each domain's validation data (16
samples) the next-token $\mathrm{KL}(p_\text{dense} \,\|\, p_\text{compressed})$ per token, and the share of tokens
whose top-1 prediction is unchanged. Unlike the per-layer error it sees DARE's noise average out, and it orders the
methods as their accuracy does (medical KL vs NLI4CT drop: RTN / GPTQ 2 bit 0.015, lossless; DARE 0.2 0.062, −0.3;
residual sign 0.099, −1.8; BitDelta 0.175, −3.3), so it is the cheap screen (a minute per method on one GPU) before
an accuracy run.

| method | bits / entry | KL, mean | KL, medical | top-1 kept, medical |
|---|---:|---:|---:|---:|
| BitDelta (row scales) | 1.00 | 0.097 | 0.175 | 91.3% |
| DARE 0.1, block 64 | 1.62 | 0.040 | 0.114 | 89.9% |
| importance DARE 0.1, block 64, rank 1 | 1.62 | 0.034 | 0.097 | 90.7% |
| ... same, $\gamma = 0.5$ | 1.62 | 0.037 | 0.102 | 90.0% |
| residual sign, 2 planes | 2.01 | 0.049 | 0.099 | 94.6% |
| **RTN 2 bit, group 128** | 2.25 | 0.0054 | 0.015 | 97.9% |
| **GPTQ 2 bit, group 128** | 2.25 | **0.0044** | **0.015** | 97.7% |
| DARE 0.1 | 2.60 | 0.038 | 0.113 | 92.6% |
| SparseGPT 0.1 | 2.60 | 0.042 | 0.098 | 94.2% |
| Wanda 0.1 | 2.60 | 0.142 | 0.229 | 88.9% |
| importance DARE 0.1, rank 1 | 2.61 | 0.032 | 0.109 | 92.8% |
| importance DARE 0.1, per entry (not storable) | 2.61 | 0.022 | 0.065 | 93.8% |
| magnitude 0.1 | 3.20 | 0.200 | 0.290 | 86.1% |
| DARE 0.2, block 64 | 3.22 | 0.024 | 0.067 | 93.3% |
| importance DARE 0.2, block 64, rank 1 | 3.22 | 0.020 | 0.057 | 94.5% |
| Wanda 0.2, block 64 | 3.22 | 0.198 | 0.313 | 86.3% |
| GPTQ 3 bit, group 128 | 3.25 | 0.0051 | 0.011 | 98.3% |
| DARE 0.2 | 4.20 | 0.020 | 0.062 | 94.2% |
| SparseGPT 0.2 | 4.20 | 0.013 | 0.039 | 96.5% |
| Wanda 0.2 | 4.20 | 0.082 | 0.145 | 92.3% |
| magnitude 0.2 | 6.40 | 0.130 | 0.199 | 89.5% |

(The importance-DARE rows were measured with the block score $\sum_{j \in b} |\Delta_{ij}|\,\|X_{:,j}\|$, the
version before the block-optimal score now in `compress.py`.)

- **Quantisation beats every sparsification** by a wide margin: 2-bit RTN has less than half the KL of the best
  pruning at about twice the storage (SparseGPT 0.2, 4.2 bits), and a sixth of any sparsification at similar storage.
- **Among the sparsifications, calibration helps only with compensation**: SparseGPT, which updates the kept entries,
  beats DARE at the same keep; Wanda, which only picks them by input-weighted magnitude, is worse than DARE (and
  magnitude worse still): dropping the small entries is a bias that random keeping avoids.
- **Importance DARE** lowers DARE's KL by 15–20% at the same storage (rank-1 probabilities); per-entry probabilities
  would nearly halve it but need the probabilities stored. Still well behind 2-bit quantisation.

### Accuracy

Accuracy in % (4 samples per question, 500 questions per set), against the dense-delta model; in parentheses the paired
difference with its 95% bootstrap interval over questions. Generation uses the same seeds for every model, so most
samples coincide and the intervals are narrow.

| deltas | bits / entry | engineering | history | law | NLI4CT |
|---|---:|---|---|---|---|
| dense bf16 | 16 | 73.5 | 75.5 | 68.4 | 79.0 |
| BitDelta, matrix scale | 1.0 | 73.5 (+0.0 [−0.8, +0.8]) | 74.2 (−1.2 [−2.4, −0.1]) | 67.5 (−0.9 [−2.0, +0.1]) | 75.8 (−3.3 [−4.8, −1.9]) |
| residual sign, 2 planes | 2.0 | 73.9 (+0.5 [−0.1, +1.1]) | 74.5 (−0.9 [−1.8, −0.1]) | 67.8 (−0.6 [−1.3, +0.1]) | 77.3 (−1.8 [−2.7, −0.9]) |
| **RTN 2 bit, group 128** | 2.25 | 73.6 (+0.1 [−0.1, +0.5]) | 75.2 (−0.3 [−0.6, +0.0]) | 68.3 (−0.1 [−0.5, +0.3]) | 79.0 (−0.1 [−0.6, +0.6]) |
| **GPTQ 2 bit, group 128** | 2.25 | 73.8 (+0.4 [+0.0, +0.8]) | 75.5 (+0.1 [−0.2, +0.5]) | 68.6 (+0.2 [−0.2, +0.6]) | 79.0 (−0.1 [−0.7, +0.4]) |
| GPTQ 3 bit, group 128 | 3.25 | 73.5 (+0.1 [−0.1, +0.3]) | 75.5 (+0.1 [−0.1, +0.3]) | 68.2 (−0.1 [−0.6, +0.3]) | 79.0 (−0.1 [−0.5, +0.4]) |

For comparison (log [35], other runs): DARE 0.2 with 1 x 64 blocks (3.2 bits) is within the intervals on three sets
and about −1.5 on NLI4CT.

- **2-bit group quantisation keeps the accuracy**, calibrated or not: every set within ±0.4 with intervals that
  include or touch 0, at 1/7 of the dense
  size, smaller than DARE 0.2 and without its NLI4CT drop. RTN needs no data and decodes from a regular layout.
- **Sign planes lose what the grids keep**: two sign planes (2 bits) still lose 1.8 on NLI4CT; a sign plane spends
  its bit on the sign and fits magnitudes with one scale per row, while a 4-level grid per 128 entries follows the
  local range.
- Below 2 bits (ternary grids, 1 bit with an offset per small group) is where calibration should start to matter;
  not measured yet.
