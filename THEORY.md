# 📐 THEORY.md — Technical Deep Dive

> Complete theoretical foundations of the Quantum-Enhanced LPR system.  
> Covers: quantum computing principles, variational circuits, image enhancement, sequence recognition, and training methodology.

---

## Table of Contents

1. [Motivation: Why Quantum for Vision?](#1-motivation-why-quantum-for-vision)
2. [Quantum Computing Foundations](#2-quantum-computing-foundations)
3. [Variational Quantum Circuits (VQC)](#3-variational-quantum-circuits-vqc)
4. [PennyLane & Quantum-Classical Interface](#4-pennylane--quantum-classical-interface)
5. [Zero-DCE: Low-Light Image Enhancement](#5-zero-dce-low-light-image-enhancement)
6. [CNN Feature Extraction](#6-cnn-feature-extraction)
7. [The 8-Qubit Quantum Layer](#7-the-8-qubit-quantum-layer)
8. [Bidirectional LSTM Sequence Decoder](#8-bidirectional-lstm-sequence-decoder)
9. [CTC Loss — Training Without Segmentation](#9-ctc-loss--training-without-segmentation)
10. [Classical Baseline Architecture](#10-classical-baseline-architecture)
11. [Training Strategy & Optimization](#11-training-strategy--optimization)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Dataset: RodoSol-ALPR](#13-dataset-rodosol-alpr)
14. [References](#14-references)

---

## 1. Motivation: Why Quantum for Vision?

Classical neural networks represent information as real-valued vectors. A quantum system represents information as a **superposition of states** in a complex Hilbert space of exponential dimension.

For an $n$-qubit system, the state space has dimension $2^n$. An 8-qubit system operates in a $2^8 = 256$-dimensional Hilbert space. This means the quantum layer can — in principle — encode and process **256-dimensional feature interactions** using only 8 "neurons."

The key hypothesis of this project:

> *Low-light license plate images contain subtle, high-frequency spatial patterns that are difficult for classical neurons to separate. A quantum circuit, operating in a richer feature space, may extract more discriminative representations from the same compressed feature vector.*

---

## 2. Quantum Computing Foundations

### 2.1 The Qubit

A classical bit is deterministically 0 or 1. A **qubit** exists in a superposition:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad \alpha, \beta \in \mathbb{C}, \quad |\alpha|^2 + |\beta|^2 = 1$$

where $|0\rangle = \begin{pmatrix}1\\0\end{pmatrix}$ and $|1\rangle = \begin{pmatrix}0\\1\end{pmatrix}$ are the **computational basis states**.

The probabilities of measuring 0 and 1 are $|\alpha|^2$ and $|\beta|^2$ respectively. Measurement **collapses** the superposition.

### 2.2 Multi-Qubit Systems

For $n$ qubits, the joint state is a **tensor product**:

$$|\psi_{01...n}\rangle \in \mathbb{C}^{2^n}$$

For 2 qubits, the basis is $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$. For our 8-qubit system, the basis has $2^8 = 256$ elements.

### 2.3 Entanglement

Two qubits are **entangled** if their joint state cannot be written as a tensor product of individual states:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \quad \text{(Bell state — maximally entangled)}$$

Entanglement is the key quantum resource — it creates **non-local correlations** between qubits that have no classical analogue. In our circuit, `StronglyEntanglingLayers` creates entanglement between all 8 qubits, allowing the quantum layer to model **all pairwise and higher-order correlations** between the 8 input features simultaneously.

### 2.4 Quantum Gates

Quantum computation applies **unitary operators** $U$ (quantum gates) to qubits. Key gates used in our circuit:

| Gate | Matrix | Effect |
|---|---|---|
| Pauli-X | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ | Bit flip |
| Pauli-Y | $\begin{pmatrix}0&-i\\i&0\end{pmatrix}$ | Bit+phase flip |
| Pauli-Z | $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ | Phase flip |
| RX(θ) | $\begin{pmatrix}\cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{pmatrix}$ | Rotation around X-axis |
| CNOT | $\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&0&1\\0&0&1&0\end{pmatrix}$ | Controlled NOT (entangler) |

### 2.5 Measurement & Expectation Values

Instead of collapsing the state (which destroys information), we compute **expectation values** of observables:

$$\langle Z_i \rangle = \langle\psi|Z_i|\psi\rangle \in [-1, +1]$$

where $Z_i$ is the Pauli-Z operator on qubit $i$. This gives a continuous, differentiable output — essential for gradient-based training.

Our circuit outputs: $[\langle Z_0 \rangle, \langle Z_1 \rangle, ..., \langle Z_7 \rangle] \in [-1,1]^8$

---

## 3. Variational Quantum Circuits (VQC)

A **Variational Quantum Circuit** (VQC), also called a **Parameterized Quantum Circuit** (PQC), is a quantum circuit with trainable parameters $\boldsymbol{\theta}$:

$$U(\boldsymbol{x}, \boldsymbol{\theta}) = U_L(\boldsymbol{\theta}_L) \cdots U_2(\boldsymbol{\theta}_2) \cdot E(\boldsymbol{x}) \cdot U_1(\boldsymbol{\theta}_1)$$

where:
- $E(\boldsymbol{x})$ = **encoding layer** (embeds classical data into quantum state)
- $U_l(\boldsymbol{\theta}_l)$ = **variational layers** (trainable rotations + entanglers)

The VQC output $f(\boldsymbol{x}, \boldsymbol{\theta}) = \langle\psi(\boldsymbol{x},\boldsymbol{\theta})|O|\psi(\boldsymbol{x},\boldsymbol{\theta})\rangle$ is differentiable with respect to $\boldsymbol{\theta}$ via the **parameter-shift rule**.

### 3.1 Parameter-Shift Rule

For a gate $G(\theta) = e^{-i\theta H/2}$ where $H^2 = I$, the gradient is:

$$\frac{\partial \langle O \rangle}{\partial \theta} = \frac{1}{2}\left[\langle O \rangle_{\theta + \pi/2} - \langle O \rangle_{\theta - \pi/2}\right]$$

This is **exact** (not an approximation like finite differences) and allows standard backpropagation through quantum circuits.

### 3.2 The Barren Plateau Problem

As circuit depth $L$ and qubit count $n$ increase, gradients vanish exponentially:

$$\text{Var}\left[\frac{\partial \mathcal{L}}{\partial \theta_k}\right] \leq O(2^{-n})$$

This "barren plateau" problem is why we use **only 8 qubits and 2 variational layers** — deep enough to be expressive, shallow enough to avoid vanishing gradients.

---

## 4. PennyLane & Quantum-Classical Interface

**PennyLane** (Bergholm et al., 2018) is the quantum ML library we use. It provides:

1. **`qml.device`** — simulator backend (`default.qubit` simulates exact quantum evolution)
2. **`qml.QNode`** — wraps a quantum function into a differentiable node
3. **`qml.qnn.TorchLayer`** — makes the QNode act as a PyTorch `nn.Module`

```python
dev = qml.device('default.qubit', wires=8)

@qml.qnode(dev, interface='torch')
def circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(8))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(8))
    return [qml.expval(qml.PauliZ(i)) for i in range(8)]

qlayer = qml.qnn.TorchLayer(circuit, {'weights': (2, 8, 3)})
```

The `TorchLayer` integrates seamlessly with PyTorch autograd — gradients flow **backward through the quantum circuit** using the parameter-shift rule, enabling end-to-end training.

---

## 5. Zero-DCE: Low-Light Image Enhancement

### 5.1 Problem Statement

Night-time images suffer from:
- **Low brightness** — pixel values clustered near 0
- **High noise** — random fluctuations dominate signal
- **Color distortion** — artificial lighting shifts color balance
- **Loss of detail** — shadows suppress texture information

### 5.2 Zero-DCE Architecture

Zero-Reference Deep Curve Estimation (Zero-DCE, Guo et al. 2020) learns **pixel-wise curve functions** to enhance each channel independently, without requiring paired normal/dark image training data.

The network learns **illumination adjustment curves** of the form:

$$\hat{x} = x + A \odot (x^2 - x)$$

where:
- $x \in [0,1]^{H \times W \times 3}$ = input low-light image
- $A \in [-1,1]^{H \times W \times 24}$ = learned curve parameter map (8 curves × 3 channels)
- $\hat{x}$ = enhanced image

Applied iteratively ($n=8$ iterations):

$$\hat{x}^{(n)} = \hat{x}^{(n-1)} + A_n \odot \left[\left(\hat{x}^{(n-1)}\right)^2 - \hat{x}^{(n-1)}\right]$$

### 5.3 Our Lightweight Implementation

```
Input (3, H, W)
    → Conv2d(3, 16, 3, padding=1) → ReLU
    → Conv2d(16, 16, 3, padding=1) → ReLU
    → Conv2d(16, 24, 3, padding=1) → Tanh      [output: A_maps, 8 curves × 3 channels]
```

The 24-channel output provides 8 curve maps ($A_1, ..., A_8$) for 8 iterative enhancement steps.

### 5.4 Why Tanh for Curves?

The Tanh activation constrains $A \in (-1, +1)$. Substituting into the curve equation:
- $A > 0$ → **brightens** the image
- $A < 0$ → **darkens** (useful for over-exposed regions)
- $A = 0$ → identity (no change)

This gives the network the ability to selectively adjust any region of the image.

---

## 6. CNN Feature Extraction

### 6.1 Architecture

```
Input: (B, 3, 64, 256)  — batch of enhanced images
  → Conv2d(3,  64, 3, padding=1) → MaxPool2d(2) → ReLU   → (B, 64, 32, 128)
  → Conv2d(64, 128, 3, padding=1) → MaxPool2d(2) → ReLU  → (B, 128, 16, 64)
  → Conv2d(128, 8, 1)                                     → (B, 8, 16, 64)
Output: (B, 8, 16, 64)
```

### 6.2 The 1×1 Bottleneck

The final `Conv2d(128, 8, 1)` is a **1×1 convolution** — it projects the 128-channel feature map into exactly 8 channels, matching `N_QUBITS = 8`.

**Why 8?** The quantum layer accepts an 8-dimensional input vector (one value per qubit). This design choice forces the CNN to distil all spatial information into exactly the right dimensionality for the quantum circuit.

### 6.3 Height Pooling

After the CNN, we pool over the height dimension:

```python
x = x.mean(dim=2)         # (B, 8, 64) — average over spatial height
x = x.permute(0, 2, 1)   # (B, 64, 8) — each of 64 column positions has 8 features
```

This yields a **sequence of 64 feature vectors**, each 8-dimensional — one vector per column position in the image. These 64 vectors are fed sequentially into the quantum circuit.

---

## 7. The 8-Qubit Quantum Layer

### 7.1 Input Encoding: AngleEmbedding

```python
qml.templates.AngleEmbedding(inputs, wires=range(8))
```

AngleEmbedding encodes each classical feature $x_i \in \mathbb{R}$ as a rotation:

$$\text{AngleEmbedding}(\boldsymbol{x}) = \prod_{i=0}^{7} R_X(x_i)$$

where $R_X(\theta) = e^{-i\theta X/2}$ rotates qubit $i$ by angle $x_i$ around the X-axis.

This maps the 8-dimensional CNN feature vector to an initial quantum state:

$$|\psi_0\rangle = \bigotimes_{i=0}^{7} R_X(x_i)|0\rangle$$

### 7.2 Variational Transformation: StronglyEntanglingLayers

```python
qml.templates.StronglyEntanglingLayers(weights, wires=range(8))
# weights shape: (n_layers=2, n_qubits=8, n_params_per_qubit=3)
```

Each layer applies:
1. **Arbitrary rotation** on each qubit: $R_Z(\theta_3) R_Y(\theta_2) R_Z(\theta_1)$ — 3 parameters per qubit
2. **CNOT entangling gates** in a "strongly entangling" pattern that ensures all pairs of qubits become correlated

The "strongly entangling" pattern uses CNOT gates connecting qubits at distances $1, 2, 3, ...$ so that after 2 layers, **every qubit has been entangled with every other qubit** at least once.

Total trainable parameters in quantum layer:
$$N_{\text{quantum params}} = N_{\text{layers}} \times N_{\text{qubits}} \times 3 = 2 \times 8 \times 3 = 48$$

### 7.3 Measurement

```python
return [qml.expval(qml.PauliZ(i)) for i in range(8)]
```

Output: $\boldsymbol{y} = [\langle Z_0 \rangle, ..., \langle Z_7 \rangle] \in [-1,1]^8$

This 8-dimensional vector is the **quantum representation** of the input CNN feature vector — the same dimensionality as input, but transformed through the 256-dimensional Hilbert space.

### 7.4 Processing Each Column

For input `x` of shape `(B, 64, 8)`:

```python
x_flat = x.reshape(-1, 8)            # (B*64, 8) — all column vectors
q_out  = quantum_layer(x_flat)       # (B*64, 8) — quantum transform applied
q_seq  = q_out.reshape(B, 64, 8)     # (B, 64, 8) — restore sequence structure
```

The quantum circuit processes each column independently but all columns share the same trained weights $\boldsymbol{\theta}$.

---

## 8. Bidirectional LSTM Sequence Decoder

### 8.1 Why LSTM for License Plates?

License plates are **variable-length sequences** (e.g., "MH12DE1234" = 10 chars, "KA01X" = 5 chars). Standard classifiers assume fixed output length. LSTM handles this naturally by processing the sequential quantum features.

**Bidirectional LSTM** runs the sequence both left-to-right and right-to-left, so each position's prediction considers both its left context and right context simultaneously — important because plate characters can disambiguate each other (e.g., "0" vs "O" depends on surrounding characters).

### 8.2 Architecture

```python
nn.LSTM(input_size=8, hidden_size=128, bidirectional=True, batch_first=True)
```

- Input: `(B, 64, 8)` — sequence of 64 quantum feature vectors
- Hidden state: 128 units per direction
- Output: `(B, 64, 256)` — 64 positions × 256 features (128 × 2 directions)

Then projected to character probabilities:
```python
nn.Linear(256, 37)   # 37 classes: blank + 0-9 + A-Z
```

Output: `(B, 64, 37)` → permuted to `(64, B, 37)` = `(T, B, C)` for CTC loss.

### 8.3 LSTM Cell Equations

At each timestep $t$:

$$\boldsymbol{f}_t = \sigma(\boldsymbol{W}_f [\boldsymbol{h}_{t-1}, \boldsymbol{x}_t] + \boldsymbol{b}_f) \quad \text{(forget gate)}$$
$$\boldsymbol{i}_t = \sigma(\boldsymbol{W}_i [\boldsymbol{h}_{t-1}, \boldsymbol{x}_t] + \boldsymbol{b}_i) \quad \text{(input gate)}$$
$$\tilde{\boldsymbol{c}}_t = \tanh(\boldsymbol{W}_c [\boldsymbol{h}_{t-1}, \boldsymbol{x}_t] + \boldsymbol{b}_c) \quad \text{(candidate cell)}$$
$$\boldsymbol{c}_t = \boldsymbol{f}_t \odot \boldsymbol{c}_{t-1} + \boldsymbol{i}_t \odot \tilde{\boldsymbol{c}}_t \quad \text{(cell state)}$$
$$\boldsymbol{o}_t = \sigma(\boldsymbol{W}_o [\boldsymbol{h}_{t-1}, \boldsymbol{x}_t] + \boldsymbol{b}_o) \quad \text{(output gate)}$$
$$\boldsymbol{h}_t = \boldsymbol{o}_t \odot \tanh(\boldsymbol{c}_t) \quad \text{(hidden state)}$$

---

## 9. CTC Loss — Training Without Segmentation

### 9.1 The Core Problem

To train a character classifier, you normally need to know **which part of the image corresponds to which character** (segmentation). For license plates, this is expensive to annotate.

**CTC (Connectionist Temporal Classification)** solves this by computing the probability of the target sequence **summed over all possible alignments**:

$$P(\boldsymbol{l}|\boldsymbol{x}) = \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\boldsymbol{l})} P(\boldsymbol{\pi}|\boldsymbol{x})$$

where $\mathcal{B}$ is the **collapse function** that:
1. Merges repeated consecutive characters
2. Removes blank tokens

### 9.2 CTC Paths Example

For target "CAT" with 5 time steps:
```
Valid CTC paths: C-CAT, CA-T, CAT-, _CAT, C_AT, CA_T, ...
(− = blank, _ = blank)
```

All paths that collapse to "CAT" are valid. CTC maximizes their total probability.

### 9.3 Forward-Backward Algorithm

CTC uses a dynamic programming algorithm analogous to HMM forward-backward:

$$\alpha_t(s) = P(\boldsymbol{\pi}_{1:t} : \text{prefix collapses to } \boldsymbol{l}'_{1:s})$$

The final loss:

$$\mathcal{L}_{\text{CTC}} = -\log P(\boldsymbol{l}|\boldsymbol{x}) = -\log \sum_{\boldsymbol{\pi} \in \mathcal{B}^{-1}(\boldsymbol{l})} \prod_{t=1}^{T} p_{\pi_t}^t$$

### 9.4 CTC Decoding (Inference)

**Greedy decoding** (used in evaluation):
1. At each timestep, take argmax over 37 classes
2. Apply collapse function $\mathcal{B}$

```python
preds = preds.argmax(dim=2)              # (T, B)
for b in range(B):
    chars = []
    prev = -1
    for t in range(T):
        c = preds[t, b].item()
        if c != 0 and c != prev:         # not blank, not repeat
            chars.append(IDX2CHAR[c])
        prev = c
    decoded.append(''.join(chars))
```

---

## 10. Classical Baseline Architecture

The classical model matches the quantum model in spirit but replaces the quantum layer with additional classical processing:

```
Input → ZeroDCE → CNN (3→64→128) → Conv2d(128, 64, 1) → [no quantum]
     → Bi-LSTM(64, 128) → Linear(256, 37) → CTC
```

Design principles for a fair comparison:
- **Same data** — identical train/val/test splits (`test_indices.json`)
- **Same training loop** — same optimizer, LR schedule, batch size
- **Same evaluation** — same CER/WER computation
- **Parameter parity** — classical CNN widened to compensate for the 48 quantum parameters

---

## 11. Training Strategy & Optimization

### 11.1 Optimizer: Adam

$$\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + (1-\beta_1)\boldsymbol{g}_t$$
$$\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2)\boldsymbol{g}_t^2$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon}\hat{\boldsymbol{m}}_t$$

Default: $\beta_1=0.9$, $\beta_2=0.999$, $\eta=10^{-3}$

### 11.2 Learning Rate Schedule: Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)$$

Gradually reduces LR from $10^{-3}$ to $\approx 0$ over 100 epochs — allows large steps early and fine-tuning late.

### 11.3 Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

Prevents gradient explosion common in RNN training — clips the global gradient norm to 5.0.

### 11.4 Checkpoint Strategy

After every epoch:
- `latest.pth` → always overwritten (for resume)
- `best.pth` → only updated when val CER improves
- `history.json` → all metrics appended

**Early stopping**: if validation CER does not improve for 10 consecutive epochs, training halts.

---

## 12. Evaluation Metrics

### 12.1 Character Error Rate (CER)

$$\text{CER} = \frac{\text{edit\_distance}(\hat{y}, y)}{|\hat{y}|}$$

where edit distance = minimum Levenshtein operations (insert, delete, substitute) to transform prediction $\hat{y}$ into ground truth $y$.

**Lower is better.** CER = 0.0 means perfect character-level accuracy.

### 12.2 Word Error Rate (WER)

$$\text{WER} = \frac{\text{incorrect plates}}{\text{total plates}}$$

A plate is "correct" only if **every character** matches exactly. WER is harsher than CER — even one wrong character in a 7-character plate counts as a full error.

**Lower is better.** WER = 1.0 means no plate was fully correct.

### 12.3 Inference Speed

Measured as **plates per second** on the test set:

$$\text{Throughput} = \frac{N_{\text{test}}}{\text{elapsed seconds}}$$

The quantum simulation adds overhead vs. classical — this metric quantifies the speed trade-off.

---

## 13. Dataset: RodoSol-ALPR

- **Source:** Brazilian roadside license plate dataset (Laroca et al., 2022)
- **Content:** High-resolution images of vehicle fronts with visible license plates
- **Plates:** Brazilian 7-character format (e.g., `ABC1234`, `ABC1D23`)
- **Conditions:** Varied lighting including night-time captures
- **Split used:**
  - Train: 80% (stratified)
  - Val: 10%
  - Test: 10% (fixed via `test_indices.json` for reproducibility)

---

## 14. References

1. **PennyLane**: Bergholm et al. (2018). *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* arXiv:1811.04968
2. **Zero-DCE**: Guo, C. et al. (2020). *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement.* CVPR 2020.
3. **CTC**: Graves, A. et al. (2006). *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks.* ICML 2006.
4. **Barren Plateaus**: McClean et al. (2018). *Barren plateaus in quantum neural network training landscapes.* Nature Communications.
5. **StronglyEntanglingLayers**: Schuld et al. (2020). *Circuit-centric quantum classifiers.* Physical Review A.
6. **RodoSol-ALPR**: Laroca, R. et al. (2022). *A Robust Real-Time Automatic License Plate Recognition.* IEEE Transactions on Intelligent Transportation Systems.
7. **LSTM**: Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation.
8. **Adam**: Kingma & Ba (2015). *Adam: A Method for Stochastic Optimization.* ICLR 2015.
