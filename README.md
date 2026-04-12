# ⚡ Quantum-Enhanced Deep Learning for Night-Time License Plate Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PennyLane](https://img.shields.io/badge/PennyLane-0.38-purple?logo=data:image/svg+xml;base64,PHN2Zy8+)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Kaggle](https://img.shields.io/badge/Training-Kaggle%20T4%20x2-20BEFF?logo=kaggle)
![HuggingFace](https://img.shields.io/badge/Storage-HuggingFace%20Hub-yellow?logo=huggingface)
![Course](https://img.shields.io/badge/Course-CSE4019%20AoAI-green)

**A hybrid quantum-classical deep learning system that reads license plates in the dark.**

[Theory Deep-Dive →](THEORY.md) · [Results & Proof →](RESULTS.md) · [HF Dataset](https://huggingface.co/datasets/Shanmuk4622/quantum-lpr-dataset) · [HF Model](https://huggingface.co/Shanmuk4622/quantum-lpr-checkpoints)

</div>

---

## 🔭 What This Project Does

Most license plate recognition (LPR) systems fail after dark — low-light images produce noise, loss of contrast, and blurred characters that confuse standard neural networks.

This project asks a research question:

> **Can a quantum-enhanced neural network read license plates in night conditions more accurately than a purely classical system?**

We answer this by building **two parallel systems** and running them head-to-head:

| System | Architecture | Training Platform |
|---|---|---|
| `HybridLPRNet_8Q` | ZeroDCE + CNN + **8-Qubit Quantum Circuit** + Bi-LSTM + CTC | Kaggle T4 x2 |
| `ClassicalLPRNet` | ZeroDCE + CNN (larger) + Bi-LSTM + CTC | Kaggle T4 x2 |

The classical model is a **fair, parameter-matched baseline** — same data, same training loop, same evaluation. Any performance gap is attributable to the quantum layer.

---

## 🗺️ Project Structure

```
Project AOAI/
│
├── README.md                    ← You are here (overview)
├── THEORY.md                    ← Deep theory: quantum circuits, ZeroDCE, CTC loss
├── RESULTS.md                   ← Training metrics, plots, evaluation results
│
├── Phase-1/                     ← Foundation work
│   ├── Quantum_Enhanced_Deep_Learning_.ipynb   ← Proof-of-concept (Colab)
│   ├── AoAI Quantam, Report 1.pdf              ← Phase 1 submission
│   ├── 8qubit_model.pth                        ← Trained quantum checkpoint
│   └── images/                                 ← Report figures
│
└── Final Phase/                 ← Production training pipeline (Kaggle)
    ├── 00_Setup_HuggingFace.ipynb    ← One-time: migrate data to HF Hub
    ├── 01_Complete_Training_Kaggle.ipynb   ← Main training (Quantum + Classical)
    ├── 02_Evaluation_Suite_Kaggle.ipynb    ← CER / WER / Speed benchmarks
    ├── 03_Visualizations_Kaggle.ipynb      ← Qubit heatmaps, training curves
    └── 04_Final_Demo_Kaggle.ipynb          ← End-to-end demo for presentation
```

---

## 🚀 Quick Start

### Prerequisites
- Google account (for Colab) or Kaggle account
- HuggingFace account with write access

### Step 1 — One-time Setup (Google Colab)
```
Open Final Phase/00_Setup_HuggingFace.ipynb in Colab
Run Phase 1 → organizes data in Google Drive
Verify the folder structure
Run Phase 2 → pushes everything to HuggingFace Hub
```

### Step 2 — Train on Kaggle
```
Upload Final Phase/01_Complete_Training_Kaggle.ipynb to Kaggle
Accelerator: GPU T4 x2
Add Secrets: HF_TOKEN_1, HF_TOKEN_2
Run all cells
→ Downloads dataset from HF, trains both models, saves checkpoints after every epoch
```

### Step 3 — Evaluate
```
Run 02_Evaluation_Suite_Kaggle.ipynb
→ Generates CER, WER, inference speed, parameter count comparison
```

### Step 4 — Demo
```
Run 04_Final_Demo_Kaggle.ipynb
→ End-to-end: input a night image → get predicted plate string + comparison table
```

---

## 🏗️ Architecture at a Glance

```
Night Image (64×256)
       │
       ▼
┌─────────────┐
│   ZeroDCE   │  ← Low-light enhancement (8 curve iterations)
│  Enhancer   │
└──────┬──────┘
       │ Enhanced Image
       ▼
┌─────────────┐
│  CNN Stem   │  ← Conv2d(3→64→128→8) — compresses to 8 channels
│  (3 layers) │     matching N_QUBITS
└──────┬──────┘
       │ Feature Map [B, 8, H, W]  →  flatten H  →  [B, W, 8]
       ▼
┌─────────────┐
│  8-Qubit    │  ← AngleEmbedding + StronglyEntanglingLayers (2 layers)
│  Circuit    │     Outputs: ⟨Z₀⟩, ⟨Z₁⟩, ..., ⟨Z₇⟩  (expectation values)
│  (PennyLane)│
└──────┬──────┘
       │ Quantum Features [B, W, 8]
       ▼
┌─────────────┐
│  Bi-LSTM    │  ← 128 units × 2 directions = 256-dim output
│  Decoder    │
└──────┬──────┘
       │ Sequence Logits [T, B, 37]   (37 = 0-9 + A-Z + blank)
       ▼
┌─────────────┐
│  CTC Loss   │  ← Handles variable-length plate strings without segmentation
│  / Decode   │
└─────────────┘
       │
       ▼
  "MH12DE1234"   ← Predicted plate string
```

---

## 📊 Key Metrics (Training → See RESULTS.md for full data)

| Metric | Quantum Model | Classical Model |
|---|---|---|
| Architecture | HybridLPRNet_8Q | ClassicalLPRNet |
| Parameters | ~1.2M | ~1.8M |
| Training Platform | Kaggle T4 x2 | Kaggle T4 x2 |
| Persistent Storage | HuggingFace Hub | HuggingFace Hub |
| Best Val CER | *updating...* | *updating...* |

> Full results available in [RESULTS.md](RESULTS.md) once training completes.

---

## 🧰 Technology Stack

| Component | Technology |
|---|---|
| Quantum Circuit | PennyLane 0.38 + `default.qubit` |
| Deep Learning | PyTorch 2.x |
| Low-light Enhancement | Zero-DCE (custom implementation) |
| Sequence Decoding | CTC Loss + greedy decode |
| Training | Kaggle Notebooks (GPU T4 x2) |
| Persistent Storage | HuggingFace Hub (dataset + checkpoints) |
| Dataset | RodoSol-ALPR (Brazilian LP dataset) |

---

## 👤 Author

**Shanmukha Srinivas** — VIT-AP University  
Course: CSE4019 — Applications of AI (AoAI)  
Semester: Winter 2025–26

---

## 📚 References

See [THEORY.md](THEORY.md) for full citations and mathematical derivations.
