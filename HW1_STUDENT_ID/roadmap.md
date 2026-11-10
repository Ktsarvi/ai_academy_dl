# Homework 1: Neural-Network Foundations in PyTorch
## Master Roadmap & Progress Tracker

**Course**: DLE-AI-202 Deep Learning (Weeks 1–2)  
**Author**: Alasgarov Teymur  
**File**: `roadmap.md`  
**Current Date**: November 2026  

---

## 1. Executive Summary & Grading Breakdown

| Component | Target Points | Current Status | Primary Artifact / Source |
| :--- | :---: | :---: | :--- |
| **Part A: Written Problems** | 30 / 30 | 🟡 **Drafted** | Solutions ready in [`part_a_solutions.md`](./part_a_solutions.md) |
| **Part B: Code Implementation** | 45 / 45 | 🟢 **100% Complete** | Modules in `src/` passing **36/36 tests** |
| **Part E: Notebook Experiments** | 15 / 15 | ⚪ **Pending Run** | Code cells in [`hw1_notebook.ipynb`](./hw1_notebook.ipynb) |
| **Part C: Reflection & Disclosure** | 10 / 10 | 🟡 **70% Drafted** | C1 & C2 ready in [`part_c_solutions.md`](./part_c_solutions.md) |
| **Bonus Tasks (Adam & MixUp)** | (+10) / 10 | 🟡 **80% Complete** | Code done in [`src/bonus.py`](./src/bonus.py), need E3 run |
| **Total Base Points** | **100 / 100** | — | — |

---

## 2. Component Progress Checklist

### [Part A] Written Problems (30 pts)
- [x] **A1: Shapes and parameter count (3 pts)** — Shapes, weights, biases, FLOPs/MACs.
- [x] **A2: Losses from likelihood (4 pts)** — Gaussian NLL to MSE, stable BCE identity, gradient $\hat{y}-y$.
- [x] **A3: Backpropagation by hand (6 pts)** — Computational graph, numerical pass, $\boldsymbol\delta^{(2)}$, $\boldsymbol\delta^{(1)}$, dying ReLU.
- [x] **A4: Activations & vanishing gradients (4 pts)** — $\sigma'$ maximum, $0.25^L < 10^{-6}$ depth, Swish & GELU derivatives.
- [x] **A5: Initialisation (4 pts)** — He variance derivation, He/Xavier std for $784\to256$, zero-weight symmetry, initial loss $\approx \ln K$.
- [x] **A6: Optimisers (4 pts)** — SGD vs. Momentum steps on $5w^2$, Adam step-size with bias correction, decoupled weight decay in AdamW.
- [x] **A7: Regularisation (3 pts)** — Weight decay as $(1-\eta\lambda)$ shrinkage, inverted dropout expectation, BatchNorm train vs. test.
- [x] **A8: Debugging reasoning (2 pts)** — Causes and first diagnostic checks for 4 common training failure modes.
- [ ] *Action Required:* Transfer LaTeX blocks from `part_a_solutions.md` into `report_template.tex`.

---

### [Part B] Code Implementation (45 pts)
- [x] **B1: Tensor Basics ([`src/tensor_basics.py`](./src/tensor_basics.py)) (10 pts)**
  - `standardize`, `one_hot`, `pairwise_sq_dists` (no loops), stable `softmax`, `log_softmax`, `cross_entropy`, stable `bce_with_logits`, `mse`.
  - *Status:* Passed all 7 tests.
- [x] **B2: Manual Backprop ([`src/manual_mlp.py`](./src/manual_mlp.py)) (10 pts)**
  - `act_derivative`, `forward`, analytical `backward` (no `.backward()`), `numerical_grad`, `SwishFn` autograd function.
  - *Status:* Passed all 7 tests.
- [x] **B3: MLP Module ([`src/model.py`](./src/model.py)) (8 pts)**
  - `MLP(nn.Module)` with configurable activations, norm, dropout; `init_weights` (He, Xavier, normal, zeros); `count_params`.
  - *Status:* Passed all 9 tests.
- [x] **B4: Training Utilities ([`src/train.py`](./src/train.py)) (12 pts)**
  - `warmup_cosine`, `accuracy_from_logits`, `train_one_epoch` with clipping, `evaluate`, `fit` with early stopping and best-weights restore.
  - *Status:* Passed all 7 tests.
- [x] **B5: Diagnostics ([`src/diagnostics.py`](./src/diagnostics.py)) (5 pts)**
  - `layer_stats` forward hooks, `dead_relu_fraction`, `overfit_one_batch` sanity check.
  - *Status:* Passed all 4 tests.
- [x] **Bonus Code ([`src/bonus.py`](./src/bonus.py)) (+10 pts)**
  - `MyAdam` matching `torch.optim.Adam` to $10^{-6}$, `mixup_batch` data augmentation.
  - *Status:* Passed all 2 tests.
- [ ] *Action Required:* Paste terminal `pytest` summary line into `report_template.tex`.

---

### [Part E] Notebook Experiments (15 pts)
*File to run:* [`hw1_notebook.ipynb`](./hw1_notebook.ipynb) (Seed set from student ID).

- [ ] **E1: Depth and Initialisation (4 pts)**
  - Build 10-layer MLP four ways: Sigmoid+Xavier, ReLU+tiny normal, ReLU+Xavier, ReLU+He.
  - Measure per-layer pre-activation std and weight gradient norm via `layer_stats`.
  - Train 5 epochs with SGD; observe failure modes (vanishing gradient vs. variance explosion).
  - Draft 3–5 sentence analysis for report citing exact numbers.
- [ ] **E2: Optimisers on Same Problem (3 pts)**
  - Compare SGD, SGD+Momentum, RMSprop, Adam, AdamW on identical model and seed.
  - Plot and compare validation-loss curves.
  - Explain why Adam converges faster and why a single fixed learning rate cannot fair-rank optimisers.
- [ ] **E3: Regularisation on Noisy Dataset (4 pts)**
  - 300 training images from digits dataset with 25% randomly corrupted labels.
  - Benchmark: Baseline, Dropout ($p=0.3$), BatchNorm, Weight Decay ($\lambda=10^{-3}$), Early Stopping, and Combination.
  - Add optional **MixUp** bonus row (+4 pts).
  - Summarize results in comparison table and write 3–5 sentence mechanism explanation.
- [ ] **E4: Final Model (4 pts)**
  - Tune hyper-parameters using validation set only (target: Test Accuracy $\ge 0.95$).
  - Train with Warmup + Cosine scheduler and gradient clipping (`clip_norm=1.0`).
  - Single final evaluation on test set.
  - Automatically generate and save [`figures/final_curves.png`](./figures/final_curves.png).

---

### [Part C] Reflection & AI Disclosure (10 pts)
- [x] **C1: Trace a Gradient (4 pts)**
  - Detailed line-by-line derivation of $W^{(1)}_{j,k}$ in `manual_mlp.backward` and activation derivative entry point. Drafted in `part_c_solutions.md`.
- [x] **C2: Silent Failure Lines (3 pts)**
  - Concrete breakdown of why removing `optimizer.zero_grad(set_to_none=True)` and `model.eval()` silently corrupts model training. Drafted in `part_c_solutions.md`.
- [ ] **C3: Experimental Surprise & AI Disclosure (3 pts)**
  - Note one unexpected experimental finding from Part E.
  - Add formal AI assistant disclosure statement.

---

### [Bonus] Optional Tasks (up to +10 pts)
- [x] **MyAdam (+6 pts)**: Custom `torch.optim.Optimizer` implemented and verified against standard PyTorch Adam.
- [ ] **MixUp in E3 (+4 pts)**: Train with `mixup_batch` in the corrupted label experiment and record accuracy in the E3 table.
- [ ] *Action Required:* Write 2–3 sentences in `report_template.tex` summarizing bonus findings.

---

## 3. Execution & Assembly Workflow

```
[Phase 1] Code & Theory (DONE)
   ├── Part A solutions drafted in part_a_solutions.md
   ├── Part B 36/36 tests passing in src/
   └── Part C1 & C2 drafted in part_c_solutions.md

[Phase 2] Experiments in hw1_notebook.ipynb (CURRENT STEP)
   ├── Set STUDENT_ID seed
   ├── Execute E1 (Depth & Init)
   ├── Execute E2 (Optimisers)
   ├── Execute E3 (Regularisation & MixUp)
   ├── Execute E4 (Final Model & save figures/final_curves.png)
   └── Run "Restart & Run All" to save notebook with outputs

[Phase 3] Report Compilation (report_template.tex)
   ├── Populate Author header & Student ID
   ├── Insert Part A LaTeX blocks (A1–A8)
   ├── Insert Part B pytest status string
   ├── Insert Part E experimental numbers, table, and embedded plot
   ├── Insert Part C reflection & AI tool disclosure
   ├── Insert Bonus remarks
   └── Compile: pdflatex report_template.tex -> report.pdf

[Phase 4] Packaging & Verification (§submit)
   ├── Fill and sign pledge.txt
   └── Create HW1_<STUDENT_ID>.zip containing:
         ├── report.pdf
         ├── src/
         ├── hw1_notebook.ipynb (with outputs)
         ├── figures/final_curves.png
         └── pledge.txt
```

---

## 4. Quick Command Reference

```powershell
# 1. Ensure fast single-thread execution
$env:DL_HW_THREADS = "1"

# 2. Run all tests
pytest tests -v

# 3. Compile LaTeX report
pdflatex report_template.tex
pdflatex report_template.tex   # Run twice for cross-references
```

