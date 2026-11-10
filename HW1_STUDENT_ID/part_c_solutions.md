# Part C: Reflection & Critical Analysis
**Course**: DLE-AI-202 Deep Learning  
**Author**: Alasgarov Teymur  
**File**: `part_c_solutions.md`

This document contains written reflections for **C1** and **C2** based on your implementations in `manual_mlp.py` and `train.py`, formatted with complete explanations and ready-to-copy LaTeX code for `report_template.tex`.

---

## C1: Trace One Weight's Gradient in `manual_mlp.backward` (4 pts)

### Prompt
> Trace one weight's gradient through your `manual_mlp.backward`, line by line, and say where the activation derivative enters. (3–5 sentences, referring to your own code).

### Detailed Explanation & Line-by-Line Trace
Consider the gradient with respect to weight $W^{(1)}_{j, k}$ connecting input feature $j$ to hidden unit $k$. In our implementation in `src/manual_mlp.py`:

1. **Output Error Vector $\boldsymbol\delta^{(2)}$**:
   ```python
   delta2 = (probs - Y) / n  # shape (n, K)
   ```
   For each sample $i$ and class $c$, $\delta_{i, c}^{(2)} = \frac{\partial L}{\partial z_{i, c}^{(2)}} = \frac{1}{n}(p_{i, c} - Y_{i, c})$. This analytically evaluates the combined derivative of the mean cross-entropy loss and softmax layer.

2. **Hidden Layer Pre-activation Error $\boldsymbol\delta^{(1)}$ & Activation Derivative Entry**:
   ```python
   delta1 = (delta2 @ params["W2"].T) * act_derivative(cache["z1"], cache["a1"], act)  # shape (n, h)
   ```
   By the chain rule, the error propagated back from layer 2 is:
   $$\delta_{i, k}^{(1)} = \frac{\partial L}{\partial z_{i, k}^{(1)}} = \left( \sum_{c=1}^K \delta_{i, c}^{(2)} W^{(2)}_{k, c} \right) \cdot f'(z_{i, k}^{(1)})$$
   **Where the activation derivative enters**: It enters exactly at the Hadamard (elementwise) product with `act_derivative(cache["z1"], cache["a1"], act)`. For $\tanh$, it multiplies by $(1 - a_{1}^2)$, and for ReLU, by $\mathbf{1}_{z_1 > 0}$. If a unit is deactivated (e.g., $z_1 \le 0$ under ReLU), this gate sets $\delta_{i, k}^{(1)} = 0$, completely halting gradient flow for that unit.

3. **Weight Gradient Accumulation $gW^{(1)}$**:
   ```python
   gW1 = cache["X"].T @ delta1  # shape (d, h)
   ```
   The gradient for scalar weight $W^{(1)}_{j, k}$ is the dot product of feature $j$ across all samples with their respective hidden errors:
   $$\frac{\partial L}{\partial W^{(1)}_{j, k}} = \sum_{i=1}^n X_{i, j} \delta_{i, k}^{(1)} = (X^T \boldsymbol\delta^{(1)})_{j, k}$$

### Ready-to-Copy LaTeX for Report (C1)
```latex
To trace the gradient of weight $W^{(1)}_{j,k}$ connecting input $j$ to hidden unit $k$, we follow \texttt{manual\_mlp.backward}:
first, \texttt{delta2 = (probs - Y) / n} computes $\delta^{(2)}_{i,c} = \frac{1}{n}(p_{i,c} - Y_{i,c}) = \partial L / \partial z^{(2)}_{i,c}$ combining softmax and mean cross-entropy.
Next, \texttt{delta1 = (delta2 @ W2.T) * act\_derivative(z1, a1, act)} backpropagates this error to the hidden layer:
here, the activation derivative enters as an elementwise multiplier $f'(z^{(1)}_{i,k})$ ($1 - a_1^2$ for tanh, $\mathbf{1}_{z_1>0}$ for ReLU), gating the incoming error $\sum_c \delta^{(2)}_{i,c} W^{(2)}_{k,c}$.
Finally, \texttt{gW1 = X.T @ delta1} computes $\partial L / \partial W^{(1)}_{j,k} = \sum_{i=1}^n X_{i,j} \delta^{(1)}_{i,k}$, accumulating the product of pre-synaptic input $X_{i,j}$ and post-synaptic error $\delta^{(1)}_{i,k}$ over the batch.
```

---

## C2: Lines That Must Not Be Removed (3 pts)

### Prompt
> Which two lines of your training or evaluation code would silently break things if removed (e.g. `zero_grad`, `model.eval()`, `no_grad`)? What exactly goes wrong? (3–5 sentences, referring to your own code).

### Detailed Explanation of Silent Failures

1. **Line 1: `optimizer.zero_grad(set_to_none=True)` in `train_one_epoch` ([`train.py`](file:///d:/.vscode/ai_academy_dl/HW1_STUDENT_ID/src/train.py))**:
   - **What happens:** In PyTorch, `.backward()` by design *accumulates* (adds) gradients into existing `.grad` buffers rather than overwriting them.
   - **Silent failure:** Without `zero_grad()`, gradients from successive mini-batches accumulate cumulatively: at batch $t$, $g_t = \sum_{s=1}^t \nabla L_s$. The step magnitude grows linearly with the number of steps, effectively multiplying the learning rate by the number of processed batches. The optimisation diverges violently into `NaN` or oscillatory chaos, without throwing any Python runtime errors.

2. **Line 2: `model.eval()` in `evaluate` ([`train.py`](file:///d:/.vscode/ai_academy_dl/HW1_STUDENT_ID/src/train.py))**:
   - **What happens:** Modules such as `nn.Dropout` and `nn.BatchNorm1d` behave fundamentally differently between training and evaluation modes.
   - **Silent failure:** Omitting `model.eval()` leaves the network in training mode (`model.training == True`). During evaluation, Dropout continues stochastically zeroing out units with probability $p$ and scaling remaining activations by $\frac{1}{1-p}$, making test predictions non-deterministic and degrading validation accuracy. Simultaneously, BatchNorm updates its running population statistics ($\mu, \sigma^2$) using the evaluation batch, corrupting internal tracking and leaking evaluation data.

*(Note on `torch.no_grad()`: If `@torch.no_grad()` were omitted in `evaluate`, the code would execute correctly and produce the same numbers, but it would build a redundant dynamic computation graph for every validation batch, quietly consuming GPU/RAM memory until potentially triggering an Out-Of-Memory error on large validation sets).*

### Ready-to-Copy LaTeX for Report (C2)
```latex
Two critical lines whose removal silently breaks behaviour are:
\textbf{(1) \texttt{optimizer.zero\_grad(set\_to\_none=True)}} in \texttt{train\_one\_epoch}: because PyTorch buffers accumulate gradients additively on \texttt{backward()}, omitting this causes gradients from previous batches to compound indefinitely ($\mathbf{g}_t = \sum_{k=1}^t \nabla L_k$), causing exploding gradient steps and immediate divergence or NaN loss without raising an exception.
\textbf{(2) \texttt{model.eval()}} in \texttt{evaluate}: leaving the model in training mode keeps \texttt{Dropout} actively zeroing random activations during evaluation (producing noisy, degraded validation metrics) and allows \texttt{BatchNorm1d} to update its running mean and variance on validation batches, invalidating metric reproducibility and leaking evaluation data.
```

---

## C3: Experimental Surprise & AI-Tool Disclosure (3 pts)

*(To be finalized after running Part E in `hw1_notebook.ipynb` so it cites actual numbers from your run).*

### Preliminary Template & AI Disclosure Draft
```latex
% Preliminary text (will update with exact numbers from Part E):
One notable result in Part~E was [e.g. how He initialisation maintained consistent activation variance (~1.0) and non-vanishing gradients across all 10 layers, whereas Xavier with ReLU suffered severe variance collapse by layer 10].
\textbf{AI-tool disclosure:} An AI coding assistant (Google Antigravity / Gemini) was used during the homework to verify tensor broadcasting shapes, assist with finite-difference formula checks in Part B, and format mathematical LaTeX templates. All derivations, code implementations, and experimental analyses were reviewed, validated, and explained by the author.
```

