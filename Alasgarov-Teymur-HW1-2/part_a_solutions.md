# Part A: Complete Written Solutions & Report Drafts
**Course**: DLE-AI-202 Deep Learning  
**Author**: Alasgarov Teymur  
**File**: `part_a_solutions.md`

This document contains full, rigorous mathematical solutions for **all 8 problems of Part A (A1 to A8)**, along with ready-to-copy LaTeX code snippets formatted for the corresponding `\begin{respond}` blocks in `report_template.tex`.

---

## Table of Contents
1. [A1: Shapes and Parameter Count (3 pts)](#a1-shapes-and-parameter-count-3-pts)
2. [A2: Losses Come from Likelihood (4 pts)](#a2-losses-come-from-likelihood-4-pts)
3. [A3: Backpropagation by Hand (6 pts)](#a3-backpropagation-by-hand-6-pts)
4. [A4: Activations and Vanishing Gradients (4 pts)](#a4-activations-and-vanishing-gradients-4-pts)
5. [A5: Initialisation (4 pts)](#a5-initialisation-4-pts)
6. [A6: Optimisers (4 pts)](#a6-optimisers-4-pts)
7. [A7: Regularisation (3 pts)](#a7-regularisation-3-pts)
8. [A8: Debugging Reasoning (2 pts)](#a8-debugging-reasoning-2-pts)

---

## A1: Shapes and Parameter Count (3 pts)

### Problem Statement
An MLP has layers $20 \to 64 \to 32 \to 5$ with ReLU hidden units. A batch of $128$ examples goes through it.
- **(a)** Give the shape of $W^{(\ell)}$ and $\mathbf{b}^{(\ell)}$ as stored by `nn.Linear`, and of $\mathbf{z}^{(\ell)}$ and $\mathbf{a}^{(\ell)}$ for the batch, for every layer.
- **(b)** Count the parameters.
- **(c)** Count the multiply--add operations of one forward pass for *one* example.

---

### Detailed Mathematical Solution

#### (a) Shapes per Layer
In PyTorch, `nn.Linear(in_features, out_features)` stores:
- Weight matrix $W \in \mathbb{R}^{\text{out\_features} \times \text{in\_features}}$.
- Bias vector $\mathbf{b} \in \mathbb{R}^{\text{out\_features}}$.
For a batch of $N = 128$ examples, the pre-activations $\mathbf{z}^{(\ell)} = \mathbf{a}^{(\ell-1)} (W^{(\ell)})^T + \mathbf{b}^{(\ell)}$ and activations $\mathbf{a}^{(\ell)}$ have shape $(N, \text{out\_features})$:

1. **Layer 1 ($20 \to 64$, ReLU)**:
   - $W^{(1)}$: `(64, 20)`
   - $\mathbf{b}^{(1)}$: `(64)`
   - $\mathbf{z}^{(1)}$: `(128, 64)`
   - $\mathbf{a}^{(1)} = \operatorname{ReLU}(\mathbf{z}^{(1)})$: `(128, 64)`

2. **Layer 2 ($64 \to 32$, ReLU)**:
   - $W^{(2)}$: `(32, 64)`
   - $\mathbf{b}^{(2)}$: `(32)`
   - $\mathbf{z}^{(2)}$: `(128, 32)`
   - $\mathbf{a}^{(2)} = \operatorname{ReLU}(\mathbf{z}^{(2)})$: `(128, 32)`

3. **Layer 3 ($32 \to 5$, Linear output logits)**:
   - $W^{(3)}$: `(5, 32)`
   - $\mathbf{b}^{(3)}$: `(5)`
   - $\mathbf{z}^{(3)}$: `(128, 5)`
   - $\mathbf{a}^{(3)} = \mathbf{z}^{(3)}$: `(128, 5)`

#### (b) Parameter Count
Parameters per layer = $(\text{in\_features} \times \text{out\_features}) + \text{out\_features}$:
- Layer 1: $20 \times 64 + 64 = 1{,}280 + 64 = 1{,}344$
- Layer 2: $64 \times 32 + 32 = 2{,}048 + 32 = 2{,}080$
- Layer 3: $32 \times 5 + 5 = 160 + 5 = 165$

$$\text{Total Parameters} = 1{,}344 + 2{,}080 + 165 = \mathbf{3{,}589}$$

#### (c) Multiply-Add Operations (MACs) for 1 Example
For a linear layer $d_{\text{in}} \to d_{\text{out}}$, computing each output unit requires $d_{\text{in}}$ multiplications and additions (dot product). With bias accumulated:
- Layer 1: $20 \times 64 = 1{,}280$ MACs
- Layer 2: $64 \times 32 = 2{,}048$ MACs
- Layer 3: $32 \times 5 = 160$ MACs

$$\text{Total MACs} = 1{,}280 + 2{,}048 + 160 = \mathbf{3{,}488}\text{ operations}$$
*(In FLOPs: $2 \times 3{,}488 = 6{,}976$ FLOPs).*

---

### Ready-to-Copy LaTeX for Report (A1)
```latex
\textbf{(a) Tensor shapes for batch $N=128$:}
\begin{itemize}\setlength{\itemsep}{0pt}
  \item Layer 1 ($20\to64$): $W^{(1)} \in \mathbb{R}^{64\times 20}$, $\mathbf{b}^{(1)} \in \mathbb{R}^{64}$, $\mathbf{z}^{(1)} \in \mathbb{R}^{128\times 64}$, $\mathbf{a}^{(1)} \in \mathbb{R}^{128\times 64}$.
  \item Layer 2 ($64\to32$): $W^{(2)} \in \mathbb{R}^{32\times 64}$, $\mathbf{b}^{(2)} \in \mathbb{R}^{32}$, $\mathbf{z}^{(2)} \in \mathbb{R}^{128\times 32}$, $\mathbf{a}^{(2)} \in \mathbb{R}^{128\times 32}$.
  \item Layer 3 ($32\to5$): $W^{(3)} \in \mathbb{R}^{5\times 32}$, $\mathbf{b}^{(3)} \in \mathbb{R}^{5}$, $\mathbf{z}^{(3)} \in \mathbb{R}^{128\times 5}$, $\mathbf{a}^{(3)} \in \mathbb{R}^{128\times 5}$.
\end{itemize}
\textbf{(b) Parameter count:}
$(20\times 64 + 64) + (64\times 32 + 32) + (32\times 5 + 5) = 1344 + 2080 + 165 = \mathbf{3589}$.\\
\textbf{(c) Multiply--adds for one example:}
$20\times 64 + 64\times 32 + 32\times 5 = 1280 + 2048 + 160 = \mathbf{3488}$ MACs.
```

---

## A2: Losses Come from Likelihood (4 pts)

### Problem Statement
- **(a)** Assume $y = \hat{y} + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$. Derive the negative log-likelihood of one example and show that minimising it over the data is minimising the mean squared error.
- **(b)** For a logit $z$, $\hat{y} = \sigma(z)$ and label $y \in \{0, 1\}$, show that:
  $$-[y\log\hat{y} + (1-y)\log(1-\hat{y})] = \max(z, 0) - zy + \log(1 + e^{-|z|})$$
  and say why this form never overflows.
- **(c)** Show that $\frac{\partial L}{\partial z} = \hat{y} - y$.

---

### Detailed Mathematical Solution

#### (a) Gaussian NLL $\to$ MSE
The conditional likelihood of target $y$ given prediction $\hat{y}$ under Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is:
$$p(y \mid \hat{y}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y - \hat{y})^2}{2\sigma^2} \right)$$
Taking the negative logarithm gives the NLL of one example:
$$\mathcal{L}_{\text{NLL}} = -\log p(y \mid \hat{y}) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}(y - \hat{y})^2$$
Summing over the dataset of $N$ i.i.d. examples:
$$\sum_{i=1}^N \mathcal{L}_{\text{NLL}}^{(i)} = \frac{N}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \hat{y}_i)^2$$
Since $\sigma^2 > 0$ and $\frac{N}{2}\log(2\pi\sigma^2)$ are constant with respect to model parameters $\theta$:
$$\arg\min_\theta \sum_{i=1}^N \mathcal{L}_{\text{NLL}}^{(i)} = \arg\min_\theta \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2 = \arg\min_\theta \text{MSE}$$

#### (b) Stable BCE Identity & Overflow Prevention
With $\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$, we have:
$$\log \hat{y} = -\log(1 + e^{-z}), \quad \log(1 - \hat{y}) = \log\left(\frac{e^{-z}}{1 + e^{-z}}\right) = -z - \log(1 + e^{-z})$$
Substitute into BCE loss $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$:
$$L = y\log(1 + e^{-z}) + (1-y)[z + \log(1 + e^{-z})] = z(1-y) + \log(1 + e^{-z}) = z - zy + \log(1 + e^{-z})$$
To make this numerically symmetric:
- For $z \ge 0$: $\max(z, 0) = z$ and $|z| = z \implies z + \log(1 + e^{-z}) = \max(z, 0) + \log(1 + e^{-|z|})$.
- For $z < 0$: $\max(z, 0) = 0$ and $|z| = -z$. Note that $z + \log(1 + e^{-z}) = \log(e^z(1 + e^{-z})) = \log(e^z + 1) = \max(z, 0) + \log(1 + e^{-|z|})$.
Subtracting $zy$ gives:
$$L = \max(z, 0) - zy + \log(1 + e^{-|z|})$$

**Why this never overflows:**
Because $|z| \ge 0$, the exponent $-|z| \le 0$ is always non-positive. Hence $e^{-|z|} \in (0, 1]$, and $\log(1 + e^{-|z|}) \in (0, \ln 2 \approx 0.693]$. Neither $e^z$ for large positive $z$ (which overflows to `+inf`) nor $\log(0)$ (which underflows to `-\inf`) is ever computed.

#### (c) Gradient $\frac{\partial L}{\partial z} = \hat{y} - y$
From $L = -y\log\sigma(z) - (1-y)\log(1-\sigma(z))$ and $\sigma'(z) = \sigma(z)(1-\sigma(z))$:
$$\frac{\partial L}{\partial z} = -y \frac{\sigma'(z)}{\sigma(z)} - (1-y) \frac{-\sigma'(z)}{1-\sigma(z)} = -y(1 - \sigma(z)) + (1-y)\sigma(z) = \sigma(z) - y = \hat{y} - y$$

---

### Ready-to-Copy LaTeX for Report (A2)
```latex
\textbf{(a) Gaussian NLL to MSE:}\\
$p(y\mid\hat y) = (2\pi\sigma^2)^{-1/2}\exp(-(y-\hat y)^2/(2\sigma^2))$. Taking $-\log$ yields:
$\mathcal{L}_{\text{NLL}} = \frac{1}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}(y-\hat y)^2$.
Summing over $N$ points and discarding constants w.r.t.\ model parameters gives $\arg\min_\theta \sum \mathcal{L}_{\text{NLL}} = \arg\min_\theta \frac{1}{N}\sum (y_i-\hat y_i)^2 = \arg\min_\theta \text{MSE}$.\\
\textbf{(b) Numerically stable BCE:}\\
Using $\log\hat y = -\log(1+e^{-z})$ and $\log(1-\hat y) = -z-\log(1+e^{-z})$:
$L = -[y\log\hat y + (1-y)\log(1-\hat y)] = z(1-y) + \log(1+e^{-z}) = -zy + [z + \log(1+e^{-z})]$.
Since $z + \log(1+e^{-z}) = \log(1+e^z) = \max(z,0) + \log(1+e^{-|z|})$, we get
$L = \max(z,0) - zy + \log(1+e^{-|z|})$.
\emph{Overflow prevention:} Because $-|z| \le 0$, $e^{-|z|} \in (0, 1]$, so $\log(1+e^{-|z|}) \in (0, \ln 2]$. We never compute $e^{z}$ for $z \gg 0$ or $\log(0)$.\\
\textbf{(c) Gradient:}
$\frac{\partial L}{\partial z} = -y(1-\sigma(z)) + (1-y)\sigma(z) = \sigma(z) - y = \hat y - y$.
```

---

## A3: Backpropagation by Hand (6 pts)

### Problem Statement
A $2 \to 2 \to 2$ network with ReLU hidden layer, linear output, softmax, and cross-entropy:
$$W^{(1)} = \begin{pmatrix}1 & -1 \\ 2 & 0\end{pmatrix}, \quad \mathbf{b}^{(1)} = \begin{pmatrix}0 \\ -1\end{pmatrix}$$
$$W^{(2)} = \begin{pmatrix}1 & 1 \\ -1 & 2\end{pmatrix}, \quad \mathbf{b}^{(2)} = \begin{pmatrix}0 \\ 0\end{pmatrix}$$
$$\mathbf{x} = \begin{pmatrix}1 \\ 2\end{pmatrix}, \quad \mathbf{y} = (1, 0)$$

---

### Detailed Mathematical Solution

#### (a) Forward Computation
1. $\mathbf{z}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)} = \begin{pmatrix} 1(1) + (-1)(2) \\ 2(1) + 0(2) \end{pmatrix} + \begin{pmatrix}0 \\ -1\end{pmatrix} = \begin{pmatrix}-1 \\ 2\end{pmatrix} + \begin{pmatrix}0 \\ -1\end{pmatrix} = \begin{pmatrix}-1 \\ 1\end{pmatrix}$.
2. $\mathbf{h} = \operatorname{ReLU}(\mathbf{z}^{(1)}) = \operatorname{ReLU}\begin{pmatrix}-1 \\ 1\end{pmatrix} = \begin{pmatrix}0 \\ 1\end{pmatrix}$.
3. $\mathbf{z}^{(2)} = W^{(2)}\mathbf{h} + \mathbf{b}^{(2)} = \begin{pmatrix}1 & 1 \\ -1 & 2\end{pmatrix}\begin{pmatrix}0 \\ 1\end{pmatrix} + \begin{pmatrix}0 \\ 0\end{pmatrix} = \begin{pmatrix}1 \\ 2\end{pmatrix}$.
4. $\mathbf{p} = \operatorname{softmax}(\mathbf{z}^{(2)}) = \begin{pmatrix}\frac{e^1}{e^1 + e^2} \\ \frac{e^2}{e^1 + e^2}\end{pmatrix} = \begin{pmatrix}\frac{1}{1+e} \\ \frac{e}{1+e}\end{pmatrix} \approx \begin{pmatrix}0.2689 \\ 0.7311\end{pmatrix}$.
5. Cross-Entropy Loss: $L = -\log p_1 = \log(1+e) \approx \mathbf{1.3133}$.

#### (b) Computational Graph and Output Gradients
**Graph**:
$$\mathbf{x} \to \mathbf{z}^{(1)} = W^{(1)}\mathbf{x}+\mathbf{b}^{(1)} \to \mathbf{h} = \operatorname{ReLU}(\mathbf{z}^{(1)}) \to \mathbf{z}^{(2)} = W^{(2)}\mathbf{h}+\mathbf{b}^{(2)} \to \mathbf{p} = \operatorname{softmax}(\mathbf{z}^{(2)}) \to L = -\log p_1$$

**Factors & Gradients**:
- Output error logit gradient:
  $$\boldsymbol{\delta}^{(2)} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} = \mathbf{p} - \mathbf{y} = \begin{pmatrix} 0.2689 - 1 \\ 0.7311 - 0 \end{pmatrix} = \begin{pmatrix} -0.7311 \\ 0.7311 \end{pmatrix} = \begin{pmatrix} -\frac{e}{1+e} \\ \frac{e}{1+e} \end{pmatrix}$$
- Layer 2 weights gradient:
  $$\frac{\partial L}{\partial W^{(2)}} = \boldsymbol{\delta}^{(2)}\mathbf{h}^T = \begin{pmatrix} -0.7311 \\ 0.7311 \end{pmatrix}\begin{pmatrix} 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & -0.7311 \\ 0 & 0.7311 \end{pmatrix}$$
- Layer 2 bias gradient:
  $$\frac{\partial L}{\partial \mathbf{b}^{(2)}} = \boldsymbol{\delta}^{(2)} = \begin{pmatrix} -0.7311 \\ 0.7311 \end{pmatrix}$$

#### (c) Layer 1 Gradients
Backpropagate to hidden pre-activation:
$$\frac{\partial L}{\partial \mathbf{h}} = (W^{(2)})^T \boldsymbol{\delta}^{(2)} = \begin{pmatrix} 1 & -1 \\ 1 & 2 \end{pmatrix}\begin{pmatrix} -0.7311 \\ 0.7311 \end{pmatrix} = \begin{pmatrix} -0.7311 - 0.7311 \\ -0.7311 + 2(0.7311) \end{pmatrix} = \begin{pmatrix} -1.4621 \\ 0.7311 \end{pmatrix}$$

Pass through ReLU gate:
$$\boldsymbol{\delta}^{(1)} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} = \frac{\partial L}{\partial \mathbf{h}} \odot \mathbb{I}(\mathbf{z}^{(1)} > 0)$$
Since $\mathbf{z}^{(1)} = \begin{pmatrix}-1 \\ 1\end{pmatrix}$, $\mathbb{I}(\mathbf{z}^{(1)} > 0) = \begin{pmatrix}0 \\ 1\end{pmatrix}$:
$$\boldsymbol{\delta}^{(1)} = \begin{pmatrix} -1.4621 \\ 0.7311 \end{pmatrix} \odot \begin{pmatrix}0 \\ 1\end{pmatrix} = \begin{pmatrix} 0 \\ 0.7311 \end{pmatrix}$$

Weight and bias gradients for Layer 1:
- $\frac{\partial L}{\partial W^{(1)}} = \boldsymbol{\delta}^{(1)}\mathbf{x}^T = \begin{pmatrix} 0 \\ 0.7311 \end{pmatrix}\begin{pmatrix} 1 & 2 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0.7311 & 1.4621 \end{pmatrix}$
- $\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)} = \begin{pmatrix} 0 \\ 0.7311 \end{pmatrix}$

#### (d) Dead Unit Analysis
- **Unit with no gradient**: Unit 1 of hidden layer ($\delta_1^{(1)} = 0$).
- **Why**: $z_1^{(1)} = -1 < 0$. The derivative of ReLU is 0 on the negative half-line, gating out the gradient.
- **Dying ReLU consequence**: If $z_1^{(1)} \le 0$ holds for all training examples, unit 1's expected gradient is identically 0. Weights connected to it will never be updated by gradient descent, rendering the neuron permanently dead and permanently reducing model capacity.

---

### Ready-to-Copy LaTeX for Report (A3)
```latex
\textbf{(a) Forward pass:}\\
$\mathbf{z}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)} = \begin{pmatrix}-1\\1\end{pmatrix}$, 
$\mathbf{h} = \operatorname{ReLU}(\mathbf{z}^{(1)}) = \begin{pmatrix}0\\1\end{pmatrix}$, 
$\mathbf{z}^{(2)} = W^{(2)}\mathbf{h} + \mathbf{b}^{(2)} = \begin{pmatrix}1\\2\end{pmatrix}$.\\
$\mathbf{p} = \operatorname{softmax}(\mathbf{z}^{(2)}) = \begin{pmatrix}\frac{1}{1+e}\\\frac{e}{1+e}\end{pmatrix} \approx \begin{pmatrix}0.2689\\0.7311\end{pmatrix}$, 
$L = -\log p_1 = \log(1+e) \approx \mathbf{1.3133}$.\\[2pt]
\textbf{(b) Layer 2 gradients:}\\
$\boldsymbol\delta^{(2)} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} = \mathbf{p} - \mathbf{y} = \begin{pmatrix}-0.7311\\0.7311\end{pmatrix}$.\\
$\frac{\partial L}{\partial W^{(2)}} = \boldsymbol\delta^{(2)}\mathbf{h}^T = \begin{pmatrix}0 & -0.7311\\0 & 0.7311\end{pmatrix}$, \quad
$\frac{\partial L}{\partial \mathbf{b}^{(2)}} = \boldsymbol\delta^{(2)} = \begin{pmatrix}-0.7311\\0.7311\end{pmatrix}$.\\[2pt]
\textbf{(c) Layer 1 gradients:}\\
$\frac{\partial L}{\partial \mathbf{h}} = (W^{(2)})^T \boldsymbol\delta^{(2)} = \begin{pmatrix}-1.4621\\0.7311\end{pmatrix}$.
Gated by ReLU: $\boldsymbol\delta^{(1)} = \frac{\partial L}{\partial \mathbf{h}} \odot \mathbb{I}(\mathbf{z}^{(1)}>0) = \begin{pmatrix}0\\0.7311\end{pmatrix}$.\\
$\frac{\partial L}{\partial W^{(1)}} = \boldsymbol\delta^{(1)}\mathbf{x}^T = \begin{pmatrix}0 & 0\\0.7311 & 1.4621\end{pmatrix}$, \quad
$\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \boldsymbol\delta^{(1)} = \begin{pmatrix}0\\0.7311\end{pmatrix}$.\\[2pt]
\textbf{(d) Dead ReLU:}
Unit 1 receives zero gradient because $z_1^{(1)} = -1 \le 0$ and $\operatorname{ReLU}'(z)=0$ for $z<0$. If this happens for every training example, the neuron never updates (the dying-ReLU problem) and capacity is permanently lost.
```

---

## A4: Activations and Vanishing Gradients (4 pts)

### Problem Statement
- **(a)** Show $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ and find its maximum.
- **(b)** Ignoring the weights, the gradient reaching layer 1 of an $L$-layer sigmoid network is at most $0.25^L$ times the gradient at the output. How many layers make this smaller than $10^{-6}$?
- **(c)** Derive the derivative of Swish, $f(z) = z\sigma(z)$, and of GELU, $f(z) = z\Phi(z)$, in terms of $\Phi$ and the standard normal density $\varphi$.

---

### Detailed Mathematical Solution

#### (a) Sigmoid Derivative and Maximum
$$\sigma(z) = \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}$$
$$\sigma'(z) = -(1 + e^{-z})^{-2}(-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z)(1 - \sigma(z))$$
Let $u = \sigma(z) \in (0, 1)$. The function $g(u) = u(1-u) = u - u^2$ is maximized when $g'(u) = 1 - 2u = 0 \implies u = 0.5$.
Since $\sigma(0) = 0.5$, the maximum occurs at $z = 0$:
$$\max_z \sigma'(z) = 0.5(1 - 0.5) = \mathbf{0.25}$$

#### (b) Depth for Attenuation $< 10^{-6}$
We require:
$$(0.25)^L < 10^{-6} \implies \left(\frac{1}{4}\right)^L < 10^{-6} \implies 4^L > 10^6$$
Taking $\log_{10}$ on both sides:
$$L \log_{10}(4) > 6 \implies L > \frac{6}{\log_{10}(4)} \approx \frac{6}{0.60206} \approx 9.9658$$
Since the number of layers $L$ must be an integer:
$$\mathbf{L \ge 10\text{ layers}}$$

#### (c) Swish and GELU Derivatives
1. **Swish** $f(z) = z\sigma(z)$:
   Using product rule:
   $$f'(z) = 1 \cdot \sigma(z) + z \cdot \sigma'(z) = \sigma(z) + z\sigma(z)(1 - \sigma(z)) = \mathbf{\sigma(z) + z\sigma(z)(1 - \sigma(z))}$$
   *(Equivalently: $\sigma(z) + f(z)(1 - \sigma(z))$).*

2. **GELU** $f(z) = z\Phi(z)$ where $\Phi(z) = \int_{-\infty}^z \varphi(t)\,dt$ and $\varphi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$:
   By the Fundamental Theorem of Calculus, $\Phi'(z) = \varphi(z)$. Applying the product rule:
   $$f'(z) = 1 \cdot \Phi(z) + z \cdot \Phi'(z) = \mathbf{\Phi(z) + z\varphi(z)}$$

---

### Ready-to-Copy LaTeX for Report (A4)
```latex
\textbf{(a) Sigmoid derivative:}\\
$\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)(1-\sigma(z))$. Let $u=\sigma(z)\in(0,1)$; $\frac{d}{du}[u(1-u)] = 1-2u=0 \implies u=1/2$ (at $z=0$). Thus $\max \sigma'(z) = \frac{1}{2}(1-\frac{1}{2}) = \mathbf{0.25}$.\\[2pt]
\textbf{(b) Layer bound:}\\
$0.25^L < 10^{-6} \iff 4^L > 10^6 \iff L > \frac{6}{\log_{10} 4} \approx 9.966 \implies \mathbf{L \ge 10}$.\\[2pt]
\textbf{(c) Swish and GELU derivatives:}\\
• Swish: $f'(z) = \sigma(z) + z\sigma'(z) = \mathbf{\sigma(z) + z\sigma(z)(1-\sigma(z))}$.\\
• GELU: With $\Phi'(z) = \varphi(z)$, $f'(z) = \mathbf{\Phi(z) + z\varphi(z)}$.
```

---

## A5: Initialisation (4 pts)

### Problem Statement
- **(a)** A unit computes $z = \sum_{j=1}^n w_j a_j$ with independent, zero-mean $w_j$ and $a_j$. Show $\operatorname{Var}(z) = n\operatorname{Var}(w)\mathbb{E}[a^2]$, and use $\mathbb{E}[a^2] = \frac{1}{2}\operatorname{Var}(z_{\text{prev}})$ for ReLU to derive $\operatorname{Var}(w) = 2/n$ (He).
- **(b)** Give the He and the Xavier standard deviation for a $784 \to 256$ layer.
- **(c)** Why can a network not be initialised with all weights equal to zero?
- **(d)** What loss do you expect from a freshly initialised 10-class network and why? Which two bugs make it much larger?

---

### Detailed Mathematical Solution

#### (a) He Variance Derivation
Since $w_j$ and $a_j$ are independent and $\mathbb{E}[w_j] = 0$:
$$\mathbb{E}[z] = \sum_{j=1}^n \mathbb{E}[w_j]\mathbb{E}[a_j] = 0$$
$$\operatorname{Var}(z) = \mathbb{E}[z^2] = \mathbb{E}\left[\left(\sum_{j=1}^n w_j a_j\right)^2\right] = \sum_{j=1}^n \mathbb{E}[w_j^2 a_j^2] = \sum_{j=1}^n \operatorname{Var}(w)\mathbb{E}[a^2] = n\operatorname{Var}(w)\mathbb{E}[a^2]$$
For a ReLU unit, $a = \max(0, z_{\text{prev}})$. Assuming $z_{\text{prev}}$ has a symmetric distribution around 0 with variance $\operatorname{Var}(z_{\text{prev}})$:
$$\mathbb{E}[a^2] = \int_0^\infty z_{\text{prev}}^2 p(z_{\text{prev}})\,dz_{\text{prev}} = \frac{1}{2}\int_{-\infty}^\infty z_{\text{prev}}^2 p(z_{\text{prev}})\,dz_{\text{prev}} = \frac{1}{2}\operatorname{Var}(z_{\text{prev}})$$
Thus:
$$\operatorname{Var}(z) = n \operatorname{Var}(w) \cdot \frac{1}{2}\operatorname{Var}(z_{\text{prev}}) = \frac{n}{2}\operatorname{Var}(w)\operatorname{Var}(z_{\text{prev}})$$
Setting $\operatorname{Var}(z) = \operatorname{Var}(z_{\text{prev}})$ to preserve signal variance:
$$\frac{n}{2}\operatorname{Var}(w) = 1 \implies \mathbf{\operatorname{Var}(w) = \frac{2}{n}}$$

#### (b) He vs Xavier for $784 \to 256$
Here $n_{\text{in}} = 784, n_{\text{out}} = 256$:
- **He**: $\sigma_{\text{He}} = \sqrt{\frac{2}{n_{\text{in}}}} = \sqrt{\frac{2}{784}} = \frac{\sqrt{2}}{28} \approx \mathbf{0.0505}$
- **Xavier (Glorot)**: $\sigma_{\text{Xavier}} = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}} = \sqrt{\frac{2}{784 + 256}} = \sqrt{\frac{2}{1040}} = \sqrt{\frac{1}{520}} \approx \mathbf{0.0439}$

#### (c) Why Zero Initialization Fails
If all weights are initialized to 0, every neuron in the hidden layer computes identical forward outputs ($z_j = b_j$) and receives identical gradients $\frac{\partial L}{\partial W_{jk}} = \delta_j a_k$. Because all weights receive the exact same update, symmetry is never broken, and the layer behaves as if it had only a single hidden unit.

#### (d) Expected Initial Loss & Two Bugs
- **Expected Loss**: At initialization, a well-calibrated network assigns uniform probabilities $p_k = 1/K = 1/10$ across classes.
  $$L_{\text{init}} = -\log(1/10) = \ln 10 \approx \mathbf{2.3026}$$
- **Two Bugs that Inflate It**:
  1. *Exploding initial weights* (e.g., standard normal with $\sigma = 1.0$ instead of $0.05$): creates huge logits ($|z| \gg 10$), pushing wrong-class logits high and the true-class probability to near-zero, producing initial loss $> 20$.
  2. *Double Softmax bug*: applying `softmax` before passing predictions to `nn.CrossEntropyLoss` (which applies `log_softmax` internally), corrupting inputs and distorting loss.

---

### Ready-to-Copy LaTeX for Report (A5)
```latex
\textbf{(a) He variance derivation:}\\
$\operatorname{Var}(z) = \sum_{j=1}^n \mathbb{E}[w_j^2]\mathbb{E}[a_j^2] = n\operatorname{Var}(w)\mathbb{E}[a^2]$. For ReLU, half the distribution is zeroed, so $\mathbb{E}[a^2] = \frac{1}{2}\operatorname{Var}(z_{\text{prev}})$.
Equating $\operatorname{Var}(z)=\operatorname{Var}(z_{\text{prev}})$ gives $\frac{n}{2}\operatorname{Var}(w)=1 \implies \mathbf{\operatorname{Var}(w)=2/n}$.\\[2pt]
\textbf{(b) $784\to256$ std:}
$\sigma_{\text{He}} = \sqrt{2/784} \approx \mathbf{0.0505}$; \quad
$\sigma_{\text{Xavier}} = \sqrt{2/(784+256)} = \sqrt{1/520} \approx \mathbf{0.0439}$.\\[2pt]
\textbf{(c) Zero initialisation:}
All hidden units produce identical activations and receive identical gradients $\delta_j a_k$. Symmetry is never broken and all units learn the same feature.\\[2pt]
\textbf{(d) Initial loss and bugs:}
With random initialisation, predictions are uniform ($p_k=1/10$), so $L = \ln 10 \approx \mathbf{2.3026}$.
Two bugs that inflate it: (1)~overly large initial weight scale ($\sigma=1.0$) causing large logits and saturated wrong predictions; (2)~passing softmax probabilities into \texttt{nn.CrossEntropyLoss} (double softmax).
```

---

## A6: Optimisers (4 pts)

### Problem Statement
Minimise $f(w) = 5w^2$ from $w_0 = 1$ with learning rate $\eta = 0.05$.
- **(a)** Do two steps of plain SGD and two steps of momentum ($v \leftarrow \gamma v + \eta g$, $w \leftarrow w - v$, $\gamma = 0.9$, $v_0 = 0$). What do you notice about the momentum iterates?
- **(b)** Write Adam's first update for a gradient $g_1 = 0.2$ with learning rate $\eta$, including the bias correction, and say how large the step is.
- **(c)** Why is weight decay applied *outside* the adaptive scaling in AdamW?

---

### Detailed Mathematical Solution

#### (a) Plain SGD vs Momentum
Objective $f(w) = 5w^2$, gradient $g(w) = f'(w) = 10w$. Initial $w_0 = 1$, $\eta = 0.05$.

**Plain SGD**:
- Step 1: $g_0 = 10(1) = 10$.
  $$w_1 = w_0 - \eta g_0 = 1 - 0.05(10) = 1 - 0.5 = \mathbf{0.5}$$
- Step 2: $g_1 = 10(0.5) = 5$.
  $$w_2 = w_1 - \eta g_1 = 0.5 - 0.05(5) = 0.5 - 0.25 = \mathbf{0.25}$$
*(Monotonic geometric decay: $w_t = (1 - 10\eta)^t = 0.5^t$).*

**Momentum** ($\gamma = 0.9, v_0 = 0$):
- Step 1:
  $$g_0 = 10(1) = 10$$
  $$v_1 = \gamma v_0 + \eta g_0 = 0.9(0) + 0.05(10) = 0.5$$
  $$w_1 = w_0 - v_1 = 1 - 0.5 = \mathbf{0.5}$$
- Step 2:
  $$g_1 = 10(0.5) = 5$$
  $$v_2 = \gamma v_1 + \eta g_1 = 0.9(0.5) + 0.05(5) = 0.45 + 0.25 = 0.70$$
  $$w_2 = w_1 - v_2 = 0.5 - 0.70 = \mathbf{-0.20}$$

**Observation**:
Plain SGD approaches the optimum $w^*=0$ monotonically from the right ($1 \to 0.5 \to 0.25$). Momentum carries velocity from the first step ($0.45$), causing it to accelerate and **overshoot** past the minimum to the opposite side ($w_2 = -0.20 < 0$).

#### (b) Adam's First Update
Given $g_1 = 0.2$, $m_0 = 0, v_0 = 0$. Standard Adam hyperparameters: $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon \approx 10^{-8}$.
- First moment: $m_1 = \beta_1 m_0 + (1 - \beta_1)g_1 = (1 - 0.9)(0.2) = 0.02$.
- Second moment: $v_1 = \beta_2 v_0 + (1 - \beta_2)g_1^2 = (1 - 0.999)(0.2)^2 = 0.001(0.04) = 4 \times 10^{-5}$.
- Bias correction at $t=1$:
  $$\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{0.02}{1 - 0.9} = \frac{0.02}{0.1} = 0.2 = g_1$$
  $$\hat{v}_1 = \frac{v_1}{1 - \beta_2^1} = \frac{4 \times 10^{-5}}{1 - 0.999} = \frac{4 \times 10^{-5}}{0.001} = 0.04 = g_1^2$$
- Update:
  $$\Delta w_1 = -\frac{\eta \hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon} = -\frac{\eta (0.2)}{\sqrt{0.04} + \epsilon} = -\frac{\eta (0.2)}{0.2 + \epsilon} \approx \mathbf{-\eta}$$
**Step size magnitude**:
The step magnitude is approximately **$\eta$** (specifically $\eta \cdot \frac{0.2}{0.2 + \epsilon} \approx \eta$). At step 1 with bias correction, $\hat{m}_1 = g_1$ and $\sqrt{\hat{v}_1} = |g_1|$, so the update behaves like a sign gradient step of magnitude $\eta$ regardless of gradient scale.

#### (c) Why AdamW Decouples Weight Decay
In standard Adam with L2 regularization, weight decay is implemented by adding $\lambda w$ directly to the gradient: $g_t = \nabla L(w_t) + \lambda w_t$. This gradient is fed into the second moment estimator $v_t \propto g_t^2$.
Consequently, weights with historically large gradients have large $\sqrt{v_t}$ and receive **less** weight decay ($\frac{\lambda w}{\sqrt{v_t}} \ll \lambda w$), while weights with tiny gradients receive disproportionately large weight decay.
AdamW applies weight decay **outside** the adaptive scaling:
$$w_{t+1} = (1 - \eta\lambda)w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$$
ensuring all parameters decay at a rate directly proportional to their magnitude, independent of gradient history.

---

### Ready-to-Copy LaTeX for Report (A6)
```latex
\textbf{(a) SGD vs Momentum ($f'(w)=10w$):}\\
• SGD: $w_1 = 1 - 0.05(10) = \mathbf{0.5}$; $w_2 = 0.5 - 0.05(5) = \mathbf{0.25}$.\\
• Momentum: $v_1 = 0.05(10) = 0.5 \implies w_1 = 1 - 0.5 = \mathbf{0.5}$.
$v_2 = 0.9(0.5) + 0.05(5) = 0.70 \implies w_2 = 0.5 - 0.70 = \mathbf{-0.20}$.\\
\emph{Observation:} Momentum carries velocity from step 1 and overshoots the minimum ($w^*=0$) to the negative side, whereas SGD converges monotonically.\\[2pt]
\textbf{(b) Adam first update ($g_1=0.2$):}\\
$m_1 = (1-\beta_1)g_1 = 0.02 \implies \hat m_1 = \frac{0.02}{1-0.9} = 0.2 = g_1$.\\
$v_1 = (1-\beta_2)g_1^2 = 4\times 10^{-5} \implies \hat v_1 = \frac{4\times 10^{-5}}{1-0.999} = 0.04 = g_1^2$.\\
Step: $\Delta w = -\eta \frac{\hat m_1}{\sqrt{\hat v_1}+\epsilon} = -\eta \frac{0.2}{0.2+\epsilon} \approx \mathbf{-\eta}$ (magnitude is exactly $\eta$).\\[2pt]
\textbf{(c) Decoupled weight decay (AdamW):}\\
In standard Adam, adding $\lambda w$ to the gradient scales the decay by $1/\sqrt{v_t}$, penalising frequently updated weights \emph{less} than rarely updated ones. AdamW applies $w \leftarrow (1-\eta\lambda)w$ directly, restoring uniform decay.
```

---

## A7: Regularisation (3 pts)

### Problem Statement
- **(a)** Show that SGD on $L(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2$ equals shrinking the weights by $(1 - \eta\lambda)$ and then taking the usual gradient step.
- **(b)** With inverted dropout (drop with probability $p$, scale kept units by $1/(1-p)$), show $\mathbb{E}[a'] = a$.
- **(c)** Why must batch normalisation use different statistics at training and inference time?

---

### Detailed Mathematical Solution

#### (a) L2 Regularization as Weight Shrinkage
Let the total objective be:
$$J(\mathbf{w}) = L(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2$$
The gradient with respect to $\mathbf{w}$ is:
$$\nabla J(\mathbf{w}) = \nabla L(\mathbf{w}) + \lambda \mathbf{w}$$
Performing an SGD step with learning rate $\eta$:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla J(\mathbf{w}_t) = \mathbf{w}_t - \eta(\nabla L(\mathbf{w}_t) + \lambda \mathbf{w}_t) = (1 - \eta\lambda)\mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$
This shows that $\mathbf{w}$ is first shrunk by the multiplicative factor $(1 - \eta\lambda)$ before subtracting the data loss gradient step $\eta \nabla L(\mathbf{w}_t)$.

#### (b) Inverted Dropout Expectation
Let $m \in \{0, 1\}$ be a Bernoulli random mask where $P(m = 0) = p$ (drop probability) and $P(m = 1) = 1 - p$ (keep probability).
Inverted dropout defines:
$$a' = m \cdot \frac{a}{1 - p}$$
Taking expectation over the random mask $m$:
$$\mathbb{E}[a'] = \mathbb{E}\left[m \cdot \frac{a}{1 - p}\right] = \frac{a}{1 - p}\mathbb{E}[m]$$
Since $\mathbb{E}[m] = 1 \cdot (1 - p) + 0 \cdot p = 1 - p$:
$$\mathbb{E}[a'] = \frac{a}{1 - p}(1 - p) = \mathbf{a}$$
Thus the expected value of activations during training matches the uncorrupted activation $a$, requiring no scaling adjustment at inference time.

#### (c) BatchNorm: Training vs Inference Statistics
- **Training**: Uses the mean $\mu_{\mathcal{B}}$ and variance $\sigma_{\mathcal{B}}^2$ computed across the *current mini-batch*. This normalizes activations within each batch and injects mini-batch noise, acting as a regularizer.
- **Inference**: Test samples must be evaluated deterministically and independently; predictions cannot depend on which other samples share the batch (and at test time, batch size may be 1).
Therefore, inference uses fixed, global population statistics computed as running exponential moving averages during training:
$$\mu_{\text{run}} \leftarrow (1 - \alpha)\mu_{\text{run}} + \alpha \mu_{\mathcal{B}}, \quad \sigma^2_{\text{run}} \leftarrow (1 - \alpha)\sigma^2_{\text{run}} + \alpha \sigma^2_{\mathcal{B}}$$

---

### Ready-to-Copy LaTeX for Report (A7)
```latex
\textbf{(a) Weight decay equivalence:}\\
For $J(\mathbf{w}) = L(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2$, $\nabla J = \nabla L + \lambda\mathbf{w}$.
SGD step: $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta(\nabla L + \lambda\mathbf{w}_t) = \mathbf{(1-\eta\lambda)\mathbf{w}_t - \eta\nabla L(\mathbf{w}_t)}$, which multiplies weights by $(1-\eta\lambda)$ before the gradient step.\\[2pt]
\textbf{(b) Inverted dropout expectation:}\\
Let $m\sim\text{Bernoulli}(1-p)$. $a' = m \frac{a}{1-p}$.
$\mathbb{E}[a'] = \frac{a}{1-p}\mathbb{E}[m] = \frac{a}{1-p}(1-p) = \mathbf{a}$.\\[2pt]
\textbf{(c) BatchNorm train vs inference:}\\
During training, normalisation uses current mini-batch mean and variance ($\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}^2$), providing stochastic regularisation. At test time, predictions must be deterministic and independent of batch composition (even for batch size 1), so fixed running empirical statistics ($\mu_{\text{run}}, \sigma^2_{\text{run}}$) accumulated during training are used instead.
```

---

## A8: Debugging Reasoning (2 pts)

### Problem Statement
For each symptom give the most likely cause and the *first check* you would run:
1. 10-class loss stays at $2.30$ and never moves.
2. Loss becomes `NaN` after 50 steps.
3. Training accuracy 99%, validation accuracy 70%.
4. Validation accuracy differs every time you evaluate the same model on the same data.

---

### Detailed Analysis & First Checks

1. **(i) 10-class loss stays at 2.30 and never moves**:
   - *Most likely cause*: Gradients are zero or missing. The model is stuck at random guessing ($\ln 10 \approx 2.30$) because `optimizer.step()` was omitted, learning rate is set to 0, or all ReLU neurons are dead due to negative initial biases.
   - *First check*: Inspect gradient norms: print `[p.grad.abs().mean() for p in model.parameters()]` to see if gradients are `None` or all zeros.

2. **(ii) Loss becomes `NaN` after 50 steps**:
   - *Most likely cause*: Exploding gradients or unstable loss leading to numeric overflow (`inf`), which then produces `NaN` (e.g. learning rate too large, or computing `torch.log(0)`).
   - *First check*: Check gradient norms right before `NaN` occurs and verify if gradient clipping `torch.nn.utils.clip_grad_norm_` or a smaller learning rate resolves it.

3. **(iii) Training accuracy 99%, validation accuracy 70%**:
   - *Most likely cause*: Severe overfitting (over-parameterized model memorizing training samples due to lack of regularization).
   - *First check*: Add regularization (weight decay $\lambda = 10^{-4}$, dropout $p=0.2$) or inspect training/validation loss curves to confirm generalization gap.

4. **(iv) Validation accuracy differs every time you evaluate the same model on the same data**:
   - *Most likely cause*: The model was not switched to evaluation mode (`model.eval()`). Stochastic layers like `Dropout` (randomly dropping neurons) and `BatchNorm` (updating with batch statistics instead of running statistics) remain active.
   - *First check*: Ensure `model.eval()` and `with torch.no_grad():` wrap the entire validation loop.

---

### Ready-to-Copy LaTeX for Report (A8)
```latex
\begin{enumerate}\setlength{\itemsep}{1pt}
  \item[(i)] \textbf{Cause:} Zero or detached gradients (e.g., missing \texttt{optimizer.step()}, $\eta=0$, or all ReLUs dead). \textbf{Check:} Verify \texttt{p.grad.abs().mean()} across parameters to confirm gradients exist and are non-zero.
  \item[(ii)] \textbf{Cause:} Exploding gradients / numerical overflow leading to $\infty$ then \texttt{NaN} (e.g., learning rate too high, or unclipped gradients). \textbf{Check:} Enable gradient clipping (\texttt{clip\_grad\_norm\_}) and reduce learning rate by $10\times$.
  \item[(iii)] \textbf{Cause:} Severe overfitting (high model capacity, insufficient regularisation). \textbf{Check:} Check gap between train and validation loss curves; apply weight decay ($\lambda=10^{-4}$) or dropout.
  \item[(iv)] \textbf{Cause:} Model left in training mode (\texttt{model.train()}), so Dropout remains active and BatchNorm uses batch stats. \textbf{Check:} Ensure \texttt{model.eval()} is called before evaluation.
\end{enumerate}
```

