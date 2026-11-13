# Part E: Experiments & Empirical Analysis
**Course**: DLE-AI-202 Deep Learning  
**Author**: Alasgarov Teymur  
**File**: `part_e_solutions.md`  

This document contains detailed write-ups and ready-to-copy LaTeX blocks for **E1–E4** based on the experimental results obtained in [`hw1_notebook.ipynb`](./hw1_notebook.ipynb).

---

## E1: Depth and Initialisation (4 pts)

### Prompt
> A 10-layer MLP built four ways (sigmoid+Xavier, ReLU+tiny normal, ReLU+Xavier, ReLU+He). Measure per-layer output std and gradient norm at initialisation with `layer_stats`, then train with SGD. Explain which configurations fail and why. (3–5 sentences, citing your numbers).

### Your Experimental Results
| Configuration | Gradient Ratio ($\text{grad}_{\text{layer1}} / \text{grad}_{\text{last}}$) | Final Validation Accuracy |
| :--- | :---: | :---: |
| `sigmoid + xavier` | $1.05 \times 10^{-6}$ | 0.107 (random guessing) |
| `relu + normal(0.01)` | $1.32 \times 10^{0}$ | 0.087 (random guessing) |
| `relu + xavier` | $1.73 \times 10^{0}$ | 0.867 |
| `relu + he` | $9.41 \times 10^{-1}$ ($\approx 1.0$) | **0.920** |

### Detailed Analysis
1. **`sigmoid + xavier` (Vanishing Gradients):**
   - The derivative of sigmoid satisfies $\sigma'(z) \le 0.25$. Across 10 layers, the backpropagated error is gated by $\prod_{\ell=1}^{10} \sigma'(z^{(\ell)}) \le 0.25^{10} \approx 9.54 \times 10^{-7}$.
   - The measured gradient norm ratio of $1.05 \times 10^{-6}$ directly matches this theoretical bound. The early layers receive effectively zero gradient update, preventing feature learning and leaving validation accuracy locked at $10.7\%$ (random chance for 10 classes).

2. **`relu + normal(0.01)` (Forward Activation Collapse):**
   - Initialising weights with tiny normal variance ($\sigma = 0.01$) causes pre-activations to shrink exponentially during the forward pass by a factor of roughly $(\sigma^2 \cdot \text{fan\_in})^{L/2} \ll 1$.
   - By layer 10, activations collapse to near-zero, destroying the representational capacity of the network. Even though the initial gradient ratio appears stable ($1.32$), the network cannot learn meaningful class separations, resulting in $8.7\%$ validation accuracy.

3. **`relu + xavier` (Variance Explosion):**
   - Xavier initialisation assumes linear or sigmoid activations with $\operatorname{Var}(W) = 2/(\text{fan\_in} + \text{fan\_out})$. Because ReLU zeros out half of the activations ($\mathbb{E}[a^2] = \frac{1}{2}\operatorname{Var}(z)$), using Xavier leads to an expansion of backpropagated gradient norms ($\text{ratio} = 1.73$). While it learns ($86.7\%$), it suffers from mild instability.

4. **`relu + he` (Stable Variance Preserving):**
   - He initialisation sets $\operatorname{Var}(W) = 2/\text{fan\_in}$, exactly counteracting the factor-of-two variance reduction introduced by ReLU.
   - The gradient ratio remains remarkably stable across all 10 layers ($0.941 \approx 1.0$), achieving the highest validation accuracy ($92.0\%$).

### Ready-to-Copy LaTeX for Report (E1)
```latex
In a 10-layer MLP, initialisation determines whether gradients survive to early layers:
\texttt{sigmoid + xavier} suffers extreme vanishing gradients ($\text{grad ratio} = 1.05\times 10^{-6}$), closely matching the theoretical limit $(0.25)^{10} \approx 9.54\times 10^{-7}$ and remaining at random guessing (val acc 0.107).
\texttt{relu + normal(0.01)} causes forward activations to collapse exponentially to zero ($\sigma^{10} \ll 1$), preventing meaningful feature representation (val acc 0.087).
\texttt{relu + xavier} maintains forward propagation but suffers variance growth across 10 layers ($\text{grad ratio} = 1.73$), reaching a respectable 0.867 val acc.
Finally, \texttt{relu + he} balances variance throughout the deep network ($\text{grad ratio} = 0.941 \approx 1.0$), achieving the highest validation accuracy of 0.920.
```

---

## E2: Optimisers (3 pts)

### Prompt
> SGD, SGD+momentum, RMSprop, Adam and AdamW on the same model, data and seed. Compare validation-loss curves; say what differs and why one learning rate cannot rank optimisers. (3–5 sentences, citing your numbers).

### Your Experimental Results
Final validation accuracy after 30 epochs ($\eta = 0.05$):
- **SGD**: $0.947$
- **SGD + Momentum**: $0.963$
- **RMSprop**: **$0.973$**
- **Adam**: $0.963$
- **AdamW**: $0.960$

### Detailed Analysis
1. **Convergence and Dynamics:**
   - Plain SGD converges slowest ($0.947$) due to oscillatory steps along high-curvature directions in the loss landscape.
   - Momentum accelerates convergence ($0.963$) by accumulating velocity along consistent gradient directions and dampening oscillations.
   - Adaptive methods (RMSprop at $0.973$, Adam at $0.963$, AdamW at $0.960$) rapidly reduce validation loss in early epochs because per-parameter second-moment scaling ($1/\sqrt{v_t + \epsilon}$) equalises effective step sizes across coordinates with varying gradient magnitudes.
2. **Why One Learning Rate Cannot Rank Optimisers:**
   - A single fixed learning rate ($\eta = 0.05$) is fundamentally unfair because adaptive algorithms automatically normalize parameter updates by their root-mean-square gradient, meaning $\eta = 0.05$ can be relatively aggressive for Adam (standard defaults are $10^{-3}$).
   - Conversely, non-adaptive SGD lacks gradient normalization and often requires substantially larger learning rates ($0.01$ to $0.1$) to escape flat regions. Ranking optimisers under a single learning rate tests hyperparameter sensitivity rather than true asymptotic capacity.

### Ready-to-Copy LaTeX for Report (E2)
```latex
All five optimisers achieved strong validation accuracy on the digits task with learning rate $\eta=0.05$: RMSprop reached the highest validation accuracy (0.973), followed by Adam (0.963), SGD+momentum (0.963), AdamW (0.960), and plain SGD (0.947). Momentum and adaptive algorithms accelerated initial convergence noticeably over plain SGD by dampening cross-gradient oscillations. However, a single fixed learning rate cannot fairly rank optimisers because adaptive methods (Adam/RMSprop) scale updates by $1/\sqrt{v_t + \epsilon}$ (operating best with smaller nominal rates like $10^{-3}$), whereas SGD requires larger steps ($\sim 10^{-1}$ to $10^{-2}$) to traverse the loss landscape efficiently.
```

---

## E3: Regularisation on Noisy Data (4 pts)

### Prompt
> 300 training images with 25% corrupted labels so that overfitting is visible. Compare baseline, dropout, batch norm, weight decay, early stopping and a combination in a table and a plot; say what each mechanism does. (Include table and 3–5 sentences).

### Your Experimental Results Table
| Configuration | Train Acc | Val Loss Min | Val Loss Last | Epochs | Test Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | $1.000$ | $0.739$ | $0.910$ | 100 | $0.809$ |
| **Dropout (0.5)** | $0.860$ | $0.601$ | $0.672$ | 100 | $0.837$ |
| **BatchNorm** | $1.000$ | $0.699$ | $0.838$ | 100 | $0.791$ |
| **Weight Decay (0.5)** | $1.000$ | $0.723$ | $0.756$ | 100 | $0.807$ |
| **Early Stopping (patience=8)** | $1.000$ | $0.750$ | $0.763$ | 19 | $0.791$ |
| **Dropout + WD + Early** | $0.740$ | $0.663$ | $0.664$ | 46 | **$0.859$** |
| **MixUp ($\alpha = 0.4$) [Bonus]** | $0.380$ | $0.680$ | $0.850$ | 100 | $0.757$ |

### Detailed Mechanism Breakdown
1. **Baseline**: Perfectly memorises corrupted training labels ($1.000$ train acc), but validation loss climbs from $0.739$ to $0.910$, demonstrating severe overfitting (test acc $0.809$).
2. **Dropout ($p=0.5$)**: Stochastically zeroes out hidden units during forward passes, breaking co-adaptation of features. This acts as an ensemble of thinned networks, dramatically reducing minimum validation loss to $0.601$ and improving test accuracy to $0.837$.
3. **BatchNorm**: While providing internal normalization, its affine parameters ($\gamma, \beta$) give sufficient capacity to fit arbitrary label noise on small datasets without penalty, degrading test accuracy to $0.791$ (worse than baseline).
4. **Weight Decay ($\lambda = 0.5$)**: Penalizes large weight norms ($L_2$ shrinkage), preventing excessive logit magnitudes and dampening overconfidence on noisy samples.
5. **Early Stopping**: Halts training at epoch 19 when validation loss begins to diverge, protecting the network from post-convergence noise memorization.
6. **Combination (`Dropout + WD + Early`)**: Synergistically combines all three inductive biases. Notably, training accuracy halts at $0.740$—almost exactly reflecting the $75\%$ true clean label fraction ($100\% - 25\%$)—and attains the highest test accuracy ($0.859$).
7. **MixUp ($\alpha=0.4$)**: Linearly interpolates feature-label pairs. With small data and high noise, it severely suppresses label memorization (train acc $0.380$), though on this small-sample setup it required longer training or lower $\alpha$ for optimal test performance.

### Ready-to-Copy LaTeX for Report (E3)
```latex
\begin{center}
\small
\begin{tabular}{lccccc}
\hline
\textbf{Configuration} & \textbf{Train Acc} & \textbf{Val Loss Min} & \textbf{Val Loss Last} & \textbf{Epochs} & \textbf{Test Acc} \\
\hline
Baseline & 1.000 & 0.739 & 0.910 & 100 & 0.809 \\
Dropout (0.5) & 0.860 & 0.601 & 0.672 & 100 & 0.837 \\
BatchNorm & 1.000 & 0.699 & 0.838 & 100 & 0.791 \\
Weight Decay (0.5) & 1.000 & 0.723 & 0.756 & 100 & 0.807 \\
Early Stopping (8) & 1.000 & 0.750 & 0.763 & 19 & 0.791 \\
Dropout + WD + Early & 0.740 & 0.663 & 0.664 & 46 & \textbf{0.859} \\
MixUp (0.4) [Bonus] & 0.380 & 0.680 & 0.850 & 100 & 0.757 \\
\hline
\end{tabular}
\end{center}
On 25\% noisy labels, the baseline overfits completely (train acc 1.000, test 0.809). Dropout randomly disables features, forcing distributed representations and reducing validation loss to 0.601 (test acc 0.837). BatchNorm accelerates memorisation of label noise, yielding worse test accuracy (0.791). Weight decay shrinks weight norms to prevent extreme logit values. The combination (\texttt{dropout+wd+early}) yields the best generalisation (test acc 0.859), naturally stopping when train accuracy reaches 0.740—almost identical to the clean data proportion (75\%).
```

---

## E4: Final Model (4 pts)

### Prompt
> Choose a configuration *using the validation set only*, train with warm-up + cosine and gradient clipping, and evaluate on the test set *once*. Target: test accuracy $\ge 0.95$. Save `figures/final_curves.png`. (3–5 sentences: configuration and why, final test accuracy, where errors occur).

### Final Results
- **Final Test Loss**: **$0.1066$**
- **Final Test Accuracy**: **$0.9799$ ($98.0\% \ge 95\%$)**
- **Best Epoch**: 34
- **Confusion Pairs**: `[(3, 8, 4), (4, 9, 1), (1, 4, 1)]`
- **Plot Saved**: [`figures/final_curves.png`](./figures/final_curves.png)

### Architecture & Hyperparameters
- **Architecture**: 3-layer MLP: $64 \to 128 \to 64 \to 10$
- **Hidden Activations**: ReLU with He initialization (`init_weights(..., "he")`)
- **Normalization & Regularization**: `LayerNorm` after linear layers, mild Dropout ($p=0.15$)
- **Optimizer**: AdamW ($\text{lr} = 2 \times 10^{-3}$, $\text{weight\_decay} = 10^{-4}$)
- **Scheduler**: `warmup_cosine` (warmup epochs = 5, total epochs = 50)
- **Gradient Clipping**: `clip_norm = 1.0` in `train_one_epoch`

### Ready-to-Copy LaTeX for Report (E4)
```latex
Selected configuration: 3-layer MLP ($64 \to 128 \to 64 \to 10$) with ReLU activation, LayerNorm, and Dropout ($p=0.15$), optimised with AdamW ($\text{lr}=2\times 10^{-3}$, $\text{weight\_decay}=10^{-4}$), warm-up cosine schedule, and gradient clipping ($\text{clip\_norm}=1.0$). Hyperparameters were tuned strictly on validation accuracy (reaching 0.983). Evaluated on the test set exactly once, the final model achieved a test loss of \textbf{0.1066} and test accuracy of \textbf{0.9799} ($98.0\% \ge 95\%$, best epoch 34). The confusion analysis indicates misclassifications are concentrated on visually similar digits: digit 3 confused for 8 (4 times), digit 4 confused for 9 (1 time), and digit 1 confused for 4 (1 time).

\begin{center}
\includegraphics[width=0.72\linewidth]{figures/final_curves.png}
\end{center}
```

