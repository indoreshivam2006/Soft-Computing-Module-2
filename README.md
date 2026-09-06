# Soft Computing Practicals — Module 2: Artificial Neural Networks (ANN)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/indoreshivam2006/Soft-Computing-Practicals)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)

**Course:** Soft Computing (EL1) — `STDA2102` | **Faculty:** Rahul Khajuria (FAST, Jagran Lakecity University)  
**Lab Focus:** 10 end-to-end practical experiments covering ANN foundations from scratch mathematical implementations in NumPy to deep learning architectures in TensorFlow/Keras. All notebooks contain pre-rendered execution outputs, plots, and evaluation metrics.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/indoreshivam2006/Soft-Computing-Practicals.git
cd Soft-Computing-Practicals

# Setup environment & install dependencies
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install numpy matplotlib scikit-learn tensorflow jupyter

# Launch notebooks
jupyter notebook
```
> **Note:** All `.ipynb` notebooks already contain saved plots and outputs. Running is optional.

---

## Experiments Master Matrix

| # | Experiment Notebook | Core Concept & Mathematical Basis | Tech Stack | Dataset | Benchmark Result |
|:---:|---|---|---|---|---|
| **01** | [`Experiment_1_Artificial_Neuron.ipynb`](Experiment_1_Artificial_Neuron.ipynb) | Biological-to-artificial mapping; Step activation $f(z) = 1 \text{ if } z \ge 0 \text{ else } 0$ | Pure Python, NumPy | Synthetic (2D) | Correct threshold firing on arbitrary test inputs |
| **02** | [`Experiment_2_Weighted_Sum_and_Bias.ipynb`](Experiment_2_Weighted_Sum_and_Bias.ipynb) | Affine transformation $z = \mathbf{w}^T\mathbf{x} + b$; bias as origin-independent threshold shifter | Pure Python, NumPy | Synthetic (3D) | Manual loop & vectorized `np.dot` equivalence verified |
| **03** | [`Experiment_3_Activation_Functions.ipynb`](Experiment_3_Activation_Functions.ipynb) | Non-linear mapping: Sigmoid $\frac{1}{1+e^{-z}}$, Tanh $\frac{e^z-e^{-z}}{e^z+e^{-z}}$, ReLU $\max(0,z)$ & derivatives | NumPy, Matplotlib | $z \in [-5, 5]$ | Characteristic S-curves and gradients visualized |
| **04** | [`Experiment_4_Single_Layer_Neural_Network.ipynb`](Experiment_4_Single_Layer_Neural_Network.ipynb) | Perceptron learning rule: $w_i \leftarrow w_i + \eta(y - \hat{y})x_i$, $b \leftarrow b + \eta(y - \hat{y})$ | Pure Python, NumPy | AND Logic Gate | Converged in **11 epochs** (100% accuracy) |
| **05** | [`Experiment_5_Multilayer_Perceptron_Keras.ipynb`](Experiment_5_Multilayer_Perceptron_Keras.ipynb) | Overcoming linear separability via hidden layer representations; binary cross-entropy | TensorFlow, Keras | XOR Logic Gate | **100% accuracy** on non-linear XOR problem |
| **06** | [`Experiment_6_Gradient_Descent.ipynb`](Experiment_6_Gradient_Descent.ipynb) | MSE loss minimization via analytical partial derivatives $\frac{\partial L}{\partial w}$, $\frac{\partial L}{\partial b}$; parameter updates | Pure Python, NumPy | Synthetic Linear | Parameters recovered: $w \to 2.82, b \to 2.64$ |
| **07** | [`Experiment_7_Backpropagation.ipynb`](Experiment_7_Backpropagation.ipynb) | Multi-layer chain rule from scratch: $\delta_o = (\hat{y}-y)\sigma'$, $\delta_h = (\delta_o W_2^T)\sigma'$, gradient accumulation | Pure Python, NumPy | XOR Logic Gate | Loss dropped **$0.29 \to 0.0015$** (5000 epochs) |
| **08** | [`Experiment_8_Train_Model_Keras.ipynb`](Experiment_8_Train_Model_Keras.ipynb) | Multi-class deep classification: `Dense(16) -> Dense(8) -> Softmax(3)`, Adam optimizer | TensorFlow, Keras | Fisher's Iris (150 samples) | **96.67% Test Accuracy** (29/30 correct) |
| **09** | [`Experiment_9_Evaluate_Model_Performance.ipynb`](Experiment_9_Evaluate_Model_Performance.ipynb) | Rigorous evaluation: Confusion Matrix, Precision $\frac{TP}{TP+FP}$, Recall $\frac{TP}{TP+FN}$, $F_1 = \frac{2PR}{P+R}$ | Scikit-learn, Keras | Fisher's Iris test split | **0.97 Macro F1-Score**, full confusion matrix |
| **10** | [`Experiment_10_Complete_ANN_Classification_System.ipynb`](Experiment_10_Complete_ANN_Classification_System.ipynb) | End-to-end clinical pipeline: `StandardScaler` $\to$ `Dense(32-16-8-1)` $\to$ validation monitoring $\to$ inference | Scikit-learn, Keras | Breast Cancer Wisconsin | **96.49% Acc**, Precision: 0.986, Recall: 0.959 |

---

## Datasets Overview

| Dataset | Type | Dimensions | Classes | Practical Relevance | Used In |
|---|---|---|---|---|---|
| **AND Gate** | Synthetic | 4 samples, 2 features | 2 (0, 1) | Baseline linearly separable logic | Exp 4 |
| **XOR Gate** | Synthetic | 4 samples, 2 features | 2 (0, 1) | Non-linear separability benchmark | Exp 5, 7 |
| **Fisher's Iris** | Botanical | 150 samples, 4 features | 3 species | Multi-class continuous feature classification | Exp 8, 9 |
| **Breast Cancer Wisconsin** | Clinical Diagnostic | 569 samples, 30 features | 2 (Malignant, Benign) | Real-world imbalanced medical diagnostic pipeline | Exp 10 |

---

## Core Theoretical Takeaways

- **Why Non-Linearity Matters:** Stacking purely linear layers collapses into a single affine transformation ($W_2(W_1X+b_1)+b_2 = W'X+b'$). Non-linear activations ($\sigma$, $\tanh$, $\text{ReLU}$) allow networks to approximate arbitrary functions.
- **Single-Layer Limits (Minsky & Papert, 1969):** A single perceptron can only form linear hyperplanes (AND/OR). Non-linearly separable problems (XOR) mathematically mandate hidden layers to transform the feature space.
- **Optimization Distinction:** *Gradient Descent* defines the parameter update policy given gradients; *Backpropagation* applies the multivariate chain rule to systematically calculate those gradients layer-by-layer backwards.
- **Clinical Evaluation Trade-Off:** In disease diagnostics (Exp 10), **Recall (0.959)** is prioritized over raw accuracy to minimize False Negatives (unidentified malignancies).

---

## Repository Structure

```text
Soft-Computing-Practicals/
├── Experiment_1_Artificial_Neuron.ipynb                      # Single neuron model & thresholding
├── Experiment_2_Weighted_Sum_and_Bias.ipynb                  # Dot product & bias shifting
├── Experiment_3_Activation_Functions.ipynb                   # Sigmoid, Tanh, ReLU & derivatives
├── Experiment_4_Single_Layer_Neural_Network.ipynb            # Perceptron learning rule (AND Gate)
├── Experiment_5_Multilayer_Perceptron_Keras.ipynb            # MLP solving non-linear XOR in Keras
├── Experiment_6_Gradient_Descent.ipynb                       # Analytical MSE optimization
├── Experiment_7_Backpropagation.ipynb                        # Scratch backpropagation algorithm
├── Experiment_8_Train_Model_Keras.ipynb                      # Deep classifier on Iris dataset
├── Experiment_9_Evaluate_Model_Performance.ipynb             # Metrics, Precision, Recall, F1
├── Experiment_10_Complete_ANN_Classification_System.ipynb    # End-to-end clinical diagnostic system
├── .gitignore                                                # Git ignore file (excludes .venv, caches)
└── README.md                                                 # Project lab manual & documentation
```
