# Week 7: Principal Component Analysis (PCA)

> **Course:** Machine Learning Foundations (IIT Madras BS Degree)  
> **Topics Covered:** Dimensionality Reduction via PCA, Viewpoint 1: Minimizing Reconstruction Error, Sample Covariance Matrix, Viewpoint 2: Maximizing Projected Variance, PCA Algorithm & Worked Example, PCA in High Dimensions ($d \gg n$ Dual/Gram Matrix Trick)

---

## 1. What is Principal Component Analysis (PCA)?

Given an unlabeled dataset $\mathcal{D} = \{x_1, x_2, \dots, x_n\}$ where each $x_i \in \mathbb{R}^d$, we want to project the data onto a lower $m$-dimensional subspace ($m < d$) such that:
1. **Reconstruction error** is **minimized**, OR equivalently,
2. **Variance of the projected data** is **maximized**.

The orthonormal basis vectors $\{u_1, \dots, u_m\}$ of this optimal subspace are called the **Principal Directions**, and the projected coordinates are the **Principal Components**.

---

## 2. Viewpoint 1: Minimizing Reconstruction Error

Let $\mathcal{B} = \{u_1, \dots, u_m\}$ be an orthonormal basis for an $m$-dimensional subspace, extended to an orthonormal basis $\{u_1, \dots, u_m, u_{m+1}, \dots, u_d\}$ for $\mathbb{R}^d$.

1. **Exact representation of $x_i$:**
   $$x_i = \sum_{j=1}^d (x_i^T u_j) u_j$$
2. **Approximation $\tilde{x}_i$ using only $m$ dimensions:**  
   Each point keeps its own coordinates $z_{ij}$ along $u_1, \dots, u_m$, and shares a single constant offset $\beta_j$ along the discarded directions $u_{m+1}, \dots, u_d$:
   $$\tilde{x}_i = \sum_{j=1}^m z_{ij} u_j + \sum_{j=m+1}^d \beta_j u_j$$
3. **Minimize Mean Squared Reconstruction Error $J$:**
   $$J = \frac{1}{n} \sum_{i=1}^n \|x_i - \tilde{x}_i\|^2 = \frac{1}{n} \sum_{i=1}^n \left[ \sum_{j=1}^m (x_i^T u_j - z_{ij})^2 + \sum_{j=m+1}^d (x_i^T u_j - \beta_j)^2 \right]$$
   *(using orthonormality of $\{u_j\}$).*
   Setting $\frac{\partial J}{\partial z_{ij}} = 0$ and $\frac{\partial J}{\partial \beta_j} = 0$ yields:
   $$z_{ij} = x_i^T u_j, \qquad \beta_j = \bar{x}^T u_j \quad \text{where } \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$$
   *(Note: If the data is **centered** so $\bar{x} = \mathbf{0}$, then $\beta_j = 0$.)*
4. **Substitute optimal $z_{ij}, \beta_j$ back into $J$:**
   $$J^* = \frac{1}{n} \sum_{i=1}^n \sum_{j=m+1}^d \big((x_i - \bar{x})^T u_j\big)^2 = \sum_{j=m+1}^d u_j^T \underbrace{\left[\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^T\right]}_{C} u_j$$
   where $C \in \mathbb{R}^{d \times d}$ is the **Sample Covariance Matrix**!
5. **Optimize over the basis $\{u_1, \dots, u_d\}$:**  
   $C$ is real, symmetric, and positive semi-definite ($v^T C v \ge 0$), so it has real eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$ and orthonormal eigenvectors $\{u_1, \dots, u_d\}$.
   * To **minimize** $\sum_{j=m+1}^d u_j^T C u_j$, we choose $\{u_{m+1}, \dots, u_d\}$ to be the eigenvectors corresponding to the **$d - m$ smallest eigenvalues** $\lambda_{m+1}, \dots, \lambda_d$ (giving minimum error $J^* = \sum_{j=m+1}^d \lambda_j$).
   * Therefore, $\{u_1, \dots, u_m\}$ are the eigenvectors corresponding to the **top-$m$ largest eigenvalues** $\lambda_1, \dots, \lambda_m$ of $C$!

---

## 3. Viewpoint 2: Maximizing Projected Variance

Suppose we project centered data onto a single unit direction $u$ ($\|u\| = 1$).
* The projected value of $x_i$ is $x_i^T u$, and the mean projected value is $\bar{x}^T u$.
* The **variance of the projected data** is:
  $$\frac{1}{n} \sum_{i=1}^n (x_i^T u - \bar{x}^T u)^2 = u^T \left[\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^T\right] u = u^T C u$$
* **Optimization Problem:**
  $$\max_{u \in \mathbb{R}^d} u^T C u \quad \text{subject to } u^T u = 1 \qquad \left(\text{or equivalently } \max_{u} \frac{u^T C u}{u^T u}\right)$$
* Using a Lagrange multiplier $\mathcal{L}(u, \lambda) = u^T C u + \lambda(1 - u^T u)$ and setting $\nabla_u \mathcal{L} = 0$:
  $$C u = \lambda u \implies u^T C u = \lambda$$
* Thus, the maximum variance is achieved by choosing $u = u_1$ (the eigenvector of $C$ with the **largest eigenvalue $\lambda_1$**), and the projected variance equals $\lambda_1$!
* Extending to $m > 1$ orthogonal unit directions: pick the eigenvectors corresponding to the **top-$m$ eigenvalues** of $C$.

---

## 4. The PCA Algorithm & Worked Example

### Algorithm Summary
1. **Input:** Data $\{x_1, \dots, x_n\} \subset \mathbb{R}^d$, target dimension $m < d$.
2. **Compute Mean & Covariance:**
   $$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i, \qquad C = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^T$$
3. **Eigendecompose $C$:** Find eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$ and orthonormal eigenvectors $\{u_1, \dots, u_d\}$.
4. **Project Data:** Each $x_i$ is compressed to $m$ coordinates $z_i = [x_i^T u_1, \dots, x_i^T u_m]^T \in \mathbb{R}^m$ (or reconstructed in $\mathbb{R}^d$ as $\tilde{x}_i = \sum_{j=1}^m (x_i^T u_j) u_j + \sum_{j=m+1}^d (\bar{x}^T u_j) u_j$).

### Worked Example (from Lecture)
Consider $\mathcal{D} = \left\{ \begin{bmatrix} -1 \\ -1 \end{bmatrix}, \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 1 \\ 1 \end{bmatrix} \right\}$ and target dimension $m = 1$:
1. **Mean:** $\bar{x} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$.
2. **Covariance:**
   $$C = \frac{1}{3} \left( \begin{bmatrix} -1 \\ -1 \end{bmatrix}\begin{bmatrix} -1 & -1 \end{bmatrix} + \mathbf{0} + \begin{bmatrix} 1 \\ 1 \end{bmatrix}\begin{bmatrix} 1 & 1 \end{bmatrix} \right) = \frac{2}{3}\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$$
3. **Eigenvalues & Eigenvectors of $C$:**
   * $\lambda_1 = \frac{4}{3}$ with $u_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$.
   * $\lambda_2 = 0$ with $u_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}$.
4. **Projection onto $u_1$:**
   * Projected coordinates: $x_1^T u_1 = -\sqrt{2}$, $x_2^T u_1 = 0$, $x_3^T u_1 = \sqrt{2}$.
   * Projected variance = $\frac{1}{3}\big((-\sqrt{2})^2 + 0^2 + (\sqrt{2})^2\big) = \frac{4}{3} = \lambda_1$.
   * Reconstruction error $J^* = \lambda_2 = 0$ (all points lie exactly on the 1D line!).

---

## 5. PCA in Higher Dimensions ($d \gg n$)

In applications like image processing (e.g., face recognition), the feature dimension $d$ (pixels) can be $100{,}000$, while the number of samples $n$ is only $500$ ($d \gg n$). Forming and eigendecomposing the $d \times d$ matrix $C$ is computationally prohibitive!

### The Gram Matrix ($n \times n$) Trick
Let $A \in \mathbb{R}^{n \times d}$ be the centered data matrix whose $i$-th row is $(x_i - \bar{x})^T$. Then:
$$C = \frac{1}{n} A^T A \quad (d \times d \text{ matrix})$$
Notice that $\text{rank}(C) \le n$, so at least $d - n$ eigenvalues of $C$ are $0$.

Instead, consider the much smaller $n \times n$ matrix $\frac{1}{n} A A^T$:
1. Find an eigenvector $v_i \in \mathbb{R}^n$ and eigenvalue $\lambda_i > 0$ of $\frac{1}{n} A A^T$:
   $$\frac{1}{n} A A^T v_i = \lambda_i v_i$$
2. Multiply both sides on the left by $A^T$:
   $$\left(\frac{1}{n} A^T A\right) (A^T v_i) = \lambda_i (A^T v_i) \implies C (A^T v_i) = \lambda_i (A^T v_i)$$
3. **Result:** $A^T v_i$ is an eigenvector of $C$ with the exact same eigenvalue $\lambda_i$! (Simply normalize $u_i = \frac{A^T v_i}{\|A^T v_i\|}$ to get the unit principal direction.)

This makes PCA fast and tractable even when $d$ is millions!
