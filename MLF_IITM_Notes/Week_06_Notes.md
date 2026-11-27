# Week 6: Singular Value Decomposition (SVD) & Positive Definite Matrices

> **Course:** Machine Learning Foundations (IIT Madras BS Degree)  
> **Topics Covered:** Singular Value Decomposition (SVD), Proof & Construction of SVD, Worked SVD Example, Quadratic Forms, Positive Definite & Semi-Definite Matrices, Eigenvalue Equivalence Theorem

---

## 1. Singular Value Decomposition (SVD)

### 1.1 Why Do We Need SVD?
Eigendecomposition ($A = Q\Lambda Q^T$) only works for **square, diagonalizable** (e.g., symmetric) matrices.  
**What about rectangular $m \times n$ matrices (like data matrices), or non-diagonalizable square matrices?**  
**Theorem (SVD):** *Every* real $m \times n$ matrix $A$ can be factored as:
$$A = Q_1 \Sigma Q_2^T \quad (\text{often written } A = U \Sigma V^T)$$
where:
* $Q_1 \in \mathbb{R}^{m \times m}$ is an **orthogonal** matrix ($Q_1^T Q_1 = I_m$). Its columns $\{y_1, \dots, y_m\}$ are called **left singular vectors**.
* $Q_2 \in \mathbb{R}^{n \times n}$ is an **orthogonal** matrix ($Q_2^T Q_2 = I_n$). Its columns $\{x_1, \dots, x_n\}$ are called **right singular vectors**.
* $\Sigma \in \mathbb{R}^{m \times n}$ is a rectangular **diagonal** matrix of the form $\begin{bmatrix} D & 0 \\ 0 & 0 \end{bmatrix}$, where $D = \text{diag}(\sigma_1, \dots, \sigma_r)$ with **singular values** $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$, and $r = \text{rank}(A)$.

### 1.2 Step-by-Step Construction (Why SVD Always Exists)
1. **Form $A^T A$ ($n \times n$):** Even if $A$ is rectangular, $A^T A$ is square, real, and **symmetric** ($(A^T A)^T = A^T A$).
2. **Apply Spectral Theorem to $A^T A$:** There exists an orthonormal basis of eigenvectors $\{x_1, \dots, x_n\}$ in $\mathbb{R}^n$ with real eigenvalues $\lambda_1, \dots, \lambda_n$:
   $$A^T A x_i = \lambda_i x_i$$
3. **Why are all $\lambda_i \ge 0$?**
   $$(A^T A x_i) \cdot x_i = (\lambda_i x_i) \cdot x_i = \lambda_i \|x_i\|^2 = \lambda_i$$
   Also:
   $$(A^T A x_i) \cdot x_i = x_i^T A^T A x_i = (A x_i)^T (A x_i) = \|A x_i\|^2 \ge 0$$
   Therefore, $\lambda_i = \|Ax_i\|^2 \ge 0$! Order them: $\lambda_1 \ge \dots \ge \lambda_r > 0$ and $\lambda_{r+1} = \dots = \lambda_n = 0$.
4. **Define Singular Values:** $\sigma_i = \sqrt{\lambda_i}$ for $i = 1, \dots, r$.
5. **Define Left Singular Vectors:** For $i = 1, \dots, r$, set:
   $$y_i = \frac{1}{\sigma_i} A x_i \in \mathbb{R}^m$$
   * Unit length: $\|y_i\| = \frac{1}{\sigma_i}\|Ax_i\| = \frac{\sqrt{\lambda_i}}{\sigma_i} = 1$.
   * Mutually orthogonal: For $i \neq j$, $y_i \cdot y_j = \frac{1}{\sigma_i \sigma_j} x_i^T A^T A x_j = \frac{\lambda_j}{\sigma_i \sigma_j} (x_i^T x_j) = 0$.
   * Extend $\{y_1, \dots, y_r\}$ to an orthonormal basis $\{y_1, \dots, y_m\}$ of $\mathbb{R}^m$.
6. **Verify $Q_1^T A Q_2 = \Sigma$:** The $(i, j)$ entry of $Q_1^T A Q_2$ is $y_i^T (Ax_j)$, which equals $\sigma_j y_i^T y_j = \sigma_i$ if $i = j \le r$, and $0$ otherwise!

#### Key Identities:
* $A^T A = Q_2 (\Sigma^T \Sigma) Q_2^T \implies$ Columns of $Q_2$ are eigenvectors of $A^T A$.
* $A A^T = Q_1 (\Sigma \Sigma^T) Q_1^T \implies$ Columns of $Q_1$ are eigenvectors of $A A^T$.

### 1.3 Worked Example: SVD of a Non-Diagonalizable Matrix
Find the SVD of $A = \begin{bmatrix} \sqrt{2} & 1 \\ 0 & \sqrt{2} \end{bmatrix}$.
*(Note: $A$ has repeated eigenvalue $\sqrt{2}$ with only one eigenvector $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$, so $A$ is **not** diagonalizable!)*
1. **Compute $A^T A$:**
   $$A^T A = \begin{bmatrix} \sqrt{2} & 0 \\ 1 & \sqrt{2} \end{bmatrix} \begin{bmatrix} \sqrt{2} & 1 \\ 0 & \sqrt{2} \end{bmatrix} = \begin{bmatrix} 2 & \sqrt{2} \\ \sqrt{2} & 3 \end{bmatrix}$$
2. **Eigenvalues of $A^T A$:** $\lambda_1 = 4, \ \lambda_2 = 1 \implies \sigma_1 = \sqrt{4} = 2, \ \sigma_2 = \sqrt{1} = 1$, so $\Sigma = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$.
3. **Orthonormal Eigenvectors of $A^T A$ ($Q_2$):**
   $$x_1 = \frac{1}{\sqrt{3}}\begin{bmatrix} 1 \\ \sqrt{2} \end{bmatrix}, \quad x_2 = \frac{1}{\sqrt{3}}\begin{bmatrix} \sqrt{2} \\ -1 \end{bmatrix} \implies Q_2 = \frac{1}{\sqrt{3}}\begin{bmatrix} 1 & \sqrt{2} \\ \sqrt{2} & -1 \end{bmatrix}$$
4. **Left Singular Vectors ($Q_1$):** Using $y_i = \frac{1}{\sigma_i} A x_i$:
   $$y_1 = \frac{1}{2} A x_1 = \frac{1}{\sqrt{3}}\begin{bmatrix} \sqrt{2} \\ 1 \end{bmatrix}, \quad y_2 = \frac{1}{1} A x_2 = \frac{1}{\sqrt{3}}\begin{bmatrix} 1 \\ -\sqrt{2} \end{bmatrix} \implies Q_1 = \frac{1}{\sqrt{3}}\begin{bmatrix} \sqrt{2} & 1 \\ 1 & -\sqrt{2} \end{bmatrix}$$
5. **Check:** $Q_1 \Sigma Q_2^T = A$. $\checkmark$

---

## 2. Positive Definite Matrices & Quadratic Forms

### 2.1 Motivation: Classifying Stationary Points
Consider a 2-variable quadratic function:
$$f(x, y) = ax^2 + 2bxy + cy^2 = \begin{bmatrix} x & y \end{bmatrix} \begin{bmatrix} a & b \\ b & c \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = v^T A v$$
At $(0, 0)$, $\nabla f(0, 0) = \mathbf{0}$ (stationary point) and $f(0, 0) = 0$. When is $(0, 0)$ a **strict global minimum** (i.e., $f(x, y) > 0$ for all $(x, y) \neq (0, 0)$)?

Completing the square (assuming $a \neq 0$):
$$f(x, y) = a\left(x + \frac{b}{a}y\right)^2 + \left(c - \frac{b^2}{a}\right)y^2$$
This is strictly positive for all $(x, y) \neq (0, 0)$ **if and only if**:
1. $a > 0$ (checked by testing $(1, 0)$), and
2. $ac - b^2 > 0$ (which is $\det(A) > 0$, and also implies $c > 0$).

* **Notice:** $\det(A) = \lambda_1 \lambda_2 = ac - b^2 > 0$ and $\text{Trace}(A) = \lambda_1 + \lambda_2 = a + c > 0 \iff \lambda_1 > 0 \text{ and } \lambda_2 > 0$!
* **Other cases:**
  * If $ac - b^2 < 0$ (eigenvalues have opposite signs): **Saddle point** at $(0, 0)$ (e.g., $f(x, y) = 2xy$).
  * If $ac - b^2 = 0$ and $a > 0$: **Positive semi-definite** ($f(x, y) \ge 0$).

### 2.2 Formal Definition in $\mathbb{R}^n$
A real **symmetric** $n \times n$ matrix $A$ is:
* **Positive Definite (PD)** ($A \succ 0$) if:
  $$v^T A v > 0 \quad \forall v \in \mathbb{R}^n, \ v \neq \mathbf{0}$$
* **Positive Semi-Definite (PSD)** ($A \succeq 0$) if:
  $$v^T A v \ge 0 \quad \forall v \in \mathbb{R}^n$$

### 2.3 Equivalence Theorem: $v^T A v > 0 \iff$ All Eigenvalues $\lambda_i > 0$

**Proof ($\implies$):**  
Suppose $v^T A v > 0$ for all $v \neq \mathbf{0}$. Let $x \neq \mathbf{0}$ be an eigenvector with $Ax = \lambda x$.
$$0 < x^T A x = x^T (\lambda x) = \lambda \|x\|^2$$
Since $\|x\|^2 > 0$, we must have $\lambda > 0$.

**Proof ($\Longleftarrow$):**  
Suppose all eigenvalues $\lambda_1, \dots, \lambda_n > 0$. Since $A$ is real symmetric, by the Spectral Theorem there exists an orthonormal basis of eigenvectors $\{q_1, \dots, q_n\}$.  
Any $v \neq \mathbf{0}$ can be written as $v = c_1 q_1 + \dots + c_n q_n$ (with not all $c_i = 0$). Then:
$$A v = c_1 \lambda_1 q_1 + \dots + c_n \lambda_n q_n$$
Taking the dot product with $v$ (using $q_i^T q_i = 1$ and $q_i^T q_j = 0$ for $i \neq j$):
$$v^T A v = c_1^2 \lambda_1 + c_2^2 \lambda_2 + \dots + c_n^2 \lambda_n > 0$$
since every $\lambda_i > 0$ and at least one $c_i^2 > 0$. $\blacksquare$
