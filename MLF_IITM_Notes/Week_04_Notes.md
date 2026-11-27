# Week 4: Linear & Polynomial Regression, Eigenvalues, Diagonalization & Symmetric Matrices

> **Course:** Machine Learning Foundations (IIT Madras BS Degree)  
> **Topics Covered:** Linear Regression (MLE connection), Polynomial Regression, Ridge Regression (Regularization), Eigenvalues & Eigenvectors, Matrix Diagonalization, Solving Recurrences (Fibonacci Sequence), Orthogonal Diagonalization of Symmetric Matrices

---

## 1. Linear, Polynomial, and Ridge Regression

### 1.1 Linear Regression (Matrix Form)
Given data $\{(x_1, y_1), \dots, (x_n, y_n)\}$ with $x_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$, we minimize the squared loss:
$$L(\theta) = \frac{1}{2} \sum_{i=1}^n (x_i^T \theta - y_i)^2 = \frac{1}{2} (A\theta - Y)^T (A\theta - Y)$$
where $A = \begin{bmatrix} x_1^T \\ \vdots \\ x_n^T \end{bmatrix} \in \mathbb{R}^{n \times d}$ is the **feature matrix** and $Y = \begin{bmatrix} y_1 \\ \vdots \\ y_n \end{bmatrix} \in \mathbb{R}^n$.
Setting $\nabla_\theta L(\theta) = 0$ gives the **Normal Equations**:
$$A^T (A\theta - Y) = 0 \implies A^T A \theta = A^T Y$$
If $A$ has full column rank, $A^T A$ is invertible (since $N(A) = N(A^T A)$), giving $\theta = (A^T A)^{-1} A^T Y$.

#### Connection to Maximum Likelihood Estimation (MLE)
If data is generated via $y_i = \theta^T x_i + \epsilon_i$, where $\epsilon_i \sim \mathcal{N}(0, 1/\beta)$ is i.i.d. zero-mean Gaussian noise, the likelihood function is:
$$\mathcal{L}(\theta) = \prod_{i=1}^n \sqrt{\frac{\beta}{2\pi}} \exp\left(-\frac{\beta}{2}(y_i - \theta^T x_i)^2\right)$$
Taking the logarithm:
$$\log \mathcal{L}(\theta) = \frac{n}{2}\log\beta - \frac{n}{2}\log(2\pi) - \beta \left[\frac{1}{2}\sum_{i=1}^n (y_i - \theta^T x_i)^2\right]$$
**Key Takeaway:** Maximizing $\log \mathcal{L}(\theta)$ is mathematically identical to **minimizing the least squares objective**!

---

### 1.2 Polynomial Regression
Suppose $x_i \in \mathbb{R}$ and we want to fit a degree-$m$ polynomial:
$$\hat{y}(x) = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_m x^m = \theta^T \phi(x)$$
Define the **transformed feature vector** $\phi(x) = [1, x, x^2, \dots, x^m]^T \in \mathbb{R}^{m+1}$.
* Even though $\hat{y}(x)$ is nonlinear in $x$, it is **linear in $\theta$**!
* So we simply replace $x_i$ with $\phi(x_i)$ in the feature matrix $A$ and solve the exact same Normal Equations: $(A^T A)\theta = A^T Y$.

---

### 1.3 Ridge Regression (L2 Regularization)
To prevent overfitting (and fix non-invertibility when $A$ is not full rank), we add a penalty on the size of $\theta$:
$$\min_{\theta} \overline{L}(\theta) = \frac{1}{2} \sum_{i=1}^n (x_i^T \theta - y_i)^2 + \frac{\lambda}{2} \|\theta\|^2$$
*(Note: Some slides write $+\lambda\|\theta\|^2$, which absorbs a factor of 2 into $\lambda$.)*
Setting the gradient to zero yields the **Ridge Regression solution**:
$$(A^T A + \lambda I)\theta_{\text{reg}} = A^T Y \implies \theta_{\text{reg}} = (A^T A + \lambda I)^{-1} A^T Y$$
* **Key Property:** For any $\lambda > 0$, $(A^T A + \lambda I)$ is **always invertible**, even if $A$ is not full rank!
* **Role of $\lambda$:**
  * $\lambda$ too small $\implies$ **Overfitting** (high variance, fits noise).
  * $\lambda$ too large $\implies$ **Underfitting** (high bias, forces $\theta \approx 0$).

---

## 2. Eigenvalues and Eigenvectors

### 2.1 Motivation: Systems of Differential Equations
Consider a coupled system $\frac{du}{dt} = Au$ with initial condition $u(0)$.
* In 1D ($\frac{du}{dt} = au$), the solution is $u(t) = e^{at} u(0)$ (unstable if $a > 0$, neutral if $a = 0$, stable if $a < 0$).
* In $n$D, try a solution of the form $u(t) = e^{\lambda t} x$ ($x \neq \mathbf{0}$):
  $$\frac{d}{dt}(e^{\lambda t} x) = A(e^{\lambda t} x) \implies \lambda e^{\lambda t} x = e^{\lambda t} A x \implies \boxed{A x = \lambda x}$$

### 2.2 Definition & Geometric Intuition
For a square matrix $A \in \mathbb{R}^{n \times n}$, a scalar $\lambda$ is an **eigenvalue** and a **non-zero** vector $x \neq \mathbf{0}$ is a corresponding **eigenvector** if:
$$A x = \lambda x$$
* **Geometry:** Most vectors change direction when multiplied by $A$. An eigenvector $x$ only gets **stretched or shrunk** (or flipped if $\lambda < 0$) by factor $\lambda$, without changing its line of direction!
* **Special Case $\lambda = 0$:** $Ax = \mathbf{0} \iff x \in N(A)$. Thus, $0$ is an eigenvalue of $A$ iff $A$ is singular (non-invertible).

### 2.3 How to Find Eigenvalues and Eigenvectors
$$Ax = \lambda x \iff (A - \lambda I)x = \mathbf{0}$$
For a non-zero solution $x \neq \mathbf{0}$ to exist, $A - \lambda I$ must be singular:
$$\det(A - \lambda I) = 0 \quad \text{(Characteristic Equation)}$$
1. Solve $\det(A - \lambda I) = 0$ to find the eigenvalues $\lambda_1, \dots, \lambda_n$.
2. For each $\lambda_i$, find the corresponding eigenvectors by finding the null space $N(A - \lambda_i I)$.

#### Classic Examples:
1. **Projection Matrix $P$ onto a plane:**
   * Any $x$ *in* the plane satisfies $Px = x \implies \lambda = 1$.
   * Any $x$ *perpendicular* to the plane satisfies $Px = 0 \implies \lambda = 0$.
2. **Permutation Matrix $B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$:**
   * $B\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \implies \lambda_1 = 1$.
   * $B\begin{bmatrix} 1 \\ -1 \end{bmatrix} = -\begin{bmatrix} 1 \\ -1 \end{bmatrix} \implies \lambda_2 = -1$.

---

## 3. Similarity and Matrix Diagonalization

### 3.1 Definition
A square matrix $A \in \mathbb{R}^{n \times n}$ is **diagonalizable** if there exists an invertible matrix $S$ and a diagonal matrix $\Lambda$ such that:
$$S^{-1} A S = \Lambda \quad \Longleftrightarrow \quad A = S \Lambda S^{-1}$$

### 3.2 When is $A$ Diagonalizable?
Suppose $A$ has **$n$ linearly independent eigenvectors** $x_1, \dots, x_n$ with eigenvalues $\lambda_1, \dots, \lambda_n$.
Form $S = [x_1 \mid x_2 \mid \dots \mid x_n]$ and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$. Then:
$$AS = [Ax_1 \mid \dots \mid Ax_n] = [\lambda_1 x_1 \mid \dots \mid \lambda_n x_n] = S \Lambda \implies S^{-1} A S = \Lambda$$
* **Key Theorem:** Eigenvectors corresponding to **distinct** eigenvalues are always linearly independent.
  * *Corollary:* If an $n \times n$ matrix $A$ has $n$ distinct eigenvalues, it is guaranteed to be diagonalizable!
  * *Warning:* Not all matrices are diagonalizable (e.g., $A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$ only has $\lambda = 0$ with 1D nullspace, so it lacks 2 independent eigenvectors).

### 3.3 Why Diagonalize? Computing $A^k$ & Solving Recurrences
If $A = S \Lambda S^{-1}$, then:
$$A^k = S \Lambda^k S^{-1}$$
Equivalently, if we write an initial vector $u_0 = c_1 x_1 + \dots + c_n x_n$ in the eigenbasis, then after $k$ steps:
$$u_k = A^k u_0 = c_1 \lambda_1^k x_1 + c_2 \lambda_2^k x_2 + \dots + c_n \lambda_n^k x_n$$

#### Worked Example: Fibonacci Sequence ($F_{k+2} = F_{k+1} + F_k$, $F_0 = 0, F_1 = 1$)
1. **Rewrite as a matrix recurrence:** Let $u_k = \begin{bmatrix} F_{k+1} \\ F_k \end{bmatrix}$. Then:
   $$u_{k+1} = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix} u_k = A u_k \implies u_k = A^k u_0, \quad u_0 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$
2. **Eigenvalues of $A$:** $\det(A - \lambda I) = \lambda^2 - \lambda - 1 = 0 \implies \lambda_1 = \frac{1 + \sqrt{5}}{2} \approx 1.618$ (golden ratio), $\lambda_2 = \frac{1 - \sqrt{5}}{2} \approx -0.618$.
3. **Eigenvectors:** $x_1 = \begin{bmatrix} \lambda_1 \\ 1 \end{bmatrix}$, $x_2 = \begin{bmatrix} \lambda_2 \\ 1 \end{bmatrix}$.
4. **Decompose $u_0$:** $\begin{bmatrix} 1 \\ 0 \end{bmatrix} = \frac{1}{\sqrt{5}} x_1 - \frac{1}{\sqrt{5}} x_2$.
5. **Closed-form $F_k$ (Binet's Formula):**
   $$F_k = \frac{1}{\sqrt{5}} \left(\frac{1 + \sqrt{5}}{2}\right)^k - \frac{1}{\sqrt{5}} \left(\frac{1 - \sqrt{5}}{2}\right)^k$$
   Since $|\lambda_2| < 1$, $\lambda_2^k \to 0$ as $k$ grows, so $F_{100} \approx \frac{1}{\sqrt{5}}(1.618)^{100}$.

---

## 4. Orthogonal Diagonalization of Symmetric Matrices

If $A \in \mathbb{R}^{n \times n}$ is a **real symmetric matrix** ($A^T = A$), it has extraordinary properties (**The Real Spectral Theorem**):
1. All eigenvalues of $A$ are **real**.
2. Eigenvectors corresponding to **distinct** eigenvalues are **orthogonal** ($x_i^T x_j = 0$).
3. $A$ is **always orthogonally diagonalizable**! That is, there exists an **orthogonal matrix** $Q$ ($Q^T Q = I \implies Q^{-1} = Q^T$) whose columns are orthonormal eigenvectors $q_1, \dots, q_n$, such that:
   $$A = Q \Lambda Q^T = \sum_{i=1}^n \lambda_i q_i q_i^T$$

#### Worked Example:
For $A = \begin{bmatrix} 1 & -2 \\ -2 & -2 \end{bmatrix}$:
* Eigenvalues: $\lambda_1 = -3$, $\lambda_2 = 2$.
* Eigenvectors: $x_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$, $x_2 = \begin{bmatrix} -2 \\ 1 \end{bmatrix}$ (notice $x_1^T x_2 = 0$!).
* Normalize to unit length: $q_1 = \frac{1}{\sqrt{5}}\begin{bmatrix} 1 \\ 2 \end{bmatrix}$, $q_2 = \frac{1}{\sqrt{5}}\begin{bmatrix} -2 \\ 1 \end{bmatrix}$.
* Then $Q = \begin{bmatrix} 1/\sqrt{5} & -2/\sqrt{5} \\ 2/\sqrt{5} & 1/\sqrt{5} \end{bmatrix}$ satisfies $Q^T Q = I$, and $A = Q \begin{bmatrix} -3 & 0 \\ 0 & 2 \end{bmatrix} Q^T$.
