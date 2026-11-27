# Week 5: Complex Matrices, Hermitian & Unitary Matrices, and the Spectral Theorem

> **Course:** Machine Learning Foundations (IIT Madras BS Degree)  
> **Topics Covered:** Complex Vector Spaces ($\mathbb{C}^n$), Complex Inner Product & Norm, Conjugate Transpose ($A^*$), Hermitian Matrices & Properties, Unitary Matrices & Properties, Schur's Theorem, Spectral Theorem (Diagonalization of Hermitian & Real Symmetric Matrices)

---

## 1. Complex Vector Spaces & Matrices

### 1.1 Complex Numbers Review
A complex number $z = a + ib$ has real part $a$, imaginary part $b$, and **complex conjugate** $\bar{z} = a - ib$.
* In polar form: $z = r e^{i\theta}$ and $\bar{z} = r e^{-i\theta}$, where $r^2 = a^2 + b^2 = |z|^2 = z\bar{z}$.

### 1.2 Complex Inner Product & Vector Length
In $\mathbb{R}^n$, length squared is $\|x\|^2 = x^T x$. **Why can't we use $x^T x$ in $\mathbb{C}^n$?**
Consider $x = \begin{bmatrix} 1 \\ i \end{bmatrix} \in \mathbb{C}^2$:
$$x^T x = 1^2 + i^2 = 1 - 1 = 0 \quad \text{even though } x \neq \mathbf{0}!$$
To fix this, we define the **complex inner product** using the conjugate of the first vector:
$$x \cdot y = \bar{x}^T y = x^* y = \sum_{k=1}^n \bar{x}_k y_k$$
Then the squared length of $x \in \mathbb{C}^n$ is:
$$\|x\|^2 = x^* x = \sum_{k=1}^n |x_k|^2 \ge 0$$
Now $\|x\| = 0 \iff x = \mathbf{0}$, and $\left\|\begin{bmatrix} 1 \\ i \end{bmatrix}\right\| = \sqrt{1^2 + |i|^2} = \sqrt{2}$.

#### Key Properties of the Complex Inner Product:
1. $x \cdot y = \overline{y \cdot x}$ (conjugate symmetry — order matters!).
2. $x \cdot (cy) = c (x \cdot y)$ (linear in the second argument).
3. $(cx) \cdot y = \bar{c} (x \cdot y)$ (conjugate-linear in the first argument).

### 1.3 Conjugate Transpose ($A^*$)
The **conjugate transpose** (or Hermitian transpose) of a matrix $A$, denoted $A^*$ (or $A^H$), is obtained by taking the complex conjugate and transposing:
$$A^* = \overline{A}^T = \overline{A^T}$$
* For a **real** matrix $A$, $A^* = A^T$.
* **Properties:** $(A^*)^* = A$ and $(AB)^* = B^* A^*$.

---

## 2. Hermitian Matrices

### 2.1 Definition
A square complex matrix $A \in \mathbb{C}^{n \times n}$ is **Hermitian** if it equals its own conjugate transpose:
$$A^* = A$$
* **Analog in $\mathbb{R}$:** Real symmetric matrices ($A^T = A$) are a special case of Hermitian matrices.
* **Note:** Diagonal entries of any Hermitian matrix must be **real** (since $a_{ii} = \bar{a}_{ii}$).

*Example:* $A = \begin{bmatrix} 2 & 3 - 3i \\ 3 + 3i & 5 \end{bmatrix}$ satisfies $A^* = A$.

### 2.2 Two Fundamental Properties of Hermitian Matrices

#### Property I: All eigenvalues of a Hermitian matrix are REAL.
* **Proof:** Suppose $Ax = \lambda x$ with $x \neq \mathbf{0}$.
  Take the conjugate transpose: $(Ax)^* = (\lambda x)^* \implies x^* A^* = \bar{\lambda} x^*$.
  Multiply on the right by $x$: $x^* A^* x = \bar{\lambda} x^* x$.
  Since $A^* = A$ and $Ax = \lambda x$:
  $$x^* (Ax) = x^* (\lambda x) = \lambda (x^* x) \implies \lambda (x^* x) = \bar{\lambda} (x^* x)$$
  Since $x \neq \mathbf{0}$, $x^* x = \|x\|^2 > 0$, so $\lambda = \bar{\lambda} \implies \lambda \in \mathbb{R}$. $\blacksquare$

#### Property II: Eigenvectors corresponding to distinct eigenvalues are ORTHOGONAL.
* **Proof:** Suppose $Ax = \lambda_1 x$ and $Ay = \lambda_2 y$ with $\lambda_1 \neq \lambda_2$.
  1. $x \cdot (Ay) = x \cdot (\lambda_2 y) = \lambda_2 (x \cdot y)$.
  2. Also, $x \cdot (Ay) = x^* A y = x^* A^* y = (Ax)^* y = (\lambda_1 x)^* y = \bar{\lambda}_1 (x^* y) = \lambda_1 (x \cdot y)$ (using $A^* = A$ and $\lambda_1 \in \mathbb{R}$).
  Equating (1) and (2): $(\lambda_2 - \lambda_1)(x \cdot y) = 0$. Since $\lambda_1 \neq \lambda_2$, we must have $x \cdot y = x^* y = 0$. $\blacksquare$

---

## 3. Unitary Matrices

### 3.1 Definition
A square matrix $U \in \mathbb{C}^{n \times n}$ is **unitary** if its columns are orthonormal:
$$U^* U = I \quad \Longleftrightarrow \quad U^{-1} = U^*$$
* **Analog in $\mathbb{R}$:** Real **orthogonal matrices** $Q$ ($Q^T Q = I \iff Q^{-1} = Q^T$).
* *Examples:*
  * $U = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & i \\ i & 1 \end{bmatrix}$ is unitary.
  * Rotation matrix $Q = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ is orthogonal (hence unitary).

### 3.2 Properties of Unitary Matrices
1. **Preserves Inner Products and Lengths:**
   $$(Ux) \cdot (Uy) = (Ux)^* (Uy) = x^* U^* U y = x^* y = x \cdot y$$
   Setting $y = x$ gives $\|Ux\| = \|x\|$ for all $x \in \mathbb{C}^n$.
2. **All Eigenvalues have Absolute Value $1$ ($|\lambda| = 1$):**
   If $Ux = \lambda x$ ($x \neq \mathbf{0}$), then $\|Ux\| = \|x\| \implies |\lambda|\|x\| = \|x\| \implies |\lambda| = 1$.
3. **Eigenvectors of Distinct Eigenvalues are Orthogonal:**
   For $Ux = \lambda_1 x, Uy = \lambda_2 y$ ($\lambda_1 \neq \lambda_2$), $x \cdot y = (Ux)\cdot(Uy) = \bar{\lambda}_1 \lambda_2 (x \cdot y)$. Since $|\lambda_1| = 1$ ($\bar{\lambda}_1 \lambda_1 = 1$), if $\bar{\lambda}_1 \lambda_2 = 1$ then $\lambda_2 = \lambda_1$ (contradiction). Thus $\bar{\lambda}_1 \lambda_2 \neq 1 \implies x \cdot y = 0$.

---

## 4. Diagonalization of Hermitian Matrices (The Spectral Theorem)

### 4.1 Schur's Theorem (Building Block)
**Theorem:** *Every* square matrix $A \in \mathbb{C}^{n \times n}$ is unitarily similar to an **upper-triangular** matrix $T$:
$$A = U T U^* \quad (\text{or } U^* A U = T)$$
where $U$ is unitary and $T$ is upper-triangular (with eigenvalues of $A$ on its diagonal).

### 4.2 The Spectral Theorem
**Theorem:** Every **Hermitian matrix** $A$ ($A^* = A$) is **unitarily diagonalizable**: there exists a unitary matrix $U$ and a **real diagonal** matrix $D$ such that:
$$U^* A U = D \quad \Longleftrightarrow \quad A = U D U^*$$

* **Proof via Schur's Theorem:**
  By Schur's Theorem, $U^* A U = T$ where $T$ is upper-triangular. Taking conjugate transpose:
  $$T^* = (U^* A U)^* = U^* A^* U = U^* A U = T$$
  Since $T$ is upper-triangular, $T^*$ is **lower-triangular**. A matrix that is both upper- and lower-triangular must be **diagonal**!
  Moreover, $T_{ii} = \overline{T_{ii}}$, so all diagonal entries (eigenvalues) are real. $\blacksquare$

#### Corollary: Real Symmetric Matrices
If $A$ is **real symmetric** ($A^T = A$), its eigenvalues are real, so $(A - \lambda I)x = 0$ yields real eigenvectors. Thus $U$ can be chosen as a **real orthogonal matrix $Q$**, giving:
$$Q^T A Q = D \quad \text{with } Q^T Q = I$$

> **Important Remark:**  
> Hermitian $\implies$ Unitarily Diagonalizable, **but the converse is FALSE!**  
> *Counterexample:* $A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ is not Hermitian ($A^* \neq A$), yet it has eigenvalues $\lambda = \pm i$ and orthonormal eigenvectors $\frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ \mp i \end{bmatrix}$, so it *is* unitarily diagonalizable ($U^* A U = \text{diag}(i, -i)$). *(In general, a matrix is unitarily diagonalizable iff it is **normal**: $A^*A = AA^*$.)*
