# Week 3: Four Fundamental Subspaces, Orthogonality, Projections & Least Squares

> **Course:** Machine Learning Foundations (IIT Madras BS Degree)  
> **Topics Covered:** Column Space, Null Space, Row Space, Left Null Space, Rank-Nullity Theorem, Orthogonal Vectors & Subspaces, Orthogonal Complements, Projections onto Lines & Subspaces, Projection Matrix Properties, Normal Equations & Least Squares Regression

---

## 1. The Four Fundamental Subspaces of a Matrix $A$

Let $A \in \mathbb{R}^{m \times n}$ be a matrix with $m$ rows and $n$ columns, and let $\text{rank}(A) = r$.

### 1.1 Column Space $C(A)$
If $A = [u_1 \mid u_2 \mid \dots \mid u_n]$ where $u_i \in \mathbb{R}^m$ are the columns of $A$:
$$C(A) = \text{span}(u_1, \dots, u_n) = \{Ax : x \in \mathbb{R}^n\} \subseteq \mathbb{R}^m$$
* **Solvability of $Ax = b$:** The linear system $Ax = b$ has a solution **if and only if** $b \in C(A)$.
* **Dimension:** $\dim(C(A)) = r = \text{rank}(A)$ (the number of pivot columns / linearly independent columns).

### 1.2 Null Space $N(A)$
The set of all vectors mapped to the zero vector by $A$:
$$N(A) = \{x \in \mathbb{R}^n : Ax = \mathbf{0}\} \subseteq \mathbb{R}^n$$
* **Why is $N(A)$ a subspace?** If $Ax_1 = 0$ and $Ax_2 = 0$, then $A(x_1 + x_2) = 0$ and $A(\alpha x_1) = 0$.
* **How to find $N(A)$:** Use Gaussian elimination to reduce $A$ to echelon form $U$. Pivot columns correspond to pivot variables; non-pivot columns correspond to **free variables**. Setting each free variable to $1$ (and others to $0$) yields basis vectors for $N(A)$.
* **Dimension:** $\dim(N(A)) = \text{nullity}(A) = n - r$ (number of free variables).
* **General Solution of $Ax = b$:**
  * If $A$ is invertible ($r = n = m$), $N(A) = \{\mathbf{0}\}$ and $Ax = b$ has a unique solution $x = A^{-1}b$.
  * Otherwise, solutions take the form $x = x_p + x_n$, where $Ax_p = b$ (particular solution) and $x_n \in N(A)$ (homogeneous solution).

### 1.3 Row Space $R(A)$ or $C(A^T)$
The span of the rows of $A$, which is identical to the column space of $A^T$:
$$R(A) = C(A^T) \subseteq \mathbb{R}^n$$
* **Crucial Theorem:** Row rank = Column rank! Thus, $\dim(C(A^T)) = r = \text{rank}(A)$.

### 1.4 Left Null Space $N(A^T)$
The null space of $A^T$ (vectors $y$ that multiply $A$ from the left to give $\mathbf{0}^T$):
$$N(A^T) = \{y \in \mathbb{R}^m : A^T y = \mathbf{0}\} = \{y \in \mathbb{R}^m : y^T A = \mathbf{0}^T\} \subseteq \mathbb{R}^m$$
* **Dimension:** $\dim(N(A^T)) = m - r$.

### Summary Table of the Four Subspaces ($A \in \mathbb{R}^{m \times n}$, $\text{rank}(A) = r$)

| Subspace | Notation | Lives in | Dimension | Rank-Nullity Identity |
| :--- | :--- | :--- | :--- | :--- |
| **Column Space** | $C(A)$ | $\mathbb{R}^m$ | $r$ | $\dim(C(A)) + \dim(N(A^T)) = m$ |
| **Left Null Space** | $N(A^T)$ | $\mathbb{R}^m$ | $m - r$ | |
| **Row Space** | $C(A^T)$ or $R(A)$ | $\mathbb{R}^n$ | $r$ | $\dim(C(A^T)) + \dim(N(A)) = n$ |
| **Null Space** | $N(A)$ | $\mathbb{R}^n$ | $n - r$ | |

---

## 2. Orthogonality

### 2.1 Orthogonal and Orthonormal Vectors
* **Vector Length (Norm):** $\|x\|^2 = x^T x = \sum_{i=1}^n x_i^2$.
* **Orthogonal Vectors ($x \perp y$):** Two vectors $x, y \in \mathbb{R}^n$ are orthogonal if their inner product is zero:
  $$x^T y = 0$$
  *(By Pythagoras: $\|x\|^2 + \|y\|^2 = \|x + y\|^2 \iff 2x^T y = 0 \iff x^T y = 0$.)*
  *Note:* The zero vector $\mathbf{0}$ is orthogonal to every vector.
* **Linear Independence:** Any set of non-zero, mutually orthogonal vectors $\{v_1, \dots, v_k\}$ is automatically linearly independent.
* **Orthonormal Vectors:** Vectors $\{u_1, \dots, u_k\}$ that are mutually orthogonal ($u_i^T u_j = 0$ for $i \neq j$) **and** have unit length ($\|u_i\| = 1$).

### 2.2 Orthogonal Subspaces & Fundamental Theorem of Linear Algebra
Two subspaces $U$ and $V$ are **orthogonal ($U \perp V$)** if $x^T y = 0$ for all $x \in U$ and all $y \in V$.

**The Big Orthogonality Theorem:**
1. **Row Space $\perp$ Null Space:** $C(A^T) \perp N(A)$ in $\mathbb{R}^n$.
   * *Why?* $Ax = \mathbf{0}$ means every row of $A$ has dot product $0$ with $x$. Therefore, any linear combination of rows (anything in $C(A^T)$) is orthogonal to $x \in N(A)$.
   * Moreover, they are **orthogonal complements**: $\dim(C(A^T)) + \dim(N(A)) = r + (n-r) = n$.
2. **Column Space $\perp$ Left Null Space:** $C(A) \perp N(A^T)$ in $\mathbb{R}^m$.
   * *Why?* Apply Claim 1 to $A^T$! They are orthogonal complements in $\mathbb{R}^m$: $\dim(C(A)) + \dim(N(A^T)) = r + (m-r) = m$.

---

## 3. Projections

### 3.1 Why Project?
In real-world ML problems, we often have an **overdetermined, inconsistent** system $Ax = b$ (more equations than unknowns, $m > n$, and $b \notin C(A)$). Since no exact solution exists, we find the vector $p = A\hat{x} \in C(A)$ that is **closest** to $b$ — i.e., the **orthogonal projection** of $b$ onto $C(A)$.

### 3.2 Projection onto a Line (1D Subspace)
Suppose we want to project $b \in \mathbb{R}^m$ onto the line spanned by vector $a \in \mathbb{R}^m$.
* The projection is $p = \hat{x} a$ for some scalar $\hat{x}$.
* The error vector $e = b - p = b - \hat{x}a$ must be **orthogonal** to $a$:
  $$a^T (b - \hat{x} a) = 0 \implies a^T b - \hat{x}(a^T a) = 0 \implies \hat{x} = \frac{a^T b}{a^T a}$$
* Therefore, the projection vector $p$ is:
  $$p = \left(\frac{a^T b}{a^T a}\right) a = \underbrace{\left(\frac{a a^T}{a^T a}\right)}_{P} b$$
  Here, $P = \frac{a a^T}{a^T a}$ is the $m \times m$ **projection matrix** onto the line through $a$.

> **Bonus — Proof of Cauchy-Schwarz Inequality:**  
> Since $\|e\|^2 = \|b - p\|^2 \ge 0$, expanding $\left\|b - \frac{a^T b}{a^T a} a\right\|^2 \ge 0$ gives $(b^T b)(a^T a) \ge (a^T b)^2$, i.e., $|a^T b| \le \|a\| \|b\|$.

### 3.3 Projection onto a Subspace $C(A)$
Now let $A \in \mathbb{R}^{m \times n}$ have linearly independent columns, and we want to project $b \in \mathbb{R}^m$ onto $S = C(A)$.
* The projection is $p = A\hat{x} \in C(A)$.
* The error vector $e = b - A\hat{x}$ must be orthogonal to **every** column of $A$ (hence orthogonal to all of $C(A)$).
* Since $C(A)^\perp = N(A^T)$, this means $e \in N(A^T)$, so $A^T e = \mathbf{0}$:
  $$A^T (b - A\hat{x}) = \mathbf{0} \implies \boxed{A^T A \hat{x} = A^T b}$$
  These are the famous **Normal Equations**!

#### Solving for $\hat{x}$ and the Projection Matrix $P$:
If the columns of $A$ are **linearly independent**, then $A^T A$ is symmetric and **invertible** ($n \times n$). Thus:
$$\hat{x} = (A^T A)^{-1} A^T b$$
$$p = A\hat{x} = \underbrace{A (A^T A)^{-1} A^T}_{P} b$$

### 3.4 Properties of Any Projection Matrix $P$
1. **Symmetric:** $P^T = P$.
2. **Idempotent:** $P^2 = P$ (projecting a point that is already in the subspace doesn't move it!).
* *Converse is also true:* Any square matrix satisfying $P^T = P$ and $P^2 = P$ is an orthogonal projection matrix onto $C(P)$.

---

## 4. Least Squares Regression

### 4.1 Connection between Calculus (Minimizing Error) and Projections
Suppose we have data points $(x_1, b_1), \dots, (x_m, b_m)$ and want to fit a line $b \approx \theta_1 x + \theta_0$.
In matrix form, we want $A\theta \approx b$:
$$\begin{bmatrix} x_1 & 1 \\ x_2 & 1 \\ \vdots & \vdots \\ x_m & 1 \end{bmatrix} \begin{bmatrix} \theta_1 \\ \theta_0 \end{bmatrix} \approx \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}$$
Because the points usually don't lie on a single straight line, $A\theta = b$ is inconsistent ($b \notin C(A)$).
The **Least Squares** objective minimizes the sum of squared errors:
$$\min_{\theta} E^2(\theta) = \|b - A\theta\|^2 = \sum_{i=1}^m (b_i - \theta_1 x_i - \theta_0)^2$$
Taking the gradient and setting it to zero yields the exact same **Normal Equations**:
$$A^T A \hat{\theta} = A^T b \implies \hat{\theta} = (A^T A)^{-1} A^T b$$

### 4.2 Worked Example (from Lecture)
Fit a line $y = \theta_1 x + \theta_0$ through the three points: $(-1, 1), (1, 1), (2, 3)$.
1. **Set up $A\theta = b$:**
   $$A = \begin{bmatrix} -1 & 1 \\ 1 & 1 \\ 2 & 1 \end{bmatrix}, \quad b = \begin{bmatrix} 1 \\ 1 \\ 3 \end{bmatrix}$$
   *(Check: Gaussian elimination on $[A \mid b]$ gives row $[0 \ 0 \mid 2]$, confirming $b \notin C(A)$ — inconsistent!)*
2. **Compute $A^T A$ and $A^T b$:**
   $$A^T A = \begin{bmatrix} -1 & 1 & 2 \\ 1 & 1 & 1 \end{bmatrix} \begin{bmatrix} -1 & 1 \\ 1 & 1 \\ 2 & 1 \end{bmatrix} = \begin{bmatrix} 6 & 2 \\ 2 & 3 \end{bmatrix}$$
   $$A^T b = \begin{bmatrix} -1 & 1 & 2 \\ 1 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 3 \end{bmatrix} = \begin{bmatrix} 6 \\ 5 \end{bmatrix}$$
3. **Solve Normal Equations $A^T A \hat{\theta} = A^T b$:**
   $$6\hat{\theta}_1 + 2\hat{\theta}_0 = 6$$
   $$2\hat{\theta}_1 + 3\hat{\theta}_0 = 5$$
   Solving gives $\hat{\theta}_1 = \frac{4}{7}$ and $\hat{\theta}_0 = \frac{9}{7}$.
4. **Best-fit Line:**
   $$y = \frac{4}{7}x + \frac{9}{7}$$
   * **Projected values:** $p = A\hat{\theta} = \left[\frac{5}{7}, \frac{13}{7}, \frac{17}{7}\right]^T$.
   * **Error vector:** $e = b - p = \left[\frac{2}{7}, -\frac{6}{7}, \frac{4}{7}\right]^T$.
   * *Sanity Check:* $e \perp (-1, 1, 2)^T$ and $e \perp (1, 1, 1)^T$ (dot products are both $0$!).
