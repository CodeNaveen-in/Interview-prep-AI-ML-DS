# Week 9: Constrained Optimization, Lagrange Multipliers & Convexity

> **Course:** Machine Learning Foundations (IIT Madras BS Degree)  
> **Topics Covered:** Geometric Optimality for Constrained Problems, Method of Lagrange Multipliers, Projected Gradient Descent (PGD), Convex Sets & Operations, Convex Hulls, Four Equivalent Definitions of Convex Functions, Local Minima = Global Minima

---

## 1. Constrained Optimization: Geometric Intuition

Consider the constrained optimization problem:
$$\min_{x \in \mathbb{R}^d} f(x) \quad \text{subject to } g(x) \le 0$$
When is a point $x^*$ optimal?
1. **Feasibility:** $g(x^*) \le 0$.
2. **No descent direction is a feasible direction:**
   * A **descent direction** $d$ decreases $f$: $d^T \nabla f(x^*) < 0$ (property of $f$).
   * A **feasible direction** $d$ (when on the boundary $g(x^*) = 0$) stays inside $g(x) \le 0$: $d^T \nabla g(x^*) < 0$ (property of $g$).
   * If any $d$ satisfied both $d^T \nabla f(x^*) < 0$ and $d^T \nabla g(x^*) < 0$, we could take a small step along $d$ and get a strictly lower objective value while remaining feasible!
   * Therefore, at a boundary optimum ($g(x^*) = 0$), $\nabla f(x^*)$ and $\nabla g(x^*)$ must point in **exact opposite directions**:
     $$\nabla f(x^*) = -\lambda \nabla g(x^*), \qquad \lambda \ge 0$$

### Equality Constraint Case: $\min_x f(x)$ subject to $g(x) = 0$
For an equality constraint $g(x) = 0$, feasible directions must keep $g(x^* + \eta d) \approx g(x^*) + \eta d^T \nabla g(x^*) = 0$, meaning $d^T \nabla g(x^*) = 0$ (orthogonal to $\nabla g(x^*)$).
For no descent direction to exist along the tangent plane $\{d : d^T \nabla g(x^*) = 0\}$, $\nabla f(x^*)$ must be parallel to $\nabla g(x^*)$ (either same or opposite direction):
$$\nabla f(x^*) = -\lambda \nabla g(x^*), \qquad \lambda \in \mathbb{R} \text{ (no sign constraint on } \lambda\text{)}$$

---

## 2. Method of Lagrange Multipliers (Worked Example)

To solve $\min_x f(x)$ subject to $g(x) = 0$:
1. $g(x^*) = 0$
2. $\nabla f(x^*) = -\lambda \nabla g(x^*)$ for some scalar $\lambda \in \mathbb{R}$.

#### Worked Example (from Lecture):
$$\min_{x_1, x_2} f(x_1, x_2) = x_1^2 + 2x_2 + 4x_2^2 \quad \text{subject to } g(x_1, x_2) = x_1^2 + x_2^2 - 1 = 0$$
1. **Compute Gradients:**
   $$\nabla f(x) = \begin{bmatrix} 2x_1 \\ 2 + 8x_2 \end{bmatrix}, \qquad \nabla g(x) = \begin{bmatrix} 2x_1 \\ 2x_2 \end{bmatrix}$$
2. **Apply $\nabla f(x) = -\lambda \nabla g(x)$:**
   * Eq (1): $2x_1 = -\lambda (2x_1) \implies 2x_1(1 + \lambda) = 0 \implies$ **either $x_1 = 0$ OR $\lambda = -1$**.
   * Eq (2): $2 + 8x_2 = -\lambda (2x_2)$.
3. **Case 1 ($\lambda = -1$):**
   $$2 + 8x_2 = 2x_2 \implies 6x_2 = -2 \implies x_2 = -\frac{1}{3}$$
   From $x_1^2 + x_2^2 = 1$: $x_1^2 = 1 - \frac{1}{9} = \frac{8}{9} \implies x_1 = \pm \frac{\sqrt{8}}{3}$.  
   Candidate points: $\left(\frac{\sqrt{8}}{3}, -\frac{1}{3}\right)$ and $\left(-\frac{\sqrt{8}}{3}, -\frac{1}{3}\right)$.
4. **Case 2 ($x_1 = 0$):**
   From $x_1^2 + x_2^2 = 1$: $x_2 = \pm 1$.  
   Candidate points: $(0, 1)$ and $(0, -1)$.
5. **Evaluate $f$ at all 4 candidates:**
   * $f(0, 1) = 6$ (Maximizer)
   * $f(0, -1) = 2$
   * $f\left(\pm\frac{\sqrt{8}}{3}, -\frac{1}{3}\right) = \frac{8}{9} - \frac{2}{3} + \frac{4}{9} = \mathbf{\frac{2}{3}}$ (**Minimizers!**)

---

## 3. Projected Gradient Descent (PGD)

In practice, solving the Lagrange multiplier equations analytically is often impossible. When optimizing over a **convex constraint set** $S = \{y : g(y) \le 0\}$, we use **Projected Gradient Descent**:

* Initialize $x_0 \in S$.
* For $t = 0, 1, \dots, T$:
  $$x_{t+1} = \Pi_S\big(x_t - \eta \nabla f(x_t)\big)$$
where the **Projection Operator** $\Pi_S(x)$ finds the closest point in $S$ to $x$:
$$\Pi_S(x) = \arg\min_{y \in S} \|x - y\|_2^2$$
In each iteration: take a standard gradient descent step, and if you land outside $S$, project straight back onto $S$.

---

## 4. Convex Sets

### 4.1 Definition
A set $S \subseteq \mathbb{R}^d$ is **convex** if for any two points $x_1, x_2 \in S$ and any $\lambda \in [0, 1]$, the line segment joining them lies entirely in $S$:
$$\lambda x_1 + (1 - \lambda) x_2 \in S \quad \forall x_1, x_2 \in S, \ \forall \lambda \in [0, 1]$$

### 4.2 Important Examples of Convex Sets
* **Intervals** $[a, b] \subset \mathbb{R}$ (a union of two disjoint intervals $[c, d] \cup [a, b]$ is *not* convex).
* **Hyperplanes:** $\{x \in \mathbb{R}^d : w^T x = b\}$.
* **Half-spaces:** $\{x \in \mathbb{R}^d : w^T x \le b\}$.
* **Euclidean Balls:** $B = \{x \in \mathbb{R}^d : \|x\|_2 \le \theta\}$.
* **Linear System Solution Sets:** $\{x \in \mathbb{R}^d : Ax = b\}$ (intersection of $m$ hyperplanes).

### 4.3 Properties & Convex Hulls
* **Intersection Property:** The intersection of any collection of convex sets is **convex**.
* **Convex Combination:** $z \in \mathbb{R}^d$ is a convex combination of $\{x_1, \dots, x_n\}$ if $z = \sum_{i=1}^n \lambda_i x_i$ with $\lambda_i \ge 0$ and $\sum_{i=1}^n \lambda_i = 1$.
* **Convex Hull $\text{CH}(S)$:** The set of all convex combinations of points in $S$ (equivalently, the intersection of all convex sets containing $S$). $\text{CH}(S)$ is always a convex set.

---

## 5. Convex Functions

A function $f: \mathbb{R}^d \to \mathbb{R}$ (or defined on a convex set) is **convex** under four equivalent definitions:

1. **Definition 1 (Epigraph):** $f$ is convex iff its **epigraph** (the set of points lying on or above the graph of $f$) is a convex set in $\mathbb{R}^{d+1}$:
   $$\text{epi}(f) = \left\{ \begin{bmatrix} x \\ z \end{bmatrix} \in \mathbb{R}^{d+1} : z \ge f(x) \right\}$$
2. **Definition 2 (Jensen's / Chord Inequality):** For all $x_1, x_2 \in \mathbb{R}^d$ and all $\lambda \in [0, 1]$, the chord lies above the graph:
   $$f\big(\lambda x_1 + (1 - \lambda) x_2\big) \le \lambda f(x_1) + (1 - \lambda) f(x_2)$$
3. **Definition 3 (First-Order Condition — Differentiable $f$):** The tangent hyperplane always lies **below** the function:
   $$f(y) \ge f(x) + (y - x)^T \nabla f(x) \quad \forall x, y \in \mathbb{R}^d$$
4. **Definition 4 (Second-Order Condition — Twice-Differentiable $f$):** The Hessian matrix $H(x) = \nabla^2 f(x)$ (where $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$) is **Positive Semi-Definite (PSD)** everywhere:
   $$H(x) \succeq 0 \quad (\text{all eigenvalues of } H(x) \ge 0) \quad \forall x \in \mathbb{R}^d$$
   *(In 1D, this simply means $f''(x) \ge 0$ everywhere, e.g., $f(x) = x^2$ has $f''(x) = 2 > 0$.)*

---

## 6. Why Convexity is the Holy Grail of Optimization

**Theorem:** If $f$ is a convex function, then **every local minimum of $f$ is also a global minimum!**

### Proof (by Contradiction):
Suppose $x^*$ is a local minimum that is **not** a global minimum. Then there exists some point $z$ such that $f(z) < f(x^*)$.
1. Since $x^*$ is a local minimum, there exists $\delta > 0$ such that for all $y$ with $\|y - x^*\| \le \delta$, $f(y) \ge f(x^*)$.
2. Choose $\lambda \in (0, 1)$ close enough to $1$ so that $y = \lambda x^* + (1 - \lambda) z$ lies inside the $\delta$-ball around $x^*$.
3. By local optimality of $x^*$ and convexity of $f$:
   $$f(x^*) \le f\big(\lambda x^* + (1 - \lambda) z\big) \le \lambda f(x^*) + (1 - \lambda) f(z)$$
4. Since $f(z) < f(x^*)$ and $1 - \lambda > 0$:
   $$f(x^*) < \lambda f(x^*) + (1 - \lambda) f(x^*) = f(x^*) \quad \text{(Contradiction!)}$$
Therefore, no non-global local minimum can exist. $\blacksquare$
