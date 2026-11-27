# Week 10: Convexity Properties, Duality & KKT Conditions

> **Course:** Machine Learning Foundations (IIT Madras BS Degree)  
> **Topics Covered:** Optimality of Convex Functions, Operations Preserving Convexity, Linear Regression & SGD, Primal & Dual Problems, Weak & Strong Duality, Karush-Kuhn-Tucker (KKT) Conditions, Support Vector Machines (SVM)

---

## 1. Properties of Convex Functions

### 1.1 Optimality Criterion for Differentiable Convex Functions
**Theorem:** Let $f: \mathbb{R}^d \to \mathbb{R}$ be differentiable and convex. Then:
$$x^* \in \mathbb{R}^d \text{ is a global minimum of } f \quad \Longleftrightarrow \quad \nabla f(x^*) = \mathbf{0}$$

* **Proof ($\implies$):** If $x^*$ is a global minimum, it is a local minimum; if $\nabla f(x^*) \neq \mathbf{0}$, stepping in $-\nabla f(x^*)$ would decrease $f$, a contradiction.
* **Proof ($\Longleftarrow$):** By the first-order convexity condition, for all $y \in \mathbb{R}^d$:
  $$f(y) \ge f(x^*) + \nabla f(x^*)^T (y - x^*) = f(x^*) + \mathbf{0}^T (y - x^*) = f(x^*)$$
  Thus $x^*$ is a global minimum! $\blacksquare$

*(Note: The set of all global minima of a convex function is also a convex set.)*

### 1.2 Operations That Preserve Convexity
1. **Sums of Convex Functions:** If $f$ and $g$ are convex, then $h(x) = f(x) + g(x)$ is convex.
2. **Composition with Non-Decreasing Convex Function:** If $g: \mathbb{R}^d \to \mathbb{R}$ is convex and $f: \mathbb{R} \to \mathbb{R}$ is **convex AND non-decreasing**, then $h(x) = f(g(x))$ is convex.
   * *Warning:* If $f$ is convex but *decreasing*, $f(g(x))$ need **not** be convex! (e.g., $g(x) = x^2$ and $f(z) = e^{-z}$ are both convex, but $e^{-x^2}$ is a bell curve, which is *not* convex!)
3. **Composition with a Linear Function:** If $f: \mathbb{R} \to \mathbb{R}$ is convex and $g: \mathbb{R}^d \to \mathbb{R}$ is linear ($g(x) = w^T x + b$), then $h(x) = f(g(x))$ is convex.

---

## 2. Applications of Optimization in Machine Learning

### 2.1 Linear Regression as Convex Optimization
Given $\{(x_1, y_1), \dots, (x_n, y_n)\}$, we minimize the squared loss:
$$f(w) = \frac{1}{2} \sum_{i=1}^n (w^T x_i - y_i)^2 = \frac{1}{2} \|Xw - y\|_2^2$$
* **Is $f(w)$ convex?** Yes! Each term $h_i(w) = (w^T x_i - y_i)^2$ is a composition of a linear function $w^T x_i - y_i$ with the convex function $z^2$, and $f(w)$ is a sum of convex functions.
* **Analytical Solution vs. Gradient Descent:**
  $$\nabla f(w) = (X^T X)w - X^T y = 0 \implies w^* = (X^T X)^{-1} X^T y$$
  * **Problem with analytical solution:** Computing $(X^T X)^{-1}$ takes $\approx \mathcal{O}(d^3)$ time (and $X^T X$ may not even be invertible).
  * **Gradient Descent:** $w_{t+1} = w_t - \eta_t \big((X^T X) w_t - X^T y\big)$ — **no matrix inversion needed!**
* **Stochastic Gradient Descent (SGD):** When $n$ is huge, computing the full gradient over all $n$ samples is slow. SGD samples a small mini-batch of data points uniformly at random and computes the gradient on that subset.

---

## 3. Primal and Dual Problems

Consider a constrained optimization problem (**The Primal Problem**):
$$\min_{x \in \mathbb{R}^d} f(x) \quad \text{subject to } h(x) \le 0$$

### 3.1 The Lagrangian & Min-Max Formulation
Define the **Lagrangian function** (with Lagrange multiplier $\lambda \ge 0$):
$$\mathcal{L}(x, \lambda) = f(x) + \lambda h(x)$$
Notice what happens if we fix $x$ and maximize $\mathcal{L}(x, \lambda)$ over $\lambda \ge 0$:
$$J(x) = \max_{\lambda \ge 0} \mathcal{L}(x, \lambda) = \begin{cases} f(x) & \text{if } h(x) \le 0 \quad (\text{best choice is } \lambda = 0) \\ +\infty & \text{if } h(x) > 0 \quad (\text{send } \lambda \to +\infty) \end{cases}$$
Therefore, the **Primal Problem** is completely equivalent to:
$$p^* = \min_{x} \left[ \max_{\lambda \ge 0} \mathcal{L}(x, \lambda) \right]$$

### 3.2 The Dual Problem
If we swap the $\min$ and $\max$, we get the **Lagrange Dual Problem**:
$$d^* = \max_{\lambda \ge 0} \left[ \min_{x} \mathcal{L}(x, \lambda) \right] = \max_{\lambda \ge 0} g(\lambda)$$
where $g(\lambda) = \min_x \mathcal{L}(x, \lambda)$ is the **Dual Function** (which is always a **concave** function of $\lambda$, even if $f$ and $h$ are non-convex!). The inner minimization over $x$ is *unconstrained*, making it often much easier to solve.

### 3.3 Weak Duality vs. Strong Duality
* **Weak Duality (ALWAYS holds):**  
  For any $\lambda \ge 0$ and any feasible $x$ ($h(x) \le 0$), $\mathcal{L}(x, \lambda) = f(x) + \lambda h(x) \le f(x) = J(x)$. Taking $\min_x$ and then $\max_{\lambda \ge 0}$:
  $$\boxed{g(\lambda^*) \le f(x^*)}$$
  The dual optimal value is always a lower bound on the primal optimal value ($d^* \le p^*$).

* **Strong Duality:**  
  When $d^* = p^*$ (i.e., $g(\lambda^*) = f(x^*)$, zero duality gap).  
  **Key Theorem:** If $f$ and $h$ are **convex** (and mild regularity conditions hold, e.g., Slater's condition), then **Strong Duality holds!**

---

## 4. Karush-Kuhn-Tucker (KKT) Conditions

When Strong Duality holds ($f(x^*) = g(\lambda^*)$), we can derive necessary conditions for optimality. Since:
$$f(x^*) = g(\lambda^*) = \min_x \big(f(x) + \lambda^* h(x)\big) \le f(x^*) + \lambda^* h(x^*) \le f(x^*)$$
the inequalities must be equalities! This forces both $\nabla_x \mathcal{L}(x^*, \lambda^*) = \mathbf{0}$ and $\lambda^* h(x^*) = 0$.

### General KKT Conditions
For the general problem:
$$\min_{x \in \mathbb{R}^d} f(x) \quad \text{s.t.} \quad h_i(x) \le 0 \ (i=1,\dots,m), \quad \ell_j(x) = 0 \ (j=1,\dots,n)$$
with Lagrangian:
$$\mathcal{L}(x, u, v) = f(x) + \sum_{i=1}^m u_i h_i(x) + \sum_{j=1}^n v_j \ell_j(x)$$
a primal-dual optimal point $(x^*, u^*, v^*)$ must satisfy the **4 KKT Conditions**:

1. **Stationarity:**
   $$\nabla f(x^*) + \sum_{i=1}^m u_i^* \nabla h_i(x^*) + \sum_{j=1}^n v_j^* \nabla \ell_j(x^*) = \mathbf{0}$$
2. **Complementary Slackness:**
   $$u_i^* h_i(x^*) = 0 \quad \forall i = 1, \dots, m$$
   *(Intuition: Either constraint $i$ is active ($h_i(x^*) = 0$), OR its multiplier is zero ($u_i^* = 0$)!)*
3. **Primal Feasibility:**
   $$h_i(x^*) \le 0 \ \forall i, \qquad \ell_j(x^*) = 0 \ \forall j$$
4. **Dual Feasibility:**
   $$u_i^* \ge 0 \quad \forall i$$

> **Sufficiency:** If $f$ and $h_i$ are convex and $\ell_j$ are affine (linear), any point $(x^*, u^*, v^*)$ satisfying the KKT conditions is **guaranteed to be globally optimal**!

---

## 5. Application: Support Vector Machine (SVM)

Given a linearly separable classification dataset $\{(x_1, y_1), \dots, (x_n, y_n)\}$ with $y_i \in \{+1, -1\}$, the **hard-margin SVM** finds the maximum-margin separating hyperplane $w^T x = 0$ by solving:
$$\begin{aligned}
\min_{w \in \mathbb{R}^d} \quad & \frac{1}{2} \|w\|^2 \\
\text{subject to} \quad & y_i (w^T x_i) \ge 1 \quad \forall i = 1, \dots, n \quad (\text{i.e., } 1 - y_i w^T x_i \le 0)
\end{aligned}$$
* **Objective:** $\frac{1}{2}\|w\|^2 = \frac{1}{2}\sum_j w_j^2$ is quadratic $\implies$ **convex**.
* **Constraints:** $1 - y_i w^T x_i \le 0$ are linear in $w$ $\implies$ **convex**.
* Therefore, SVM is a **convex optimization problem** where Strong Duality and KKT conditions apply cleanly!
