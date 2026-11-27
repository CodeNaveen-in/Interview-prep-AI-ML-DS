# Week 8: Pillars of ML, Unconstrained Optimization & Gradient Descent

> **Course:** Machine Learning Foundations (IIT Madras BS Degree)  
> **Topics Covered:** The Three Pillars of ML, General Optimization Formulation, Unconstrained Optimization, Gradient Descent Algorithm, Step-Size (Learning Rate) Selection, Local vs. Global Minima, Taylor Series Justification & Steepest Descent

---

## 1. The Three Pillars of Machine Learning

Machine Learning rests on three foundational mathematical pillars:
1. **Linear Algebra:** Captures the **structure and relationships** between data points and features.
2. **Probability & Statistics:** Models **noise and uncertainty** in real-world data.
3. **Optimization:** Converts **Data $\longrightarrow$ Decisions** (finding the "best" model parameters that minimize loss or maximize reward).

---

## 2. Introduction to Optimization

### 2.1 General Form of an Optimization Problem
$$\begin{aligned}
\min_{x \in \mathbb{R}^d} \quad & f(x) && \text{(Objective Function)} \\
\text{subject to} \quad & g_i(x) \le 0, \quad i = 1, \dots, k && \text{(Inequality Constraints)} \\
& h_j(x) = 0, \quad j = 1, \dots, l && \text{(Equality Constraints)}
\end{aligned}$$
Here, $x \in \mathbb{R}^d$ is the **decision variable / parameter vector**.

* **Intuitive Example (Cow & Grass):** A cow is tied to a 10-unit rope at $(20, 30)$, blocked by a vertical fence at $x_1 \le 25$, and wants to get as close as possible to grass at $(40, 40)$:
  * Objective: $\min_{(x_1, x_2)} (x_1 - 40)^2 + (x_2 - 40)^2$
  * Rope constraint: $(x_1 - 20)^2 + (x_2 - 30)^2 \le 10^2$
  * Fence constraint: $x_1 \le 25$

### 2.2 Unconstrained Optimization
When there are **no constraints**, the problem is simply:
$$\min_{x \in \mathbb{R}^d} f(x)$$
* For simple functions like $f(x) = (x - 5)^2$, setting $f'(x) = 2(x - 5) = 0$ immediately gives $x^* = 5$.
* **Why do we need iterative algorithms?** For complex functions (e.g., $3x^6 + 2x^5 + 3x^3 + 5x^2 + 2$, or neural network loss functions with millions of parameters), solving $\nabla f(x) = \mathbf{0}$ analytically is impossible!

---

## 3. Gradient Descent (1D Case)

To minimize $f(x)$ iteratively, start at an arbitrary initial point $x_0 \in \mathbb{R}$ and repeatedly update:
$$x_{t+1} = x_t + \eta_t d_t$$
where $d_t$ is the **update direction** and $\eta_t > 0$ is the **step size** (learning rate).

### 3.1 Choosing the Direction: $d_t = -f'(x_t)$
Consider $f(x) = (x - 5)^2$, so $f'(x) = 2(x - 5)$.
* If $x > 5$: we want to move **left** ($d < 0$). Notice $-f'(x) = -2(x-5) < 0$.
* If $x < 5$: we want to move **right** ($d > 0$). Notice $-f'(x) = -2(x-5) > 0$.
Thus, $d_t = -f'(x_t)$ always points toward the minimum!

### 3.2 Why Step Size ($\eta_t$) Matters!
1. **Without scaling ($\eta_t = 1$):** For $f(x) = (x-5)^2$ starting at $x_0 = 10$:
   * $x_1 = 10 - 2(10 - 5) = 0$
   * $x_2 = 0 - 2(0 - 5) = 10$
   It **oscillates forever** between $0$ and $10$!
2. **Step size decaying too fast ($\eta_t = \frac{1}{2^t}$):**  
   Since $\sum_{t=0}^{\infty} \frac{1}{2^t} = 2 < \infty$, the total distance the algorithm can *ever* travel is bounded! If you start far from the minimum, it **gets stuck before reaching $x^*$**.
3. **Good decaying step size ($\eta_t = \frac{1}{t+1}$):**  
   Since $\sum_{t=0}^{\infty} \frac{1}{t+1} = \infty$ (divergent harmonic series) and $\eta_t \to 0$, it can travel arbitrarily far yet slows down to settle at the minimum.

### 3.3 The 1D Gradient Descent Algorithm
* Initialize $x_0 \in \mathbb{R}$.
* For $t = 0, 1, 2, \dots$:
  $$x_{t+1} = x_t - \eta_t f'(x_t)$$

---

## 4. Local vs. Global Minima

* **Local Minimum ($\hat{x}$):** $\exists \epsilon > 0$ such that $f(\hat{x}) \le f(x)$ for all $x \in [\hat{x} - \epsilon, \hat{x} + \epsilon]$.
* **Global Minimum ($x^*$):** $f(x^*) \le f(x)$ for **all** $x$ in the domain.
* **Key Fact:** In general, Gradient Descent is only guaranteed to converge to a **local minimum** (which one depends on the initialization $x_0$).
* **Special Case — Convex Functions:** For **convex functions** (like $(x - 5)^2$), **every local minimum is a global minimum!**

---

## 5. Taylor Series & Multivariate Gradient Descent

### 5.1 Why Does $d = -\nabla f(x)$ Work? (Taylor Series Justification)
For a multivariate function $f: \mathbb{R}^d \to \mathbb{R}$, the first-order **Taylor Series** expansion around $x$ for a small step $\eta > 0$ in direction $d \in \mathbb{R}^d$ is:
$$f(x + \eta d) = f(x) + \eta d^T \nabla f(x) + \text{higher-order terms (negligible for small } \eta\text{)}$$
Rearranging:
$$f(x + \eta d) - f(x) \approx \eta \, d^T \nabla f(x)$$
To make the function **decrease** ($f(x + \eta d) < f(x)$), we need a descent direction $d$ such that:
$$d^T \nabla f(x) < 0$$
Choosing $d = -\nabla f(x)$ guarantees:
$$d^T \nabla f(x) = -\nabla f(x)^T \nabla f(x) = -\|\nabla f(x)\|_2^2 < 0 \quad (\text{whenever } \nabla f(x) \neq \mathbf{0})$$

### 5.2 Steepest Descent Property
Among all unit-length directions ($\|d\|_2 = 1$), $d^T \nabla f(x) = \|d\|\|\nabla f(x)\|\cos\theta$ is **most negative** when $\theta = 180^\circ$, i.e., when $d$ points in the exact direction of $-\nabla f(x)$.
* Therefore, $-\nabla f(x)$ is the direction of **steepest descent** (which is why Gradient Descent is also called **Steepest Descent**).

### 5.3 Multivariate Gradient Descent Algorithm
* Initialize $x_0 \in \mathbb{R}^d$.
* For $t = 0, 1, 2, \dots$:
  $$x_{t+1} = x_t - \eta_t \nabla f(x_t)$$
  where $\nabla f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1}(x) \\ \vdots \\ \frac{\partial f}{\partial x_d}(x) \end{bmatrix} \in \mathbb{R}^d$.
