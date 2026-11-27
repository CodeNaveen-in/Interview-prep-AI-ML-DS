# Week 2: Sets, Functions, Univariate & Multivariate Calculus

> **Course:** Machine Learning Foundations (IIT Madras BS Degree)  
> **Topics Covered:** Sets & Metric Spaces, Sequences, Vector Spaces, Functions & Contours, Continuity & Differentiability, Linear & Higher-Order Approximations (Taylor Series), Product & Chain Rules, Lines & Hyperplanes, Partial Derivatives, Gradients, Directional Derivatives, Steepest Ascent/Descent, Hessian Matrix

---

## 1. Sets, Metric Spaces, and Vector Spaces

### 1.1 Common Sets & Intervals
* $\mathbb{R}$: Set of all real numbers.
* $\mathbb{R}_+$ (or $[0, \infty)$): Set of non-negative real numbers (or positive reals depending on context).
* $\mathbb{Z}$: Set of integers; $\mathbb{Z}_+$: Positive integers.
* **Closed interval:** $[a, b] = \{x \in \mathbb{R} : a \le x \le b\}$.
* **Open interval:** $(a, b) = \{x \in \mathbb{R} : a < x < b\}$.
* $\mathbb{R}^d$: Set of $d$-dimensional real vectors ($d$-tuples of real numbers).

### 1.2 Metric Spaces & Distance
To talk about "closeness" or convergence in $\mathbb{R}^d$, we define a distance (metric) between two vectors $x, y \in \mathbb{R}^d$:
$$D(x, y) = \|x - y\|_2 = \sqrt{\sum_{i=1}^d (x_i - y_i)^2}$$
* An **open ball** of radius $\epsilon > 0$ centered at $x$ is:
  $$B(x, \epsilon) = \{y \in \mathbb{R}^d : D(x, y) < \epsilon\}$$

### 1.3 Logic & Set Operations
* $\forall$ ("for all"), $\exists$ ("there exists"), $\implies$ ("implies"), $\iff$ ("if and only if / equivalent").
* Union ($A \cup B$), Intersection ($A \cap B$), Complement ($A^c$).

### 1.4 Vector Spaces & Inner Products
A **vector space** $V$ is closed under linear combinations: for any $u, v \in V$ and scalars $\alpha, \beta \in \mathbb{R}$, $\alpha u + \beta v \in V$. ($\mathbb{R}^d$ is a vector space.)
* **Dot Product (Inner Product):** $x \cdot y = x^T y = \sum_{i=1}^d x_i y_i$.
* **Norm (Length):** $\|x\| = \sqrt{x^T x}$.
* **Orthogonality:** $x$ and $y$ are perpendicular (orthogonal) if and only if $x^T y = 0$.

### 1.5 Functions and Graphs
A function $f: A \to B$ maps every element in the **domain** $A$ to an element in the **co-domain** $B$.
* **Graph of $f$:** $G_f = \{(x, f(x)) : x \in A\}$.
* For 2D functions $f: \mathbb{R}^2 \to \mathbb{R}$, we visualize using **heat maps** and **contour plots** (curves where $f(x_1, x_2) = c$ is constant).

---

## 2. Univariate Calculus ($f: \mathbb{R} \to \mathbb{R}$)

### 2.1 Continuity
A function $f: \mathbb{R} \to \mathbb{R}$ is **continuous at $a \in \mathbb{R}$** if for every sequence $\{x_n\}$ converging to $a$, the sequence $\{f(x_n)\}$ converges to $f(a)$:
$$\lim_{x \to a} f(x) = f(a)$$
Intuitively, there are no sudden "jumps" or "holes" at $a$.
* *Example of discontinuity:* $\text{sign}(x)$ has a jump at $x = 0$.

### 2.2 Differentiability
A function $f: \mathbb{R} \to \mathbb{R}$ is **differentiable at $a$** if the following limit exists:
$$f'(a) = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}$$
* **Key Fact:** Differentiability $\implies$ Continuity. (If $f$ is *not* continuous at $a$, it cannot be differentiable at $a$.)
* **Converse is FALSE:** Continuity does *not* imply differentiability!
  * *Classic Example:* $f(x) = |x|$ is continuous at $x = 0$, but **not differentiable** at $x = 0$ because the left-hand derivative ($-1$) $\neq$ right-hand derivative ($+1$) — there is a sharp corner.

### 2.3 Linear Approximation (Linearization)
Near a point $a$, any smooth differentiable function behaves like its tangent line!
$$L(x) = f(a) + f'(a)(x - a)$$
* This is the equation of the **tangent line** to $G_f$ at $(a, f(a))$ with slope $m = f'(a)$.
* **Why do we use it?** Linear functions are easy to compute and analyze, and give very accurate approximations close to $a$.

#### Worked Examples (from Lecture & Tutorial):
1. **Approximate $\sqrt{50}$:**
   * Let $f(x) = \sqrt{x}$ and choose closest known point $a = 49$.
   * $f(49) = 7$, and $f'(x) = \frac{1}{2\sqrt{x}} \implies f'(49) = \frac{1}{14}$.
   * $L(50) = 7 + \frac{1}{14}(50 - 49) = 7 + \frac{1}{14} \approx \mathbf{7.071}$ (Actual: $7.071$).
   * *Note:* Far away from $a = 49$, accuracy drops (e.g., $L(100) = 10.64$, actual $\sqrt{100} = 10$).
2. **Approximate $e^{0.017}$:**
   * Let $f(x) = e^x$, $a = 0$. $f(0) = 1, f'(0) = 1 \implies L(x) = 1 + x$.
   * $e^{0.017} \approx L(0.017) = \mathbf{1.017}$.
3. **Approximate $\sin(x)$ around $x = 0$:**
   * $f(0) = 0, f'(0) = \cos(0) = 1 \implies \sin(x) \approx x$ for small $x$.

### 2.4 Higher-Order Approximations (Taylor Series)
To get better approximations further from $a$, we include curvature (second derivative) and higher derivatives:
* **Quadratic Approximation:**
  $$Q(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2$$
* **General Taylor Approximation ($n$-th order):**
  $$P_n(x) = \sum_{k=0}^n \frac{f^{(k)}(a)}{k!}(x - a)^k = f(a) + f'(a)(x-a) + \frac{f''(a)}{2}(x-a)^2 + \frac{f'''(a)}{6}(x-a)^3 + \dots$$

### 2.5 Differentiation Rules
* **Product Rule:** If $f(x) = g(x)h(x)$, then:
  $$f'(x) = g'(x)h(x) + g(x)h'(x)$$
* **Chain Rule:** If $f(x) = g(h(x))$, then:
  $$f'(x) = g'(h(x)) \cdot h'(x)$$

### 2.6 Maxima, Minima, and Saddle Points
* A point $x^*$ where $f'(x^*) = 0$ is called a **critical point** (or stationary point).
* A critical point can be:
  * A **local minimum** ($f''(x^*) > 0$, curve bends upward).
  * A **local maximum** ($f''(x^*) < 0$, curve bends downward).
  * A **saddle / inflection point** (e.g., $f(x) = x^3$ at $x = 0$, where $f'(0) = 0$ and $f''(0) = 0$, neither max nor min).

---

## 3. Multivariate Calculus ($f: \mathbb{R}^d \to \mathbb{R}$)

### 3.1 Lines and Hyperplanes in $\mathbb{R}^d$
* **Line** passing through point $u \in \mathbb{R}^d$ along direction vector $v \in \mathbb{R}^d$:
  $$L = \{x \in \mathbb{R}^d : x = u + t v, \quad t \in \mathbb{R}\}$$
  * A line through two points $u$ and $v$: $x = u + t(v - u) = (1-t)u + tv$.
* **Hyperplane** (a $(d-1)$-dimensional flat subspace) normal (perpendicular) to vector $w \in \mathbb{R}^d$ with offset $b \in \mathbb{R}$:
  $$H = \{x \in \mathbb{R}^d : w^T x = b\}$$
  * In $\mathbb{R}^2$, this is a line; in $\mathbb{R}^3$, a 2D plane.
  * For any two points $x_1, x_2$ on the hyperplane, $w^T(x_1 - x_2) = b - b = 0$, confirming $w$ is perpendicular to every vector lying inside the hyperplane!

### 3.2 Partial Derivatives & Gradients
* **Partial Derivative:** The rate of change of $f$ along the $j$-th coordinate axis, holding all other coordinates fixed:
  $$\frac{\partial f}{\partial x_j}(x) = \lim_{h \to 0} \frac{f(x + h e_j) - f(x)}{h}$$
  where $e_j$ is the $j$-th standard basis vector.
* **Gradient Vector ($\nabla f$):** The column vector stacking all $d$ partial derivatives:
  $$\nabla f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1}(x) \\ \frac{\partial f}{\partial x_2}(x) \\ \vdots \\ \frac{\partial f}{\partial x_d}(x) \end{bmatrix} \in \mathbb{R}^d$$

### 3.3 Multivariate Linear Approximation & Tangent Planes
For a differentiable function $f: \mathbb{R}^d \to \mathbb{R}$, the linear approximation around $a \in \mathbb{R}^d$ is:
$$L(x) = f(a) + \nabla f(a)^T (x - a) = f(a) + \sum_{j=1}^d \frac{\partial f}{\partial x_j}(a)(x_j - a_j)$$
Geometrically, $z = L(x)$ is the **tangent hyperplane** to the surface $z = f(x)$ at $x = a$.

#### Worked Example (from Tutorial):
Find the linearization of $f(x, y) = x e^{xy}$ at $(1, 0)$ and approximate $f(1.1, -0.1)$:
1. $\frac{\partial f}{\partial x} = e^{xy} + xy e^{xy} \implies \frac{\partial f}{\partial x}(1, 0) = 1$.
2. $\frac{\partial f}{\partial y} = x^2 e^{xy} \implies \frac{\partial f}{\partial y}(1, 0) = 1$.
3. $f(1, 0) = 1 \cdot e^0 = 1$.
4. $L(x, y) = 1 + 1(x - 1) + 1(y - 0) = x + y$.
5. $f(1.1, -0.1) \approx L(1.1, -0.1) = 1.1 - 0.1 = \mathbf{1.0}$ (Actual: $0.9854$).

### 3.4 Directional Derivative
What is the rate of change of $f$ at point $x$ if we move along an arbitrary **unit vector** $u \in \mathbb{R}^d$ ($\|u\| = 1$)?
$$D_u f(x) = \lim_{t \to 0} \frac{f(x + t u) - f(x)}{t} = \nabla f(x)^T u = \sum_{j=1}^d u_j \frac{\partial f}{\partial x_j}(x)$$
> **Watch Out!** Always normalize $u$ to a **unit vector** ($\hat{u} = \frac{u}{\|u\|}$) before computing $D_{\hat{u}} f = \nabla f \cdot \hat{u}$!

### 3.5 Geometric Meaning of the Gradient
Using the **Cauchy-Schwarz Inequality**:
$$|a^T b| \le \|a\| \|b\| \implies -\|\nabla f(x)\| \le \nabla f(x)^T \hat{u} \le \|\nabla f(x)\|$$
(since $\|\hat{u}\| = 1$). Equality holds when $\hat{u}$ is parallel to $\nabla f(x)$. This gives three crucial geometric insights:
1. **Direction of Steepest Ascent:** $D_{\hat{u}} f(x)$ is maximized when $\hat{u} = \frac{\nabla f(x)}{\|\nabla f(x)\|}$. Thus, **$\nabla f(x)$ points in the direction of steepest increase** of $f$, and $\|\nabla f(x)\|$ is that maximum rate of increase.
2. **Descent Directions:** A direction $u$ is a descent direction (where $f$ decreases) if $\nabla f(x)^T u < 0$. The **direction of steepest descent** is $-\nabla f(x)$.
3. **Orthogonal to Contours:** Moving along a contour (level curve where $f(x) = c$) produces zero change in $f$, so $\nabla f(x)^T u = 0$. Therefore, **the gradient $\nabla f(x)$ is always perpendicular (normal) to the contour lines** passing through $x$.

### 3.6 Higher-Order Multivariate Approximation & The Hessian
Including the second-order Taylor term for $f: \mathbb{R}^d \to \mathbb{R}$ around $a$:
$$f(x) \approx f(a) + \nabla f(a)^T (x - a) + \frac{1}{2} (x - a)^T \mathbf{H}_f(a) (x - a)$$
where $\mathbf{H}_f(a) = \nabla^2 f(a) \in \mathbb{R}^{d \times d}$ is the **Hessian Matrix** of second-order partial derivatives:
$$\big(\nabla^2 f(x)\big)_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}(x)$$
* For smooth functions, the Hessian is symmetric: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$.
* **Critical Point:** $x^*$ is a critical point if $\nabla f(x^*) = \mathbf{0}$. (Whether it's a min, max, or saddle point depends on the eigenvalues / definiteness of the Hessian $\nabla^2 f(x^*)$, which we study in Weeks 6 and 8!)
