# Week 11: Probability — Discrete & Continuous Random Variables

> **Course:** Machine Learning Foundations (IIT Madras BS Degree)  
> **Topics Covered:** Probability Space & Axioms, Conditional Probability & Bayes' Rule, Independence, Discrete Random Variables (PMF, CDF, Expectation, Variance, Covariance), Standard Discrete Distributions (Bernoulli, Binomial, Poisson, Geometric), Continuous Random Variables (PDF, CDF, Joint/Marginal/Conditional PDFs, Convolution, Transformations), Standard Continuous Distributions (Uniform, Exponential, Normal/Gaussian)

---

## 1. Probability Space & Conditioning

### 1.1 The Probability Triple $(\Omega, \mathcal{F}, P)$
1. **Sample Space ($\Omega$):** Set of all elementary outcomes of an experiment (e.g., $\Omega = \{H, T\}$ for a coin toss; $\Omega = \{1, 2, 3, 4, 5, 6\}$ for a die roll).
2. **Events ($\mathcal{F}$):** A collection of subsets of $\Omega$ (forming a $\sigma$-algebra: closed under complements and countable unions, containing $\emptyset$ and $\Omega$).
3. **Probability Measure ($P : \mathcal{F} \to [0, 1]$):** Satisfies Kolmogorov's Axioms:
   * $P(A) \ge 0$ for all $A \in \mathcal{F}$.
   * $P(\Omega) = 1$.
   * For mutually disjoint events $A_1, A_2, \dots$: $P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i)$.

#### Essential Identities:
* $P(A^c) = 1 - P(A)$
* $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ (Inclusion-Exclusion)

### 1.2 Conditional Probability, Total Probability & Bayes' Rule
* **Conditional Probability** (assuming $P(B) > 0$):
  $$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$
* **Independence:** Events $A$ and $B$ are independent iff $P(A \cap B) = P(A)P(B)$ (equivalently, $P(A \mid B) = P(A)$).
* **Law of Total Probability:** If $\{B_1, B_2, \dots, B_k\}$ partition $\Omega$:
  $$P(A) = \sum_{i=1}^k P(A \mid B_i) P(B_i)$$
* **Bayes' Rule:**
  $$P(B_i \mid A) = \frac{P(A \mid B_i) P(B_i)}{\sum_{j=1}^k P(A \mid B_j) P(B_j)}$$

---

## 2. Discrete Random Variables

A **Random Variable (RV)** $X : \Omega \to \mathbb{R}$ maps outcomes in the sample space to real numbers.

### 2.1 PMF and CDF
* **Probability Mass Function (PMF):** $p_X(x) = P(X = x)$, satisfying $p_X(x) \ge 0$ and $\sum_{x} p_X(x) = 1$.
* **Cumulative Distribution Function (CDF):** $F_X(x) = P(X \le x) = \sum_{t \le x} p_X(t)$.
* **Indicator RV:** $\mathbf{1}_A(\omega) = 1$ if $\omega \in A$, else $0$. Note that $\mathbb{E}[\mathbf{1}_A] = P(A)$.

### 2.2 Expectation, Variance, and Covariance
* **Expectation (Mean):** $\mu_X = \mathbb{E}[X] = \sum_{x} x \, p_X(x)$.
  * **LOTUS (Law of the Unconscious Statistician):** $\mathbb{E}[g(X)] = \sum_x g(x) \, p_X(x)$.
  * **Linearity of Expectation (ALWAYS holds, even if dependent!):**
    $$\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c$$
* **Variance:**
  $$\text{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$
  * **Scaling Property:** $\text{Var}(aX + b) = a^2 \text{Var}(X)$.
* **Joint, Marginal & Conditional PMFs:**
  * Joint PMF: $p_{XY}(x, y) = P(X = x, Y = y)$.
  * Marginals: $p_X(x) = \sum_y p_{XY}(x, y)$, $\quad p_Y(y) = \sum_x p_{XY}(x, y)$.
  * Conditional PMF: $p_{X|Y}(x \mid y) = \frac{p_{XY}(x, y)}{p_Y(y)}$.
  * **Independence:** $X \perp Y \iff p_{XY}(x, y) = p_X(x) p_Y(y) \ \forall x, y$.
    * If $X \perp Y$: $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ and $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$.
* **Covariance:**
  $$\text{Cov}(X, Y) = \mathbb{E}\big[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])\big] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$
  * In general: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$.

### 2.3 Standard Discrete Distributions

| Distribution | Notation & Meaning | PMF $p_X(k)$ | Mean $\mathbb{E}[X]$ | Variance $\text{Var}(X)$ |
| :--- | :--- | :--- | :--- | :--- |
| **Bernoulli** | $\text{Bern}(p)$: 1 coin flip ($k \in \{0, 1\}$) | $p^k (1-p)^{1-k}$ | $p$ | $p(1-p)$ |
| **Binomial** | $\text{Bin}(n, p)$: \# successes in $n$ i.i.d. trials ($k \in \{0,\dots,n\}$) | $\binom{n}{k} p^k (1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| **Poisson** | $\text{Pois}(\lambda)$: rare event counts ($k \in \{0, 1, 2, \dots\}$) | $\frac{e^{-\lambda} \lambda^k}{k!}$ | $\lambda$ | $\lambda$ |
| **Geometric** | $\text{Geom}(p)$: trials until 1st success ($k \in \{1, 2, \dots\}$) | $(1-p)^{k-1} p$ | $\frac{1}{p}$ | $\frac{1-p}{p^2}$ |

---

## 3. Continuous Random Variables

### 3.1 PDF and CDF
For a continuous RV $X$, the probability of hitting any exact single point is $0$: $P(X = x) = 0$.
* **Probability Density Function (PDF):** $f_X(x) \ge 0$ such that $\int_{-\infty}^{\infty} f_X(x) \, dx = 1$, and:
  $$P(a \le X \le b) = \int_a^b f_X(x) \, dx$$
* **Cumulative Distribution Function (CDF):**
  $$F_X(x) = P(X \le x) = \int_{-\infty}^x f_X(t) \, dt \quad \implies \quad f_X(x) = \frac{d}{dx} F_X(x)$$
* **Expectation & Variance:**
  $$\mathbb{E}[X] = \int_{-\infty}^{\infty} x f_X(x) \, dx, \qquad \text{Var}(X) = \int_{-\infty}^{\infty} x^2 f_X(x) \, dx - (\mathbb{E}[X])^2$$

### 3.2 Multiple Continuous Random Variables
* **Joint PDF:** $f_{XY}(x, y) \ge 0$ with $\iint_{\mathbb{R}^2} f_{XY}(x, y)\,dx\,dy = 1$.
* **Marginals:** $f_X(x) = \int_{-\infty}^{\infty} f_{XY}(x, y) \, dy$, $\quad f_Y(y) = \int_{-\infty}^{\infty} f_{XY}(x, y) \, dx$.
* **Conditional PDF:** $f_{X|Y}(x \mid y) = \frac{f_{XY}(x, y)}{f_Y(y)}$ (for $f_Y(y) > 0$).
* **Independence:** $X \perp Y \iff f_{XY}(x, y) = f_X(x) f_Y(y)$ everywhere.
* **Sum of Independent RVs ($Z = X + Y$ — Convolution):**
  $$f_Z(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z - x) \, dx$$
* **Transformed RV ($Y = g(X)$):** Find the CDF $F_Y(y) = P(g(X) \le y)$ in terms of $F_X$, then differentiate $f_Y(y) = F_Y'(y)$ (or for strictly monotonic $g$: $f_Y(y) = f_X(g^{-1}(y)) \left|\frac{d}{dy} g^{-1}(y)\right|$).

---

## 4. Standard Continuous Distributions

### 4.1 Uniform Distribution: $X \sim \text{Unif}(a, b)$
Equal density over $[a, b]$:
$$f_X(x) = \begin{cases} \frac{1}{b - a} & \text{if } a \le x \le b \\ 0 & \text{otherwise} \end{cases}$$
$$\mathbb{E}[X] = \frac{a + b}{2}, \qquad \text{Var}(X) = \frac{(b - a)^2}{12}$$

### 4.2 Exponential Distribution: $X \sim \text{Exp}(\lambda)$ ($\lambda > 0$)
Models waiting times between Poisson events:
$$f_X(x) = \begin{cases} \lambda e^{-\lambda x} & \text{if } x \ge 0 \\ 0 & \text{otherwise} \end{cases}, \qquad F_X(x) = 1 - e^{-\lambda x} \ (x \ge 0)$$
$$\mathbb{E}[X] = \frac{1}{\lambda}, \qquad \text{Var}(X) = \frac{1}{\lambda^2}$$
* **Memorylessness Property:** $P(X > a + b \mid X > a) = P(X > b) = e^{-\lambda b}$.
* **Minimum of Independent Exponentials:** If $X \sim \text{Exp}(\lambda_1)$ and $Y \sim \text{Exp}(\lambda_2)$ are independent, then $\min(X, Y) \sim \text{Exp}(\lambda_1 + \lambda_2)$.

### 4.3 Normal (Gaussian) Distribution: $X \sim \mathcal{N}(\mu, \sigma^2)$
The bell curve with mean $\mu$ and variance $\sigma^2$:
$$f_X(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \qquad x \in \mathbb{R}$$
* **Standard Normal ($Z \sim \mathcal{N}(0, 1)$):** $f_Z(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$, with $\mathbb{E}[Z] = 0$ and $\text{Var}(Z) = 1$.
* **Standardization:** If $X \sim \mathcal{N}(\mu, \sigma^2)$, then $Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$ (and conversely $X = \mu + \sigma Z$).
* **Gaussian Integral Fact:** $\int_{-\infty}^{\infty} e^{-x^2/2}\,dx = \sqrt{2\pi}$ (proved by squaring the integral and converting to polar coordinates $r, \theta$).
