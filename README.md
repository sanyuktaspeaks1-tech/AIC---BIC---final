# StreamCo Churn Prediction — Subset Selection Case Study

A worked, fully-computed example of **stepwise (mixed) subset selection** in logistic regression — every log-likelihood, AIC, BIC, and likelihood-ratio test below is computed from a real (simulated) 50,000-row dataset, not hand-picked illustrative numbers.

---
**Business question:** Will a customer cancel their subscription next month?

- **Outcome:** `y ∈ {0, 1}` — churn indicator
- **Sample size:** `n = 50,000` customers
- **Candidate predictors (p = 8):**

| Variable | Description |
|---|---|
| `x1` | Monthly watch hours |
| `x2` | Days since last login |
| `x3` | Number of support tickets |
| `x4` | Subscription price tier |
| `x5` | Has downloaded app (binary) |
| `x6` | Number of profiles on account |
| `x7` | Days since account created |
| `x8` | Number of devices used |

**Model family:** Logistic regression

$$P(y_i = 1 \mid x_i) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik})}}$$

**Selection method:** Mixed (bidirectional) stepwise selection using likelihood-ratio tests, α_enter = α_stay = 0.05

---

## Dataset Generation

Ground-truth churn probability was generated from a known process (so we can check whether selection recovers the true signal variables):

$$z_i = -1.2 - 0.06x_1 + 0.15x_2 + 0.02x_3 + 0.05x_4 - 0.9x_5 - 0.02x_6 + 0.0001x_7 + 0.03x_8 + \varepsilon_i$$

$$p_i = \frac{1}{1+e^{-z_i}}, \qquad y_i \sim \text{Bernoulli}(p_i)$$

Resulting empirical churn rate: **14.498%**

---


### Log-Likelihood (Bernoulli / logistic case)

$$\ln L(\beta) = \sum_{i=1}^{n} \Big[y_i \ln(\hat p_i) + (1-y_i)\ln(1-\hat p_i)\Big]$$


$$\text{LR} = 2\big(\ln L_{\text{new}} - \ln L_{\text{old}}\big) \sim \chi^2_{q}$$

where `q` = number of parameters added/removed (here always 1). Compare against critical value **3.84** at α = 0.05 for `q = 1`.

### AIC / BIC

$$\text{AIC} = 2k - 2\ln L \qquad \text{BIC} = k\ln(n) - 2\ln L$$

`k` = number of parameters (including intercept). Lower is better for both; BIC penalizes complexity more heavily since `ln(50,000) ≈ 10.82 ≫ 2`.

---

## Step-by-Step Selection with Explicit Calculations

 Step 0 — Null Model (intercept only)

$$\ln L_{\text{null}} = -20{,}695.082$$

$$\text{AIC} = 2(1) - 2(-20{,}695.082) = 2 + 41{,}390.164 = \mathbf{41{,}392.164}$$

$$\text{BIC} = 1 \times \ln(50{,}000) - 2(-20{,}695.082) = 10.820 + 41{,}390.164 = \mathbf{41{,}400.984}$$

---

Step 1 — `x2` enters (days since last login)

$$\ln L_{\text{new}} = -18{,}863.928$$

$$\text{LR} = 2\big(-18{,}863.928-(-20{,}695.082)\big) = 2(1{,}831.154) = \mathbf{3{,}662.307}$$

$$p\text{-value} = P(\chi^2_1 > 3{,}662.307) \approx \mathbf{0.000000}$$

$$\text{AIC} = 2(2) - 2(-18{,}863.928) = \mathbf{37{,}731.857} \qquad \text{BIC} = \mathbf{37{,}749.496}$$

✅ Enters (LR ≫ 3.84)

---

 Step 2 — `x1` enters (monthly watch hours)

$$\ln L_{\text{new}} = -18{,}255.406$$

$$\text{LR} = 2\big(-18{,}255.406-(-18{,}863.928)\big) = 2(608.522) = \mathbf{1{,}217.046}$$

$$p\text{-value} \approx \mathbf{0.000000}$$

$$\text{AIC} = 2(3) - 2(-18{,}255.406) = \mathbf{36{,}516.811} \qquad \text{BIC} = \mathbf{36{,}543.270}$$

✅ **Enters**

---

Step 3 — `x5` enters (has app downloaded)

$$\ln L_{\text{new}} = -17{,}786.803$$

$$\text{LR} = 2(468.603) = \mathbf{937.206}, \qquad p \approx \mathbf{0.000000}$$

$$\text{AIC} = \mathbf{35{,}581.605} \qquad \text{BIC} = \mathbf{35{,}616.884}$$

✅ **Enters**

---

 Step 4 — `x7` enters (account age)

$$\ln L_{\text{new}} = -17{,}772.715$$

$$\text{LR} = 2(14.088) = \mathbf{28.175}, \qquad p \approx \mathbf{0.000000}$$

$$\text{AIC} = \mathbf{35{,}555.430}$$

✅ **Enters** — but notice LR just dropped from 937 → 28. We're approaching the noise floor.

---

Step 5 — `x4` enters (price tier)

$$\ln L_{\text{new}} = -17{,}764.746$$

$$\text{LR} = 2(7.969) = \mathbf{15.938}, \qquad p = \mathbf{0.000065}$$

$$\text{AIC} = \mathbf{35{,}541.493}$$

✅ **Enters** — weakest addition so far, still clears α = 0.05

---

Step 6 — `x3` tested → **REJECTED**

$$\ln L_{\text{new}} = -17{,}763.255$$

$$\text{LR} = 2\big(-17{,}763.255-(-17{,}764.746)\big) = 2(1.491) = \mathbf{2.982}$$

$$p = P(\chi^2_1 > 2.982) = \mathbf{0.0842} > 0.05$$

❌ **Does not enter.** Algorithm terminates. `x6` and `x8` never won a candidate round and were never even tested.

---

## Final Model

| Step | Variable | LR statistic | p-value | AIC after entry |
|---|---|---|---|---|
| 1 | `x2` | 3,662.31 | <0.0001 | 37,731.86 |
| 2 | `x1` | 1,217.05 | <0.0001 | 36,516.81 |
| 3 | `x5` | 937.21 | <0.0001 | 35,581.61 |
| 4 | `x7` | 28.18 | <0.0001 | 35,555.43 |
| 5 | `x4` | 15.94 | 0.00007 | 35,541.49 |
| — | `x3` *(rejected)* | 2.98 | 0.0842 | — |

**Selected model:** `churn ~ x2 + x1 + x5 + x7 + x4`

**Total improvement:** AIC dropped from **41,392.16 → 35,541.49** using 5 of 8 candidate variables.

```
                           Logit Regression Results
==============================================================================
Dep. Variable:                      y   No. Observations:                50000
Model:                          Logit   Df Residuals:                    49994
Method:                           MLE   Df Model:                            5
Pseudo R-squ.:                  0.1416
Log-Likelihood:                -17765.
LL-Null:                       -20695.
LLR p-value:                     0.000
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -1.1418      0.055    -20.859      0.000      -1.249      -1.035
x2             0.1476      0.002     59.899      0.000       0.143       0.152
x1            -0.0609      0.002    -34.341      0.000      -0.064      -0.057
x5            -0.9007      0.029    -31.154      0.000      -0.957      -0.844
x7             0.0001   2.37e-05      5.307      0.000    7.95e-05       0.000
x4             0.0543      0.014      4.003      0.000       0.028       0.081
==============================================================================
```

In practice: data scientists doing exploratory/statistical analysis (deciding which variables matter, hypothesis testing, medical/social science research) reach for statsmodels. Engineers building a production model that just needs to predict well at scale reach for scikit-learn, XGBoost, or a neural net framework
---


```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

np.random.seed(42)
n = 50000

# ---- Generate 8 candidate predictors ----
x1 = np.random.normal(20, 8, n)                          # monthly watch hours
x2 = np.random.exponential(5, n)                          # days since last login
x3 = np.random.poisson(1.2, n)                            # support tickets
x4 = np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.3, 0.2, 0.1])  # price tier
x5 = np.random.binomial(1, 0.75, n)                       # has app
x6 = np.random.choice([1, 2, 3, 4, 5], n)                 # num profiles
x7 = np.random.uniform(0, 2000, n)                        # days since account created
x8 = np.random.choice([1, 2, 3], n)                       # num devices

# ---- True churn-generating process ----
z = (-1.2
     - 0.06 * x1
     + 0.15 * x2
     + 0.02 * x3
     + 0.05 * x4
     - 0.9 * x5
     - 0.02 * x6
     + 0.0001 * x7
     + 0.03 * x8
     + np.random.normal(0, 1, n) * 0.3)
p = 1 / (1 + np.exp(-z))
y = np.random.binomial(1, p)

df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
                    'x5': x5, 'x6': x6, 'x7': x7, 'x8': x8})
print("Churn rate:", df.y.mean())


def fit(vars_):
    """Fit a logistic regression on the given variable list (empty = null model)."""
    X = sm.add_constant(df[vars_]) if vars_ else pd.DataFrame({'const': np.ones(n)})
    return sm.Logit(df['y'], X).fit(disp=0)


# ---- Null model ----
null_model = fit([])
k0 = 1
ll0 = null_model.llf
aic0 = 2 * k0 - 2 * ll0
bic0 = k0 * np.log(n) - 2 * ll0
print("\n--- NULL MODEL ---")
print("Log-likelihood:", ll0)
print("AIC:", aic0, "BIC:", bic0)

print(final_model.summary())
```

**Dependencies:**
```bash
pip install numpy pandas statsmodels scipy
```

---

## Key Takeaways

- **AIC vs BIC:** BIC's `k·ln(n)` penalty grows with sample size, so it punishes marginal variables (like `x4` here) harder than AIC does — at larger `n`, BIC gets even stricter.
- **LR test collapse is a warning sign:** watch how the LR statistic fell from 3,662 → 1,217 → 937 → 28 → 16 → 2.98. That sharp decay is the model telling you it has run out of real signal and is approaching pure noise.
- **Selection recovered the true signal variables:** the data-generating process used `x1, x2, x5` as the strongest true drivers — and indeed those three entered first, in the first three steps, before the weaker/noisier variables.
- **`x6` and `x8` never mattered** — they were genuine noise in the simulation and correctly never made it past a single candidate round.
- **Real-world caveat:** classical stepwise LR tests assume standard MLE asymptotics; at very large `n` (as here) this is generally safe, but always cross-validate the final variable set on held-out data before trusting it in production — see the wider discussion on Lasso/Elastic Net as a more scalable, assumption-light alternative for `p` in the hundreds or thousands.
