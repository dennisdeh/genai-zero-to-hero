## Model Explainability
A model with good test metrics can still be right for the wrong reasons: it may rely on a leaky feature, on an
artefact of data collection, or behave implausibly for some customers. Explainability methods help to debug models,
to build trust with stakeholders, to meet regulatory requirements (e.g. explaining an individual credit decision) and
to generate hypotheses about the domain. It is useful to distinguish

- **global** explanations - which features matter overall and how do they shape the prediction on average - from
  **local** explanations - why did the model make *this* prediction;
- **model-specific** methods (coefficients of a linear model, TreeSHAP for trees) from **model-agnostic** methods
  (permutation importance, partial dependence, KernelSHAP), which only need the ability to call `predict`.

An explanation always explains the **model**, not the real world. Whether a pattern is causal is a separate
question that observational data alone cannot answer.

Both scripts in this chapter use the synthetic churn data from chapter c01 (with the categorical columns as integer
codes, e.g. contract 0 = month-to-month, 1 = one-year, 2 = two-year). Since we know the true data-generating process,
we can check every explanation against the truth: `total_charges`, `num_products` and `noise` have no effect at all,
churn decreases with tenure (log-shaped), the age effect is U-shaped, the price effect is four times stronger for
month-to-month contracts, and fibre customers with three or more support calls get an extra jump in churn risk.

- `main_1_shap.py`: SHAP values for a LightGBM model - bar and beeswarm summary plots, dependence plots, SHAP
  interaction values and a waterfall plot for one customer.
- `main_2_permutation_pdp.py`: impurity vs permutation importance, correlated features, partial dependence and ICE.

### Shapley values
SHAP (Lundberg & Lee, 2017) transfers the **Shapley value** from cooperative game theory to model predictions. The
"players" are the features, and the "payout" to be distributed is the difference between the prediction for $x$ and
the average prediction. For a coalition $S\subseteq\{1,\dots,p\}$ of features let $v(S)$ be the expected model
output when only the features in $S$ are known. The Shapley value of feature $j$ is its average marginal
contribution over all orders in which features can be added:
$$\phi_j = \sum_{S\subseteq \{1,\dots,p\}\setminus\{j\}} \frac{|S|!\,(p-|S|-1)!}{p!}\,\left[v(S\cup\{j\}) - v(S)\right].$$
It is the unique attribution satisfying four axioms: *efficiency* (the attributions add up to the prediction),
*symmetry* (features with identical contributions get identical values), *dummy* (a feature that never changes the
output gets zero) and *additivity* (for a sum of models the values add up). Efficiency gives the **local accuracy**
property used everywhere in SHAP:
$$f(x) = \phi_0 + \sum_{j=1}^{p}\phi_j(x), \qquad \phi_0 = \mathbb{E}[f(X)].$$
For classifiers, SHAP usually explains the raw output (log-odds), where effects are additive; in probability space
they are not.

Exact computation needs $2^p$ coalitions. **TreeSHAP** (Lundberg et al., 2020) exploits the tree structure to
compute exact Shapley values for tree ensembles in polynomial time, $O(TLD^2)$ per sample for $T$ trees with $L$
leaves and depth $D$ - explaining 2,000 customers takes well under a second. By default it estimates $v(S)$ with the
*path-dependent* approach (using the training-data cover stored in the trees); with a background dataset it can
instead compute *interventional* values that break feature dependencies. KernelSHAP and other model-agnostic
estimators work for any model but are much slower and approximate.

### SHAP in practice
`main_1_shap.py` trains LightGBM with early stopping, creates `shap.TreeExplainer(model)` and computes an
`Explanation` object for 2,000 test customers. It first verifies local accuracy: base value plus the sum of the SHAP
values reproduces the model's log-odds to $10^{-14}$. Then:

- **Global importance** - mean $|\phi_j|$ per feature (`outputs/shap_bar.png`). The contract type dominates; the
  three features without a true effect come last (mean $|\phi|$: total_charges 0.06, noise 0.03, num_products 0.02,
  versus 0.27 to 0.80 for the real drivers).
- **Beeswarm plot** (`outputs/shap_beeswarm.png`) - one dot per customer and feature, positioned by its SHAP value and
  coloured by the feature value. It shows the *direction* as well: long tenure (red) pushes churn down, many
  support calls push it up, and for age both ends are red on the positive side - the U-shape.
- **Dependence plots** (`outputs/shap_dependence_*.png`) - SHAP value of one feature against its value, coloured by
  another feature. The plot for `support_calls` shows the saturating effect with a jump at three calls, larger for
  fibre customers; the plot for `age` shows the U-shape; the one for `monthly_charges` coloured by contract shows the
  steeper price effect for month-to-month contracts.
- **SHAP interaction values** - a matrix $\Phi_{ij}$ per sample whose diagonal holds the main effects and whose
  off-diagonal entries split each pairwise interaction equally between the two features. The strongest pairs found
  are contract $\times$ monthly_charges and (third) has_fibre $\times$ support_calls - the two true interactions.
  The printed table shows that for $\ge 3$ calls the interaction term is positive for fibre customers and negative
  for the others. A Shapley value of a single feature always contains half of its interactions, which is why the
  dependence plot alone does not separate main effect and interaction.
- **Waterfall plot** (`outputs/shap_waterfall_highest_risk.png`) - a local explanation of the highest-risk customer:
  starting at the base value $\mathbb{E}[f(X)] \approx -1.19$ log-odds (about 23%), five support calls add $+1.35$,
  the month-to-month contract $+0.80$, age 73 $+0.68$, fibre $+0.49$ and the price $+0.43$, ending at $f(x)=2.94$
  log-odds, a churn probability of 95%. Explanations like this can be shown to a customer-service agent next to the
  score.

### Permutation importance
**Permutation importance** (Breiman, 2001; Fisher et al., 2019) measures how much a metric drops when the values of
one feature are randomly shuffled, breaking its relation with the target while keeping its distribution:
$$\text{PI}_j = s(f, X, y) - \frac{1}{K}\sum_{k=1}^K s\left(f, X^{(\pi_k, j)}, y\right).$$
It is model-agnostic, measured in units of the metric (here: drop in ROC-AUC) and - when computed on **held-out**
data - reflects what the model has learned that *generalises*. `sklearn.inspection.permutation_importance` repeats the
shuffling `n_repeats` times to give a mean and standard deviation.

#### Pitfall 1: impurity-based importance is biased
The `feature_importances_` attribute of scikit-learn tree ensembles is the **mean decrease in impurity (MDI)**: the
total impurity reduction of all splits on a feature, computed on the training data. It has two well-known
problems: it favours continuous and high-cardinality features, which offer many candidate split points, and it
rewards splits that only fit training noise. In `main_2_permutation_pdp.py` a fully grown random forest gives the
pure-noise column an MDI of 0.12, ranking it **4th of 10** - above real drivers such as tenure, contract type and
support calls - and ranks `total_charges` (no true effect) second. Permutation importance on the test set ranks
`noise` last with an importance of about zero and gives `total_charges` zero as well
(`outputs/importance_mdi_vs_permutation.png`). Gain-based importances of XGBoost/LightGBM suffer from the same
bias. Prefer permutation importance on held-out data or mean $|$SHAP$|$.

#### Pitfall 2: correlated features share importance
When we add `tenure_copy`, a near-duplicate of `tenure_months` (correlation $\approx$ 0.999), the model is as good as
before (ROC-AUC 0.846 in both cases), but the permutation importance of tenure drops from 0.040 to 0.006, with 0.018
going to the copy: when one copy is shuffled, the model can still use the other. The information did not become
less useful, it is just spread over two columns, and *both* now look unimportant. Shuffling a feature that is
correlated with others also creates unrealistic data points (e.g. a 2-month customer with 5,000 in total charges),
on which the model is evaluated far outside the data it was trained on. Remedies: inspect the correlation structure
first, permute groups of correlated features together, cluster features and keep one representative, or use
conditional importance methods. SHAP values are affected in a similar way (credit is split between correlated
features, and interventional SHAP also evaluates unrealistic combinations).

### Partial dependence and ICE
The **partial dependence** function of a feature $x_S$ is the average prediction when $x_S$ is set to a value and
all other features $x_C$ keep their observed values:
$$\hat f_S(x_S) = \frac{1}{n}\sum_{i=1}^n f\left(x_S, x_C^{(i)}\right).$$
The PDP plots this average curve. **Individual conditional expectation (ICE)** curves plot $f(x_S, x_C^{(i)})$ for
each sample $i$ separately; the PDP is their mean. If ICE curves are parallel the feature acts additively; crossing
or fanning curves reveal interactions. *Centred* ICE curves (`centered=True`) start every curve at zero, which makes
differences in shape visible. `outputs/pdp_ice.png` shows the decreasing tenure effect, the increasing price effect
with a widening fan of ICE curves (the contract interaction), the U-shaped age effect and the saturating effect of
support calls - all on the log-odds scale (`response_method="decision_function"`; the default would show
probabilities, on which additive effects are distorted by the sigmoid).

To make an interaction explicit, the script computes the PDP of `monthly_charges` separately within each contract
group and of `support_calls` within fibre/non-fibre customers (`outputs/pdp_interactions.png`). The fitted slopes
(0.043, 0.018 and 0.010 log-odds per unit of price) are close to the true values (0.04, 0.01, 0.01).

Pitfalls of PDPs:

- **Correlated features / extrapolation.** The PDP evaluates the model on combinations that may never occur (a
  72-month tenure with a total charge of 100). Accumulated local effects (ALE) plots avoid this by averaging local
  differences within the data distribution.
- **Averages hide heterogeneity** - always look at ICE curves as well.
- **`method="recursion"` vs `"brute"`.** For tree models scikit-learn computes PDPs by default with a fast
  recursion over the trees, which uses the *training* distribution stored in the trees and ignores the data you
  pass. A PDP on a subset of customers (as in the interaction plots) therefore requires `method="brute"` - with the
  default, all three contract groups would have shown the same slope.

### Which method when?
| question | method |
|---|---|
| which features does the model rely on (globally)? | permutation importance on held-out data, mean $\lvert$SHAP$\rvert$ |
| how does the prediction change with a feature? | PDP + ICE, SHAP dependence plot, ALE |
| are there interactions? | ICE fans, group-wise PDPs, SHAP interaction values |
| why this individual prediction? | SHAP waterfall / force plot |
| linear model | standardised coefficients (but beware of correlated features) |

Explanations are diagnostic tools. They are only as reliable as the model and the data, and they should be
sanity-checked - for example, as here, by adding a feature that is known to be pure noise.
